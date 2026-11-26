#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_

#include "lane-sync.h"
#include "common/assert.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <algorithm>
#include <atomic>
#include <compare>
#include <immintrin.h>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
struct ImageTestAccess;
namespace Spec {
struct ImageEntry;
} // namespace Spec

using ImageId = Common::SlotId;

struct CachedImageView {
	ImageViewInfo info;
	vk::ImageView view = nullptr;
};

struct ImageUsage {
	bool texture       = false;
	bool storage       = false;
	bool render_target = false;
	bool depth_target  = false;
	bool video_out     = false;
};

struct ImageBinding {
	vk::ImageLayout  attachment_layout = vk::ImageLayout::eUndefined;
	vk::AccessFlags2 attachment_access;
	bool             is_bound      = false;
	bool             is_target     = false;
	bool             needs_rebind  = false;
	bool             force_general = false;
	bool             shader_write  = false;
	vk::ImageAspectFlags pixel_sampled_aspects;
	vk::ImageAspectFlags other_sampled_aspects;
};

// Scopes a group of transitions that all precede the same recorded command. No
// command executes between transitions in one group, so repeating an identical
// transition of the same image inside a group cannot be covering a hazard.
// Transitions recorded outside any group are never treated as repeats, which is
// what keeps the copy and upload paths correct.
void BeginTransitGroup() noexcept;
void EndTransitGroup() noexcept;

// Moves whenever an image's barrier state (layout, access) is set outside a speculation's own view: what was found
// in place holds while it does not.
[[nodiscard]] uint64_t ImageStateEpoch() noexcept;
void                   MoveImageStateEpoch() noexcept;

class TransitGroup final {
public:
	TransitGroup() noexcept { BeginTransitGroup(); }
	~TransitGroup() { EndTransitGroup(); }
	KYTY_CLASS_NO_COPY(TransitGroup);
};

class Image final {
public:
	// `may_fail`: an image the GPU cannot make (its memory full, more layers or bytes than the format allows) is left
	// without one (CreateFailed) instead of ending the emulator: a sampled texture's lookup binds a null image instead.
	// One that must exist is tried past those limits, and left without one only when its memory could not be had
	// (TextureCache::InsertImage makes room and tries again).
	Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& info, bool may_fail = false);
	~Image();
	KYTY_CLASS_NO_COPY(Image);

	[[nodiscard]] bool CreateFailed() const noexcept { return m_create_failed; }

	[[nodiscard]] vk::ImageView FindView(const ImageViewInfo& view_info);
	// A parked image taken back for `image_info` (TextureCache::TakeParkedImage, the same Vulkan image): the state the
	// constructor gives a new one, but for its Vulkan image, views and barrier state (and serial: the same object).
	void PrepareReuse(const ImageInfo& image_info);
	// (ram1b, KYTY_RAM_NATIVE_TEX) A sparse image (backing.sparse): its layers whose guest slices meet [begin, end) bound
	// while committed, unbound when not; [gone_begin, gone_end) about to be unmapped (GPU thread; nothing for a dense one).
	void SparseUpdate(uint64_t begin, uint64_t end, uint64_t gone_begin = 0, uint64_t gone_end = 0);
	// (NEGRAS-LOG d) KYTY_TEX_NEGRAS_MAGENTA=1: patron visible en los mips nativos recien ligados.
	void NegrasMagentaFill(uint32_t levels);
	[[nodiscard]] bool IsSparse() const noexcept { return backing.sparse != nullptr; }
	using Barriers = std::vector<vk::ImageMemoryBarrier2>;
	[[nodiscard]] const Barriers& GetBarriers(vk::ImageLayout                      destination_layout,
	                                   vk::AccessFlags2                     destination_access,
	                                   vk::PipelineStageFlags2              destination_stage,
	                                   std::optional<ImageSubresourceRange> range);
	void Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
	             std::optional<ImageSubresourceRange> range, vk::CommandBuffer command_buffer);
	// (p7, KYTY_P7_C7) A new image's contents cleared to 0 before its first use that may read them (Transit).
	bool ClearUndefined(vk::CommandBuffer command_buffer);
	// The whole-image barrier state the calling thread's next transition starts from (a speculative translation's
	// own once it used the image: speculation-state.h).
	[[nodiscard]] const VulkanImageState& CurrentState() const;
	// A speculative translation's work uses the image as it is now (its commit puts it there first: a use without a
	// transition needs this too); null outside one.
	Spec::ImageEntry* SpeculativeEntry();
	// The barrier state set to `state` (each subresource's: `subresources`, unless empty), with the barriers there
	// from the image's (a speculative translation's work is committed where it found the image).
	void RestoreState(const VulkanImageState& state, const std::vector<VulkanImageState>& subresources,
	                  vk::CommandBuffer command_buffer);
	void Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	            uint64_t size);
	void Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	              uint64_t size);
	void CopyImage(Image& source);
	void Resolve(Image& source, const ImageSubresourceRange& source_range,
	             const ImageSubresourceRange& destination_range);
	void CopyImageWithBuffer(Image& source, Buffer& buffer);
	void CopyMip(Image& source, uint32_t mip, uint32_t layer);

	// Moves whenever an image becomes (maybe) CPU-dirty: a proof that bound images were clean
	// holds while it stays (native XPR records used again within a frame).
	[[nodiscard]] static uint64_t CpuDirtyEpoch() noexcept { return s_cpu_dirty_epoch.load(std::memory_order_acquire); }

	void InvalidateCpuWrite(uint64_t vaddr, uint64_t size) {
		if (ImageRangeOverlaps(info.data.address, info.data.size, vaddr, size)) {
			m_cpu_dirty        = true;
			m_maybe_cpu_dirty  = false;
			m_maybe_hash_valid = false;
			DropPartialDirty();
			NoteCpuDirty();
			++validity;
		} else if (ImagePageRangesOverlap(info.data.address, info.data.size, vaddr, size)) {
			m_maybe_cpu_dirty = true;
			NoteCpuDirty();
			++validity;
		}
	}

	// Partial CPU writes (KYTY_PARTIAL_IMAGE_DIRTY): while the image is dirty only in the
	// recorded guest ranges, it keeps watching its other pages, so a refresh uploads just the
	// subresources over those ranges. A clean image or one already dirty this way can take one.
	[[nodiscard]] bool CanTakePartialDirty() const noexcept {
		return !m_maybe_cpu_dirty && (!m_cpu_dirty || m_partial_dirty);
	}
	void InvalidateCpuWritePartial(uint64_t begin, uint64_t end) {
		if (!CanTakePartialDirty() || begin >= end) {
			EXIT("image cannot take a partial CPU write\n");
		}
		if (!m_cpu_dirty) {
			m_cpu_dirty        = true;
			m_partial_dirty    = true;
			m_maybe_hash_valid = false;
			m_dirty_ranges.clear();
		}
		AddRange(m_dirty_ranges, begin, end);
		NoteCpuDirty();
		++validity;
	}
	[[nodiscard]] bool IsPartiallyCpuDirty() const noexcept { return m_cpu_dirty && m_partial_dirty; }
	[[nodiscard]] const std::vector<std::pair<uint64_t, uint64_t>>& CpuDirtyRanges() const noexcept {
		return m_dirty_ranges;
	}
	// A whole-image CPU-dirty state as the one dirty range it amounts to (a refresh of part of it completes that part:
	// TextureCache::RefreshTextureLevels).
	void NarrowCpuDirty() {
		if (!m_cpu_dirty || m_partial_dirty || m_maybe_cpu_dirty) {
			EXIT("image cannot narrow its dirty state\n");
		}
		m_partial_dirty = true;
		m_dirty_ranges.assign(1, {info.data.address, info.data.End()});
		++validity;
	}
	// The whole image must be uploaded again (the image stopped watching all of its pages).
	void DropPartialDirty() noexcept {
		m_partial_dirty = false;
		m_dirty_ranges.clear();
		++validity;
	}
	// A refresh of [begin, end) only (the layers one binding covers): the other ranges stay dirty.
	void RefreshRangeComplete(uint64_t begin, uint64_t end) {
		if (!IsPartiallyCpuDirty() || m_maybe_cpu_dirty || begin >= end) {
			EXIT("image cannot complete a partial refresh\n");
		}
		SubtractRange(m_dirty_ranges, begin, end);
		if (m_dirty_ranges.empty()) {
			RefreshComplete();
		}
	}
	// Removes [begin, end) from sorted, disjoint ranges.
	static void SubtractRange(std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t begin,
	                          uint64_t end) {
		std::vector<std::pair<uint64_t, uint64_t>> kept;
		kept.reserve(ranges.size() + 1);
		for (const auto& [range_begin, range_end]: ranges) {
			if (range_end <= begin || range_begin >= end) {
				kept.emplace_back(range_begin, range_end);
				continue;
			}
			if (range_begin < begin) kept.emplace_back(range_begin, begin);
			if (end < range_end) kept.emplace_back(end, range_end);
		}
		ranges = std::move(kept);
	}
	// Sorted, disjoint [begin, end) ranges; merges adjacent and overlapping ones.
	static void AddRange(std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t begin,
	                     uint64_t end) {
		auto at = ranges.begin();
		while (at != ranges.end() && at->second < begin) ++at;
		auto last = at;
		while (last != ranges.end() && last->first <= end) {
			begin = std::min(begin, last->first);
			end   = std::max(end, last->second);
			++last;
		}
		at = ranges.erase(at, last);
		ranges.insert(at, {begin, end});
	}

	[[nodiscard]] bool IsCpuDirty() const { return m_cpu_dirty || m_maybe_cpu_dirty; }
	[[nodiscard]] bool IsDefinitelyCpuDirty() const { return m_cpu_dirty; }
	[[nodiscard]] bool IsMaybeCpuDirty() const { return m_maybe_cpu_dirty; }
	void               MarkMaybeCpuDirty() {
		if (!m_cpu_dirty) {
			m_maybe_cpu_dirty = true;
			NoteCpuDirty();
			++validity;
		}
	}
	[[nodiscard]] bool NeedsMaybeCpuHash() const {
		return m_maybe_cpu_dirty && !m_maybe_hash_valid;
	}
	void SetMaybeCpuHash(uint64_t hash) {
		if (!NeedsMaybeCpuHash()) {
			EXIT("image cannot initialize maybe-dirty hash\n");
		}
		m_maybe_cpu_hash   = hash;
		m_maybe_hash_valid = true;
	}
	[[nodiscard]] bool ResolveMaybeCpuHash(uint64_t hash) {
		if (!m_maybe_cpu_dirty || !m_maybe_hash_valid || m_cpu_dirty) {
			EXIT("image cannot resolve maybe-dirty hash\n");
		}
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		m_cpu_dirty |= hash != m_maybe_cpu_hash;
		return m_cpu_dirty;
	}

	void RefreshComplete() {
		if (!IsCpuDirty()) {
			EXIT("clean image cannot complete a refresh\n");
		}
		m_cpu_dirty        = false;
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		DropPartialDirty();
	}

	[[nodiscard]] bool IsGpuModified() const noexcept { return m_gpu_modified; }
	void               MarkGpuModified() noexcept {
		if (!m_gpu_modified) ++validity; // (a texture GPU work writes refreshes whole: TextureCache::LevelRefreshCandidate)
		m_gpu_modified = true;
	}
	void               ClearGpuModified() noexcept {
		m_gpu_modified = false;
		MoveImageStateEpoch();
	}

	[[nodiscard]] bool IsBufferModified() const noexcept { return m_buffer_modified; }
	// Bytes GPU work wrote through a buffer: all of the image (where it is not known), or [begin, end) ranges of it, which
	// TextureCache::InitializeImage uploads alone (BufferModifiedRanges: sorted, merged; empty when all of it).
	void               MarkBufferModified() noexcept {
		m_buffer_modified = true;
		m_buffer_whole    = true;
		m_buffer_ranges.clear();
		m_buffer_fill     = BufferFill::Mixed;
		++validity;
		MoveImageStateEpoch();
	}
	void MarkBufferModified(uint64_t begin, uint64_t end) {
		AddBufferRange(begin, end);
		m_buffer_fill = BufferFill::Mixed;
	}
	// The same from a compute fill of one 32-bit value: while every buffer write since the last upload was a fill of that
	// value and they cover all of the image, BufferFilledWith gives the value (InitializeImage clears to it).
	void MarkBufferFilled(uint64_t begin, uint64_t end, uint32_t value) {
		const bool first = !m_buffer_modified;
		AddBufferRange(begin, end);
		if (first) {
			m_buffer_fill       = BufferFill::Value;
			m_buffer_fill_value = value;
		} else if (m_buffer_fill != BufferFill::Value || m_buffer_fill_value != value) {
			m_buffer_fill = BufferFill::Mixed;
		}
	}
	[[nodiscard]] std::optional<uint32_t> BufferFilledWith() const noexcept {
		if (!m_buffer_modified || m_buffer_fill != BufferFill::Value || m_buffer_whole || m_buffer_ranges.size() != 1 ||
		    m_buffer_ranges.front().first > info.data.address || m_buffer_ranges.front().second < info.data.End())
			return std::nullopt;
		return m_buffer_fill_value;
	}
	[[nodiscard]] const std::vector<std::pair<uint64_t, uint64_t>>& BufferModifiedRanges() const noexcept {
		return m_buffer_ranges;
	}
	void ClearBufferModified() noexcept {
		m_buffer_modified = false;
		m_buffer_whole    = false;
		m_buffer_ranges.clear();
		m_buffer_fill = BufferFill::None;
	}

private:
	void AddBufferRange(uint64_t begin, uint64_t end) {
		if (!m_buffer_modified) {
			m_buffer_whole = false;
			m_buffer_ranges.clear();
		}
		m_buffer_modified = true;
		if (!m_buffer_whole && begin < end) {
			auto at = std::lower_bound(m_buffer_ranges.begin(), m_buffer_ranges.end(), begin,
			                           [](const auto& range, uint64_t value) { return range.second < value; });
			auto last = at;
			while (last != m_buffer_ranges.end() && last->first <= end) {
				begin = std::min(begin, last->first);
				end   = std::max(end, last->second);
				++last;
			}
			at = m_buffer_ranges.erase(at, last);
			m_buffer_ranges.insert(at, {begin, end});
			// (Past a few, all of it: what a whole upload costs. More in an image uploaded a piece at a time anyway (256 MiB
			// and up, TextureCache's ChunkedUploadMin), where the whole is GBs: a texture pool's range over memory the GPU
			// writes in many places, issue #49.)
			if (m_buffer_ranges.size() > (info.data.size >= (uint64_t {256} << 20u) ? 256u : 16u)) {
				m_buffer_whole = true;
				m_buffer_ranges.clear();
			}
		}
		++validity;
		MoveImageStateEpoch();
	}

public:
	[[nodiscard]] bool IsStencilModified() const noexcept { return m_stencil_modified; }
	void               MarkStencilModified() noexcept {
		m_stencil_modified = true;
		++validity;
		MoveImageStateEpoch();
	}
	void               ClearStencilModified() noexcept { m_stencil_modified = false; }

	[[nodiscard]] bool Overlaps(uint64_t address, uint64_t size,
	                            bool pages = false) const noexcept {
		return pages ? ImagePageRangesOverlap(info.data.address, info.data.size, address, size)
		             : ImageRangeOverlaps(info.data.address, info.data.size, address, size);
	}
	[[nodiscard]] bool SafeToDownload() const noexcept {
		return IsGpuModified() && !IsBufferModified() && !IsCpuDirty();
	}
	[[nodiscard]] bool IsTracked() const noexcept { return track_addr != 0 && track_addr_end != 0; }
	[[nodiscard]] uint64_t AccountedSize() const noexcept {
		return backing.image == nullptr ? 0 : (info.data.size + 1023) & ~uint64_t {1023};
	}
	[[nodiscard]] uint64_t HashGuestEdges() const;

	// Unique per image object: a deleted image's slot id goes to later images. (First, with validity: on the line of
	// the slot's liveness, which a lookup reads anyway.)
	uint64_t         serial             = 0;
	// Moves with each change of what a use of the image was proven by (RenderExecutor::TableValidateSet): bytes the CPU
	// wrote or may have written, a GPU buffer or stencil-plane write over them, the first GPU write or use as a target or
	// storage (a texture used so refreshes whole), its metadata, a stencil association. Registrations over its pages are
	// stamped instead (TextureCache::RegistrationsSince). (Atomic: the fault handler's threads move it too.)
	std::atomic<uint64_t> validity {0};
	ImageInfo        info;
	LaneSync::Marks  lane_marks; // KYTY_ASYNC_COMPUTE (lane-sync.h)
	VulkanImage      backing;
	std::vector<CachedImageView> views;
	ImageUsage       usage;
	ImageBinding     binding;
	bool             registered     = false;
	bool             guest_alias    = false; // TextureCache::GuestTexture: another view of its memory is an image too
	mutable uint32_t query_epoch    = 0;
	uint64_t         track_addr     = 0;
	uint64_t         track_addr_end = 0;
	// Page-aligned ranges inside the tracked pages the image stopped watching after partial
	// CPU writes; always inside CpuDirtyRanges while the image is partially dirty.
	std::vector<std::pair<uint64_t, uint64_t>> untracked_holes;
	// KYTY_PARTIAL_IMAGE_DIRTY=2: guest data hashes per partial-dirty granule at the last upload.
	std::vector<uint64_t> partial_hashes;
	ImageId          depth_id {};
	uint64_t         frame_accessed_last = 0; // TextureCache::AdvanceFrame count at the last use
	size_t           lru_id             = 0;
	uint64_t         lru_tick           = 0; // the collection tick of its last use (TextureCache)
	// Transit group that last set the whole-image state; see BeginTransitGroup.
	uint64_t         transit_group      = 0;
	// (p7b, C2) The recording thread's count of writing commands (LocalVulkanRecording::RecordedWrites) when the
	// whole-image state was last set: a repeat in the same transit group is proven redundant only while it is unchanged.
	uint64_t         transit_writes     = UINT64_MAX;
	// The staging copy of the last whole-image upload (null: none or not refillable) and the command buffer it was
	// recorded in (CommandScheduler::CommandSerial): TextureCache::InitializeImage.
	const Buffer*    staged_ring        = nullptr;
	uint64_t         staged_offset      = 0;
	uint64_t         staged_serial      = 0;
	// A colour (or other) image that took over a GPU-rendered layered depth image's memory (the spot-shadow pool
	// recreated for a storage/target use): guarded like the depth array (DepthArrayGuarded).
	bool             depth_array_alias  = false;
	// The barrier state (backing.state, backing.subresource_states, transit_group): changed under this lock, which a
	// speculative translation's thread reads it under (SpeculativeEntry).
	struct StateLock {
		explicit StateLock(const Image& owner) noexcept: image(owner) {
			while (image.state_busy.exchange(true, std::memory_order_acquire)) _mm_pause();
		}
		~StateLock() { image.state_busy.store(false, std::memory_order_release); }
		KYTY_CLASS_NO_COPY(StateLock);
		const Image& image;
	};
	mutable std::atomic<bool> state_busy {false};
	// RegisterImage calls on this object: a proof names one registration.
	uint32_t         registrations      = 0;
	// (NEGRAS-LOG) Origen de su creacion (TexNativa::t_neg_origen al registrarse por primera vez) y mips ya rellenados en
	// magenta (KYTY_TEX_NEGRAS_MAGENTA).
	uint8_t          neg_origen         = 0;
	uint32_t         neg_magenta_levels = 0;
	// TextureCache::ViewLevelBytes's last answer: the view's levels (base | count << 8; UINT32_MAX: none yet) and their
	// bytes [level_begin, level_end) (level_end 0: the view needs the whole image).
	mutable uint32_t level_view         = UINT32_MAX;
	mutable uint64_t level_begin        = 0;
	mutable uint64_t level_end          = 0;
	// (M7, GestorMem) Id del gestor central de memoria (0: sin registrar), tipo y flags comunicados (para no llamar al
	// gestor en cada uso: solo cuando cambian). Los pone TextureCache (RegisterImage / UnregisterImage / Find*).
	uint64_t         gestor_id          = 0;
	int8_t           gestor_tipo        = -1;
	uint8_t          gestor_flags       = 0;

	// (M7, TEX-DEDUP) Dos direcciones del juego con el mismo contenido (hash, formato y forma) comparten la VkImage: la
	// imagen nueva se crea como alias de `SetShareSource` (cuenta de referencias en Share; la memoria se libera con la
	// ultima). El estado de barreras es uno solo para el grupo (PushSharedState lo copia a todos tras cada transicion).
	// Copy-on-write: TextureCache::FindImage borra el miembro que se vaya a escribir (CPU, GPU, bufer, RT o storage) antes
	// de usarlo; se crea otra vez con su propia VkImage. Las paginas de cada direccion las vigila su propia imagen.
	struct Share;
	Share* share       = nullptr;
	bool   share_fresh = false; // recien unida: su contenido ya esta en la VkImage (TextureCache la da por subida)
	[[nodiscard]] bool SharedWithOthers() const noexcept;
	// Hilo que crea la imagen: el siguiente constructor de este hilo intenta ser alias de `source` (nullptr: nada).
	static void SetShareSource(Image* source) noexcept;
	struct DedupStats {
		uint64_t ahora_n = 0, ahora_bytes = 0; // imagenes que ahora usan la VkImage de otra, y sus bytes (VRAM ahorrada)
		uint64_t unidas = 0, unidas_bytes = 0, separadas = 0, escrituras_tarde = 0; // acumulado
	};
	[[nodiscard]] static DedupStats Dedup() noexcept;
	static void NoteDedupSeparated() noexcept;

	void PushSharedState(); // (M7) el estado de barreras de esta imagen a los demas miembros de su grupo

	// (RT-ALIAS-MEM, KYTY_RT_ALIAS_MEM=1) Aliasing de memoria de Vulkan para los render targets que el juego solapa en
	// sus arenas (RTSharedAllocator/RTPersistAllocator: un asignador lineal que se reinicia, Ghidra 0x91c680/0x915930;
	// en PS5 varias superficies viven en los mismos bytes). Una imagen nueva cuyo rango del juego cae entero dentro del
	// de otra viva de la misma arena se crea como VkImage propia ligada (vmaBindImageMemory2) a la VkDeviceMemory de esa
	// otra, en el desplazamiento del juego: sin asignacion propia. Cuando un miembro escribe, los que solapan su rango de
	// memoria quedan "pisados": su siguiente uso parte de VK_IMAGE_LAYOUT_UNDEFINED con dependencia completa (lo que
	// pide Vulkan para memoria con alias). Si un miembro pisado se LEE antes de escribirse (el juego aun necesitaba su
	// contenido), se marca malo: TextureCache lo recrea con memoria propia y ese rango no vuelve a compartir.
	struct MemAlias;
	std::atomic<MemAlias*> mem_alias {nullptr};
	uint64_t          mem_alias_off   = 0; // desplazamiento y tamano en la VkDeviceMemory del grupo
	uint64_t          mem_alias_size  = 0;
	std::atomic<bool> mem_alias_pisada {false};
	std::atomic<bool> mem_alias_mala {false};
	// Creada con memoria que admite alias (VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT: sin VkMemoryDedicatedAllocateInfo, que
	// ataria la memoria a una sola VkImage): solo estas pueden prestar su memoria a otras.
	bool              mem_alias_capable = false;
	// Hilo que crea la imagen: la siguiente imagen de este hilo intenta ligarse a la memoria de `host` en `offset` (host
	// nullptr: ninguna); `capable`: si no se liga, se crea con memoria que admite alias (RT de las arenas del juego).
	// `for_address`: solo la imagen de esa direccion del juego lo toma (otra creada antes en el mismo hilo, no).
	static void SetMemAliasHost(Image* host, uint64_t offset, bool capable, uint64_t for_address) noexcept;
	[[nodiscard]] bool MemAliasGuest() const noexcept; // ligada a la memoria de otra (no tiene asignacion propia)
	[[nodiscard]] bool MemAliasBad() const noexcept { return mem_alias_mala.load(std::memory_order_relaxed); }
	[[nodiscard]] static bool MemAliasBlocked(uint64_t address, uint64_t size);
	struct MemAliasStats {
		uint64_t ahora_n = 0, ahora_bytes = 0;                        // imagenes sin memoria propia ahora, y sus bytes
		uint64_t unidas = 0, rechazadas = 0, reinicios = 0, malas = 0; // acumulado
	};
	[[nodiscard]] static MemAliasStats MemAliasCounters() noexcept;

private:
	friend struct ImageTestAccess;
	bool TryJoinShare(Image& source, const vk::ImageCreateInfo& create);
	bool TryJoinMemAlias(Image& host, uint64_t offset, const vk::ImageCreateInfo& create);
	bool CreateAliasCapable(const vk::ImageCreateInfo& create);
	void FillMemAliasBacking(const vk::ImageCreateInfo& create, vk::Image image, VmaAllocation allocation);
	void MemAliasBeforeUse(vk::AccessFlags2 destination_access, VulkanImageState* spec_state,
	                       std::vector<VulkanImageState>* spec_subresources);
	void NoteLateWrite() noexcept;

	[[nodiscard]] static vk::ImageAspectFlags FullAspectMask(vk::Format format) noexcept;
	[[nodiscard]] static uint32_t             CopyRows(uint64_t row_size, uint32_t rows,
	                                                   uint64_t capacity) noexcept;
	[[nodiscard]] static std::pair<uint32_t, uint32_t>
	SanitizeCopyLayers(const Image& source, const Image& destination, uint32_t depth);

	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;
	bool              m_create_failed    = false;
	uint64_t          m_maybe_cpu_hash   = 0;
	static void NoteCpuDirty() noexcept { s_cpu_dirty_epoch.fetch_add(1, std::memory_order_release); }
	inline static std::atomic<uint64_t> s_cpu_dirty_epoch {1};
	bool              m_cpu_dirty        = false;
	bool              m_maybe_cpu_dirty  = false;
	bool              m_maybe_hash_valid = false;
	bool              m_gpu_modified     = false;
	bool              m_buffer_modified  = false;
	bool              m_buffer_whole     = false; // (with m_buffer_modified: where is not known)
	std::vector<std::pair<uint64_t, uint64_t>> m_buffer_ranges;
	enum class BufferFill : uint8_t { None, Value, Mixed };
	BufferFill        m_buffer_fill       = BufferFill::None;
	uint32_t          m_buffer_fill_value = 0;
	bool              m_stencil_modified  = false;
	bool              m_partial_dirty     = false;
	std::vector<std::pair<uint64_t, uint64_t>> m_dirty_ranges;
};

namespace ImageOps {

void                                 Validate(const ImageInfo& info);
[[nodiscard]] Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element);

} // namespace ImageOps

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
