// --- Native XPR draws --------------------------------------------------------
//
// Included by renderDraw.cpp (local Vulkan recording builds only). The normal
// path, asked to store, leaves two entries behind:
//  * a draw state, keyed by all graphics registers except user data: the render
//    targets, their attachment layouts and the dynamic state;
//  * an object record, keyed by the shader addresses and user data: one
//    descriptor set written once with the object's buffers, images, samplers
//    and fixed renderer buffers; the constants (flattened SRT, shader data,
//    push data); per draw state, the pipeline variant with native_bindings -
//    the same SPIR-V, the flattened-SRT and shader-data bindings dynamic storage
//    buffers in the shader upload ring; and what proves the record current.
// A native draw copies the constants into the upload ring, binds the set with
// the ring offsets as dynamic offsets, pushes the push data and draws from
// GPU-side indirect arguments.
//
// Runtime (KYTY_NATIVE_XPR=1): the command processor offers every
// DRAW_INDEX_INDIRECT run (consecutive packets differing only in the argument
// offset share one object). A record is proven current once per frame:
//  * every guest word its SRT evaluation read still has the logged value, read
//    the way the evaluation reads it; words learned to land only in the
//    flattened SRT (from an earlier re-store of the same key) are patched;
//  * the guest mapping is unchanged and no guest readback is pending;
//  * every buffer is still the same live game buffer and CPU-clean (proved by
//    the tracker's region epochs, checked precisely when they moved);
//  * every sampled image is still registered with the same backing and is not
//    CPU-dirty.
// Per draw only the live-object epochs are compared. Anything else - a missing
// record, state or pipeline, a failed proof - falls back to the normal path,
// which stores again (reads logged through ShaderReadObserver).
//
// Verification (KYTY_NATIVE_XPR=2): a draw the native path would emit takes the
// normal path instead, which compares what it prepared with the record
// (NativeXprVerify). The offline replay (xpr_replay_tests --native) checks the
// emission pixel by pixel against the normal path.

extern "C" {
// 0 off, 1 native XPR draws, 2 verification.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_native_xpr_mode {0};
// Negative controls for the offline replay only: bit 0 zeroes the vertex
// stage's flattened SRT in the upload, bit 1 skips the draw call.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_native_xpr_debug {0};
// Diagnosis of a glitch that native XPR causes even in verification (KYTY_NATIVE_XPR_DIAG): bit 0 stores no
// record (the normal path draws everything, nothing is logged or bound for a record), bit 1 leaves CPU-dirty SRT
// buffers to the upload ring when a record is stored (no EnsureBufferContents/ObtainBuffer cleaning of their owner),
// bit 2 validates no record (a found record takes the normal path before NativeXprValidate), bit 3 tries no native
// draw at all (NativeXprTry refuses at once), bit 4 leaves the command processor as with native XPR off (no packet
// observation, no tries).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_native_xpr_diag {0};
// 1: objects are drawn in last frame's order, so the record that followed this
// one last frame is prefetched while this draw runs.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_native_xpr_predict_mode {0};
// 1: native image proofs check the image's own registration (see NativeXprValidateResources).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_native_image_proof_mode {0};
// 1: a store whose pipeline variant is not compiled yet hands it to the compile workers and
// is skipped; the draw takes the normal path until a later store finds the variant ready.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_async_xpr_pipelines_mode {0};
// 1: relocation. Objects whose user data points at a new copy of the same tables every frame (a ring)
// had keys new every frame, so no record; the pointer words (user data pairs the SRT reads through and
// uses for nothing else) leave the key, and the record reads the same offsets from the new base.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_native_xpr_relocate_mode {0};
// N > 0: at most N store requests a frame. Walking brings hundreds of meshes and LODs
// into view at once; storing them all in one frame (about 16 us each) made it a hitch. The others ask again
// in later frames (their draws take the normal path until then).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_native_xpr_store_budget {0};
// N > 2: records stay N frames unused (2 before: a turn of the camera dropped every record behind it, and
// turning back stored them all again within the store budget, thousands of normal draws a frame for seconds).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_native_xpr_keep_frames {0};
// 1: a relocated key holds the object's descriptor bases (read through the draw's pointers) before its use in
// the frame: the instances of one mesh keep their records when the frame draws them in another order (a culled
// instance shifted every later use onto another instance's record, which rebound its descriptors).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_native_xpr_instance_mode {0};
// 1: table draws (src/local/table-xpr.inc) for the pairs that allow them.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_table_xpr_mode {0};
// 1: table dispatches (src/local/table-xpr.inc) for the compute shaders that allow them.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_table_dispatch_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_table_indirect_mode {1};
// N > 0: at most N table store requests a frame (each takes its draw to the normal path; the others draw from native
// records meanwhile). A new area brings hundreds of pairs whose programs and pipelines compile in the background.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_table_store_budget {32};
}
// Keys that are XXH3 hashes already: their own bucket index (std::hash<uint64_t> is FNV-1a over the bytes on MSVC).
struct HashedKey {
	[[nodiscard]] size_t operator()(uint64_t key) const noexcept { return static_cast<size_t>(key); }
};

namespace {
// word_patch value of a logged word whose change reaches nothing the evaluation produces.
constexpr int32_t NativeXprIgnore = -2;

// T# fields that only feed the LOD statistics (MIN_LOD_WARN, MIP_STATS_CNT_EN,
// MIP_STATS_CNT_ID): the game reassigns the counters every frame. They reach the
// shader data only, never the image resolution or view.
constexpr std::array<uint32_t, 8> NativeXprLodStatsMask {0, 0, 0, 0, 0, (0xfffu << 8u) | (1u << 25u), 0xffu, 0};

// Gather feed kind of a flattened-SRT copy of a structural word (beside the
// SrtReadTrace::Kind descriptor kinds).
constexpr uint32_t NativeXprFlatFeed = 0xf;

// A gather run or span read at a fixed address (not relative to a relocated root).
constexpr uint8_t NativeXprNoRoot = 0xff;

// Longest gather span, in words.
constexpr uint32_t NativeXprSpanWords = 256;

// word_patch values below this encode a T# LOD-statistics dword:
// NativeXprImagePatchBase - (stage << 20 | image << 4 | dword).
constexpr int32_t NativeXprImagePatchBase = -16;

int32_t NativeXprImagePatch(uint32_t stage, uint32_t image, uint32_t dword) {
	return NativeXprImagePatchBase - static_cast<int32_t>(stage << 20u | image << 4u | dword);
}

// The LOD-statistics shader-data words of `images`, as RebindBuffers computes them.
void NativeXprApplyLodStats(const ShaderRecompiler::IR::CompiledShaderInfo&            program,
                            const std::vector<ShaderRecompiler::IR::DescriptorValue>& images,
                            std::vector<uint32_t>&                                     shader_data) {
	const auto& layout = program.bindings;
	for (uint32_t i = 0; i < layout.lod_stats_count && i < images.size(); ++i) {
		const auto desc = DecodeNativeDescriptor<ShaderTextureResource>(images[i]);
		shader_data[layout.LodStatsDword() + i] =
		    desc.MipStatsCntEn() ? (0x80000000u | desc.MipStatsCntId() | (uint32_t(desc.BaseLevel()) << 8u) |
		                            (uint32_t(desc.MinLodWarn5()) << 12u))
		                         : 0u;
	}
}

bool NativeXprSameImagesButLodStats(const std::vector<ShaderRecompiler::IR::DescriptorValue>& a,
                                    const std::vector<ShaderRecompiler::IR::DescriptorValue>& b) {
	if (a.size() != b.size()) return false;
	for (size_t i = 0; i < a.size(); ++i) {
		if (a[i].dword_count != b[i].dword_count) return false;
		for (uint32_t w = 0; w < a[i].dwords.size(); ++w)
			if ((a[i].dwords[w] & ~NativeXprLodStatsMask[w]) != (b[i].dwords[w] & ~NativeXprLodStatsMask[w]))
				return false;
	}
	return true;
}

// Identity of a set's images: the image descriptors without the LOD-statistics fields.
uint64_t NativeXprImageSignature(std::span<const ShaderRecompiler::IR::ResourceSnapshot* const> snapshots) {
	uint64_t hash = 0xcbf29ce484222325ull;
	for (const auto* snapshot: snapshots)
		for (const auto& image: snapshot->images) {
			for (uint32_t w = 0; w < image.dwords.size(); ++w)
				hash = (hash ^ (image.dwords[w] & ~NativeXprLodStatsMask[w])) * 0x100000001b3ull;
			hash = (hash ^ 0xffu) * 0x100000001b3ull;
		}
	return hash;
}

void NativeXprPrefetch(const void* data, size_t bytes) {
	const auto* at = static_cast<const char*>(data);
	for (size_t offset = 0; offset < bytes; offset += 64) __builtin_prefetch(at + offset);
}

template <typename T>
void NativeXprPrefetch(const std::vector<T>& values) {
	NativeXprPrefetch(values.data(), std::min<size_t>(values.size() * sizeof(T), 1024));
}

// The reader ReadShaderRawGuestMemory uses for every SRT word. (A speculative translation's: not memory its own work
// or guest memory writes write, refused; else noted, as read as no GPU work wrote it: speculation-state.h.)
uint32_t NativeXprReadGuestWord(uint64_t address) {
	uint32_t value = 0;
	if (!Libs::LibKernel::Memory::TryReadGpuCleanBackingOnWatchedPage(address, &value, sizeof(value))) {
		if (auto* spec = Spec::Current()) {
			// (Its processor's pending writes as they leave it; not what its own work writes. A speculation's thread
			// does not read what the GPU wrote: only the GPU thread's read fault reads it back.)
			if (!GuestGpu::IsGpuThread() && Libs::LibKernel::Memory::ReadFaults(address, sizeof(value)))
				return Spec::Refuse("GPU-written guest memory"), 0u;
			if (!spec->ReadJournaled(address, 1, &value)) return Spec::Refuse("memory the processor writes"), 0u;
			return value;
		}
		std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
	}
	return value;
}

// The way the SRT evaluation reads a range: spans through the watched-page aware
// span reader, single words otherwise.
void NativeXprReadWords(uint64_t address, uint32_t count, uint32_t* out) {
	for (uint32_t done = 0; done < count;) {
		const uint32_t chunk = std::min<uint32_t>(count - done, 16);
		const auto     at    = address + 4ull * done;
		if (chunk < 2 || !Libs::LibKernel::Memory::TryReadGpuShaderSpan(at, out + done, chunk * 4ull, false))
			for (uint32_t i = 0; i < chunk; ++i) out[done + i] = NativeXprReadGuestWord(at + 4ull * i);
		done += chunk;
	}
}

// Retrace's lists mostly come out of address-ordered runs already sorted: the check is one pass. Equal
// elements are identical or merged independently of their order.
template <typename Range, typename Less = std::less<>>
void NativeXprSort(Range& range, Less less = {}) {
	if (!std::ranges::is_sorted(range, less)) std::ranges::sort(range, less);
}
} // namespace

// Draw state, identified by every graphics register except user data: the
// render targets as a draw under that state left them (reopened renderings
// load), their attachment layouts and the dynamic state.
struct RenderExecutor::NativeXprDrawState {
	struct Target {
		ImageId               id;
		vk::ImageLayout       layout = vk::ImageLayout::eUndefined;
		vk::AccessFlags2      access;
		ImageSubresourceRange range;
		uint64_t              metadata = 0; // surface metadata (HTile/DCC) address, 0 none
		uint32_t              slice    = 0;
		uint64_t              serial   = 0; // the image object the attachment views belong to
		ImageInfo             requested;    // what the registers ask for (TextureCache::IsTargetCurrent)
	};
	RenderState                      rendering;       // reopened: loads
	RenderState                      first_rendering; // as the stored draw began it
	std::vector<Target>              targets;
	LocalDrawRecording::DynamicState dynamic;
	uint64_t                         dynamic_hash = 0; // of the dynamic state's bytes: two states compare it first
	ResolvedDynamicState             resolved;         // the dynamic state's Vulkan values (once, at capture)
	// Metadata epoch under which no target had a pending fast clear.
	mutable uint64_t                 meta_epoch = 0;
	// Texture cache resolution epoch under which the targets were what a lookup finds (0: not yet).
	mutable uint64_t                 targets_epoch = 0;
	// The resolution epoch NativeXprTargetsCurrent last found the targets current at (0: not yet): images are made and
	// deleted (a slot reused) only with a registration change, which moves it.
	mutable uint64_t                 targets_checked = 0;
};

// One object: its resources under the compiled programs it was translated with.
struct RenderExecutor::NativeXprRecord {
	struct Stage {
		const ShaderRecompiler::IR::CompiledShaderInfo* program = nullptr;
		std::vector<uint32_t> flattened;   // flattened SRT
		std::vector<uint32_t> shader_data; // push data and/or the shader-data binding
		bool     srt_binding  = false;
		bool     data_binding = false;
		uint32_t push_start   = ShaderRecompiler::IR::PushData::NoStart;
		ShaderRecompiler::IR::ResourceSnapshot snapshot; // structure, for learning
		// SRT evaluation inputs, for re-evaluation when read words changed.
		std::vector<uint32_t> user_data;
		uint64_t              shader_base = 0;
		// What the set holds for this stage, for verification (mode 2).
		std::vector<vk::ImageView> views;
		std::vector<vk::Sampler>   samplers;
		// Resolved images and, per image, its (position in set_images, mip element);
		// per buffer, its positions in set_buffers.
		std::vector<TextureBinding>                            textures;
		std::vector<std::vector<std::pair<uint32_t, uint32_t>>> image_positions;
		std::vector<std::vector<uint32_t>>                      buffer_positions;
	};
	struct BufferDep {
		uint64_t   address = 0, size = 0;
		BufferId   id;
		vk::Buffer handle = nullptr;
		bool       texel  = false;
		uint32_t   stage = 0, slot = 0;
	};
	// The fields every use reads come first (one cache line); the descriptor, read only
	// when the proof is redone, last.
	struct ImageDep {
		ImageId               id;
		vk::ImageLayout       layout = vk::ImageLayout::eUndefined;
		ImageSubresourceRange range;
		// The image object the set's views belong to: a deleted image's slot can
		// hold another image with the same backing, which IsSampledImageCurrent accepts.
		uint64_t              serial = 0;
		uint32_t              registration = 0; // the image's registration the proof saw
		TextureCache::ImageDesc desc;
	};
	// The images' ids alone, contiguous: every use checks them against the draw's targets
	// (an ImageDep holds a whole ImageDesc, a cache line or more apart from the next id).
	std::vector<ImageId> image_ids;
	void                 IndexImages() {
		image_ids.clear();
		for (const auto& dep: images) image_ids.push_back(dep.id);
	}
	std::vector<uint32_t> key;
	// Relocation: a stage's user data pair (user, user + 1) the key leaves out (its words low_word and
	// high_word), whose reads the gather keeps as offsets from `base`, the pointer the record now holds.
	struct Root {
		uint8_t stage = 0, user = 0, low_word = 0, high_word = 0;
		// The pair's shader data words (user data the shader gets), or NativeXprNoRoot.
		uint8_t low_data = NativeXprNoRoot, high_data = NativeXprNoRoot;
		// An earlier root this one equals (the stages get the same pointer): reads through either
		// are one read, so the record holds while they stay equal.
		uint8_t  same = NativeXprNoRoot;
		uint64_t base = 0; // 48 bits, dword aligned
	};
	std::vector<Root> roots;
	// Relocation: user data words nothing reads (the key leaves them out; the record copies them).
	struct Unread {
		uint8_t stage = 0, user = 0, word = 0;
	};
	std::vector<Unread> unread;
	// The first key words again, inline: a lookup compares most keys without
	// the vector's cache miss.
	static constexpr uint32_t KeyHead = 36;
	uint32_t                  key_size = 0;
	std::array<uint32_t, KeyHead> key_head {};
	void                      HoldKey() {
        key_size = static_cast<uint32_t>(key.size());
        std::copy_n(key.begin(), std::min(key_size, KeyHead), key_head.begin());
	}
	[[nodiscard]] bool SameKey(const std::vector<uint32_t>& other) const {
		if (other.size() != key_size) return false;
		const auto head = std::min(key_size, KeyHead);
		return std::memcmp(key_head.data(), other.data(), head * 4ull) == 0 &&
		       (key_size == head ||
		        std::memcmp(key.data() + head, other.data() + head, (key_size - head) * 4ull) == 0);
	}
	uint64_t              vertex_program = 0, pixel_program = 0;
	vk::DescriptorSet     set    = nullptr;
	vk::DescriptorPool    pool   = nullptr;
	vk::Buffer            upload = nullptr; // ring the dynamic bindings point into
	std::array<Stage, 2>  stages;
	uint32_t              stage_count = 1;
	vk::ShaderStageFlags  push_stages;
	bool                  has_push = false;
	// Native pipeline variant per draw state this object was drawn under.
	std::vector<std::pair<uint64_t, const PipelineCache::Pipeline*>> pipelines;
	// Direct draws (DRAW_INDEX_2 / DRAW_INDEX_OFFSET_2): the vertex and instance offsets the
	// normal path resolved under a draw state. Both come from the index-offset register or the
	// vertex shader's offset user-data SGPRs, which the state and object keys cover.
	struct DirectOffsets {
		uint64_t state    = 0;
		int32_t  vertex   = 0;
		uint32_t instance = 0;
	};
	std::vector<DirectOffsets> direct_offsets;
	// Validation (runtime).
	std::vector<std::pair<uint64_t, uint32_t>> reads; // (address, words)
	std::vector<uint32_t>  words;
	std::vector<int32_t>   word_patch; // stage << 24 | flattened index, -1 unknown
	std::vector<BufferDep> buffers;
	std::vector<ImageDep>  images;
	uint64_t mapping_epoch      = 0;
	uint64_t cpu_epoch          = 0; // sum of the buffer regions' CPU epochs, 0 unknown
	uint64_t registration_epoch = 0; // buffer cache registrations when proven live
	uint64_t resolution_epoch   = 0; // texture cache registrations when proven live
	uint64_t start_epoch        = 0; // texture cache image starts when texel ranges were proven
	uint64_t validated_frame    = UINT64_MAX;
	uint64_t retire_tick        = 0;
	uint64_t used_frame         = 0; // last frame a draw used the record
	uint64_t uses               = 0;
	uint64_t next_hash          = 0; // key hash of the record the next native draw used
	uint64_t next2_hash         = 0; // ... the one after it (its index slot is fetched a draw ahead)
	uint64_t image_dirty_epoch  = 0;     // Image::CpuDirtyEpoch() when the images were proven clean
	// The payload the set was written with; an identical rebind keeps the set.
	std::vector<vk::DescriptorBufferInfo> set_buffers;
	std::vector<vk::DescriptorImageInfo>  set_images;
	struct WriteSlot {
		uint32_t           binding = 0, count = 0, start = 0;
		vk::DescriptorType type {};
		bool               image = false;
	};
	std::vector<WriteSlot> set_writes;
	// Earlier sets that differ from the current one only in images (ring
	// textures rotate through a few bases): switching back to one needs no
	// resolution and no new set. Dropped when anything else changes.
	struct Variant {
		uint64_t                                   signature = 0;
		vk::DescriptorSet                          set       = nullptr;
		vk::DescriptorPool                         pool      = nullptr;
		std::vector<vk::DescriptorImageInfo>       set_images;
		std::array<std::vector<TextureBinding>, 2> textures;
		std::array<std::vector<vk::ImageView>, 2>  views;
		std::vector<ImageDep>                      images;
	};
	std::vector<Variant> variants;
	uint64_t             image_signature = 0; // of the current set's images
	// Traced records (every stage evaluated by a single linear SRT plan): the
	// structural reads are compared on each use, the data reads are copied
	// straight into the flattened SRT; no learning or patching.
	struct Gather {
		bool traced = false;
		bool broken = false; // descriptors patched but not resolved: retire
		struct Run {
			uint64_t address = 0;
			uint32_t count   = 0;
			uint32_t first   = 0; // into values (structural) / targets_at (data)
			bool     clean   = false;
			uint8_t  root    = NativeXprNoRoot; // into roots: the address is an offset from its base
		};
		std::vector<Run>      structural;
		std::vector<uint32_t> values;
		std::vector<uint8_t>  uses; // per structural value: SrtReadTrace::Use bits
		// Per structural value: the snapshot descriptor dwords it is copied to,
		// stage << 28 | kind << 24 | index << 8 | dword.
		std::vector<uint32_t> feeds_at; // one extra end entry
		std::vector<uint32_t> feeds;
		std::vector<Run>      data;
		std::vector<uint32_t> targets_at; // per data dword: start in targets; one extra end entry
		std::vector<uint32_t> targets;    // stage << 24 | flat index
		// The non-clean structural and the data words, read as page-local spans
		// (small gaps included). `words` is what the record currently holds for
		// them: a validation handles only the words that differ.
		struct Span {
			uint64_t address = 0; // an offset from the base of `root`, if any
			uint32_t count   = 0;
			uint32_t first   = 0; // into words
			uint8_t  root    = NativeXprNoRoot;
			uint8_t  level   = 0; // read after every span of a lower one (pointers it is read through)
		};
		std::vector<Span>     spans;
		std::vector<uint32_t> words;
		std::vector<int32_t>  word_value; // structural value index, or -1
		std::vector<int32_t>  word_data;  // data dword index (targets_at), or -1
		std::vector<uint32_t> value_word; // per non-clean structural value: its index in words
		// Relocation: pointers read from memory that later spans are read through (a span root past
		// the record's roots), as the structural values of their halves; per structural value, whether
		// it is such a half, read for nothing else (a change moves what is read through it).
		struct Pointer {
			uint32_t low = 0, high = 0;
		};
		std::vector<Pointer> pointers;
		std::vector<uint8_t> pointer_value;
		// The clean structural words, as page-local spans; per span word its
		// structural value index, or -1 for a gap.
		std::vector<Span>    clean_spans;
		std::vector<int32_t> clean_value;
	} gather;
};

namespace {
// A root's base: its pointer as the SRT's pointer reads take it (LinearSrtPlan::ReadAddress).
uint64_t NativeXprBase(uint32_t low, uint32_t high) {
	return ((uint64_t {high} << 32u) | low) & 0x0000fffffffffffcull;
}

// Where a gather run or span is now: its address, or its offset from its root's current base (a
// record root, or a pointer the gather read). (Templates: the record type is private to RenderExecutor.)
template <typename Record, typename Span>
uint64_t NativeXprAddress(const Record& record, const Span& span) {
	if (span.root == NativeXprNoRoot) return span.address;
	if (span.root < record.roots.size()) return record.roots[span.root].base + span.address;
	const auto& gather  = record.gather;
	const auto& pointer = gather.pointers[span.root - record.roots.size()];
	return NativeXprBase(gather.values[pointer.low], gather.values[pointer.high]) + span.address;
}

// A program's user data (bit k: user[k]) a record may take from each draw: the pairs (user[k],
// user[k + 1]) its SRT only reads through, and the words nothing reads (not the SRT, the shader or
// the draw offsets). Recorded programs make no address loads: the shader gets user data only as
// shader data, which a relocation updates.
struct NativeXprUserData {
	uint32_t pointers = 0, unread = 0;
};
NativeXprUserData NativeXprUserDataOf(PipelineCache& pipelines, const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	const auto use    = pipelines.UserDataUse(program);
	const auto base   = int64_t {program.user_data_base};
	uint32_t   offset = 0, shader = 0;
	for (const auto reg: {int64_t {program.info.vertex_offset_sgpr} - base, int64_t {program.info.instance_offset_sgpr} - base})
		if (reg >= 0 && reg < 32) offset |= 1u << reg;
	for (const auto reg: program.bindings.user_data_registers)
		if (reg >= base && reg - base < 32) shader |= 1u << (reg - base);
	return {use.pointers & ~offset & ~(offset >> 1), ~(use.read | shader | offset)};
}

// Where the shader data holds user data `user` of the program, or NativeXprNoRoot.
uint8_t NativeXprUserDataWord(const ShaderRecompiler::IR::CompiledShaderInfo& program, uint32_t user) {
	const auto& registers = program.bindings.user_data_registers;
	const auto  found     = std::ranges::find(registers, program.user_data_base + user);
	const auto  index     = found - registers.begin();
	return found == registers.end() || index >= NativeXprNoRoot ? NativeXprNoRoot : static_cast<uint8_t>(index);
}

// The lines of the record's fields a use reads (addresses only: no load waits).
template <typename Record>
void NativeXprPrefetchFields(const Record& record) {
	for (const void* field: {static_cast<const void*>(&record.key), static_cast<const void*>(&record.key_head[16]),
	                         static_cast<const void*>(&record.key_head[32]), static_cast<const void*>(&record.set),
	                         static_cast<const void*>(&record.stages[0].flattened),
	                         static_cast<const void*>(&record.stages[0].push_start),
	                         static_cast<const void*>(&record.stages[1].flattened),
	                         static_cast<const void*>(&record.stages[1].push_start),
	                         static_cast<const void*>(&record.pipelines), static_cast<const void*>(&record.buffers),
	                         static_cast<const void*>(&record.images), static_cast<const void*>(&record.cpu_epoch),
	                         static_cast<const void*>(&record.used_frame), static_cast<const void*>(&record.gather),
	                         static_cast<const void*>(&record.gather.clean_spans), static_cast<const void*>(&record.gather.spans),
	                         static_cast<const void*>(&record.gather.words)})
		__builtin_prefetch(field);
}

// The arrays a use reads in full (once the fields above are in).
template <typename Record>
void NativeXprPrefetchArrays(const Record& record) {
	const auto& gather = record.gather;
	NativeXprPrefetch(gather.spans);
	NativeXprPrefetch(gather.words);
	NativeXprPrefetch(gather.clean_spans);
	// Read only for changed words, but constants change every frame.
	NativeXprPrefetch(gather.word_data.data(), std::min<size_t>(gather.word_data.size() * 4, 512));
	NativeXprPrefetch(gather.word_value.data(), std::min<size_t>(gather.word_value.size() * 4, 512));
	NativeXprPrefetch(gather.targets_at.data(), std::min<size_t>(gather.targets_at.size() * 4, 512));
	NativeXprPrefetch(gather.targets.data(), std::min<size_t>(gather.targets.size() * 4, 512));
	NativeXprPrefetch(gather.uses.data(), std::min<size_t>(gather.uses.size(), 256));
	NativeXprPrefetch(gather.feeds_at.data(), std::min<size_t>(gather.feeds_at.size() * 4, 256));
	NativeXprPrefetch(gather.feeds.data(), std::min<size_t>(gather.feeds.size() * 4, 256));
	NativeXprPrefetch(gather.value_word.data(), std::min<size_t>(gather.value_word.size() * 4, 256));
	NativeXprPrefetch(record.pipelines);
	// An image dependency is large (its descriptor): the first line of each holds what a use reads.
	NativeXprPrefetch(record.image_ids);
	for (size_t i = 0; i < record.images.size() && i < 16; ++i) __builtin_prefetch(&record.images[i]);
	NativeXprPrefetch(record.buffers);
	if (!gather.traced) {
		NativeXprPrefetch(record.reads);
		NativeXprPrefetch(record.words);
		NativeXprPrefetch(record.word_patch);
	}
	for (uint32_t st = 0; st < record.stage_count; ++st) {
		NativeXprPrefetch(record.stages[st].flattened);
		NativeXprPrefetch(record.stages[st].shader_data);
	}
}
} // namespace

struct RenderExecutor::NativeXprCache {
	vk::Device                                                        device;
	std::unordered_map<uint64_t, std::unique_ptr<NativeXprRecord>, HashedKey> records;
	// Past this many records the sweep keeps only recently used ones (NativeXprCollect); at
	// RecordsHardCap a store drops them all (NativeXprStore).
	static constexpr size_t RecordsSoftCap = 49152, RecordsHardCap = 65536;
	// Open-addressing index over `records` (linear probing, at most half full):
	// one cache line per lookup instead of bucket, node and owner.
	struct IndexSlot {
		uint64_t         hash   = 0; // 0 empty (hash 0 is stored as 1)
		NativeXprRecord* record = nullptr;
	};
	std::vector<IndexSlot> index = std::vector<IndexSlot>(8192);
	size_t                 index_count = 0;
	static uint64_t        IndexHash(uint64_t hash) { return hash != 0 ? hash : 1; }
	// Where a lookup of `hash` starts probing (to fetch that line ahead of the lookup).
	[[nodiscard]] const void* IndexSlotOf(uint64_t hash) const {
		return &index[IndexHash(hash) & (index.size() - 1)];
	}
	[[nodiscard]] NativeXprRecord* IndexFind(uint64_t hash) const {
		hash            = IndexHash(hash);
		const auto mask = index.size() - 1;
		for (auto at = hash & mask;; at = (at + 1) & mask) {
			if (index[at].hash == hash) return index[at].record;
			if (index[at].hash == 0) return nullptr;
		}
	}
	void IndexInsert(uint64_t hash, NativeXprRecord* record) {
		if ((index_count + 1) * 2 > index.size()) {
			std::vector<IndexSlot> old(index.size() * 2);
			old.swap(index);
			index_count = 0;
			for (const auto& slot: old)
				if (slot.hash != 0) IndexInsert(slot.hash, slot.record);
		}
		hash            = IndexHash(hash);
		const auto mask = index.size() - 1;
		for (auto at = hash & mask;; at = (at + 1) & mask) {
			if (index[at].hash == hash) {
				index[at].record = record;
				return;
			}
			if (index[at].hash == 0) {
				index[at] = {hash, record};
				++index_count;
				return;
			}
		}
	}
	void IndexErase(uint64_t hash) {
		hash            = IndexHash(hash);
		const auto mask = index.size() - 1;
		auto       at   = hash & mask;
		for (; index[at].hash != hash; at = (at + 1) & mask)
			if (index[at].hash == 0) return;
		// Backward shift: later entries of the probe run move into the hole.
		for (auto next = (at + 1) & mask; index[next].hash != 0; next = (next + 1) & mask) {
			const auto home = index[next].hash & mask;
			if (((next - home) & mask) >= ((next - at) & mask)) {
				index[at] = index[next];
				at        = next;
			}
		}
		index[at] = {};
		--index_count;
	}
	std::unordered_map<uint64_t, std::unique_ptr<NativeXprDrawState>, HashedKey> states;
	// Invalidated records: their sets retire once the GPU passed retire_tick;
	// their logged reads stay for learning until the key is stored again.
	std::unordered_map<uint64_t, std::unique_ptr<NativeXprRecord>, HashedKey> stale;
	struct RetiredSet {
		vk::DescriptorPool pool;
		vk::DescriptorSet  set;
		uint64_t           tick = 0;
	};
	std::vector<RetiredSet> retired;
	// Sets the GPU is done with, by layout, for that layout's next allocation (a driver allocation and
	// free are a call each); the layout of every set allocated here.
	struct FreeSet {
		vk::DescriptorPool pool;
		vk::DescriptorSet  set;
	};
	// A camera turn rebinds ~1500 sets a frame (Latria): with 32 per layout most retired sets were freed and
	// allocated again a frame later.
	static constexpr size_t FreeSetsPerLayout = 2048, FreeSetsMax = 16384;
	std::unordered_map<VkDescriptorSetLayout, std::vector<FreeSet>> free_sets;
	std::unordered_map<VkDescriptorSet, VkDescriptorSetLayout>      set_layouts;
	size_t                                                          free_count = 0;
	std::vector<vk::DescriptorPool>                                pools;
	std::vector<uint32_t>                                          key;
	// Store request: the reads of the normal draw's SRT evaluation.
	std::vector<std::pair<uint64_t, uint64_t>> reads;
	bool                                       logging = false;
	// Scratch for validation.
	// Copy of the texture cache's image start log: every (epoch, address) with
	// starts_from < epoch <= starts_seen.
	std::vector<std::pair<uint64_t, uint64_t>> recent_starts;
	uint64_t                                   starts_from = 0, starts_seen = 0;
	std::vector<vk::DescriptorImageInfo>           previous_images;
	std::vector<NativeXprRecord::ImageDep>         previous_deps;
	std::array<std::vector<TextureBinding>, 2>     previous_textures;
	std::array<std::vector<vk::ImageView>, 2>      previous_views;
	std::vector<std::pair<uint32_t, uint32_t>> rebind_images;  // (stage, image)
	std::vector<std::pair<uint32_t, uint32_t>> rebind_buffers; // (stage, buffer)
	std::vector<TextureBinding>                rebind_textures;
	std::vector<uint32_t>                      span;
	std::vector<std::pair<uint32_t, uint32_t>> changed;
	// What the last native emission left bound in which command buffer.
	struct Emit {
		vk::CommandBuffer              command;
		uint64_t                       generation = 0;
		const NativeXprDrawState*      state      = nullptr;
		const PipelineCache::Pipeline* pipeline   = nullptr;
		uint64_t                       index_base = 0, index_bytes = 0;
		vk::IndexType                  index_type = vk::IndexType::eUint16;
		// The dynamic state the emissions recorded into `command` (valid in the same command and generation, when
		// `dynamic_held`): another state records only the values that differ (NativeXprEmitDynamic).
		ResolvedDynamicState           dynamic {};
		vk::ImageAspectFlags           feedback {};
		bool                           dynamic_held = false, feedback_held = false;
	} emit;
	// Programs the SRT trace cannot cover (no single linear plan): never retried.
	std::unordered_set<const ShaderRecompiler::IR::CompiledShaderInfo*> untraceable;
	// Per shader-address pair (key words 0-3): records stored and records that
	// left unused. A pair whose records are never reused (user data that changes
	// every frame) stops storing.
	struct PairStats {
		uint64_t stored = 0, wasted = 0;
		bool     blocked = false;
		bool     unstorable = false; // a program writes memory or uses DMA/GDS (NativeXprStore)
		// Relocation: the key words the pair's records leave out (bit i: word i), known once a store
		// looked at the pair's programs (NativeXprStore).
		bool     relocation_known = false;
		bool     pointers_off     = false; // a record could not be traced through its pointers
		uint64_t relocate         = 0;
		// Relocation: descriptor base addresses (buffer and image dword 0) a record reads through a pointer of
		// its roots' tables: the key words of the root, the pointer's offset from the root's base and the word's
		// from the pointer's. They tell the pair's objects apart where their use in the frame does not (the
		// tables are new copies every frame, the buffers they describe are the object's). From the pair's
		// first traced record.
		struct Identity {
			uint8_t  low_word = 0, high_word = 0;
			uint32_t pointer = 0, offset = 0;
		};
		std::vector<Identity> identity;
		bool                  identity_known = false;
	};
	std::unordered_map<uint64_t, PairStats, HashedKey> pairs;
	// Keys that missed, with the frame of their latest miss. A record pays off only
	// when its key comes again, so it is stored at the second sighting within the
	// frames a record outlives unused: keys new every frame (user data pointing at
	// per-frame allocations) are never stored, even in pairs whose other objects are.
	struct SeenKey {
		uint64_t hash = 0, frame = 0;
	};
	std::vector<SeenKey> seen = std::vector<SeenKey>(8192);
	bool SeenAgain(uint64_t hash, uint64_t frame) {
		hash             = IndexHash(hash);
		const auto mask  = seen.size() - 1;
		SeenKey*   empty = nullptr;
		for (uint64_t i = 0, at = hash & mask; i < 8; ++i, at = (at + 1) & mask) {
			auto& entry = seen[at];
			if (entry.hash == hash) {
				const bool again = entry.frame + 2 >= frame;
				entry.frame      = frame;
				return again;
			}
			if (empty == nullptr && (entry.hash == 0 || entry.frame + 2 < frame)) empty = &entry;
		}
		*(empty != nullptr ? empty : &seen[hash & mask]) = {hash, frame};
		return false;
	}
	// Relocation (direct draws, and indirect runs whose whole key missed): the latest try's key before
	// its pointer words left it, the words it left out, the words that joined the rest (its mesh or the
	// run's first argument address, which use of them in the frame it was) and its hash (for the store a
	// miss requests); per program, the user data a record may take from each draw.
	std::vector<uint32_t> full_key;
	uint64_t              tried_relocate = 0;
	std::vector<uint32_t> tried_suffix;
	uint64_t              tried_hash     = 0;
	// Keys whose store was refused: the same refusal would come every frame.
	static constexpr uint64_t RefuseFrames = 64;
	std::vector<SeenKey>      refused      = std::vector<SeenKey>(4096);
	[[nodiscard]] bool        Refused(uint64_t hash, uint64_t frame) const {
		hash            = IndexHash(hash);
		const auto mask = refused.size() - 1;
		for (uint64_t i = 0, at = hash & mask; i < 8; ++i, at = (at + 1) & mask)
			if (refused[at].hash == hash) return refused[at].frame + RefuseFrames > frame;
		return false;
	}
	void Refuse(uint64_t hash, uint64_t frame) {
		hash            = IndexHash(hash);
		const auto mask = refused.size() - 1;
		auto*      slot = &refused[hash & mask];
		for (uint64_t i = 0, at = hash & mask; i < 8; ++i, at = (at + 1) & mask)
			if (refused[at].hash == hash || refused[at].hash == 0 || refused[at].frame + RefuseFrames <= frame) {
				slot = &refused[at];
				break;
			}
		*slot = {hash, frame};
	}
	std::unordered_map<const ShaderRecompiler::IR::CompiledShaderInfo*, NativeXprUserData> user_data;
	// Uses of each relocated key in the frame: the n-th use finds the n-th record (objects that differ
	// only in their pointers keep their own records).
	struct Occurrence {
		uint64_t hash  = 0;
		uint32_t frame = 0, count = 0;
	};
	std::vector<Occurrence> occurrences = std::vector<Occurrence>(8192);
	uint32_t                NextOccurrence(uint64_t hash, uint64_t frame) {
		hash            = IndexHash(hash);
		const auto mask = occurrences.size() - 1;
		const auto now  = static_cast<uint32_t>(frame);
		// Entries of earlier frames are free: within a frame none is freed, so a probe run never breaks.
		for (uint64_t i = 0, at = hash & mask; i < 16; ++i, at = (at + 1) & mask) {
			auto& entry = occurrences[at];
			if (entry.hash == 0 || entry.frame != now) {
				entry = {hash, now, 1};
				return 0;
			}
			if (entry.hash == hash) return entry.count++;
		}
		return UINT32_MAX;
	}
	// Relocated objects (pair and mesh) whose record rebound its descriptors: their uses tell instances apart by
	// their descriptor bases (NativeXprTry). Others keep their uses' records (no reads for the key).
	std::vector<uint64_t> rebound = std::vector<uint64_t>(4096);
	[[nodiscard]] bool    Rebound(uint64_t hash) const {
		hash            = IndexHash(hash);
		const auto mask = rebound.size() - 1;
		for (uint64_t i = 0, at = hash & mask; i < 8; ++i, at = (at + 1) & mask) {
			if (rebound[at] == hash) return true;
			if (rebound[at] == 0) return false;
		}
		return false;
	}
	void MarkRebound(uint64_t hash) {
		hash            = IndexHash(hash);
		const auto mask = rebound.size() - 1;
		for (uint64_t i = 0, at = hash & mask; i < 8; ++i, at = (at + 1) & mask) {
			if (rebound[at] == hash) return;
			if (rebound[at] == 0) {
				rebound[at] = hash;
				return;
			}
		}
		rebound[hash & mask] = hash;
	}
	// The latest relocated try: its object (key without the use) and, keyed by descriptor bases, its key by use.
	uint64_t              object_hash = 0;
	std::vector<uint32_t> use_key;
	size_t                suffix_at = 0; // where the words the try added start (tried_suffix)
	uint64_t                                swept_frame = UINT64_MAX;
	// Draw-state key of the latest draw (native or normal), for clean draws.
	uint64_t state_key       = 0;
	bool     state_key_valid = false;
	uint64_t last_hash       = 0; // key hash of the previous native draw's record
	uint64_t last2_hash      = 0; // ... and of the one before it
	uint64_t                  last_state_key = 0; // the latest states lookup (entries never move)
	const NativeXprDrawState* last_state     = nullptr;
	uint64_t collected_frame = UINT64_MAX;
	// Store requests in store_frame (kyty_local_native_xpr_store_budget).
	uint64_t store_frame    = UINT64_MAX;
	uint32_t store_requests = 0;
	size_t   reported_records = 0;
	~NativeXprCache() {
		for (auto pool: pools) device.destroyDescriptorPool(pool);
	}
};

namespace {
// The shader addresses and the user-data counts. The same programs come with different numbers of user data
// words (1-1: a pair's draws with 0 or 2 PS words), which moves the words in the key: relocation learns which
// are pointers per layout (learned from the 0-word draws, the others' per-frame PS pointer stayed in their keys).
uint64_t NativeXprPairKey(const std::vector<uint32_t>& key) {
	if (key.size() < 4) return 0;
	std::array<uint32_t, 7> layout {key[0], key[1], key[2], key[3], 0, 0, 0};
	for (size_t i = 0, at = 4; i < 3 && at < key.size(); ++i) {
		layout[4 + i] = key[at];
		at += 1 + key[at];
	}
	return XXH3_64bits(layout.data(), sizeof(layout));
}
} // namespace

RenderExecutor::NativeXprCache& RenderExecutor::NativeXprState() {
	if (!m_native_xpr) {
		m_native_xpr         = std::make_shared<NativeXprCache>();
		m_native_xpr->device = m_context.GetGraphics().device;
	}
	return *m_native_xpr;
}

void RenderExecutor::NativeXprRequestStore() {
	if ((kyty_local_native_xpr_diag.load(std::memory_order_relaxed) & 1u) != 0) return;
	m_native_xpr_store = true;
}

std::vector<std::pair<uint64_t, uint64_t>>* RenderExecutor::NativeXprReadLog() {
	if (!m_native_xpr_store) return nullptr;
	auto& cache   = NativeXprState();
	cache.logging = true;
	cache.reads.clear();
	return &cache.reads;
}

void RenderExecutor::NativeXprEndPacket() {
	m_native_xpr_store  = false;
	m_native_xpr_verify = {};
	if (m_native_xpr) m_native_xpr->logging = false;
}

// Object identity: both shader addresses and every user-data SGPR (the XPR
// geometry root and whatever else the SRT chains start from), and the one the
// CP writes the start instance into (the vertex shader is translated for it).
void RenderExecutor::NativeXprKey(CommandBuffer& buffer, std::vector<uint32_t>& key) const {
	const auto& sh = buffer.GetShaders();
	const auto  es = sh.GetVs().es_regs.data_addr;
	const auto  vs = es != 0 ? es : sh.GetVs().gs_regs.data_addr;
	const auto  ps = sh.GetPs().ps_regs.data_addr;
	key.clear();
	key.push_back(static_cast<uint32_t>(vs));
	key.push_back(static_cast<uint32_t>(vs >> 32u));
	key.push_back(static_cast<uint32_t>(ps));
	key.push_back(static_cast<uint32_t>(ps >> 32u));
	for (const auto* sgprs: {&sh.GetVs().hs_user_sgpr, &sh.GetVs().gs_user_sgpr, &sh.GetPs().ps_user_sgpr}) {
		const auto count = std::min<uint32_t>(sgprs->count, HW::UserSgprInfo::SGPRS_MAX);
		key.push_back(count);
		key.insert(key.end(), sgprs->value, sgprs->value + count);
	}
	key.push_back(static_cast<uint32_t>(sh.GetVs().start_instance_user_sgpr));
}

// Draw-state identity: the whole graphics context, the user configuration and
// the graphics shader stage registers with the user-data layout (types and
// counts) but not the user-data values, which identify the object instead.
uint64_t RenderExecutor::NativeXprStateKey(CommandBuffer& buffer) const {
	const auto& ctx = buffer.GetRegisters();
	const auto& cfg = buffer.GetUserConfig();
	const auto& vs  = buffer.GetShaders().GetVs();
	const auto& ps  = buffer.GetShaders().GetPs();
	auto hash = XXH3_64bits(&ctx, sizeof(ctx));
	hash      = XXH3_64bits_withSeed(&cfg, sizeof(cfg), hash);
	hash      = XXH3_64bits_withSeed(&vs.es_regs, sizeof(vs.es_regs), hash);
	hash      = XXH3_64bits_withSeed(&vs.ls_regs, sizeof(vs.ls_regs), hash);
	hash      = XXH3_64bits_withSeed(&vs.hs_regs, sizeof(vs.hs_regs), hash);
	hash      = XXH3_64bits_withSeed(&vs.gs_regs, sizeof(vs.gs_regs), hash);
	hash      = XXH3_64bits_withSeed(&ps.ps_regs, sizeof(ps.ps_regs), hash);
	for (const auto* sgprs: {&vs.hs_user_sgpr, &vs.gs_user_sgpr, &ps.ps_user_sgpr}) {
		hash = XXH3_64bits_withSeed(sgprs->type, sizeof(sgprs->type), hash);
		hash = XXH3_64bits_withSeed(&sgprs->count, sizeof(sgprs->count), hash);
	}
	return hash;
}

vk::DescriptorSet RenderExecutor::NativeXprAllocateSet(vk::DescriptorSetLayout layout,
                                                       vk::DescriptorPool& pool_out) {
	auto&      cache  = NativeXprState();
	const auto device = cache.device;
	if (const auto found = cache.free_sets.find(static_cast<VkDescriptorSetLayout>(layout));
	    found != cache.free_sets.end() && !found->second.empty()) {
		const auto item = found->second.back();
		found->second.pop_back();
		--cache.free_count;
		pool_out = item.pool;
		return item.set;
	}
	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts        = &layout;
	vk::DescriptorSet set       = nullptr;
	// Freed sets return to their pool; try the newest pools first.
	for (size_t i = 0; i < cache.pools.size() && i < 4; ++i) {
		const auto pool         = cache.pools[cache.pools.size() - 1 - i];
		allocate.descriptorPool = pool;
		if (device.allocateDescriptorSets(&allocate, &set) == vk::Result::eSuccess) {
			pool_out = pool;
			cache.set_layouts[static_cast<VkDescriptorSet>(set)] = static_cast<VkDescriptorSetLayout>(layout);
			return set;
		}
	}
	constexpr uint32_t Sets = 1024;
	const std::array<vk::DescriptorPoolSize, 5> sizes {{
	    {vk::DescriptorType::eStorageBuffer, Sets * 16},
	    {vk::DescriptorType::eStorageBufferDynamic, Sets * 4},
	    {vk::DescriptorType::eSampledImage, Sets * 32},
	    {vk::DescriptorType::eSampler, Sets * 8},
	    {vk::DescriptorType::eStorageImage, Sets * 4},
	}};
	vk::DescriptorPoolCreateInfo create {};
	create.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
	create.maxSets       = Sets;
	create.poolSizeCount = static_cast<uint32_t>(sizes.size());
	create.pPoolSizes    = sizes.data();
	vk::DescriptorPool pool = nullptr;
	EXIT_IF(device.createDescriptorPool(&create, nullptr, &pool) != vk::Result::eSuccess);
	cache.pools.push_back(pool);
	allocate.descriptorPool = pool;
	EXIT_IF(device.allocateDescriptorSets(&allocate, &set) != vk::Result::eSuccess);
	pool_out = pool;
	cache.set_layouts[static_cast<VkDescriptorSet>(set)] = static_cast<VkDescriptorSetLayout>(layout);
	return set;
}

// A descriptor set leaves service; it may be in flight until the current tick.
void RenderExecutor::NativeXprRetireSet(vk::DescriptorPool pool, vk::DescriptorSet set) {
	if (!set) return;
	Spec::NoteRetiredSet(reinterpret_cast<uint64_t>(static_cast<VkDescriptorSet>(set)));
	NativeXprState().retired.push_back({pool, set, m_context.GetCommandScheduler().CurrentTick()});
}

void RenderExecutor::NativeXprDropVariants(NativeXprRecord& record) {
	for (auto& variant: record.variants) NativeXprRetireSet(variant.pool, variant.set);
	record.variants.clear();
}

// A record leaves service; its reads stay for learning when the key is stored again.
void RenderExecutor::NativeXprRetire(std::unique_ptr<NativeXprRecord> record, bool keep_for_learning) {
	auto& cache = NativeXprState();
	NativeXprRetireSet(record->pool, record->set);
	record->set = nullptr;
	NativeXprDropVariants(*record);
	if (!keep_for_learning) return;
	const auto hash = XXH3_64bits(record->key.data(), record->key.size() * sizeof(uint32_t));
	if (cache.stale.size() >= 65536) cache.stale.clear();
	cache.stale[hash] = std::move(record);
}

void RenderExecutor::ResolvePendingTicks(uint64_t pending, uint64_t tick) {
	if (!m_native_xpr) return;
	for (auto& item: m_native_xpr->retired)
		if (item.tick == pending) item.tick = tick;
}

// Once per frame: the retired sets the GPU is done with are kept for reuse or freed, one call per
// pool.
void RenderExecutor::NativeXprCollect() {
	if (!m_native_xpr) return;
	auto&      cache = *m_native_xpr;
	const auto frame = m_context.FrameNumber();
	if (frame - cache.swept_frame >= 8) {
		// Every eighth frame, records no draw used for `keep` frames leave (their
		// objects are gone or their keys changed); a pair whose records keep
		// leaving unused stops storing. Past the soft cap only the last two frames' stay
		// (at the hard cap NativeXprStore drops them all).
		cache.swept_frame = frame;
		const uint64_t keep = cache.records.size() > NativeXprCache::RecordsSoftCap
		                          ? 2
		                          : std::max(2u, kyty_local_native_xpr_keep_frames.load(std::memory_order_relaxed));
		if (SlowLog::HitchThreshold() > 0.0 && cache.records.size() / 4096 != cache.reported_records / 4096) {
			std::printf("[tsc %llu] XPR records %zu\n", static_cast<unsigned long long>(__rdtsc()), cache.records.size());
			cache.reported_records = cache.records.size();
		}
		for (auto it = cache.records.begin(); it != cache.records.end();) {
			auto& record = *it->second;
			if (record.used_frame + keep >= frame) {
				++it;
				continue;
			}
			auto& pair = cache.pairs[NativeXprPairKey(record.key)];
			if (record.uses == 0 && ++pair.wasted > 32 && pair.wasted * 10 > pair.stored * 9 && !pair.blocked) {
				pair.blocked = true;
			}
			cache.IndexErase(it->first);
			NativeXprRetire(std::move(it->second), false);
			it = cache.records.erase(it);
		}
	}
	if (cache.collected_frame == frame || cache.retired.empty()) return;
	cache.collected_frame = frame;
	auto& scheduler       = m_context.GetCommandScheduler();
	std::vector<NativeXprCache::RetiredSet> done;
	std::erase_if(cache.retired, [&](const NativeXprCache::RetiredSet& item) {
		if (!scheduler.IsFree(item.tick)) return false;
		const auto layout = cache.set_layouts.find(static_cast<VkDescriptorSet>(item.set));
		if (layout != cache.set_layouts.end() && cache.free_count < NativeXprCache::FreeSetsMax) {
			auto& sets = cache.free_sets[layout->second];
			if (sets.size() < NativeXprCache::FreeSetsPerLayout) {
				sets.push_back({item.pool, item.set});
				++cache.free_count;
				return true;
			}
		}
		if (layout != cache.set_layouts.end()) cache.set_layouts.erase(layout);
		done.push_back(item);
		return true;
	});
	std::ranges::sort(done, {}, [](const auto& item) { return static_cast<VkDescriptorPool>(item.pool); });
	std::vector<vk::DescriptorSet> batch;
	for (size_t first = 0; first < done.size();) {
		size_t last = first;
		batch.clear();
		while (last < done.size() && done[last].pool == done[first].pool) batch.push_back(done[last++].set);
		(void)cache.device.freeDescriptorSets(done[first].pool, static_cast<uint32_t>(batch.size()), batch.data());
		first = last;
	}
}

// A fresh record replaced a stale one whose logged reads were at the same
// addresses. Words whose change moved only the flattened SRT, each to a unique
// position carrying its new value, are learned as patchable constant data.
void RenderExecutor::NativeXprLearn(const NativeXprRecord& stale, NativeXprRecord& fresh) {
	fresh.word_patch.assign(fresh.words.size(), -1);
	if (stale.reads != fresh.reads || stale.stage_count != fresh.stage_count ||
	    stale.vertex_program != fresh.vertex_program || stale.pixel_program != fresh.pixel_program)
		return;
	// Everything but the flattened constants and the T# LOD-statistics fields (and
	// the shader-data words derived from them) must be unchanged.
	for (uint32_t st = 0; st < fresh.stage_count; ++st) {
		const auto& a = stale.stages[st];
		const auto& b = fresh.stages[st];
		if (!(a.snapshot.buffers == b.snapshot.buffers) || !(a.snapshot.samplers == b.snapshot.samplers) ||
		    a.snapshot.user_data != b.snapshot.user_data || !(a.snapshot.uniform_fill == b.snapshot.uniform_fill) ||
		    a.flattened.size() != b.flattened.size() ||
		    !NativeXprSameImagesButLodStats(a.snapshot.images, b.snapshot.images) ||
		    a.shader_data.size() != b.shader_data.size()) {
			return;
		}
		const auto& layout = b.program->bindings;
		for (uint32_t i = 0; i < a.shader_data.size(); ++i) {
			const bool lod = i >= layout.LodStatsDword() && i < layout.LodStatsDword() + layout.lod_stats_count;
			if (!lod && a.shader_data[i] != b.shader_data[i]) return;
		}
	}
	// What moved: flattened positions and T# LOD dwords, each to be explained by
	// exactly one changed word carrying its new value.
	std::vector<int32_t> positions;
	for (uint32_t st = 0; st < fresh.stage_count; ++st) {
		const auto& a = stale.stages[st];
		const auto& b = fresh.stages[st];
		for (size_t j = 0; j < a.flattened.size(); ++j)
			if (a.flattened[j] != b.flattened[j]) positions.push_back(static_cast<int32_t>(st << 24u | j));
		for (uint32_t i = 0; i < b.snapshot.images.size(); ++i)
			for (uint32_t w = 0; w < b.snapshot.images[i].dwords.size(); ++w)
				if (a.snapshot.images[i].dwords[w] != b.snapshot.images[i].dwords[w])
					positions.push_back(NativeXprImagePatch(st, i, w));
	}
	const auto value_at = [&](int32_t position) {
		if (position >= 0)
			return fresh.stages[static_cast<uint32_t>(position) >> 24u].flattened[static_cast<uint32_t>(position) & 0xffffffu];
		const auto code = static_cast<uint32_t>(NativeXprImagePatchBase - position);
		return fresh.stages[code >> 20u].snapshot.images[(code >> 4u) & 0xffffu].dwords[code & 0xfu];
	};
	// A changed word is patchable when exactly one moved position took its new
	// value, and ignorable (NativeXprIgnore) when nothing moved with it: flattened
	// entries and descriptor dwords are whole-word copies, so a word that reached
	// any of them would have changed it.
	std::vector<uint8_t> explained(positions.size(), 0);
	bool                 ambiguous = false;
	for (size_t k = 0; k < fresh.words.size(); ++k) {
		if (stale.word_patch.size() == fresh.words.size() && stale.word_patch[k] != -1 &&
		    stale.words[k] == fresh.words[k]) {
			fresh.word_patch[k] = stale.word_patch[k];
			continue;
		}
		if (stale.words[k] == fresh.words[k]) continue;
		int32_t match   = -1;
		size_t  matches = 0;
		for (size_t p = 0; p < positions.size(); ++p) {
			if (value_at(positions[p]) == fresh.words[k]) {
				match        = positions[p];
				explained[p] = 1;
				++matches;
			}
		}
		if (matches == 1) fresh.word_patch[k] = match;
		else if (matches == 0) fresh.word_patch[k] = NativeXprIgnore;
		else ambiguous = true;
	}
	// A change no word explains means the data is not a plain copy.
	if (ambiguous || std::ranges::find(explained, uint8_t {0}) != explained.end()) {
		fresh.word_patch.assign(fresh.words.size(), -1);
	}
}

// Resolved bindings -> record: the stage data, the runtime proof dependencies and
// the descriptor set (kept when an identical payload is already written).
// Returns a reject reason, or nullptr.
const char* RenderExecutor::NativeXprBindResources(NativeXprRecord& record,
                                                   std::span<PreparedBindings* const> stages,
                                                   vk::DescriptorSetLayout layout, bool runtime) {
	auto& buffers   = m_context.GetBufferCache();
	auto& textures  = m_context.GetTextureCache();
	auto& resources = m_context.GetGpuResources();
	for (const auto* prepared: stages) {
		const auto& program = *prepared->runtime->program;
		if (program.info.uses_dma || prepared->gds.buffer != nullptr) return "dma_gds";
		for (const auto& resource: program.info.buffers)
			if (resource.written || resource.atomic) return "written_buffer";
		for (const auto& image: program.info.images)
			if (image.written || image.atomic) return "written_image";
		for (const auto& image: prepared->images)
			if (image.desc.type == TextureCache::BindingType::Storage) return "storage_image";
	}
	std::vector<NativeXprRecord::BufferDep> buffer_deps;
	std::vector<NativeXprRecord::ImageDep>  image_deps;
	const auto                              start_epoch = textures.StartEpoch();
	if (runtime) {
		for (uint32_t st = 0; st < stages.size(); ++st) {
			auto*       prepared = stages[st];
			const auto& program  = *prepared->runtime->program;
			for (size_t i = 0; i < program.info.buffers.size(); ++i) {
				const auto& source = prepared->buffer_sources[i];
				if (source.address == 0 || source.size == 0) continue;
				// Served by the upload ring or pending CPU changes: not a fixed descriptor.
				const auto* owner = source.id ? buffers.GetBufferIfLive(source.id) : nullptr;
				bool        cpu   = buffers.IsRegionCpuModified(source.address, source.size);
				if (owner != nullptr && cpu && (kyty_local_native_xpr_diag.load(std::memory_order_relaxed) & 2u) == 0 &&
				    !buffers.IsRegionGpuModified(source.address, source.size)) {
					// The normal path streams small ranges of CPU-dirty pages through the
					// upload ring on every draw and never cleans the owner. Clean it once:
					// a later CPU write faults and the record's proof notices.
					buffers.EnsureBufferContents(source.address, source.size);
					const auto [resolved, offset] = buffers.ObtainBuffer(
					    source.address, source.size, false, program.info.buffers[i].formatted, source.id);
					cpu = buffers.IsRegionCpuModified(source.address, source.size);
					if (resolved == owner && !cpu) {
						const auto alignment  = m_context.GetGraphics().StorageMinAlignment();
						const auto aligned    = offset - offset % alignment;
						const auto adjustment = offset - aligned;
						if (adjustment < 256) {
							prepared->buffers[i] = {resolved->Handle(), aligned, source.size + adjustment};
							auto& word = prepared->shader_data[program.bindings.memory_offset_dword + i / 4u];
							const auto shift = (i % 4u) * 8u;
							word = (word & ~(0xffu << shift)) | (static_cast<uint32_t>(adjustment) << shift);
						}
					}
				}
				if (owner == nullptr || owner->Handle() != prepared->buffers[i].buffer || cpu) {
					return "dynamic_buffer";
				}
				// ObtainBuffer refreshes a texel buffer from a GPU-owned image that starts
				// exactly at its address (SynchronizeBufferFromImage); the record cannot.
				// Any image starting there rejects it.
				const bool texel = program.info.buffers[i].formatted;
				if (texel && textures.HasImageStartingAt(source.address)) {
					return "texel_image";
				}
				buffer_deps.push_back({source.address, source.size, source.id, prepared->buffers[i].buffer, texel, st,
				                       static_cast<uint32_t>(i)});
			}
			for (const auto& image: prepared->images) {
				// A null image (an empty texture descriptor: a texture not streamed in yet) is bound as the normal
				// path binds it (the null image, in GENERAL); a texture arriving there rebinds the record.
				if (image.desc.info.data.Empty()) continue;
				if (image.layout == vk::ImageLayout::eGeneral) {
					return "general_image";
				}
				if (!textures.IsSampledImageCurrent(image.image_id, image.desc)) return "image_not_current";
				const auto& view = image.desc.view_info;
				image_deps.push_back({image.image_id, image.layout,
				                      {view.base_level, view.level_count, view.base_layer, view.layer_count},
				                      textures.m_slot_images[image.image_id].serial,
				                      textures.m_slot_images[image.image_id].registrations, image.desc});
			}
		}
	}

	// Stage data and the descriptor payload; pointers once the vectors stop growing.
	struct Pending {
		vk::WriteDescriptorSet write;
		size_t                 start = 0;
		bool                   image = false;
	};
	std::vector<Pending>                  pending;
	std::vector<vk::DescriptorBufferInfo> buffer_infos;
	std::vector<vk::DescriptorImageInfo>  image_infos;
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	const auto fixed_view = [](const Buffer* value) -> vk::DescriptorBufferInfo {
		EXIT_IF(value == nullptr);
		return {value->Handle(), 0, value->Size()};
	};
	const auto upload = buffers.GetShaderUploadBuffer().Handle();
	std::array<NativeXprRecord::Stage, 2> data;
	std::array<std::vector<std::vector<std::pair<uint32_t, uint32_t>>>, 2> positions;
	std::array<std::vector<std::vector<uint32_t>>, 2>                     buffer_positions;
	vk::ShaderStageFlags push_stages = vk::ShaderStageFlagBits::eFragment;
	bool                 has_push    = false;
	for (uint32_t index = 0; index < stages.size(); ++index) {
		const auto& prepared = *stages[index];
		const auto& program  = *prepared.runtime->program;
		auto&       stage    = data[index];
		stage.program        = &program;
		stage.flattened      = prepared.runtime->resources.flattened_srt;
		stage.shader_data    = prepared.shader_data;
		stage.snapshot       = prepared.runtime->resources;
		EXIT_IF(stage.shader_data.size() != program.bindings.ShaderDataDwords());
		if (program.bindings.UsesPushData()) {
			stage.push_start = program.bindings.push_data_start_dword;
			has_push         = true;
		}
		push_stages |= program.stage == ShaderType::Pixel  ? vk::ShaderStageFlagBits::eFragment
		               : program.stage == ShaderType::Mesh ? vk::ShaderStageFlagBits::eMeshEXT
		                                                   : vk::ShaderStageFlagBits::eVertex;
		std::vector<uint32_t> occurrences(prepared.images.size(), 0);
		positions[index].assign(prepared.images.size(), {});
		buffer_positions[index].assign(prepared.buffers.size(), {});
		for (const auto& binding: program.bindings.descriptors) {
			Pending item;
			item.write.dstBinding      = ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind);
			item.write.descriptorType  = NativeDescriptorType(binding.kind);
			item.write.descriptorCount = NativeDescriptorCount(binding);
			item.image = ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) !=
			                 ShaderRecompiler::IR::ImageResourceClass::None ||
			             binding.kind == Kind::Samplers;
			item.start = item.image ? image_infos.size() : buffer_infos.size();
			if (binding.kind == Kind::Samplers) {
				for (const auto resource: binding.resources) {
					const auto sampler = prepared.samplers.at(resource);
					EXIT_IF(sampler == nullptr);
					image_infos.emplace_back(sampler, nullptr, vk::ImageLayout::eUndefined);
				}
			} else if (item.image) {
				for (const auto resource: binding.resources) {
					const auto element = occurrences.at(resource)++;
					positions[index].at(resource).emplace_back(static_cast<uint32_t>(image_infos.size()), element);
					image_infos.push_back(MakeImageInfo(prepared.images.at(resource), element));
				}
			} else {
				switch (binding.kind) {
					case Kind::Buffers:
						for (const auto resource: binding.resources) {
							const auto& view = prepared.buffers.at(resource);
							EXIT_IF(view.buffer == nullptr);
							buffer_positions[index].at(resource).push_back(static_cast<uint32_t>(buffer_infos.size()));
							buffer_infos.push_back(view);
						}
						break;
					case Kind::LodStats: buffer_infos.push_back(fixed_view(buffers.GetLodStatsBuffer())); break;
					case Kind::BdaPagetable: buffer_infos.push_back(fixed_view(buffers.GetBdaPageTableBuffer())); break;
					case Kind::FaultBuffer: buffer_infos.push_back(fixed_view(buffers.GetFaultBuffer())); break;
					case Kind::FlattenedSrt:
					case Kind::ShaderData: {
						const auto& values = binding.kind == Kind::FlattenedSrt ? stage.flattened : stage.shader_data;
						if (values.empty()) return "empty_dynamic";
						(binding.kind == Kind::FlattenedSrt ? stage.srt_binding : stage.data_binding) = true;
						item.write.descriptorType = vk::DescriptorType::eStorageBufferDynamic;
						buffer_infos.emplace_back(upload, 0, values.size() * sizeof(uint32_t));
						break;
					}
					default: return "binding_kind";
				}
			}
			EXIT_IF((item.image ? image_infos.size() : buffer_infos.size()) - item.start != item.write.descriptorCount);
			pending.push_back(item);
		}
	}

	// Accepted: publish into the record.
	for (uint32_t index = 0; index < stages.size(); ++index) {
		auto& stage       = record.stages[index];
		stage.views.clear();
		for (const auto& image: stages[index]->images) stage.views.push_back(image.image_view);
		stage.samplers        = stages[index]->samplers;
		stage.textures         = stages[index]->images;
		stage.image_positions  = std::move(positions[index]);
		stage.buffer_positions = std::move(buffer_positions[index]);
		stage.program     = data[index].program;
		stage.flattened   = std::move(data[index].flattened);
		stage.shader_data = std::move(data[index].shader_data);
		stage.snapshot    = std::move(data[index].snapshot);
		stage.srt_binding  = data[index].srt_binding;
		stage.data_binding = data[index].data_binding;
		stage.push_start   = data[index].push_start;
	}
	record.stage_count = static_cast<uint32_t>(stages.size());
	record.push_stages = push_stages;
	record.has_push    = has_push;
	record.upload      = upload;
	// A new image object can get a destroyed view's handle, and a new buffer a destroyed buffer's (the
	// driver's handles are its objects' addresses): an identical payload keeps the set only over the same
	// image and buffer objects. A set kept over a buffer the cache destroyed and made again at the same
	// range holds the old one's device address: the GPU reads freed memory and the device is lost.
	const bool same_objects =
	    !runtime || (std::ranges::equal(record.images, image_deps,
	                                    [](const auto& a, const auto& b) { return a.id == b.id && a.serial == b.serial; }) &&
	                 std::ranges::equal(record.buffers, buffer_deps, [](const auto& a, const auto& b) { return a.id == b.id; }));
	if (runtime) {
		record.buffers            = std::move(buffer_deps);
		record.images             = std::move(image_deps);
		record.IndexImages();
		record.mapping_epoch      = resources.MappingEpoch();
		record.registration_epoch = buffers.RegistrationEpoch();
		record.resolution_epoch   = textures.ResolutionEpoch();
		record.start_epoch        = start_epoch;
		uint64_t cpu_epoch        = 0;
		for (const auto& dep: record.buffers) {
			const auto epoch = buffers.CpuModificationEpoch(dep.address, dep.size);
			if (epoch == 0) {
				cpu_epoch = 0;
				break;
			}
			cpu_epoch += epoch;
		}
		record.cpu_epoch       = cpu_epoch;
		record.validated_frame = m_context.FrameNumber();
	}
	const auto same_payload = [](const auto& a, const auto& b) {
		return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
	};
	{
		std::array<const ShaderRecompiler::IR::ResourceSnapshot*, 2> snapshots {};
		for (uint32_t st = 0; st < record.stage_count; ++st) snapshots[st] = &record.stages[st].snapshot;
		record.image_signature = NativeXprImageSignature({snapshots.data(), record.stage_count});
	}
	if (record.set && same_objects && same_payload(record.set_buffers, buffer_infos) &&
	    same_payload(record.set_images, image_infos))
		return nullptr;
	NativeXprDropVariants(record);
	NativeXprRetireSet(record.pool, record.set);
	record.set = NativeXprAllocateSet(layout, record.pool);
	std::vector<vk::WriteDescriptorSet> writes;
	writes.reserve(pending.size());
	for (auto& item: pending) {
		item.write.dstSet = record.set;
		if (item.image) item.write.pImageInfo = image_infos.data() + item.start;
		else item.write.pBufferInfo = buffer_infos.data() + item.start;
		writes.push_back(item.write);
	}
	if (!writes.empty())
		m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0,
		                                                    nullptr);
	record.set_buffers = std::move(buffer_infos);
	record.set_images  = std::move(image_infos);
	record.set_writes.clear();
	for (const auto& item: pending)
		record.set_writes.push_back({item.write.dstBinding, item.write.descriptorCount, static_cast<uint32_t>(item.start),
		                             item.write.descriptorType, item.image});
	return nullptr;
}

// The record's images changed back to those of an earlier set: bind that set
// again if its images are all still current (as a record proof holds them) and
// CPU-clean; the current set becomes the variant.
bool RenderExecutor::NativeXprSwitchVariant(NativeXprRecord& record, uint64_t signature) {
	const auto found = std::ranges::find(record.variants, signature, &NativeXprRecord::Variant::signature);
	if (found == record.variants.end()) return false;
	auto& textures = m_context.GetTextureCache();
	for (const auto& dep: found->images) {
		const auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || image->serial != dep.serial || image->IsCpuDirty() ||
		    !textures.IsSampledImageCurrent(dep.id, dep.desc)) {
			NativeXprRetireSet(found->pool, found->set);
			record.variants.erase(found);
			return false;
		}
	}
	for (const auto& dep: found->images) (void)textures.GetImage(dep.id); // LRU, as FindTexture does
	auto& variant = *found;
	std::swap(variant.signature, record.image_signature);
	std::swap(variant.set, record.set);
	std::swap(variant.pool, record.pool);
	std::swap(variant.set_images, record.set_images);
	std::swap(variant.images, record.images);
	record.IndexImages();
	for (uint32_t st = 0; st < record.stage_count; ++st) {
		std::swap(variant.textures[st], record.stages[st].textures);
		std::swap(variant.views[st], record.stages[st].views);
	}
	record.resolution_epoch = textures.ResolutionEpoch();
	return true;
}

// Only image or buffer base addresses changed: resolve those slots again (the
// texture resolve cache makes a repeated image cheap), replace their
// descriptors in the stored payload and bind a new set. False when a slot does
// not resolve to what a record may hold; the caller then resolves everything.
bool RenderExecutor::NativeXprRebindSlots(NativeXprRecord& record,
                                          std::span<const std::pair<uint32_t, uint32_t>> images,
                                          std::span<const std::pair<uint32_t, uint32_t>> buffer_slots) {
	if (record.pipelines.empty() || record.set_writes.empty()) return false;
	auto& textures = m_context.GetTextureCache();
	auto& buffers  = m_context.GetBufferCache();
	// The set's other buffers are written again as they are (and an earlier set holds them too): each must
	// still be the object resolved, else the write names a destroyed buffer. Otherwise resolve everything.
	for (const auto& dep: record.buffers) {
		if (std::ranges::find(buffer_slots, std::pair {dep.stage, dep.slot}) != buffer_slots.end()) continue;
		const auto* owner = buffers.GetBufferIfLive(dep.id);
		if (owner == nullptr || owner->Handle() != dep.handle) return false;
	}
	std::array<const ShaderRecompiler::IR::ResourceSnapshot*, 2> snapshots {};
	for (uint32_t st = 0; st < record.stage_count; ++st) snapshots[st] = &record.stages[st].snapshot;
	const auto signature = NativeXprImageSignature({snapshots.data(), record.stage_count});
	if (buffer_slots.empty() && NativeXprSwitchVariant(record, signature)) return true;
	// All or nothing: the full rebind keeps the set when its payload equals the
	// stored one, so the stored payload must never run ahead of the set.
	auto& resolved = m_native_xpr->rebind_textures;
	resolved.clear();
	for (const auto& [st, index]: images) {
		auto& stage = record.stages[st];
		if (index >= stage.textures.size() || index >= stage.image_positions.size() ||
		    index >= stage.program->info.images.size())
			return false;
		const auto& resource = stage.program->info.images[index];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::DynamicStorage) return false;
		auto binding = ResolveTexture(resource, stage.snapshot.images[index]);
		if (binding.desc.type != TextureCache::BindingType::Texture) return false;
		if (binding.desc.info.data.Empty()) {
			// The normal path binds the null image in GENERAL; one it never bound yet is not.
			auto* null_image = textures.m_slot_images.try_get(binding.image_id);
			if (null_image == nullptr || null_image->backing.state.layout != vk::ImageLayout::eGeneral)
				return false;
			binding.image_view = textures.FindTexture(binding.image_id, binding.desc);
			binding.mip_views.clear();
			binding.layout = vk::ImageLayout::eGeneral;
			resolved.push_back(std::move(binding));
			continue;
		}
		auto* image = textures.m_slot_images.try_get(binding.image_id);
		if (image == nullptr || image->binding.is_target || image->binding.force_general) return false;
		binding.image_view = textures.FindTexture(binding.image_id, binding.desc);
		binding.mip_views.clear();
		auto& owner = textures.GetImage(binding.image_id);
		owner.usage.texture = true;
		binding.layout = owner.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
		                                      : vk::ImageLayout::eShaderReadOnlyOptimal;
		if (!textures.IsSampledImageCurrent(binding.image_id, binding.desc)) return false;
		resolved.push_back(std::move(binding));
	}
	// Buffers as FindBuffers/RebindBuffers resolve them, held as a store holds
	// them: a live owner, CPU-clean (cleaned here as the store cleans it), no
	// image starting at a texel range.
	struct BufferSlot {
		vk::DescriptorBufferInfo  info;
		uint32_t                  adjustment = 0;
		NativeXprRecord::BufferDep dep;
	};
	std::array<BufferSlot, 16> fresh_buffers;
	if (buffer_slots.size() > fresh_buffers.size()) return false;
	const auto  alignment = m_context.GetGraphics().StorageMinAlignment();
	const auto  max_range = m_context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	for (size_t i = 0; i < buffer_slots.size(); ++i) {
		const auto [st, index] = buffer_slots[i];
		auto& stage            = record.stages[st];
		if (index >= stage.program->info.buffers.size() || index >= stage.buffer_positions.size() ||
		    index >= stage.snapshot.buffers.size())
			return false;
		const auto& resource = stage.program->info.buffers[index];
		if (resource.written || resource.atomic) return false;
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(stage.snapshot.buffers[index]);
		const auto address    = descriptor.Base48();
		const auto stride     = descriptor.Stride();
		const auto records    = descriptor.NumRecords();
		const auto requested  = stride != 0 ? static_cast<uint64_t>(stride) * records : records;
		if (address == 0 || requested == 0) return false;
		const auto size  = Libs::LibKernel::Memory::ClampRangeSize(address, requested);
		if (size == 0) return false;
		const auto id    = buffers.FindBuffer(address, size);
		const auto* owner = buffers.GetBufferIfLive(id);
		if (owner == nullptr) return false;
		if (buffers.IsRegionCpuModified(address, size)) {
			if (buffers.IsRegionGpuModified(address, size)) return false;
			buffers.EnsureBufferContents(address, size);
		}
		const auto [served, offset] = buffers.ObtainBuffer(address, size, false, resource.formatted, id);
		if (served != owner || buffers.IsRegionCpuModified(address, size)) return false;
		if (resource.formatted && textures.HasImageStartingAt(address)) return false;
		const auto aligned    = offset - offset % alignment;
		const auto adjustment = offset - aligned;
		if (adjustment >= 256 || size > max_range - adjustment) return false;
		fresh_buffers[i] = {{owner->Handle(), aligned, size + adjustment},
		                    static_cast<uint32_t>(adjustment),
		                    {address, size, id, owner->Handle(), resource.formatted, st, index}};
	}
	// The set's other images are written again as they are: their objects (and
	// so their views) must still be the ones resolved. Otherwise resolve everything.
	for (const auto& dep: record.images) {
		const auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || !image->registered || image->serial != dep.serial) return false;
	}
	if (buffer_slots.empty()) {
		// What the current set holds, kept with it as a variant below.
		auto& cache            = *m_native_xpr;
		cache.previous_images  = record.set_images;
		cache.previous_deps    = record.images;
		for (uint32_t st = 0; st < record.stage_count; ++st) {
			cache.previous_textures[st] = record.stages[st].textures;
			cache.previous_views[st]    = record.stages[st].views;
		}
	}
	for (size_t i = 0; i < images.size(); ++i) {
		const auto [st, index] = images[i];
		auto& stage            = record.stages[st];
		for (const auto& [position, element]: stage.image_positions[index])
			record.set_images[position] = MakeImageInfo(resolved[i], element);
		stage.views[index]    = resolved[i].image_view;
		stage.textures[index] = std::move(resolved[i]);
	}
	for (size_t i = 0; i < buffer_slots.size(); ++i) {
		const auto [st, index] = buffer_slots[i];
		auto&       stage      = record.stages[st];
		const auto& update     = fresh_buffers[i];
		for (const auto position: stage.buffer_positions[index]) record.set_buffers[position] = update.info;
		auto&      word  = stage.shader_data[stage.program->bindings.memory_offset_dword + index / 4u];
		const auto shift = (index % 4u) * 8u;
		word             = (word & ~(0xffu << shift)) | (update.adjustment << shift);
		const auto dep   = std::ranges::find_if(record.buffers, [&](const NativeXprRecord::BufferDep& item) {
            return item.stage == st && item.slot == index;
        });
		if (dep != record.buffers.end()) *dep = update.dep;
		else record.buffers.push_back(update.dep);
	}
	if (!images.empty()) {
		// Proof dependencies follow the resolved images.
		record.images.clear();
		for (uint32_t st = 0; st < record.stage_count; ++st)
			for (const auto& image: record.stages[st].textures) {
				if (image.desc.info.data.Empty()) continue;
				const auto& view = image.desc.view_info;
				record.images.push_back({image.image_id, image.layout,
				                         {view.base_level, view.level_count, view.base_layer, view.layer_count},
				                         textures.m_slot_images[image.image_id].serial,
				                         textures.m_slot_images[image.image_id].registrations, image.desc});
			}
		record.IndexImages();
		record.resolution_epoch = textures.ResolutionEpoch();
	}
	// New buffer dependencies are proven again before the draw.
	if (!buffer_slots.empty()) record.validated_frame = 0;
	if (buffer_slots.empty() && record.set) {
		// The old set differs only in images: keep it for a switch back.
		if (record.variants.size() >= 3) {
			NativeXprRetireSet(record.variants.front().pool, record.variants.front().set);
			record.variants.erase(record.variants.begin());
		}
		auto& kept      = record.variants.emplace_back();
		kept.signature  = record.image_signature;
		kept.set        = record.set;
		kept.pool       = record.pool;
		kept.set_images = m_native_xpr->previous_images;
		kept.images     = std::move(m_native_xpr->previous_deps);
		for (uint32_t st = 0; st < record.stage_count; ++st) {
			kept.textures[st] = std::move(m_native_xpr->previous_textures[st]);
			kept.views[st]    = std::move(m_native_xpr->previous_views[st]);
		}
	} else {
		NativeXprDropVariants(record);
		NativeXprRetireSet(record.pool, record.set);
	}
	record.image_signature = signature;
	record.set = NativeXprAllocateSet(record.pipelines.front().second->descriptor_set_layout, record.pool);
	std::vector<vk::WriteDescriptorSet> writes;
	writes.reserve(record.set_writes.size());
	for (const auto& slot: record.set_writes) {
		vk::WriteDescriptorSet write {};
		write.dstSet          = record.set;
		write.dstBinding      = slot.binding;
		write.descriptorCount = slot.count;
		write.descriptorType  = slot.type;
		if (slot.image) write.pImageInfo = record.set_images.data() + slot.start;
		else write.pBufferInfo = record.set_buffers.data() + slot.start;
		writes.push_back(write);
	}
	m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0,
	                                                    nullptr);
	return true;
}

// A record whose descriptors changed: resolve every slot again the way the
// normal path's preparation does, with plain sampled-image layouts, and bind.
bool RenderExecutor::NativeXprRebind(NativeXprRecord&                                        record,
                                     std::array<ShaderRecompiler::IR::ResourceSnapshot, 2>& fresh) {
	if (record.pipelines.empty()) return false;
	std::array<ShaderStageRuntime, 2> runtimes;
	std::array<PreparedBindings, 2>   prepared;
	std::array<PreparedBindings*, 2>  stages {};
	auto&                             textures = m_context.GetTextureCache();
	bool                              ok       = true;
	for (uint32_t st = 0; st < record.stage_count && ok; ++st) {
		runtimes[st].program   = record.stages[st].program;
		runtimes[st].resources = std::move(fresh[st]);
		PrepareBindingsInto(runtimes[st], prepared[st]);
		stages[st] = &prepared[st];
	}
	for (uint32_t st = 0; st < record.stage_count && ok; ++st) FindBuffers(prepared[st]);
	for (uint32_t st = 0; st < record.stage_count && ok; ++st) RebindBuffers(prepared[st]);
	for (uint32_t st = 0; st < record.stage_count && ok; ++st) RebindImages(prepared[st]);
	for (uint32_t st = 0; st < record.stage_count && ok; ++st)
		for (auto& binding: prepared[st].images) {
			auto& image = textures.GetImage(binding.image_id);
			if (image.info.data.Empty() && binding.desc.type != TextureCache::BindingType::Storage &&
			    image.backing.state.layout == vk::ImageLayout::eGeneral) {
				binding.layout = vk::ImageLayout::eGeneral;
				continue;
			}
			if (image.info.data.Empty() || binding.desc.type == TextureCache::BindingType::Storage ||
			    image.binding.is_target || image.binding.force_general) {
				ok = false;
				break;
			}
			binding.layout = image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
			                                      : vk::ImageLayout::eShaderReadOnlyOptimal;
		}
	ResetBindings();
	return ok && NativeXprBindResources(record, {stages.data(), record.stage_count},
	                                    record.pipelines.front().second->descriptor_set_layout,
	                                    true) == nullptr;
}

// A draw state's dynamic state into `vk_buffer`: all of it in another command buffer or generation (what any other
// path recorded is unknown), else the values that differ from what the table and native emissions recorded there.
// (A state change recorded all ~13 values, most unchanged: ~2% of the GPU thread on the 1-1 walk, 0.8% now.)
void RenderExecutor::NativeXprEmitDynamic(NativeXprCache& cache, vk::CommandBuffer vk_buffer,
                                          const NativeXprDrawState& draw_state, bool same_command) {
	auto&      emit = cache.emit;
	const bool diff = same_command && emit.dynamic_held;
	if (!diff) emit.dynamic = draw_state.resolved;
	EmitGraphicsDynamicParams(vk_buffer, draw_state.resolved, diff ? &emit.dynamic : nullptr, VULKAN_HPP_DEFAULT_DISPATCHER);
	emit.dynamic_held = true;
	const auto& state = draw_state.dynamic;
	if (state.feedback_enabled) {
		if (!diff || !emit.feedback_held || emit.feedback != state.feedback_aspects)
			vk_buffer.setAttachmentFeedbackLoopEnableEXT(state.feedback_aspects);
		emit.feedback      = state.feedback_aspects;
		emit.feedback_held = true;
	} else if (!diff) {
		emit.feedback_held = false;
	}
}

// The draw state under `state_key`: the render targets as this draw left them (a reopened rendering loads),
// their layouts and the dynamic state.
void RenderExecutor::NativeXprCaptureState(CommandBuffer& buffer, const DrawRenderState& state,
                                           const RenderState& rendering, uint64_t state_key) {
	auto& cache    = NativeXprState();
	auto& textures = m_context.GetTextureCache();
	NativeXprDrawState draw_state;
	draw_state.first_rendering = rendering;
	draw_state.rendering       = rendering;
	for (auto& attachment: draw_state.rendering.color_attachments) attachment.is_clear = false;
	draw_state.rendering.depth_stencil_attachment.depth_clear   = false;
	draw_state.rendering.depth_stencil_attachment.stencil_clear = false;
	const auto metadata_of = [](const TextureCache::ImageDesc& desc) {
		return desc.info.metadata.kind == ImageMetadataKind::None ? uint64_t {0}
		                                                          : desc.info.metadata.range.address;
	};
	for (uint32_t i = 0; i < state.color_count; ++i) {
		const auto& desc  = state.color_info[i].desc;
		const auto& view  = desc.view_info;
		const auto& image = textures.GetImage(state.color_info[i].image_id);
		draw_state.targets.push_back({state.color_info[i].image_id,
		                              rendering.color_attachments[i].image_layout,
		                              image.binding.attachment_access,
		                              {view.base_level, view.level_count, view.base_layer, view.layer_count},
		                              metadata_of(desc), view.base_layer, image.serial, desc.info});
	}
	if (state.depth_info.image_id) {
		const auto& desc  = state.depth_info.desc;
		const auto& view  = desc.view_info;
		const auto& image = textures.GetImage(state.depth_info.image_id);
		draw_state.targets.push_back({state.depth_info.image_id,
		                              rendering.depth_stencil_attachment.image_layout,
		                              image.binding.attachment_access,
		                              {view.base_level, view.level_count, view.base_layer, view.layer_count},
		                              metadata_of(desc), view.base_layer, image.serial, desc.info});
	}
	draw_state.meta_epoch = 0;
	draw_state.dynamic = LocalDrawRecording::Capture(
	    buffer, state.vs_input_info, state.color_info, state.color_count, state.depth_info,
	    m_context.GetGraphics().attachment_feedback_loop_enabled,
	    rendering.depth_stencil_attachment.image_layout == vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
	        ? vk::ImageAspectFlags {vk::ImageAspectFlagBits::eDepth}
	        : vk::ImageAspectFlags {});
	draw_state.dynamic_hash = XXH3_64bits(&draw_state.dynamic, sizeof(draw_state.dynamic));
	draw_state.resolved     = ResolveGraphicsDynamicParams(draw_state.dynamic.buffer, draw_state.dynamic.indexed_viewports,
	                                                       draw_state.dynamic.colors, draw_state.dynamic.color_count,
	                                                       draw_state.dynamic.depth);
	// Entries stay at their address (emission tracking compares pointers); a
	// replaced state must be emitted again.
	const auto write = CatalogWrite();
	auto&      slot  = cache.states[state_key];
	if (!slot) slot = std::make_unique<NativeXprDrawState>();
	*slot = std::move(draw_state);
	if (cache.emit.state == slot.get()) cache.emit.state = nullptr;
}

// Called by ExecutePreparedDraw after CommitBindings (image layouts are final)
// for a draw whose caller requested a store.
void RenderExecutor::NativeXprStore(CommandBuffer& buffer, const DrawRenderState& state,
                                    vk::PrimitiveTopology topology, bool primitive_restart_enable,
                                    const RenderState& rendering,
                                    std::span<PreparedBindings* const> stages, const DrawEmitInfo& emit) {
	LiveCounters::Add(LiveCounters::XprStores);
	auto&      cache   = NativeXprState();
	const bool runtime = cache.logging;
	// What the programs are (stage layout, DMA/GDS, written resources) never changes for
	// the shader pair: it stops asking for stores (every request logs the draw's reads).
	const auto refuse_program = [&] {
		LiveCounters::Add(LiveCounters::XprRefuseProgram);
		std::vector<uint32_t> key;
		NativeXprKey(buffer, key);
		auto& pair = cache.pairs[NativeXprPairKey(key)];
		pair.unstorable = true;
	};
	if (stages.empty() || stages.size() > 2 || (stages.size() == 2) != state.ps_active) return refuse_program();
	// Only stages without DMA, GDS or any written resource are recorded.
	for (const auto* prepared: stages) {
		const auto& program = *prepared->runtime->program;
		if (program.info.uses_dma || prepared->gds.buffer != nullptr) return refuse_program();
		for (const auto& resource: program.info.buffers)
			if (resource.written || resource.atomic) return refuse_program();
		for (const auto& image: program.info.images)
			if (image.written || image.atomic) return refuse_program();
		for (const auto& image: prepared->images)
			if (image.desc.type == TextureCache::BindingType::Storage) return refuse_program();
	}

	auto       record    = std::make_unique<NativeXprRecord>();
	const auto state_key = NativeXprStateKey(buffer);
	auto&      pipelines = m_context.GetPipelineCache();
	NativeXprKey(buffer, record->key);
	auto& pair_stats = cache.pairs[NativeXprPairKey(record->key)];
	// A refused key is not requested again for a while (NativeXprCache::Refused).
	const auto refuse = [&] { cache.Refuse(cache.tried_hash, m_context.FrameNumber()); };
	// Relocation (NativeXprTry): the key words of user data pairs the SRTs only read through
	// (pointers to a new copy of the same tables every frame) and of user data nothing reads leave
	// the key; the gather reads through the pointers. The VS's user data are the GS SGPRs (NGG ones
	// from s8); the vertex tables a static vertex input reads (ShaderGetStaticInputInfoVS) stay.
	std::vector<NativeXprRecord::Root>   roots;
	std::vector<NativeXprRecord::Unread> unread;
	if (kyty_local_native_xpr_relocate_mode.load(std::memory_order_relaxed) != 0) {
		const auto&    key   = record->key;
		const uint32_t hs    = key[4], gs = key[5 + hs], ps = key[6 + hs + gs];
		const uint32_t shift = (buffer.GetRegisters().GetShaderStages() & 0x20u) != 0 ? 8 : 0;
		uint64_t       words = 0;
		for (uint32_t st = 0; st < stages.size(); ++st) {
			const auto* program   = stages[st]->runtime->program;
			auto [use, unknown]   = cache.user_data.try_emplace(program);
			if (unknown) use->second = NativeXprUserDataOf(pipelines, *program);
			uint32_t fetch = 0;
			if (st == 0 && state.vs_input_info.fetch_embedded)
				for (const auto reg: {state.vs_input_info.fetch_attrib_reg, state.vs_input_info.fetch_buffer_reg})
					if (reg >= 0 && reg + shift < 31) fetch |= 3u << (reg + shift);
			// The key word of the stage's user data k, or -1.
			const auto word_of = [&](uint32_t k) {
				const auto j    = st == 0 ? int64_t {k} - shift : int64_t {k};
				const auto word = (st == 0 ? 6 + hs : 7 + hs + gs) + j;
				return j < 0 || j >= (st == 0 ? gs : ps) || word >= 64 ? int64_t {-1} : word;
			};
			for (auto pairs = pair_stats.pointers_off ? 0u : use->second.pointers & ~fetch & ~(fetch >> 1); pairs != 0;
			     pairs &= pairs - 1) {
				const auto k = static_cast<uint32_t>(std::countr_zero(pairs));
				const auto w = word_of(k);
				if (w < 0 || word_of(k + 1) < 0) continue;
				words |= 3ull << w;
				roots.push_back({static_cast<uint8_t>(st), static_cast<uint8_t>(k), static_cast<uint8_t>(w),
				                 static_cast<uint8_t>(w + 1), NativeXprUserDataWord(*program, k),
				                 NativeXprUserDataWord(*program, k + 1)});
			}
			for (auto bits = use->second.unread & ~fetch; bits != 0; bits &= bits - 1) {
				const auto k = static_cast<uint32_t>(std::countr_zero(bits));
				const auto w = word_of(k);
				if (w < 0) continue;
				words |= 1ull << w;
				unread.push_back({static_cast<uint8_t>(st), static_cast<uint8_t>(k), static_cast<uint8_t>(w)});
			}
		}
		if (!pair_stats.relocation_known) {
			pair_stats.relocation_known = true;
			pair_stats.relocate         = words;
		}
		// A relocated record comes from a try under the same words (its use in the frame); other tries
		// (indirect runs, a pair's first) store whole keys.
		if (cache.tried_relocate == 0) {
			roots.clear();
			unread.clear();
		} else if (words != cache.tried_relocate) {
			return refuse();
		}
	}
	const auto colors    = std::span {state.color_info, state.color_count};
	const auto* pixel    = state.ps_active ? &state.ps_input_info : nullptr;
	auto* const variant  = kyty_local_async_xpr_pipelines_mode.load(std::memory_order_relaxed) != 0
	                          ? pipelines.TryCreateGraphicsPipeline(colors, state.depth_info, state.vs_input_info,
	                                                                buffer, pixel, topology, primitive_restart_enable,
	                                                                state.programs.vertex, state.programs.pixel, true)
	                          : &pipelines.CreateGraphicsPipeline(colors, state.depth_info, state.vs_input_info,
	                                                              buffer, pixel, topology, primitive_restart_enable,
	                                                              state.programs.vertex, state.programs.pixel, true);
	if (variant == nullptr) {
		LiveCounters::Add(LiveCounters::XprStorePending);
		return;
	}
	auto& pipeline = *variant;
	EXIT_IF(!pipeline.native_bindings || pipeline.uses_push_descriptors);
	record->vertex_program = state.programs.vertex.id;
	record->pixel_program  = state.ps_active ? state.programs.pixel.id : 0;
	record->pipelines.emplace_back(state_key, &pipeline);
	if (emit.state_offsets)
		record->direct_offsets.push_back({state_key, emit.vertex_offset, emit.first_instance});
	record->stage_count = static_cast<uint32_t>(stages.size());
	if (runtime) {
		uint64_t total_words = 0;
		for (const auto& [address, size]: cache.reads) {
			// (Reads of the null page are zeros the record could not read again.)
			if (address % 4 != 0 || size % 4 != 0 || size == 0 || size > 4096 || address < 0x10000) {
				refuse();
				return LiveCounters::Add(LiveCounters::XprRefuseReads);
			}
			total_words += size / 4;
		}
		if (total_words > 2048) {
			refuse();
			return LiveCounters::Add(LiveCounters::XprRefuseReads);
		}
		record->reads.reserve(cache.reads.size());
		record->words.reserve(total_words);
		for (const auto& [address, size]: cache.reads) {
			const auto count = static_cast<uint32_t>(size / 4);
			record->reads.emplace_back(address, count);
			record->words.resize(record->words.size() + count);
			NativeXprReadWords(address, count, record->words.data() + record->words.size() - count);
		}
		// What RefreshShaders evaluated each stage's SRT with.
		const auto& sh  = buffer.GetShaders();
		const auto& ctx = buffer.GetRegisters();
		ShaderVertexInputInfo vertex_info;
		ResetNativeVertexInput(vertex_info);
		const auto vertex_params = PrepareProgram(sh.GetVs(), ctx, buffer.GetUserConfig(), vertex_info);
		record->stages[0].user_data.assign(vertex_params.user_data.begin(), vertex_params.user_data.end());
		record->stages[0].shader_base = vertex_params.Base();
		if (stages.size() == 2) {
			std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
			for (uint32_t i = 0; i < state.color_count; i++)
				mapping[state.color_info[i].target_slot] = state.color_info[i].export_mapping;
			ShaderPixelInputInfo pixel_info;
			ResetNativeStageInput(pixel_info);
			const auto pixel_params = PrepareProgram(sh.GetPs(), ctx.GetShaderRegisters(), mapping, pixel_info);
			record->stages[1].user_data.assign(pixel_params.user_data.begin(), pixel_params.user_data.end());
			record->stages[1].shader_base = pixel_params.Base();
		}
	}
	if (NativeXprBindResources(*record, stages, pipeline.descriptor_set_layout, runtime) != nullptr) {
		refuse();
		return LiveCounters::Add(LiveCounters::XprRefuseBind);
	}
	// The roots' pointers: the user data the SRTs were evaluated with, as the key and the shader data
	// hold them.
	for (auto& root: roots) {
		const auto& stage = record->stages[root.stage];
		const auto& user  = stage.user_data;
		const auto  held  = [&](uint8_t data, uint32_t value) {
			return data == NativeXprNoRoot || (data < stage.shader_data.size() && stage.shader_data[data] == value);
		};
		if (!runtime || root.user + 1u >= user.size() || user != stage.snapshot.user_data ||
		    user[root.user] != record->key[root.low_word] || user[root.user + 1] != record->key[root.high_word] ||
		    !held(root.low_data, user[root.user]) || !held(root.high_data, user[root.user + 1]))
			return refuse();
		root.base = NativeXprBase(user[root.user], user[root.user + 1]);
		for (size_t r = 0; r < static_cast<size_t>(&root - roots.data()) && root.same == NativeXprNoRoot; ++r)
			if (roots[r].same == NativeXprNoRoot && record->key[roots[r].low_word] == record->key[root.low_word] &&
			    record->key[roots[r].high_word] == record->key[root.high_word])
				root.same = static_cast<uint8_t>(r);
	}
	// Unread words past the stage's user data reach no part of the draw.
	for (const auto& word: unread) {
		const auto& stage = record->stages[word.stage];
		if (!runtime || stage.user_data != stage.snapshot.user_data ||
		    (word.user < stage.user_data.size() && stage.user_data[word.user] != record->key[word.word]))
			return refuse();
	}
	record->roots  = std::move(roots);
	record->unread = std::move(unread);
	if (runtime) NativeXprRetrace(*record);
	if (!record->roots.empty() && !record->gather.traced) {
		// Without the reads through them the pointers stay in the pair's keys.
		pair_stats.pointers_off = true;
		for (const auto& root: record->roots) pair_stats.relocate &= ~(3ull << root.low_word);
		return refuse();
	}
	if (!record->roots.empty() && !pair_stats.identity_known) {
		using Kind                = ShaderRecompiler::IR::SrtReadTrace::Kind;
		pair_stats.identity_known = true;
		const auto& gather        = record->gather;
		// The span of a record root that holds the word at `index` (in `words`), or none.
		const auto table_of = [&](uint32_t index) -> const NativeXprRecord::Gather::Span* {
			for (const auto& span: gather.spans)
				if (span.level == 0 && span.root < record->roots.size() && index >= span.first && index < span.first + span.count)
					return &span;
			return nullptr;
		};
		for (const auto& span: gather.spans) {
			if (span.level != 1 || span.root < record->roots.size()) continue;
			const auto& pointer = gather.pointers[span.root - record->roots.size()];
			const auto  low = gather.value_word[pointer.low], high = gather.value_word[pointer.high];
			const auto* table = table_of(low);
			if (table == nullptr || high != low + 1 || table != table_of(high)) continue;
			const auto& root = record->roots[table->root];
			for (uint32_t i = 0; i < span.count && pair_stats.identity.size() < 8; ++i) {
				const auto v = gather.word_value[span.first + i];
				if (v < 0 || gather.pointer_value[v] != 0) continue;
				for (auto f = gather.feeds_at[v]; f < gather.feeds_at[v + 1]; ++f) {
					const auto kind = static_cast<Kind>((gather.feeds[f] >> 24u) & 0xfu);
					if ((kind == Kind::Buffer || kind == Kind::Image) && (gather.feeds[f] & 0xffu) == 0) {
						pair_stats.identity.push_back({root.low_word, root.high_word,
						                               static_cast<uint32_t>(table->address + 4u * (low - table->first)),
						                               static_cast<uint32_t>(span.address + 4u * i)});
						break;
					}
				}
			}
		}
	}

	NativeXprCaptureState(buffer, state, rendering, state_key);

	// A relocated key: without the roots' and the unread words, with the words the try added (NativeXprTry).
	if (!record->roots.empty() || !record->unread.empty()) {
		for (const auto& root: record->roots) record->key[root.low_word] = record->key[root.high_word] = 0;
		for (const auto& word: record->unread) record->key[word.word] = 0;
		record->key.insert(record->key.end(), cache.tried_suffix.begin(), cache.tried_suffix.end());
		LiveCounters::Add(LiveCounters::XprRelocatedStored);
	}
	record->HoldKey();
	const auto hash    = XXH3_64bits(record->key.data(), record->key.size() * sizeof(uint32_t));
	record->used_frame = m_context.FrameNumber();
	++pair_stats.stored;
	if (auto old = cache.records.find(hash); old != cache.records.end()) {
		// Pipelines depend on the programs and the draw state only; direct-draw offsets also on
		// the user data, which the equal key holds.
		if (old->second->vertex_program == record->vertex_program &&
		    old->second->pixel_program == record->pixel_program) {
			for (const auto& entry: old->second->pipelines)
				if (entry.first != state_key) record->pipelines.push_back(entry);
			for (const auto& entry: old->second->direct_offsets)
				if (entry.state != state_key) record->direct_offsets.push_back(entry);
		}
		cache.IndexErase(hash);
		NativeXprRetire(std::move(old->second), false);
		cache.records.erase(old);
	}
	if (runtime) {
		if (auto stale = cache.stale.find(hash); stale != cache.stale.end()) {
			if (stale->second->key == record->key) {
				NativeXprLearn(*stale->second, *record);
				if (stale->second->vertex_program == record->vertex_program &&
				    stale->second->pixel_program == record->pixel_program) {
					for (const auto& entry: stale->second->pipelines)
						if (std::ranges::find(record->pipelines, entry.first,
						                      &std::pair<uint64_t, const PipelineCache::Pipeline*>::first) ==
						    record->pipelines.end())
							record->pipelines.push_back(entry);
					for (const auto& entry: stale->second->direct_offsets)
						if (std::ranges::find(record->direct_offsets, entry.state, &NativeXprRecord::DirectOffsets::state) ==
						    record->direct_offsets.end())
							record->direct_offsets.push_back(entry);
				}
			}
			cache.stale.erase(stale);
		}
		if (record->word_patch.size() != record->words.size())
			record->word_patch.assign(record->words.size(), -1);
	}
	if (cache.records.size() >= NativeXprCache::RecordsHardCap) {
		if (SlowLog::HitchThreshold() > 0.0) {
			std::printf("[tsc %llu] SLOW XprReset 0.0 ms records=%zu\n", static_cast<unsigned long long>(__rdtsc()),
			            cache.records.size());
			std::fflush(stdout);
		}
		for (auto& [key, value]: cache.records) NativeXprRetire(std::move(value), false);
		cache.records.clear();
		cache.index.assign(8192, {});
		cache.index_count = 0;
	}
	cache.IndexInsert(hash, record.get());
	cache.records[hash] = std::move(record);
	LiveCounters::Add(LiveCounters::XprStored);
}

// Clean (specialization) reads go through the clean backing only; false when
// the bytes are GPU-owned and a fresh evaluation would fail too.
bool NativeXprReadClean(uint64_t address, uint32_t count, uint32_t* out) {
	for (uint32_t done = 0; done < count;) {
		const uint32_t chunk = std::min<uint32_t>(count - done, 16);
		const auto     at    = address + 4ull * done;
		if (chunk < 2 || !Libs::LibKernel::Memory::TryReadGpuShaderSpan(at, out + done, chunk * 4ull, true))
			for (uint32_t i = 0; i < chunk; ++i)
				if (!Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(at + 4ull * i, out + done + i, 4))
					return false;
		done += chunk;
	}
	return true;
}

// Traces every stage again and rebuilds the gather plan; not traced on failure.
void RenderExecutor::NativeXprRetrace(NativeXprRecord& record) {
	auto& gather  = record.gather;
	gather.traced = false;
	if (record.stages[0].shader_base == 0) return;
	// `root`: the record root the address is relative to (relocation), or NativeXprNoRoot.
	struct StructuralRead {
		uint64_t address;
		uint32_t value;
		uint8_t  use;
		uint8_t  root;
		bool     operator<(const StructuralRead& other) const { return address < other.address; }
	};
	struct DataRead {
		uint64_t address;
		uint32_t target; // stage << 24 | flat
		uint8_t  root;
		bool     operator<(const DataRead& other) const {
			return address != other.address ? address < other.address : target < other.target;
		}
	};
	// Spans over the non-clean structural words and the data words.
	struct Word {
		uint64_t address;
		int32_t  value;
		int32_t  data;
		uint8_t  root;
		uint8_t  level;
	};
	// Relocated records: a pointer read from memory (its halves' addresses and the roots they are read
	// through), whether what is read through it moves with it, and its root and level as a span root.
	struct Pointer {
		uint64_t low, high;
		uint8_t  low_root, high_root;
		bool     usable = true;
		uint8_t  root = NativeXprNoRoot, level = 0;
	};
	using Trace = ShaderRecompiler::IR::SrtReadTrace;
	// Scratch kept across calls (a retrace per stored record: tens to hundreds a frame
	// while walking, each with some twenty temporary vectors otherwise).
	thread_local struct {
		std::vector<StructuralRead>                structural, clean, unique;
		std::vector<std::pair<uint64_t, uint32_t>> feeds, index_of;
		std::vector<DataRead>                      data;
		std::vector<std::pair<uint64_t, uint8_t>>  roots;
		std::vector<uint32_t>                      feed_value, cursor;
		std::vector<Word>                          words;
		std::vector<Pointer>                       pointers;
		std::vector<uint8_t>                       read_root, levels;
		Trace                                      trace;
	} scratch;
	auto& structural = scratch.structural;
	auto& clean      = scratch.clean;
	auto& feeds      = scratch.feeds; // (address, packed feed)
	auto& data       = scratch.data;
	auto& pointers   = scratch.pointers;
	structural.clear();
	clean.clear();
	feeds.clear();
	data.clear();
	pointers.clear();
	const auto user_roots = static_cast<uint8_t>(record.roots.size());
	// The record root of stage `st`'s reads through user data pair `user`.
	const auto root_of = [&](uint32_t st, uint8_t user) {
		for (size_t r = 0; user != Trace::NoRoot && r < record.roots.size(); ++r) {
			const auto& root = record.roots[r];
			if (root.stage == st && root.user == user) return root.same != NativeXprNoRoot ? root.same : static_cast<uint8_t>(r);
		}
		return NativeXprNoRoot;
	};
	auto& pipelines   = m_context.GetPipelineCache();
	auto& untraceable = m_native_xpr->untraceable;
	for (uint32_t st = 0; st < record.stage_count; ++st)
		if (untraceable.contains(record.stages[st].program)) return;
	for (uint32_t st = 0; st < record.stage_count; ++st) {
		const auto& stage = record.stages[st];
		auto&       trace = scratch.trace;
		if (!pipelines.TraceStage(*stage.program, stage.user_data, stage.shader_base, trace)) {
			untraceable.insert(stage.program);
			return;
		}
		if (trace.flat_words != stage.flattened.size() || (trace.unrooted && !record.roots.empty())) {
			return;
		}
		// A read's root: a user data pair's, or (relocated records) the pointer two earlier reads make.
		auto&      read_root = scratch.read_root;
		const auto base_of   = [&](const Trace::Base& base) {
			if (base.root != Trace::NoRoot) return root_of(st, base.root);
			if (base.low == Trace::NoBase || user_roots == 0) return NativeXprNoRoot;
			const auto& low  = trace.structural[base.low];
			const auto& high = trace.structural[base.high];
			if (low.clean || high.clean) return NativeXprNoRoot;
			for (size_t p = 0; p < pointers.size(); ++p)
				if (pointers[p].low == low.address && pointers[p].high == high.address)
					return static_cast<uint8_t>(user_roots + p);
			if (user_roots + pointers.size() >= NativeXprNoRoot) return NativeXprNoRoot;
			pointers.push_back({low.address, high.address, read_root[base.low], read_root[base.high]});
			return static_cast<uint8_t>(user_roots + pointers.size() - 1);
		};
		read_root.assign(trace.structural.size(), NativeXprNoRoot);
		for (size_t i = 0; i < trace.structural.size(); ++i) read_root[i] = base_of(trace.structural[i].base);
		for (size_t i = 0; i < trace.data.size(); ++i)
			data.push_back({trace.data[i].second, st << 24u | trace.data[i].first, base_of(trace.data_bases[i])});
		for (size_t i = 0; i < trace.structural.size(); ++i) {
			const auto& read = trace.structural[i];
			(read.clean ? clean : structural).push_back({read.address, read.value, read.use, read_root[i]});
		}
		for (const auto& feed: trace.feeds) {
			if (feed.index >= 0x10000 || feed.dword >= 8) continue;
			feeds.emplace_back(trace.structural[feed.read].address,
			                   st << 28u | static_cast<uint32_t>(feed.kind) << 24u | feed.index << 8u | feed.dword);
		}
		// Flat copies of structural words: kind NativeXprFlatFeed, the flat index in bits 0-23.
		for (const auto& [read, flat]: trace.flat_feeds) {
			if (flat >= 0x1000000) continue;
			feeds.emplace_back(trace.structural[read].address, st << 28u | NativeXprFlatFeed << 24u | flat);
		}
	}
	// A pointer moves what is read through it when its halves are read for nothing else (no other
	// address, descriptor or predicate use, no clean read; flattened copies follow); else those reads
	// stay at their addresses, and a change of the pointer evaluates again.
	auto& levels = scratch.levels;
	levels.clear();
	if (!pointers.empty()) {
		for (auto& pointer: pointers) {
			const auto half = [&](uint64_t address) { return address == pointer.low || address == pointer.high; };
			for (const auto& read: structural)
				if (half(read.address) && read.use != Trace::Pointer) pointer.usable = false;
			for (const auto& read: clean)
				if (half(read.address)) pointer.usable = false;
			for (const auto& [address, packed]: feeds)
				if (half(address) && ((packed >> 24u) & 0xfu) != NativeXprFlatFeed) pointer.usable = false;
		}
		// The levels (a pointer's halves are read at a lower one) and roots of the usable ones.
		uint8_t    next     = user_roots;
		const auto level_of = [&](uint8_t root) {
			return root != NativeXprNoRoot && root >= user_roots && pointers[root - user_roots].usable
			           ? pointers[root - user_roots].level
			           : uint8_t {0};
		};
		for (auto& pointer: pointers) {
			pointer.level = static_cast<uint8_t>(1 + std::max(level_of(pointer.low_root), level_of(pointer.high_root)));
			if (!pointer.usable) continue;
			pointer.root = next++;
			levels.push_back(pointer.level);
		}
		const auto remap = [&](uint8_t& root) {
			if (root != NativeXprNoRoot && root >= user_roots) root = pointers[root - user_roots].root;
		};
		for (auto* reads: {&structural, &clean})
			for (auto& read: *reads) remap(read.root);
		for (auto& read: data) remap(read.root);
	}
	// A word read both at a fixed address and through a root (or through two roots) would part on
	// relocation: such a record is not traced.
	if (!record.roots.empty()) {
		auto& roots = scratch.roots;
		roots.clear();
		for (const auto* reads: {&structural, &clean})
			for (const auto& read: *reads) roots.emplace_back(read.address, read.root);
		for (const auto& read: data) roots.emplace_back(read.address, read.root);
		NativeXprSort(roots);
		for (size_t i = 1; i < roots.size(); ++i)
			if (roots[i].first == roots[i - 1].first && roots[i].second != roots[i - 1].second) return;
	}
	const auto runs_of = [&](std::vector<StructuralRead>& reads, bool is_clean) {
		NativeXprSort(reads);
		// One entry per address, with the union of its uses.
		auto& unique = scratch.unique;
		unique.clear();
		for (const auto& read: reads) {
			if (!unique.empty() && unique.back().address == read.address) unique.back().use |= read.use;
			else unique.push_back(read);
		}
		for (const auto& read: unique) {
			if (gather.structural.empty() || gather.structural.back().clean != is_clean ||
			    gather.structural.back().root != read.root || gather.structural.back().count >= 16 ||
			    gather.structural.back().address + 4ull * gather.structural.back().count != read.address)
				gather.structural.push_back(
				    {read.address, 0, static_cast<uint32_t>(gather.values.size()), is_clean, read.root});
			++gather.structural.back().count;
			gather.values.push_back(read.value);
			gather.uses.push_back(read.use);
		}
	};
	gather.structural.clear();
	gather.values.clear();
	gather.uses.clear();
	runs_of(structural, false);
	runs_of(clean, true);
	// Spans read through a pointer come after those its halves are in.
	const auto level_of = [&](uint8_t root) {
		return root != NativeXprNoRoot && root >= user_roots ? levels[root - user_roots] : uint8_t {0};
	};
	const auto by_level = [](const Word& a, const Word& b) {
		return std::tie(a.level, a.root, a.address) < std::tie(b.level, b.root, b.address);
	};
	auto& words = scratch.words;
	words.clear();
	for (const auto& run: gather.structural)
		if (run.clean)
			for (uint32_t i = 0; i < run.count; ++i)
				words.push_back({run.address + 4ull * i, static_cast<int32_t>(run.first + i), -1, run.root, level_of(run.root)});
	NativeXprSort(words, by_level);
	gather.clean_spans.clear();
	gather.clean_value.clear();
	for (const auto& word: words) {
		auto*      span = gather.clean_spans.empty() ? nullptr : &gather.clean_spans.back();
		const auto end  = span != nullptr ? span->address + 4ull * span->count : 0;
		// The gap words count too: a span never exceeds NativeXprSpanWords (the gather's buffer).
		if (span == nullptr || span->root != word.root || word.address < end || word.address - end > 32 ||
		    word.address / 4096 != span->address / 4096 || span->count + (word.address - end) / 4 + 1 > NativeXprSpanWords) {
			gather.clean_spans.push_back(
			    {word.address, 0, static_cast<uint32_t>(gather.clean_value.size()), word.root, word.level});
			span = &gather.clean_spans.back();
		}
		for (; span->address + 4ull * span->count < word.address; ++span->count) gather.clean_value.push_back(-1);
		gather.clean_value.push_back(word.value);
		++span->count;
	}
	// Feeds per structural value (clean values never feed a patch).
	auto& index_of = scratch.index_of; // (address, value index), non-clean runs
	index_of.clear();
	for (const auto& run: gather.structural)
		if (!run.clean)
			for (uint32_t i = 0; i < run.count; ++i) index_of.emplace_back(run.address + 4ull * i, run.first + i);
	NativeXprSort(index_of);
	// The usable pointers' halves as structural values.
	const auto value_at = [&](uint64_t address) {
		const auto it = std::ranges::lower_bound(index_of, std::pair {address, 0u});
		return it != index_of.end() && it->first == address ? it->second : UINT32_MAX;
	};
	gather.pointers.clear();
	gather.pointer_value.assign(gather.values.size(), 0);
	for (const auto& pointer: pointers) {
		if (!pointer.usable) continue;
		const auto low = value_at(pointer.low), high = value_at(pointer.high);
		if (low == UINT32_MAX || high == UINT32_MAX) return;
		gather.pointers.push_back({low, high});
		gather.pointer_value[low] = gather.pointer_value[high] = 1;
	}
	NativeXprSort(feeds);
	{
		// Grouped by value in feed order: a count per value, its prefix sums as the
		// starts, then the feeds placed at their value's cursor.
		const auto values_count = gather.values.size();
		auto&      feed_value   = scratch.feed_value;
		feed_value.clear();
		gather.feeds_at.assign(values_count + 1, 0);
		for (const auto& [address, packed]: feeds) {
			const auto     it    = std::ranges::lower_bound(index_of, std::pair {address, 0u});
			const uint32_t value = it != index_of.end() && it->first == address ? it->second : UINT32_MAX;
			feed_value.push_back(value);
			if (value != UINT32_MAX) ++gather.feeds_at[value + 1];
		}
		for (size_t v = 0; v < values_count; ++v) gather.feeds_at[v + 1] += gather.feeds_at[v];
		gather.feeds.resize(gather.feeds_at[values_count]);
		auto& cursor = scratch.cursor;
		cursor.assign(gather.feeds_at.begin(), gather.feeds_at.end() - 1);
		for (size_t f = 0; f < feeds.size(); ++f)
			if (feed_value[f] != UINT32_MAX) gather.feeds[cursor[feed_value[f]]++] = feeds[f].second;
	}
	NativeXprSort(data);
	gather.data.clear();
	gather.targets_at.clear();
	gather.targets.clear();
	for (size_t i = 0; i < data.size();) {
		const auto address = data[i].address;
		if (gather.data.empty() || gather.data.back().root != data[i].root || gather.data.back().count >= 16 ||
		    gather.data.back().address + 4ull * gather.data.back().count != address)
			gather.data.push_back({address, 0, static_cast<uint32_t>(gather.targets_at.size()), false, data[i].root});
		++gather.data.back().count;
		gather.targets_at.push_back(static_cast<uint32_t>(gather.targets.size()));
		for (; i < data.size() && data[i].address == address; ++i) gather.targets.push_back(data[i].target);
	}
	gather.targets_at.push_back(static_cast<uint32_t>(gather.targets.size()));

	words.clear();
	for (const auto& run: gather.structural)
		if (!run.clean)
			for (uint32_t i = 0; i < run.count; ++i)
				words.push_back({run.address + 4ull * i, static_cast<int32_t>(run.first + i), -1, run.root, level_of(run.root)});
	for (const auto& run: gather.data)
		for (uint32_t i = 0; i < run.count; ++i)
			words.push_back({run.address + 4ull * i, -1, static_cast<int32_t>(run.first + i), run.root, level_of(run.root)});
	NativeXprSort(words, by_level);
	gather.spans.clear();
	gather.words.clear();
	gather.word_value.clear();
	gather.word_data.clear();
	constexpr uint64_t Gap = 32, Page = 4096;
	for (const auto& word: words) {
		auto*          span = gather.spans.empty() ? nullptr : &gather.spans.back();
		const uint64_t end  = span != nullptr ? span->address + 4ull * span->count : 0;
		if (span != nullptr && span->root == word.root && word.address < end) {
			// Read both structurally and as data.
			const auto at = span->first + span->count - 1;
			if (word.value >= 0) gather.word_value[at] = word.value;
			if (word.data >= 0) gather.word_data[at] = word.data;
			continue;
		}
		// The gap words count too: a span never exceeds NativeXprSpanWords (the gather's buffer).
		if (span == nullptr || span->root != word.root || word.address - end > Gap ||
		    word.address / Page != span->address / Page || span->count + (word.address - end) / 4 + 1 > NativeXprSpanWords) {
			gather.spans.push_back({word.address, 0, static_cast<uint32_t>(gather.words.size()), word.root, word.level});
			span = &gather.spans.back();
		}
		for (; span->address + 4ull * span->count < word.address; ++span->count) {
			gather.words.push_back(0);
			gather.word_value.push_back(-1);
			gather.word_data.push_back(-1);
		}
		gather.words.push_back(0);
		gather.word_value.push_back(word.value);
		gather.word_data.push_back(word.data);
		++span->count;
	}
	// Start from what the record holds (the evaluation's results), not from a
	// new read: a word the guest rewrote since then shows up as a change.
	using Kind = ShaderRecompiler::IR::SrtReadTrace::Kind;
	const auto held = [&](uint32_t v) {
		for (auto f = gather.feeds_at[v]; f < gather.feeds_at[v + 1]; ++f) {
			const auto feed  = gather.feeds[f];
			const auto& stage = record.stages[feed >> 28u];
			const auto kind  = (feed >> 24u) & 0xfu;
			if (kind == NativeXprFlatFeed) {
				if ((feed & 0xffffffu) < stage.flattened.size()) return stage.flattened[feed & 0xffffffu];
				continue;
			}
			const auto& snapshot = stage.snapshot;
			const auto& values   = static_cast<Kind>(kind) == Kind::Buffer  ? snapshot.buffers
			                       : static_cast<Kind>(kind) == Kind::Image ? snapshot.images
			                                                                : snapshot.samplers;
			const auto index = (feed >> 8u) & 0xffffu;
			if (index < values.size() && (feed & 0xffu) < values[index].dword_count)
				return values[index].dwords[feed & 0xffu];
		}
		return gather.values[v];
	};
	gather.value_word.assign(gather.values.size(), 0);
	for (size_t w = 0; w < gather.words.size(); ++w) {
		if (const auto v = gather.word_value[w]; v >= 0) {
			gather.value_word[v] = static_cast<uint32_t>(w);
			const auto value     = held(static_cast<uint32_t>(v));
			gather.words[w] = gather.values[v] = value;
		} else if (const auto d = gather.word_data[w]; d >= 0 && gather.targets_at[d] != gather.targets_at[d + 1]) {
			const auto target = gather.targets[gather.targets_at[d]];
			gather.words[w]   = record.stages[target >> 24u].flattened[target & 0xffffffu];
		}
	}
	// Spans through a root keep their offset from its base (NativeXprAddress).
	for (auto* spans: {&gather.spans, &gather.clean_spans})
		for (auto& span: *spans)
			if (span.root != NativeXprNoRoot) span.address -= NativeXprAddress(record, NativeXprRecord::Gather::Span {0, 0, 0, span.root});
	gather.traced = true;
}

// Structural words that changed feed only descriptor dwords (m_native_xpr->changed).
// T# LOD-statistics fields only move shader-data words; a new image or buffer base
// address is resolved again with a new set. Anything else is left to a full
// evaluation (false, nothing changed).
RenderExecutor::NativeXprPatch RenderExecutor::NativeXprPatchDescriptors(NativeXprRecord& record) {
	using Kind   = ShaderRecompiler::IR::SrtReadTrace::Kind;
	auto& gather = record.gather;
	const auto is_flat = [](uint32_t feed) { return ((feed >> 24u) & 0xfu) == NativeXprFlatFeed; };
	const auto target  = [&](uint32_t feed) -> ShaderRecompiler::IR::DescriptorValue* {
		auto&      snapshot = record.stages[feed >> 28u].snapshot;
		const auto kind     = static_cast<Kind>((feed >> 24u) & 0xfu);
		const auto index    = (feed >> 8u) & 0xffffu;
		auto&      values   = kind == Kind::Buffer ? snapshot.buffers : kind == Kind::Image ? snapshot.images : snapshot.samplers;
		return index < values.size() && (feed & 0xffu) < values[index].dword_count ? &values[index] : nullptr;
	};
	bool     rebind     = false;
	uint32_t lod_stages = 0;
	auto&    rebind_images  = m_native_xpr->rebind_images;
	auto&    rebind_buffers = m_native_xpr->rebind_buffers;
	rebind_images.clear();
	rebind_buffers.clear();
	for (const auto& [index, value]: m_native_xpr->changed)
		for (auto f = gather.feeds_at[index]; f < gather.feeds_at[index + 1]; ++f) {
			const auto feed  = gather.feeds[f];
			if (is_flat(feed)) continue;
			const auto kind  = static_cast<Kind>((feed >> 24u) & 0xfu);
			const auto dword = feed & 0xffu;
			auto*      desc  = target(feed);
			if (desc == nullptr) return NativeXprPatch::Evaluate;
			const auto diff = desc->dwords[dword] ^ value;
			if (kind == Kind::Image) {
				// A zeroed (invalid) image or a change beyond the base address and
				// the LOD-statistics fields needs the evaluation's validity check.
				if (std::ranges::all_of(desc->dwords, [](uint32_t w) { return w == 0; })) return NativeXprPatch::Evaluate;
				if ((diff & ~NativeXprLodStatsMask[dword]) == 0) {
					lod_stages |= 1u << (feed >> 28u);
				} else if (dword == 0 || (dword == 1 && (diff & ~0xffu) == 0)) {
					rebind = true;
					rebind_images.emplace_back(feed >> 28u, (feed >> 8u) & 0xffffu);
				} else {
					return NativeXprPatch::Evaluate;
				}
			} else if (kind == Kind::Buffer) {
				if (dword == 0 || (dword == 1 && (diff & ~0xffffu) == 0)) {
					rebind = true;
					rebind_buffers.emplace_back(feed >> 28u, (feed >> 8u) & 0xffffu);
				} else {
					return NativeXprPatch::Evaluate;
				}
			} else {
				return NativeXprPatch::Evaluate;
			}
		}
	// Another instance's buffers or images through this use's pointers: the object is keyed by its descriptor
	// bases from now on (NativeXprTry).
	if (rebind && m_native_xpr->tried_relocate != 0) m_native_xpr->MarkRebound(m_native_xpr->object_hash);
	for (const auto& [index, value]: m_native_xpr->changed) {
		for (auto f = gather.feeds_at[index]; f < gather.feeds_at[index + 1]; ++f) {
			const auto feed = gather.feeds[f];
			if (is_flat(feed)) record.stages[feed >> 28u].flattened[feed & 0xffffffu] = value;
			else target(feed)->dwords[feed & 0xffu] = value;
		}
		gather.values[index] = value;
	}
	for (uint32_t st = 0; st < record.stage_count; ++st)
		if (lod_stages & (1u << st))
			NativeXprApplyLodStats(*record.stages[st].program, record.stages[st].snapshot.images,
			                       record.stages[st].shader_data);
	if (!rebind) return NativeXprPatch::Done;
	{
		for (auto* slots: {&rebind_images, &rebind_buffers}) {
			std::ranges::sort(*slots);
			slots->erase(std::unique(slots->begin(), slots->end()), slots->end());
		}
		if (NativeXprRebindSlots(record, rebind_images, rebind_buffers)) return NativeXprPatch::Done;
	}
	std::array<ShaderRecompiler::IR::ResourceSnapshot, 2> snapshots;
	for (uint32_t st = 0; st < record.stage_count; ++st) {
		snapshots[st]               = record.stages[st].snapshot;
		snapshots[st].flattened_srt = record.stages[st].flattened;
	}
	// The snapshot already carries the new descriptors: a failed resolution
	// leaves nothing an evaluation could compare against.
	return NativeXprRebind(record, snapshots) ? NativeXprPatch::Done : NativeXprPatch::Invalid;
}

// Validation reads: a page without read watchers is loaded directly (every path
// of the evaluation's readers ends in that load for it); other pages take the
// evaluation's readers. The page answer is kept for one validation.
class NativeXprReader {
public:
	explicit NativeXprReader(GpuResourceManager& resources)
	    : m_resources(resources), m_granules(resources.ReadGranules()) {}
	void Read(uint64_t address, uint32_t count, uint32_t* out) {
		const uint64_t bytes = uint64_t {count} * 4u;
		const uint64_t first = address >> 12u, last = (address + bytes - 1u) >> 12u;
		if (first == last) {
			if (first != m_page) {
				m_page = first;
				// (The page's granule bit first, here: a call per page was ~1% of the GPU thread at 1-1.)
				const auto granule = first >> (PageManager::READ_GRANULE_BITS - 12u);
				m_direct           = (address < TRACKER_ADDRESS_SIZE &&
				            ((m_granules[granule / 64u].load(std::memory_order_acquire) >> (granule % 64u)) & 1u) == 0) ||
				           !m_resources.HasReadWatchers(first << 12u, 4096);
			}
			if (m_direct) {
				// (A speculation's: as no GPU work wrote it, which a hole that writes it after disproves, and as its
				// processor's pending writes leave it.)
				if (auto* spec = Spec::Current()) {
					if (!spec->ReadJournaled(address, count, out)) return Spec::Refuse("memory the processor writes");
					return;
				}
				std::memcpy(out, reinterpret_cast<const void*>(address), bytes);
				return;
			}
		}
		NativeXprReadWords(address, count, out);
	}

private:
	GpuResourceManager&          m_resources;
	const std::atomic<uint64_t>* m_granules;
	uint64_t                     m_page   = UINT64_MAX;
	bool                         m_direct = false;
};

// Traced records: false when a structural read changed (the caller evaluates
// again); otherwise the data reads are copied into the flattened SRT.
bool RenderExecutor::NativeXprGather(NativeXprRecord& record) {
	LiveCensus::Scope census(LiveCensus::NativeGather, 0);
	auto&           gather = record.gather;
	NativeXprReader reader(m_context.GetGpuResources());
	auto& changed = m_native_xpr->changed; // (value index, new value)
	changed.clear();
	// Records are mostly used once per frame: their guest words are cold. Start
	// every line's load before comparing any.
	for (const auto& item: gather.spans) {
		const auto address = NativeXprAddress(record, item);
		for (uint64_t line = address & ~63ull; line < address + 4ull * item.count; line += 64)
			__builtin_prefetch(reinterpret_cast<const void*>(line));
	}
	std::array<uint32_t, NativeXprSpanWords> fresh;
	// A pointer read from memory (a root past the record's) may hold anything: what is read through it
	// must be mapped.
	const auto readable = [&](const NativeXprRecord::Gather::Span& item, uint64_t address) {
		return item.count <= fresh.size() && // never past the buffer: evaluate again
		       (item.root == NativeXprNoRoot || item.root < record.roots.size() ||
		        Libs::LibKernel::Memory::IsFullyMapped(address, 4ull * item.count));
	};
	// Level by level: the spans read through a pointer after those holding it.
	size_t next_clean = 0, next = 0;
	while (next_clean < gather.clean_spans.size() || next < gather.spans.size()) {
		const auto level = std::min(next_clean < gather.clean_spans.size() ? gather.clean_spans[next_clean].level : 0xff,
		                            next < gather.spans.size() ? gather.spans[next].level : 0xff);
		// Clean structural reads (GPU-written memory, read as the evaluation reads
		// it) are never patched.
		for (; next_clean < gather.clean_spans.size() && gather.clean_spans[next_clean].level == level; ++next_clean) {
			const auto& item    = gather.clean_spans[next_clean];
			const auto* values  = gather.clean_value.data() + item.first;
			const auto  address = NativeXprAddress(record, item);
			if (!readable(item, address)) return false;
			// A span that reads clean has clean words; otherwise each word is read
			// as the evaluation reads it.
			if (Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(address, fresh.data(), 4ull * item.count)) {
				for (uint32_t i = 0; i < item.count; ++i)
					if (values[i] >= 0 && fresh[i] != gather.values[values[i]]) return false;
				continue;
			}
			for (uint32_t i = 0; i < item.count; ++i) {
				if (values[i] < 0) continue;
				uint32_t value = 0;
				if (!NativeXprReadClean(address + 4ull * i, 1, &value) || value != gather.values[values[i]]) return false;
			}
		}
		for (; next < gather.spans.size() && gather.spans[next].level == level; ++next) {
			const auto& item    = gather.spans[next];
			const auto  address = NativeXprAddress(record, item);
			if (!readable(item, address)) return false;
			reader.Read(address, item.count, fresh.data());
			auto* held = gather.words.data() + item.first;
			if (std::memcmp(fresh.data(), held, item.count * 4ull) == 0) continue;
			for (uint32_t i = 0; i < item.count; ++i) {
				const auto value = fresh[i];
				if (value == held[i]) continue;
				const auto at = item.first + i;
				if (const auto d = gather.word_data[at]; d >= 0)
					for (auto t = gather.targets_at[d]; t < gather.targets_at[d + 1]; ++t) {
						const auto target = gather.targets[t];
						record.stages[target >> 24u].flattened[target & 0xffffffu] = value;
					}
				if (const auto v = gather.word_value[at]; v >= 0) {
					// A pointer's half: what later spans read through it moves (its feeds are flattened copies).
					if (gather.pointer_value[v] != 0) {
						for (auto f = gather.feeds_at[v]; f < gather.feeds_at[v + 1]; ++f)
							record.stages[gather.feeds[f] >> 28u].flattened[gather.feeds[f] & 0xffffffu] = value;
						gather.values[v] = held[i] = value;
						continue;
					}
					// Only words that feed nothing but descriptor dwords may be patched;
					// anything else is evaluated again (which traces again).
					if ((gather.uses[v] & ~ShaderRecompiler::IR::SrtReadTrace::Descriptor) != 0 ||
					    gather.feeds_at[v] == gather.feeds_at[v + 1]) {
						return false;
					}
					changed.emplace_back(v, value);
					continue; // held once the patch is done
				}
				held[i] = value;
			}
		}
	}
	if (!changed.empty()) {
		const auto patched = NativeXprPatchDescriptors(record);
		if (patched == NativeXprPatch::Evaluate) return false;
		if (patched == NativeXprPatch::Invalid) {
			record.gather.broken = true;
			return false;
		}
		for (const auto& [v, value]: changed) gather.words[gather.value_word[v]] = value;
	}
	return true;
}

bool RenderExecutor::NativeXprReevaluate(NativeXprRecord& record) {
	auto& cache = *m_native_xpr;
	if (record.stages[0].shader_base == 0) return false;
	std::array<ShaderRecompiler::IR::ResourceSnapshot, 2> fresh;
	auto& reads = cache.reads;
	reads.clear();
	// Traced records need no read log: the trace is rebuilt instead.
	const bool traced = record.gather.traced;
	{
		std::optional<ShaderReadObserver> observer;
		if (!traced)
			observer.emplace(
			    [](void* userdata, uint64_t address, uint64_t size) {
				    static_cast<std::vector<std::pair<uint64_t, uint64_t>>*>(userdata)->emplace_back(address, size);
			    },
			    &reads);
		auto& pipelines = m_context.GetPipelineCache();
		for (uint32_t st = 0; st < record.stage_count; ++st) {
			const auto& stage = record.stages[st];
			if (!pipelines.RematerializeStage(*stage.program, stage.user_data, stage.shader_base, fresh[st]))
				return false;
		}
	}
	// Same structure: the descriptors up to the LOD-statistics fields, user data,
	// uniform fill and the flattened SRT size are unchanged.
	bool same_structure = true;
	for (uint32_t st = 0; st < record.stage_count && same_structure; ++st) {
		const auto& a  = record.stages[st].snapshot;
		const auto& b  = fresh[st];
		same_structure = a.buffers == b.buffers && NativeXprSameImagesButLodStats(a.images, b.images) &&
		                 a.samplers == b.samplers && a.user_data == b.user_data &&
		                 a.uniform_fill == b.uniform_fill &&
		                 b.flattened_srt.size() == record.stages[st].flattened.size();
	}
	if (traced) {
		if (!same_structure) {
			if (!NativeXprRebind(record, fresh)) return false;
		} else {
			for (uint32_t st = 0; st < record.stage_count; ++st) {
				auto& stage = record.stages[st];
				if (!(stage.snapshot.images == fresh[st].images)) {
					NativeXprApplyLodStats(*stage.program, fresh[st].images, stage.shader_data);
					stage.snapshot.images = std::move(fresh[st].images);
				}
				stage.flattened = std::move(fresh[st].flattened_srt);
			}
		}
		NativeXprRetrace(record);
		return true;
	}
	std::vector<std::pair<uint64_t, uint32_t>> new_reads;
	std::vector<uint32_t>                      new_words;
	new_reads.reserve(reads.size());
	for (const auto& [address, size]: reads) {
		if (address % 4 != 0 || size % 4 != 0 || size == 0 || size > 4096) return false;
		const auto count = static_cast<uint32_t>(size / 4);
		new_reads.emplace_back(address, count);
		const auto at = new_words.size();
		new_words.resize(at + count);
		NativeXprReadWords(address, count, new_words.data() + at);
	}
	if (!same_structure) {
		// New descriptors: resolve them and bind a new set (flattened constants,
		// shader data and proofs come with it); no patch knowledge carries over.
		if (!NativeXprRebind(record, fresh)) return false;
		record.word_patch.assign(new_words.size(), -1);
		record.reads = std::move(new_reads);
		record.words = std::move(new_words);
		NativeXprRetrace(record);
		return true;
	}
	// Same structure up to the LOD-statistics fields. With an unchanged read layout
	// the old and new evaluations teach which words are plain copies (flattened
	// constants, LOD dwords), so the next change of them is only patched.
	std::array<std::vector<uint32_t>, 2> shader_data;
	for (uint32_t st = 0; st < record.stage_count; ++st) {
		shader_data[st] = record.stages[st].shader_data;
		NativeXprApplyLodStats(*record.stages[st].program, fresh[st].images, shader_data[st]);
	}
	if (new_reads == record.reads) {
		NativeXprRecord fresh_record;
		fresh_record.reads          = new_reads;
		fresh_record.words          = new_words;
		fresh_record.stage_count    = record.stage_count;
		fresh_record.vertex_program = record.vertex_program;
		fresh_record.pixel_program  = record.pixel_program;
		for (uint32_t st = 0; st < record.stage_count; ++st) {
			fresh_record.stages[st].program     = record.stages[st].program;
			fresh_record.stages[st].snapshot    = fresh[st];
			fresh_record.stages[st].shader_data = shader_data[st];
			fresh_record.stages[st].flattened   = fresh[st].flattened_srt;
		}
		NativeXprLearn(record, fresh_record);
		record.word_patch = std::move(fresh_record.word_patch);
	} else {
		record.word_patch.assign(new_words.size(), -1);
	}
	for (uint32_t st = 0; st < record.stage_count; ++st) {
		record.stages[st].snapshot.images = fresh[st].images;
		record.stages[st].shader_data     = std::move(shader_data[st]);
	}
	for (uint32_t st = 0; st < record.stage_count; ++st) record.stages[st].flattened = std::move(fresh[st].flattened_srt);
	record.reads = std::move(new_reads);
	record.words = std::move(new_words);
	if (record.gather.traced) NativeXprRetrace(record);
	return true;
}

// Once per frame: is the record's content still what a fresh translation gives?
bool RenderExecutor::NativeXprValidate(NativeXprRecord& record, uint64_t frame) {
	auto& cache     = *m_native_xpr;
	auto& resources = m_context.GetGpuResources();
	// A guest mapping change forces the buffer and image proofs below; the words
	// are compared again anyway.
	const bool mapping_moved = resources.MappingEpoch() != record.mapping_epoch;
	// An unmap can take words the record reads (tables the SRT no longer reaches):
	// those are read directly, so none may have been unmapped since the proof.
	if (mapping_moved) {
		const auto unmapped = [&](uint64_t address, uint32_t count) {
			return resources.UnmappedSince(record.mapping_epoch, address, 4ull * count);
		};
		if (record.gather.traced ? std::ranges::any_of(record.gather.spans,
		                                               [&](const auto& span) {
			                                               return unmapped(NativeXprAddress(record, span), span.count);
		                                               })
		                         : std::ranges::any_of(record.reads,
		                                               [&](const auto& read) { return unmapped(read.first, read.second); }))
			return false;
	}
	// The read log of a relocated record is at its first pointers.
	if (!record.roots.empty() && !record.gather.traced) return false;
	if (record.gather.traced) {
		if (!NativeXprGather(record) && (record.gather.broken || !NativeXprReevaluate(record))) {
			return false;
		}
		return NativeXprValidateResources(record, frame, mapping_moved);
	}
	// Every logged word, read the way the evaluation reads it: a span through the
	// watched-page aware span reader, single words otherwise.
	auto& changed = cache.changed;
	auto& span    = cache.span;
	changed.clear();
	size_t cursor = 0;
	for (const auto& [address, count]: record.reads) {
		span.resize(count);
		NativeXprReadWords(address, count, span.data());
		if (std::memcmp(span.data(), record.words.data() + cursor, count * 4ull) != 0) {
			for (uint32_t i = 0; i < count; ++i)
				if (span[i] != record.words[cursor + i])
					changed.emplace_back(static_cast<uint32_t>(cursor + i), span[i]);
		}
		cursor += count;
	}
	if (!changed.empty() &&
	    std::ranges::any_of(changed, [&](const auto& change) { return record.word_patch[change.first] == -1; })) {
		// A word no patch explains: evaluate the SRT again. Unchanged structure
		// (descriptors, user data, uniform fill, sizes) keeps the descriptor set;
		// only the flattened constants and the logged reads are replaced.
		if (NativeXprReevaluate(record)) {
			changed.clear();
		}
	}
	if (!changed.empty()) {
		for (const auto& [index, value]: changed) {
			if (record.word_patch[index] == -1) {
				return false;
			}
		}
		// Only learned words moved: patch the flattened SRT copies and the T# LOD
		// dwords (with the shader-data words derived from them).
		uint32_t lod_stages = 0;
		for (const auto& [index, value]: changed) {
			const auto patch = record.word_patch[index];
			if (patch >= 0) {
				const auto at = static_cast<uint32_t>(patch);
				record.stages[at >> 24u].flattened[at & 0xffffffu] = value;
			} else if (patch <= NativeXprImagePatchBase) {
				const auto code = static_cast<uint32_t>(NativeXprImagePatchBase - patch);
				record.stages[code >> 20u].snapshot.images[(code >> 4u) & 0xffffu].dwords[code & 0xfu] = value;
				lod_stages |= 1u << (code >> 20u);
			}
			record.words[index] = value;
		}
		for (uint32_t st = 0; st < record.stage_count; ++st)
			if (lod_stages & (1u << st))
				NativeXprApplyLodStats(*record.stages[st].program, record.stages[st].snapshot.images,
				                       record.stages[st].shader_data);
	}
	return NativeXprValidateResources(record, frame, mapping_moved);
}

// No image started at a texel dependency since the record's proof.
bool RenderExecutor::NativeXprTexelRangesClear(const NativeXprRecord& record) {
	if (std::ranges::none_of(record.buffers, &NativeXprRecord::BufferDep::texel)) return true;
	auto& cache    = *m_native_xpr;
	auto& textures = m_context.GetTextureCache();
	if (cache.starts_seen != textures.StartEpoch()) {
		const auto seen = cache.starts_seen;
		if (seen != 0 && textures.NewImageStartsSince(seen, cache.recent_starts)) {
			if (!cache.recent_starts.empty()) cache.starts_seen = std::max(seen, cache.recent_starts.back().first);
		} else {
			cache.recent_starts.clear();
			cache.starts_from = cache.starts_seen = textures.StartEpoch();
		}
		if (cache.recent_starts.size() > 4096) {
			const auto drop   = cache.recent_starts.size() - 2048;
			cache.starts_from = cache.recent_starts[drop - 1].first;
			cache.recent_starts.erase(cache.recent_starts.begin(), cache.recent_starts.begin() + static_cast<ptrdiff_t>(drop));
		}
	}
	if (record.start_epoch == 0 || record.start_epoch < cache.starts_from) {
		for (const auto& dep: record.buffers)
			if (dep.texel && textures.HasImageStartingAt(dep.address)) return false;
		return true;
	}
	const auto first = std::ranges::upper_bound(cache.recent_starts, record.start_epoch, {},
	                                            &std::pair<uint64_t, uint64_t>::first);
	for (auto it = first; it != cache.recent_starts.end(); ++it)
		for (const auto& dep: record.buffers)
			if (dep.texel && dep.address == it->second) return false;
	return true;
}

// Buffers and images: once per frame, or when the live-object epochs moved.
bool RenderExecutor::NativeXprValidateResources(NativeXprRecord& record, uint64_t frame, bool mapping_moved) {
	auto& buffers   = m_context.GetBufferCache();
	auto& textures  = m_context.GetTextureCache();
	auto& resources = m_context.GetGpuResources();
	// The rest holds for the whole frame unless the live-object epochs move, or the guest
	// wrote a bound buffer since the proof (its region's CPU epoch moved).
	if (!mapping_moved && record.validated_frame == frame &&
	    record.registration_epoch == buffers.RegistrationEpoch() &&
	    record.resolution_epoch == textures.ResolutionEpoch()) {
		uint64_t cpu_epoch = 0;
		for (const auto& dep: record.buffers) {
			const auto epoch = buffers.CpuModificationEpoch(dep.address, dep.size);
			if (epoch == 0) {
				cpu_epoch = 0;
				break;
			}
			cpu_epoch += epoch;
		}
		// The epochs cannot prove it: the ranges themselves (what the normal path's
		// ObtainBuffer checks per draw).
		const bool buffers_clean = record.buffers.empty() || (cpu_epoch != 0 && cpu_epoch == record.cpu_epoch) ||
		                           std::ranges::none_of(record.buffers, [&](const auto& dep) {
			                           return buffers.IsRegionCpuModified(dep.address, dep.size);
		                           });
		// Images: no image became CPU-dirty since the proof, or none of these did.
		const auto image_epoch  = Image::CpuDirtyEpoch();
		bool       images_clean = image_epoch == record.image_dirty_epoch;
		if (buffers_clean && !images_clean) {
			images_clean = std::ranges::none_of(record.images, [&](const auto& dep) {
				const auto* image = textures.m_slot_images.try_get(dep.id);
				return image == nullptr || image->serial != dep.serial || image->IsCpuDirty();
			});
			if (images_clean) record.image_dirty_epoch = image_epoch;
		}
		if (buffers_clean && images_clean) return true;
	}
	const auto image_epoch_at_proof = Image::CpuDirtyEpoch();
	// Buffers: live owners, CPU-clean. The region epochs prove both unchanged
	// cheaply; when they moved, check each range.
	uint64_t cpu_epoch = 0;
	for (const auto& dep: record.buffers) {
		const auto epoch = buffers.CpuModificationEpoch(dep.address, dep.size);
		if (epoch == 0) {
			cpu_epoch = 0;
			break;
		}
		cpu_epoch += epoch;
	}
	// After a mapping change, the ranges as FindBuffers gives them now: none unmapped since
	// the proof (the refresh below reads them), and one it clamped to the mappings not grown.
	const auto range_current = [&](const NativeXprRecord::BufferDep& dep) {
		const auto& descriptors = record.stages[dep.stage].snapshot.buffers;
		if (dep.slot >= descriptors.size() || resources.UnmappedSince(record.mapping_epoch, dep.address, dep.size))
			return false;
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(descriptors[dep.slot]);
		const auto stride     = descriptor.Stride();
		const auto requested  = stride != 0 ? uint64_t(stride) * descriptor.NumRecords() : uint64_t(descriptor.NumRecords());
		return requested == dep.size || Libs::LibKernel::Memory::ClampRangeSize(dep.address, requested) == dep.size;
	};
	// TouchLiveBuffer also keeps the owners from the garbage collector, as the
	// normal path's per-draw ObtainBuffer would.
	const bool cpu_proven = !mapping_moved && cpu_epoch != 0 && cpu_epoch == record.cpu_epoch;
	// A buffer or image the record binds is gone or replaced (textures streaming in and out): its slots are
	// resolved again from the descriptors it holds, as for a descriptor change, instead of dropping the record
	// (whose draws then took the normal path until a store, within the store budget, made it again).
	const auto rebind = [&] {
		std::array<ShaderRecompiler::IR::ResourceSnapshot, 2> snapshots;
		for (uint32_t st = 0; st < record.stage_count; ++st) {
			snapshots[st]               = record.stages[st].snapshot;
			snapshots[st].flattened_srt = record.stages[st].flattened;
		}
		if (!NativeXprRebind(record, snapshots)) return false;
		if (record.gather.traced) NativeXprRetrace(record);
		return true;
	};
	for (const auto& dep: record.buffers) {
		if (mapping_moved && !range_current(dep)) return false;
		if (!buffers.TouchLiveBuffer(dep.id, dep.handle, dep.address, dep.size)) return rebind();
		if (cpu_proven || !buffers.IsRegionCpuModified(dep.address, dep.size)) continue;
		// The guest rewrote the range: bring the owner up to date as the store
		// did (the normal path would copy the range for its draw instead).
		if (buffers.IsRegionGpuModified(dep.address, dep.size)) {
			return false;
		}
		buffers.EnsureBufferContents(dep.address, dep.size);
		const auto [served, offset] = buffers.ObtainBuffer(dep.address, dep.size, false, dep.texel, dep.id);
		if (served == nullptr || served->Handle() != dep.handle || buffers.IsRegionCpuModified(dep.address, dep.size)) {
			return false;
		}
	}
	record.cpu_epoch          = cpu_epoch;
	record.registration_epoch = buffers.RegistrationEpoch();
	// Texel buffers: still no image starting at one (checked against the image
	// starts logged since the record's proof).
	if (const auto start_epoch = textures.StartEpoch(); record.start_epoch != start_epoch) {
		if (!NativeXprTexelRangesClear(record)) {
			return false;
		}
		record.start_epoch = start_epoch;
	}
	// Images: the same objects (views alive), registered with the same backing,
	// contents current.
	const bool registrations_moved = textures.ResolutionEpoch() != record.resolution_epoch;
	// KYTY_NATIVE_IMAGE_PROOF: other images' registrations move the global epoch every
	// frame; an image still in the registration the proof saw passes the same checks.
	const bool per_image = kyty_local_native_image_proof_mode.load(std::memory_order_relaxed) != 0;
	for (const auto& dep: record.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || image->serial != dep.serial) return rebind();
		// The guest rewrote it (a texture streaming in): upload it as the normal path's FindTexture does,
		// instead of dropping the record and storing it again.
		if (image->IsCpuDirty()) {
			if (!image->registered || image->depth_id || image->binding.needs_rebind) return false;
			(void)textures.FindTexture(dep.id, dep.desc);
			image = textures.m_slot_images.try_get(dep.id);
			if (image == nullptr || image->serial != dep.serial || image->IsCpuDirty()) return false;
		}
		if (registrations_moved) {
			if (per_image && image->registered && image->registrations == dep.registration &&
			    !image->depth_id && !image->binding.needs_rebind) {
				image->frame_accessed_last = textures.CurrentFrame();
			} else if (!textures.IsSampledImageCurrent(dep.id, dep.desc)) {
				return rebind();
			}
		}
		(void)textures.GetImage(dep.id); // LRU, as FindTexture does
	}
	record.resolution_epoch = textures.ResolutionEpoch();
	record.mapping_epoch    = resources.MappingEpoch();
	record.validated_frame  = frame;
	record.image_dirty_epoch = image_epoch_at_proof;
	return true;
}

// Emits the draws of one record under one draw state. `commands` are the
// GPU-side DrawIndexedIndirectCommand addresses, all drawn with the same bindings.
struct RenderExecutor::NativeXprDirect {
	NativeXprDirectDraw draw;
	int32_t             vertex_offset  = 0;
	uint32_t            first_instance = 0;
};

// What the normal path's target acquisition establishes must still hold: the image is registered,
// GPU-owned (so CPU reads download it) and CPU-clean, and no fast clear is pending in its metadata.
// Otherwise the normal path acquires. Then the targets in the state's layouts.
bool RenderExecutor::NativeXprEmitTargets(const NativeXprDrawState& draw_state, vk::CommandBuffer vk_buffer) {
	auto&      textures   = m_context.GetTextureCache();
	const auto meta_epoch = textures.MetaEpoch();
	auto*      spec       = Spec::Current();
	for (const auto& target: draw_state.targets) {
		auto* image = textures.m_slot_images.try_get(target.id);
		if (image == nullptr || image->serial != target.serial || !image->registered)
			return Spec::Refuse("table: target released"), false;
		if (image->binding.needs_rebind) return Spec::Refuse("table: target rebind"), false;
		if (!image->SafeToDownload()) return Spec::Refuse("table: target download"), false;
		// (A speculation asks each time: its answers are noted.)
		if (target.metadata != 0 && (spec != nullptr || draw_state.meta_epoch != meta_epoch) &&
		    textures.IsMetaCleared(target.metadata, target.slice)) {
			// The normal path clears it first. A speculation's draw is a hole that does: the draws after it find it
			// cleared (speculation-state.h).
			if (spec != nullptr) spec->ConsumeMeta(target.metadata, target.slice);
			return Spec::Refuse("table: cleared target"), false;
		}
	}
	// (Not a speculation's answer: it holds as after holes the commit may not reach.)
	if (spec == nullptr) draw_state.meta_epoch = meta_epoch;
	for (const auto& target: draw_state.targets) {
		auto& image = textures.GetImage(target.id); // LRU, as acquisition does
		// (The whole image's layout stands for its subresources while they share one state. Else, as after a
		// speculation's commit that left a layer elsewhere (rendering to layer 0 of four left the image's layout the
		// target's, layers 1-3 sampled), the target's own subresources decide; a speculation's always do: it does not
		// see its holes' transitions.)
		if (spec != nullptr || image.CurrentState().layout != target.layout || !image.backing.subresource_states.empty())
			image.Transit(target.layout, target.access, target.range, vk_buffer);
	}
	return true;
}

// A deleted target image took the state's attachment views along (its slot may hold another image by
// now), and a newer image with the same backing as a target is what the registers now select: the
// normal path captures the state again.
bool RenderExecutor::NativeXprTargetsCurrent(const NativeXprDrawState& draw_state) {
	auto&      textures = m_context.GetTextureCache();
	const auto checked  = textures.ResolutionEpoch();
	// (Every table draw asked: a slot lookup and a cold image object per target, ~28K a frame at 1-1.)
	if (draw_state.targets_checked == checked) return true;
	// (A live id is the same object: an erase moves its slot's generation, which the id names. The image itself, a cold
	// line per target, is not read.)
	for (const auto& target: draw_state.targets)
		if (!textures.m_slot_images.is_allocated(target.id)) return false;
	if (draw_state.targets_epoch != textures.ResolutionEpoch()) {
		uint64_t proven = UINT64_MAX;
		for (const auto& target: draw_state.targets) {
			auto epoch = draw_state.targets_epoch;
			if (!textures.IsTargetCurrent(target.id, target.requested, epoch)) return false;
			proven = std::min(proven, epoch);
		}
		draw_state.targets_epoch = proven != UINT64_MAX ? proven : textures.ResolutionEpoch();
	}
	draw_state.targets_checked = checked;
	return true;
}

bool RenderExecutor::NativeXprEmit(CommandBuffer& buffer, NativeXprRecord& record,
                                   const NativeXprDrawState& draw_state,
                                   const PipelineCache::Pipeline& pipeline,
                                   std::span<const uint64_t> commands, uint64_t index_base,
                                   uint64_t index_bytes, vk::IndexType index_type, bool keep_clears,
                                   const NativeXprDirect* direct) {
	auto&           cache = *m_native_xpr;
	auto& upload = m_context.GetBufferCache().GetShaderUploadBuffer();
	if (upload.Handle() != record.upload) {
		return false;
	}
	// Constants into the ring; dynamic offsets follow binding order (vertex
	// flattened SRT, vertex shader data, pixel ...), as the bindings are numbered.
	const auto& limits    = m_context.GetGraphics().physical_device_properties.limits;
	const auto  alignment = std::max<uint64_t>({256, limits.minUniformBufferOffsetAlignment,
	                                            limits.minStorageBufferOffsetAlignment});
	std::array<uint32_t, 4> offsets {};
	uint32_t                offset_count = 0;
	ShaderRecompiler::IR::PushData push_data;
	const auto debug = kyty_local_native_xpr_debug.load(std::memory_order_relaxed);
	for (uint32_t i = 0; i < record.stage_count; ++i) {
		const auto& stage = record.stages[i];
		if (stage.srt_binding) {
			const std::vector<uint32_t> zeroes(i == 0 && (debug & 1u) != 0 ? stage.flattened.size() : 0, 0u);
			const auto* data = zeroes.empty() ? stage.flattened.data() : zeroes.data();
			offsets[offset_count++] = static_cast<uint32_t>(
			    upload.Copy(data, stage.flattened.size() * sizeof(uint32_t), alignment));
		}
		if (stage.data_binding)
			offsets[offset_count++] = static_cast<uint32_t>(upload.Copy(
			    stage.shader_data.data(), stage.shader_data.size() * sizeof(uint32_t), alignment));
		if (stage.push_start != ShaderRecompiler::IR::PushData::NoStart) {
			EXIT_IF(stage.push_start + stage.shader_data.size() > push_data.dwords.size());
			std::ranges::copy(stage.shader_data, push_data.dwords.begin() + stage.push_start);
		}
	}

	// Indices and arguments are read by the GPU; lookups may restart the
	// scheduler, so the command buffer handle is taken after them.
	auto& buffer_cache = m_context.GetBufferCache();
	// One lookup per group of consecutive arguments: the same range the draws
	// read, synchronized as the per-draw lookups would.
	constexpr uint64_t stride = sizeof(vk::DrawIndexedIndirectCommand);
	// Only the first `groups` entries are written and read: no zero fill (1.25 KB per draw).
	std::array<std::pair<Buffer*, uint64_t>, 64> arguments;
	std::array<uint32_t, 64>                     group_sizes;
	uint32_t                                     groups = 0;
	EXIT_IF(commands.size() > arguments.size() || (direct != nullptr && !commands.empty()));
	for (size_t first = 0; first < commands.size();) {
		size_t count = 1;
		while (first + count < commands.size() && commands[first + count] == commands[first] + count * stride) ++count;
		const auto [owner, offset] = buffer_cache.ObtainBuffer(commands[first], count * stride, false);
		if (owner == nullptr) return false;
		arguments[groups]   = {owner, offset};
		group_sizes[groups] = static_cast<uint32_t>(count);
		++groups;
		first += count;
	}
	auto&      emit       = cache.emit;
	const bool same_index = emit.command == buffer.Handle() && emit.generation == buffer.GraphicsGeneration() &&
	                        emit.index_base == index_base && emit.index_bytes == index_bytes &&
	                        emit.index_type == index_type;
	std::pair<Buffer*, uint64_t> indices {};
	if (!same_index) {
		// A small CPU-dirty index range is copied into the upload ring at every use and never cleaned
		// (ObtainBuffer): ~1300 copies of ~250 ranges a frame (1-1, Latria). Clean it once, as a store cleans
		// the SRT's buffers: a later CPU write faults and the range is copied again.
		if (index_bytes <= BufferCache::CACHING_PAGESIZE && buffer_cache.IsRegionCpuModified(index_base, index_bytes) &&
		    !buffer_cache.IsRegionGpuModified(index_base, index_bytes))
			buffer_cache.EnsureBufferContents(index_base, index_bytes);
		const auto [owner, offset] = buffer_cache.ObtainBuffer(index_base, index_bytes, false);
		if (owner == nullptr) return false;
		indices = {owner, offset};
	}
	if (buffer.IsInvalid()) return false;
	auto vk_buffer = buffer.Handle();

	auto& textures  = m_context.GetTextureCache();
	auto& scheduler = m_context.GetCommandScheduler();
	if (!NativeXprEmitTargets(draw_state, vk_buffer)) return false;
	for (const auto& dep: record.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr) return false;
		// A DCC image with a fast clear pending: the normal path's FindTexture applies it first.
		if (dep.desc.info.metadata.kind == ImageMetadataKind::Dcc) {
			if (image->info.IsVolume()) return false;
			for (uint32_t layer = dep.range.base_layer; layer < dep.range.base_layer + dep.range.layer_count; ++layer)
				if (layer >= 32 || textures.IsMetaCleared(dep.desc.info.metadata.range.address, layer)) return false;
		}
		image->Transit(dep.layout, vk::AccessFlagBits2::eShaderRead, dep.range, vk_buffer);
	}
	scheduler.BeginRendering(keep_clears ? draw_state.first_rendering : draw_state.rendering);

	// Everything bound since the last native emission in this command buffer
	// stays bound unless another path bound graphics state (generation).
	const bool same_command = emit.command == vk_buffer && emit.generation == buffer.GraphicsGeneration();
	if (same_command && emit.state != &draw_state && emit.state != nullptr &&
	    emit.state->dynamic_hash == draw_state.dynamic_hash &&
	    std::memcmp(&emit.state->dynamic, &draw_state.dynamic, sizeof(draw_state.dynamic)) == 0) {
		// Another draw state with the same dynamic state is bound already.
		emit.state = &draw_state;
	}
	if (!same_command || emit.state != &draw_state) NativeXprEmitDynamic(cache, vk_buffer, draw_state, same_command);
	if (!same_command || emit.pipeline != &pipeline) {
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	}
	vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.pipeline_layout, 0, 1,
	                             &record.set, offset_count, offsets.data());
	if (record.has_push)
		vk_buffer.pushConstants(pipeline.pipeline_layout, record.push_stages, 0, sizeof(push_data),
		                        push_data.dwords.data());
	if (!same_index) {
		vk_buffer.bindIndexBuffer(indices.first->Handle(), indices.second, index_type);
	}
	if ((debug & 2u) == 0) {
		if (direct != nullptr) {
			// The index buffer is bound at the draw's first index (index_base).
			vk_buffer.drawIndexed(direct->draw.index_count, direct->draw.instance_count, 0, direct->vertex_offset,
			                      direct->first_instance);
		}
		// Each group of consecutive arguments goes out as one multi-draw.
		for (uint32_t group = 0; group < groups; ++group)
			vk_buffer.drawIndexedIndirect(arguments[group].first->Handle(), arguments[group].second,
			                              group_sizes[group], static_cast<uint32_t>(stride));
	}
	emit.command     = vk_buffer;
	emit.generation  = buffer.GraphicsGeneration();
	emit.state       = &draw_state;
	emit.pipeline    = &pipeline;
	emit.index_base  = index_base;
	emit.index_bytes = index_bytes;
	emit.index_type  = index_type;
	// KYTY_FRAMEGEN: the camera, which the normal path's draw hook reads from the G-buffer vertex
	// shader's SRT. Without it frames drawn from records had no generated frames (and were not
	// capped by KYTY_FLIP_RATE).
	if (FrameGen::Enabled() && record.stage_count > 0 && record.stages[0].program != nullptr) {
		FrameGen::OnDraw(record.stages[0].program->shader_hash, record.stages[0].flattened);
	}
	return true;
}

// Offline replay: draws the object whose record matches the current registers
// under the draw state they select, without runtime proofs. `keep_clears`
// begins rendering with the stored draw's clears as well.
bool RenderExecutor::NativeXprDraw(CommandBuffer& buffer, const NativeXprDrawArgs& args) {
	if (!m_native_xpr || buffer.IsInvalid() || args.index_address == 0 || args.index_bytes == 0 ||
	    args.args_address == 0)
		return false;
	auto& cache = *m_native_xpr;
	m_context.GetCommandScheduler().PopPendingOperations();
	RenderLockGuard lock(m_context.GetMutex());

	NativeXprKey(buffer, cache.key);
	const auto hash  = XXH3_64bits(cache.key.data(), cache.key.size() * sizeof(uint32_t));
	const auto found = cache.records.find(hash);
	if (found == cache.records.end() || found->second->key != cache.key) {
		return false;
	}
	auto&      record    = *found->second;
	const auto state_key = NativeXprStateKey(buffer);
	const auto state_it  = cache.states.find(state_key);
	const auto pipeline_it =
	    std::ranges::find(record.pipelines, state_key, &std::pair<uint64_t, const PipelineCache::Pipeline*>::first);
	if (state_it == cache.states.end() || pipeline_it == record.pipelines.end()) {
		return false;
	}
	cache.emit = {};
	return NativeXprEmit(buffer, record, *state_it->second, *pipeline_it->second, {&args.args_address, 1},
	                     args.index_address, args.index_bytes, args.index_type, args.keep_clears);
}

// Runtime: a DRAW_INDEX_INDIRECT run of the current object. `clean`: no graphics
// register changed since the previous indexed indirect draw except user data.
bool RenderExecutor::NativeXprTry(CommandBuffer& buffer, bool clean, std::span<const uint64_t> commands,
                                  uint64_t index_base, uint64_t index_bytes, vk::IndexType index_type,
                                  const NativeXprDirectDraw* direct) {
	if ((kyty_local_native_xpr_diag.load(std::memory_order_relaxed) & 8u) != 0) return false;
	auto& cache = NativeXprState();
	// Store requests within the frame's budget (kyty_local_native_xpr_store_budget); the others ask again
	// in a later frame.
	const auto request_store = [&] {
		if ((kyty_local_native_xpr_diag.load(std::memory_order_relaxed) & 1u) != 0) return;
		if (const auto budget = kyty_local_native_xpr_store_budget.load(std::memory_order_relaxed); budget != 0) {
			if (cache.store_frame != m_context.FrameNumber()) {
				cache.store_frame    = m_context.FrameNumber();
				cache.store_requests = 0;
			}
			if (cache.store_requests++ >= budget) {
				LiveCounters::Add(LiveCounters::XprMissBudget);
				return;
			}
		}
		m_native_xpr_store = true;
	};
	// A changed state leaves no key behind, even when this try stops before computing it.
	if (!clean) cache.state_key_valid = false;
	const bool indexed = direct == nullptr || direct->indexed;
	if (buffer.IsInvalid() || (direct == nullptr ? commands.empty() : !commands.empty()) || commands.size() > 64 ||
	    (indexed && (index_base == 0 || index_bytes == 0)))
		return false;
	m_context.GetCommandScheduler().PopPendingOperations();
	// (The GPU thread's: a speculation's thread does not take it.)
	std::optional<RenderLockGuard> lock;
	if (Role() == 0) lock.emplace(m_context.GetMutex());
	NativeXprCollect();

	if (!clean || !cache.state_key_valid) {
		cache.state_key       = NativeXprStateKey(buffer);
		cache.state_key_valid = true;
	}
	if (kyty_local_table_xpr_mode.load(std::memory_order_relaxed) != 0) {
		const auto table = TableTry(buffer, commands, index_base, index_bytes, index_type, direct);
		if (table == TableResult::Drawn) return true;
		if (table == TableResult::Store) return LiveCounters::Add(LiveCounters::TableStores), false;
	}
	// A speculative translation draws only the table way (speculation-state.h), as does a draw of vertices.
	if (Spec::Current() != nullptr) return Spec::Refuse("native record"), false;
	if (!indexed) return false;
	NativeXprKey(buffer, cache.key);
	// Relocation: the pair's pointer words leave the key, and which use of the rest in the frame this
	// is joins it.
	const bool relocation = kyty_local_native_xpr_relocate_mode.load(std::memory_order_relaxed) != 0;
	const bool relocating = relocation && direct != nullptr;
	auto       pair       = relocating ? cache.pairs.find(NativeXprPairKey(cache.key)) : cache.pairs.end();
	cache.tried_relocate  = 0;
	cache.use_key.clear();
	const auto relocate   = [&] {
		cache.full_key = cache.key;
		for (auto words = pair->second.relocate; words != 0; words &= words - 1) cache.key[std::countr_zero(words)] = 0;
		cache.tried_relocate = pair->second.relocate;
		const auto masked    = cache.key.size();
		// The mesh (the draw's first index) joins the key, so the n-th use counts the uses of one mesh: a
		// record keeps its object while the frame's other objects of the pair come and go (counting the
		// pair's uses, a walk re-evaluated most relocated records every frame). A run's object is its first
		// draw's arguments: their address stays while its tables move every frame (and a run tried again
		// from a later draw after a miss is another key, so the uses count the same in every frame).
		const uint64_t object = direct != nullptr ? index_base : commands[0];
		cache.key.push_back(static_cast<uint32_t>(object));
		cache.key.push_back(static_cast<uint32_t>(object >> 32u));
		const auto frame  = m_context.FrameNumber();
		cache.object_hash = XXH3_64bits(cache.key.data(), cache.key.size() * sizeof(uint32_t));
		auto use          = cache.NextOccurrence(cache.object_hash, frame);
		cache.use_key.clear();
		// An object whose record rebound its descriptors (the instances of a mesh drawn in another order): its
		// descriptor bases through this draw's pointers join the key and the uses count among draws of the
		// same ones, so each instance keeps its record whatever order the frame draws them in (instances that
		// share all of them differ in data words only, which a use patches). The key by use stays a second
		// choice: an object whose descriptors move every frame keeps that record.
		if (const auto& identity = pair->second.identity;
		    !identity.empty() && kyty_local_native_xpr_instance_mode.load(std::memory_order_relaxed) != 0) {
			if (cache.Rebound(cache.object_hash)) {
				cache.use_key = cache.key;
				cache.use_key.push_back(use);
				std::array<uint32_t, 8> words {};
				uint64_t                table  = 0;
				bool                    mapped = false;
				for (size_t i = 0; i < identity.size(); ++i) {
					const auto& id = identity[i];
					if (i == 0 || id.low_word != identity[i - 1].low_word || id.high_word != identity[i - 1].high_word ||
					    id.pointer != identity[i - 1].pointer) {
						const auto root = NativeXprBase(cache.full_key[id.low_word], cache.full_key[id.high_word]);
						table           = root == 0 ? 0
						                            : NativeXprBase(NativeXprReadGuestWord(root + id.pointer),
						                                            NativeXprReadGuestWord(root + id.pointer + 4));
						// A pointer read from memory may hold anything: what is read through it must be mapped.
						mapped = table != 0 && Libs::LibKernel::Memory::IsFullyMapped(table + id.offset, 4);
					}
					if (mapped) words[i] = NativeXprReadGuestWord(table + id.offset);
				}
				cache.key.push_back(static_cast<uint32_t>(XXH3_64bits(words.data(), identity.size() * sizeof(uint32_t))));
				use = cache.NextOccurrence(XXH3_64bits(cache.key.data(), cache.key.size() * sizeof(uint32_t)), frame);
			}
		}
		cache.key.push_back(use);
		cache.suffix_at = masked;
		cache.tried_suffix.assign(cache.key.begin() + static_cast<std::ptrdiff_t>(masked), cache.key.end());
	};
	if (pair != cache.pairs.end() && pair->second.relocate != 0) relocate();
	auto hash  = XXH3_64bits(cache.key.data(), cache.key.size() * sizeof(uint32_t));
	auto found = cache.IndexFind(hash);
	// A run whose whole key misses can be an object whose tables move every frame: its relocated key. Runs
	// whose whole keys come again keep them (relocating those too made every use read through the pointers:
	// Boletaria 1-1 23.6 -> 32.3 ms a frame).
	uint64_t whole_hash = 0;
	if (relocation && direct == nullptr && (found == nullptr || !found->SameKey(cache.key))) {
		pair = cache.pairs.find(NativeXprPairKey(cache.key));
		if (pair != cache.pairs.end() && pair->second.relocate != 0 && !pair->second.blocked && !pair->second.unstorable) {
			whole_hash = hash;
			relocate();
			hash  = XXH3_64bits(cache.key.data(), cache.key.size() * sizeof(uint32_t));
			found = cache.IndexFind(hash);
		}
	}
	// A key by descriptor bases that misses: one the last frames had is stored (the normal path draws once);
	// a new one (a new instance, or an object whose descriptors move every frame) takes the object's use
	// instead, its record or a store by use.
	std::optional<bool> seen_again;
	if ((found == nullptr || !found->SameKey(cache.key)) && !cache.use_key.empty()) {
		seen_again = cache.SeenAgain(hash, m_context.FrameNumber());
		if (!*seen_again) {
			seen_again.reset();
			cache.key.swap(cache.use_key);
			cache.tried_suffix.assign(cache.key.begin() + static_cast<std::ptrdiff_t>(cache.suffix_at), cache.key.end());
			hash  = XXH3_64bits(cache.key.data(), cache.key.size() * sizeof(uint32_t));
			found = cache.IndexFind(hash);
		}
	}
	if (found != nullptr) NativeXprPrefetchFields(*found);
	LiveCounters::Add(direct != nullptr ? LiveCounters::XprDirectTries : LiveCounters::XprTries);
	// The key a store this try requests is refused under (a found record's too: dropped, it stores again).
	cache.tried_hash = hash;
	if (found == nullptr || !found->SameKey(cache.key)) {
		LiveCounters::Add(LiveCounters::XprMissKey);
		if (!relocating && whole_hash == 0) pair = cache.pairs.find(NativeXprPairKey(cache.key));
		if (pair != cache.pairs.end() && (pair->second.blocked || pair->second.unstorable)) {
			LiveCounters::Add(LiveCounters::XprMissBlocked);
			return false;
		}
		// A run's whole key that comes again is stored as it is.
		if (whole_hash != 0 && !cache.Refused(whole_hash, m_context.FrameNumber()) &&
		    cache.SeenAgain(whole_hash, m_context.FrameNumber())) {
			cache.tried_relocate = 0;
			cache.tried_hash     = whole_hash;
			request_store();
			return false;
		}
		// The first store of a pair learns its pointer words (a run's too: a run whose tables move every frame
		// never comes again under its whole key, so only a store learns that it can be relocated).
		if (relocation && (relocating || whole_hash == 0) && (pair == cache.pairs.end() || !pair->second.relocation_known)) {
			request_store();
			return false;
		}
		if (cache.Refused(hash, m_context.FrameNumber())) {
			LiveCounters::Add(LiveCounters::XprMissRefused);
			return false;
		}
		if (!(seen_again ? *seen_again : cache.SeenAgain(hash, m_context.FrameNumber()))) {
			LiveCounters::Add(LiveCounters::XprMissUnseen);
			return false;
		}
		request_store();
		return false;
	}
	auto& record = *found;
	// A record that no longer holds: the normal path draws and stores again.
	const auto drop = [&] {
		LiveCounters::Add(LiveCounters::XprMissValidate);
		cache.IndexErase(hash);
		const auto owner = cache.records.find(hash);
		NativeXprRetire(std::move(owner->second), true);
		cache.records.erase(owner);
		request_store();
		return false;
	};
	// A relocated record reads through this draw's pointers (NativeXprGather) and holds its unread words.
	if (!record.roots.empty() || !record.unread.empty()) {
		if (cache.tried_relocate == 0) return false;
		LiveCounters::Add(LiveCounters::XprRelocated);
		const auto& key = cache.full_key;
		for (const auto& word: record.unread) {
			auto& stage = record.stages[word.stage];
			if (word.user < stage.user_data.size())
				stage.user_data[word.user] = stage.snapshot.user_data[word.user] = key[word.word];
		}
		for (const auto& root: record.roots)
			if (root.same != NativeXprNoRoot && (key[root.low_word] != key[record.roots[root.same].low_word] ||
			                                     key[root.high_word] != key[record.roots[root.same].high_word]))
				return drop();
		for (auto& root: record.roots) {
			const auto low   = cache.full_key[root.low_word];
			const auto high  = cache.full_key[root.high_word];
			auto&      stage = record.stages[root.stage];
			if (stage.user_data[root.user] == low && stage.user_data[root.user + 1] == high) continue;
			stage.user_data[root.user] = stage.snapshot.user_data[root.user] = low;
			stage.user_data[root.user + 1] = stage.snapshot.user_data[root.user + 1] = high;
			if (root.low_data != NativeXprNoRoot) stage.shader_data[root.low_data] = low;
			if (root.high_data != NativeXprNoRoot) stage.shader_data[root.high_data] = high;
			root.base = NativeXprBase(low, high);
		}
	}
	record.used_frame = m_context.FrameNumber();
	++record.uses;
	if (kyty_local_native_xpr_predict_mode.load(std::memory_order_relaxed) != 0) {
		// Learn the order, then fetch the fields of the predicted next record while
		// this draw runs. Records are found through the index only (never a stale pointer).
		if (cache.last_hash != 0 && cache.last_hash != hash) {
			if (auto* previous = cache.IndexFind(cache.last_hash)) previous->next_hash = hash;
			if (cache.last2_hash != 0 && cache.last2_hash != hash)
				if (auto* before = cache.IndexFind(cache.last2_hash)) before->next2_hash = hash;
			cache.last2_hash = cache.last_hash;
		}
		cache.last_hash = hash;
		// Two ahead: the index line of the record after next, so the lookup below finds the
		// next record's slot in cache (the index is too large to stay there between frames).
		if (record.next2_hash != 0) __builtin_prefetch(cache.IndexSlotOf(record.next2_hash));
		if (record.next_hash != 0)
			if (auto* next = cache.IndexFind(record.next_hash)) NativeXprPrefetchFields(*next);
	}
	// States are never erased (a replaced one keeps its address): the last lookup's entry
	// stands while the state key is the same.
	if (cache.last_state == nullptr || cache.last_state_key != cache.state_key) {
		const auto state_it = cache.states.find(cache.state_key);
		if (state_it == cache.states.end()) {
			LiveCounters::Add(LiveCounters::XprMissState);
			request_store();
			return false;
		}
		cache.last_state_key = cache.state_key;
		cache.last_state     = state_it->second.get();
	}
	const auto pipeline_it = std::ranges::find(record.pipelines, cache.state_key,
	                                           &std::pair<uint64_t, const PipelineCache::Pipeline*>::first);
	if (pipeline_it == record.pipelines.end()) {
		LiveCounters::Add(LiveCounters::XprMissState);
		request_store();
		return false;
	}
	// A direct draw also needs the offsets the normal path resolved under this state.
	NativeXprDirect direct_draw;
	if (direct != nullptr) {
		const auto offsets =
		    std::ranges::find(record.direct_offsets, cache.state_key, &NativeXprRecord::DirectOffsets::state);
		if (offsets == record.direct_offsets.end()) {
			LiveCounters::Add(LiveCounters::XprMissState);
			request_store();
			return false;
		}
		direct_draw = {*direct, offsets->vertex, offsets->instance};
	}
	const auto& draw_state = *cache.last_state;
	if (!NativeXprTargetsCurrent(draw_state)) {
		LiveCounters::Add(LiveCounters::XprMissTarget);
		request_store();
		return false;
	}
	// Records are mostly used once per frame, so their data is cold: load the
	// arrays together instead of missing on each in turn.
	NativeXprPrefetchArrays(record);
	// A sampled image that is also a render target takes the normal path.
	for (const auto id: record.image_ids)
		for (const auto& target: draw_state.targets)
			if (id == target.id) {
				LiveCounters::Add(LiveCounters::XprMissTarget);
				return false;
			}
	const auto frame = m_context.FrameNumber();
	// Every use reads the SRT words again: the guest may rewrite constants between
	// two draws of one frame (the normal path reads them at each draw).
	if ((kyty_local_native_xpr_diag.load(std::memory_order_relaxed) & 4u) != 0) return false;
	if (!NativeXprValidate(record, frame)) return drop();
	if (kyty_local_native_xpr_mode.load(std::memory_order_relaxed) == 2) {
		m_native_xpr_verify = {&record, &draw_state};
		return false;
	}
	const bool emitted = NativeXprEmit(buffer, record, draw_state, *pipeline_it->second, commands, index_base,
	                                   index_bytes, index_type, false, direct != nullptr ? &direct_draw : nullptr);
	if (emitted) LiveCounters::Add(direct != nullptr ? LiveCounters::XprDirectHits : LiveCounters::XprHits);
	return emitted;
}

// Mode 2: the normal path prepared the draw the native path would have emitted
// from `m_native_xpr_verify`; every difference is a record the proofs kept
// although it went stale (or state they do not cover).
void RenderExecutor::NativeXprVerify(CommandBuffer& buffer, const DrawRenderState& state,
                                     const RenderState& rendering, std::span<PreparedBindings* const> stages,
                                     const DrawEmitInfo& emit) {
	const auto pending = m_native_xpr_verify;
	m_native_xpr_verify = {};
	if (pending.record == nullptr) return;
	const auto& record     = *pending.record;
	const auto& draw_state = *pending.state;
	static uint64_t                 verified = 0, mismatches = 0;
	static std::array<uint64_t, 12> kinds {};
	++verified;
	uint32_t mismatch = 0;
	if (stages.size() != record.stage_count) mismatch |= 1u;
	for (uint32_t st = 0; st < record.stage_count && st < stages.size(); ++st) {
		const auto& prepared = *stages[st];
		const auto& stage    = record.stages[st];
		const auto& fresh    = prepared.runtime->resources;
		const auto& layout   = stage.program->bindings;
		if (prepared.runtime->program != stage.program) mismatch |= 2u;
		if (!(fresh.buffers == stage.snapshot.buffers) || !(fresh.images == stage.snapshot.images) ||
		    !(fresh.samplers == stage.snapshot.samplers) || fresh.user_data != stage.snapshot.user_data ||
		    !(fresh.uniform_fill == stage.snapshot.uniform_fill))
			mismatch |= 4u;
		if (fresh.flattened_srt != stage.flattened) mismatch |= 8u;
		// Shader data, except the buffer alignment bytes (streamed slots differ).
		if (prepared.shader_data.size() != stage.shader_data.size()) {
			mismatch |= 16u;
		} else {
			const auto first = layout.memory_offset_dword;
			const auto last  = first + (static_cast<uint32_t>(stage.program->info.buffers.size()) + 3u) / 4u;
			for (uint32_t i = 0; i < prepared.shader_data.size(); ++i)
				if ((i < first || i >= last) && prepared.shader_data[i] != stage.shader_data[i]) mismatch |= 16u;
		}
		if (prepared.images.size() != stage.views.size()) mismatch |= 32u;
		else
			for (size_t i = 0; i < stage.views.size(); ++i)
				if (prepared.images[i].image_view != stage.views[i]) mismatch |= 32u;
		if (prepared.samplers != stage.samplers) mismatch |= 64u;
		for (size_t i = 0; i < prepared.buffer_sources.size(); ++i) {
			const auto& source = prepared.buffer_sources[i];
			if (source.address == 0 || source.size == 0) continue;
			if (std::ranges::none_of(record.buffers, [&](const auto& dep) {
				    return dep.address == source.address && dep.size == source.size;
			    }))
				mismatch |= 128u;
		}
	}
	const auto& a = draw_state.rendering;
	if (a.num_color_attachments != rendering.num_color_attachments || a.width != rendering.width ||
	    a.height != rendering.height || a.num_layers != rendering.num_layers ||
	    a.depth_stencil_attachment.image_view != rendering.depth_stencil_attachment.image_view)
		mismatch |= 256u;
	for (uint32_t i = 0; i < a.num_color_attachments && i < rendering.num_color_attachments; ++i)
		if (a.color_attachments[i].image_view != rendering.color_attachments[i].image_view) mismatch |= 256u;
	if (a.depth_stencil_attachment.image_layout != rendering.depth_stencil_attachment.image_layout)
		mismatch |= 512u;
	const auto dynamic = LocalDrawRecording::Capture(
	    buffer, state.vs_input_info, state.color_info, state.color_count, state.depth_info,
	    m_context.GetGraphics().attachment_feedback_loop_enabled,
	    rendering.depth_stencil_attachment.image_layout == vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
	        ? vk::ImageAspectFlags {vk::ImageAspectFlagBits::eDepth}
	        : vk::ImageAspectFlags {});
	if (std::memcmp(&dynamic, &draw_state.dynamic, sizeof(dynamic)) != 0) mismatch |= 1024u;
	// Direct draws: the offsets the record holds for this state.
	if (emit.state_offsets) {
		const auto offsets = std::ranges::find(record.direct_offsets, m_native_xpr->state_key,
		                                       &NativeXprRecord::DirectOffsets::state);
		if (offsets != record.direct_offsets.end() &&
		    (offsets->vertex != emit.vertex_offset || offsets->instance != emit.first_instance))
			mismatch |= 2048u;
	}
	if (mismatch != 0) {
		++mismatches;
		for (uint32_t bit = 0; bit < kinds.size(); ++bit)
			if (mismatch & (1u << bit)) ++kinds[bit];
		if (mismatches <= 40) {
			std::printf("NATIVE_XPR_VERIFY mismatch=0x%x vs=%016" PRIx64 " ps=%016" PRIx64 "\n", mismatch,
			            record.stages[0].program->shader_hash,
			            record.stage_count > 1 ? record.stages[1].program->shader_hash : 0);
			if (mismatch & 8u)
				for (uint32_t st = 0; st < record.stage_count && st < stages.size(); ++st) {
					const auto& fresh = stages[st]->runtime->resources.flattened_srt;
					const auto& old   = record.stages[st].flattened;
					for (size_t i = 0; i < fresh.size() && i < old.size(); ++i)
						if (fresh[i] != old[i]) {
							std::printf("  flat stage=%u index=%zu record=%08x normal=%08x\n", st, i, old[i],
							            fresh[i]);
							break;
						}
				}
			std::fflush(stdout);
		}
	}
	if (verified % 20000 == 0) {
		std::printf("NATIVE_XPR_VERIFY_SUMMARY verified=%" PRIu64 " mismatches=%" PRIu64, verified, mismatches);
		static constexpr const char* names[] = {"stages", "program", "structure", "flattened", "shader_data", "views",
		                                        "samplers", "buffers", "targets", "depth_layout", "dynamic", "offsets"};
		for (size_t i = 0; i < kinds.size(); ++i) std::printf(" %s=%" PRIu64, names[i], kinds[i]);
		std::printf("\n");
		std::fflush(stdout);
	}
}

void RenderExecutor::NativeXprForgetState() {
	if (m_native_xpr) m_native_xpr->state_key_valid = false;
}
