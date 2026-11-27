// --- Table draws (GPU-driven XPR, docs/GPU-DRIVEN-XPR.md) ------------------------------
//
// Included by renderDraw.cpp after native-xpr.inc (local Vulkan recording builds only).
//
// KYTY_TABLE_XPR=1: the XPR draws of a shader pair whose programs allow table mode
// (ShaderRecompiler::IR::EnterTableMode) need no record. For each draw the renderer evaluates the programs' table
// plans (IR::TablePlan): the slots their shaders read and the T#/S# words, from guest memory as it is when the draw
// is translated (as the normal path's SRT evaluation, which the game's command buffers patch between draws); the
// T#/S# words pick the pair's image set - a descriptor set with the images and the samplers, proven current once
// a frame - and each V# is resolved to its buffer's device address (cached while its buffer and region stay the
// same). The slots and buffer addresses go into a block in the table upload ring whose device address is pushed
// with the user data: the shaders read it, no flattened SRT binding, no buffer descriptors.
//
// The normal path, asked to store (TableStore), compiles the pair's table mode programs and the pipeline of each
// draw state, and captures the draw state (NativeXprCaptureState).
//
// KYTY_TABLE_XPR=2 also continues a run (TableContinue): a draw the observer calls clean (no register but user
// data and index state changed) right after a table draw whose emission is the last one in the command buffer
// has the same pair and draw state, with its pipeline and dynamic state still bound.

// Per role (RenderExecutor::Role): none proven yet.
static constexpr std::array<uint64_t, RenderExecutor::Roles> NoRoles() {
	std::array<uint64_t, RenderExecutor::Roles> all {};
	all.fill(UINT64_MAX);
	return all;
}

// A draw state's pipeline.
struct RenderExecutor::TableVariant {
	uint64_t                       state          = 0;
	const PipelineCache::Pipeline* pipeline       = nullptr; // null while a worker compiles it
	bool                           fetch_embedded = false;   // the vertex program fetches (a direct draw's offsets)
	bool                           refused        = false;   // another specialization than the pair's programs
	// The pixel program's export mapping under this state (GetGraphicsPrograms).
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
	// KYTY_HWRT (local/hwrt-scene.h): a G-buffer state's capturing pipeline (null: none, or a worker still compiles it:
	// hwrt_make asks again, with the registers of a draw of this same state), and its pixel shader (the material hook).
	const PipelineCache::Pipeline*                                                 hwrt_pipeline = nullptr;
	std::shared_ptr<std::function<const PipelineCache::Pipeline*(CommandBuffer&)>> hwrt_make;
	uint64_t                                                                       hwrt_ps    = 0;
	bool                                                                           hwrt_scene = false;
	vk::ShaderModule                                                               hwrt_ps_module = nullptr; // (hwrt3: material)
};

// Words a draw writes whole after resizing (TableEvaluate's T#/S# words): resize leaves the new ones uninitialized. (With
// std::allocator it zeroed them first, a memset call per draw: TableEvaluate 7.1 -> 6.8% of the GPU thread at Latria.)
template <typename T>
struct TableUninitialized: std::allocator<T> {
	using std::allocator<T>::allocator;
	template <typename U>
	struct rebind {
		using other = TableUninitialized<U>;
	};
	template <typename U>
	void construct(U* at) noexcept {
		::new (static_cast<void*>(at)) U;
	}
	template <typename U, typename... Args>
	void construct(U* at, Args&&... args) {
		::new (static_cast<void*>(at)) U(std::forward<Args>(args)...);
	}
};
using TableWords = std::vector<uint32_t, TableUninitialized<uint32_t>>;

// The descriptor set of one choice of images and samplers (the T#/S# words of every stage).
struct RenderExecutor::TableImageSet {
	struct Image {
		ImageId                 id;
		uint64_t                serial = 0;
		vk::ImageLayout         layout = vk::ImageLayout::eUndefined;
		ImageSubresourceRange   range;
		TextureCache::ImageDesc desc;
		vk::AccessFlags2        access = vk::AccessFlagBits2::eShaderRead;
		bool                    target  = false; // a target of the draw state the set was made under (its access)
		bool                    storage = false; // the shader writes it: GPU-modified at every use (TableTransit)
		bool                    whole   = false; // bound both as a texture and as storage: the whole image's state
		// The main executor's proof of its use (a target, or a texture without DCC found current): the image's validity
		// then, which the proof holds with while no registration stamped the view's pages since (TableValidateSet).
		bool                    proven   = false;
		uint64_t                validity = 0;
	};
	TableWords            words; // without the LOD statistics fields (NativeXprLodStatsMask)
	vk::DescriptorSet     set  = nullptr;
	vk::DescriptorPool    pool = nullptr;
	std::vector<Image>    images;
	// (hwrt3, KYTY_HWRT_REFL_MATERIAL: HwScene::MaterialActive only) the pixel stage's sampled images, by SPIR-V binding and
	// element (the base-colour texture of a captured mesh is chosen among them)
	std::vector<HwScene::PsImage> hwrt_ps_images;
	// BufferCache::BdaTableGeneration when the set got the BDA page table: made again once the table grew.
	uint64_t              bda_generation = 0;
	// Proven current in that frame under those registrations, by role (RenderExecutor::Role: [0] also the frame it was
	// made or refused in); used in that frame (by any executor).
	std::array<uint64_t, Roles> validated_frame = NoRoles();
	std::array<uint64_t, Roles> resolution_epoch {};
	std::atomic<uint64_t>       used_frame {0};
	// The texture cache's flip count (TextureCache::GcTick) and frame (CurrentFrame) when the main executor last checked
	// every image in full, which touches their uses (the LRU, the last use): again within TouchTicks of either (an image
	// unused for 32 frames may go to an overlap, for 600 flips to the collector).
	uint64_t                  touched_tick  = 0;
	uint64_t                  touched_frame = 0;
	static constexpr uint64_t TouchTicks    = 8;
	// The memory its images' views span (from the lowest to the highest byte): registrations elsewhere reach none of
	// its proofs (TableValidateSet). 0, 0: not known yet.
	uint64_t                  span_begin = 0, span_end = 0;
	// TextureCache::ImagesEpoch after the main executor last transited its images (TableTransit) for a draw under
	// transit_state (null: a dispatch): they stay in place while it does not move (UINT64_MAX: not yet).
	uint64_t                  transit_epoch = UINT64_MAX;
	const NativeXprDrawState* transit_state = nullptr;
	// The words select another specialization, or an image no set can hold: the draws take the native path
	// (TableResolveSet tries again after TableImageSet::RetryFrames).
	bool                  refused = false;
	static constexpr uint64_t RetryFrames = 300;
	// A dispatch's words that select another specialization of the pair's program whose table form is the same
	// (TableInterchangeable: its images of other dimensions or formats): that program and its pipeline, which the set
	// is made for (null: the pair's).
	const ShaderRecompiler::IR::CompiledShaderInfo* program  = nullptr;
	const PipelineCache::Pipeline*                  pipeline = nullptr;
};

// Two specializations of a program the table way alike but for their images' and samplers' bindings: the same slots,
// buffers (and their decoded formats), descriptor words, page table reads and shader data. The words TableEvaluate
// reads and the blocks TableBlocks makes for one serve the other.
static bool TableInterchangeable(const ShaderRecompiler::IR::CompiledShaderInfo& a,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& b) {
	const auto& x = a.bindings;
	const auto& y = b.bindings;
	return a.table_mode && b.table_mode && a.stage == b.stage && a.shader_hash == b.shader_hash &&
	       a.user_data_base == b.user_data_base && a.table_plan == b.table_plan && a.info.buffers == b.info.buffers &&
	       a.info.images.size() == b.info.images.size() && a.info.samplers.size() == b.info.samplers.size() &&
	       a.bda_read_plan == b.bda_read_plan && x.push_data_start_dword == y.push_data_start_dword &&
	       x.memory_offset_dword == y.memory_offset_dword && x.memory_offset_count == y.memory_offset_count &&
	       x.buffer_word_count == y.buffer_word_count && x.lod_stats_count == y.lod_stats_count &&
	       x.table_block_dwords == y.table_block_dwords && x.user_data_registers == y.user_data_registers;
}

// A stage's flat form as TableRuns makes it (TablePack packs it into the pair): every operand an index into one of
// three arrays (the stage's user data, its block's slots, the stage's constants).
struct RenderExecutor::TableFlatStage {
	struct Operand {
		uint32_t index = 0;
		uint32_t base  = 2; // 0: user data, 1: slots, 2: constants
	};
	struct Run {
		std::array<Operand, 3> address; // low, high, offset
		int32_t                immediate = 0;
		uint32_t               first = 0, count = 0;
		Operand                records; // a bounded run's (TablePlan::Slot::bounded)
		bool                   bounded = false;
	};
	std::vector<Run>      runs;         // the runs in level order
	std::vector<uint32_t> level_ends;   // the run after each level
	std::vector<Operand>  image_words;  // every image's words, then every sampler's
	std::vector<uint32_t> image_counts;
	std::vector<Operand>  buffer_words; // every buffer's four V# words (TableBlocks)
	std::vector<uint32_t> constants;
	uint32_t              user_words = 0; // the user data it reads: below this
};

struct RenderExecutor::TablePair {
	// Evaluated slots at consecutive addresses (the same base, immediates 4 apart): one read, at the address of the
	// first. Ordered by level: a run whose address is made of slots is read after the runs of those slots.
	struct Run {
		ShaderRecompiler::IR::TablePlan::Slot read;
		uint32_t                              first = 0, count = 0, level = 0;
	};
	uint64_t                                                    vs = 0, ps = 0; // (a dispatch's: its shader, 0)
	// The GS user SGPR the CP writes an indirect draw's start instance into (START_INST_LOC, -1: none): the vertex
	// program is translated for it (ShaderVertexInputInfo::start_instance_sgpr).
	int32_t                                                     start_instance = -1;
	bool                                                        refused  = false;
	const char* special = nullptr; // a dispatch's: the normal path's special path that takes it (refused)
	std::array<const ShaderRecompiler::IR::CompiledShaderInfo*, 2> programs {}; // vertex, pixel (null: inactive)
	std::array<std::vector<Run>, 2>                             runs;
	// TableEvaluate's form of a stage's plan (TableRuns, TablePack): every operand an index into one of three arrays (the
	// stage's user data, its block's slots, the stage's constants), so a draw's evaluation branches on no operand kind.
	// Both stages' arrays in one block, each operand 16 bits (its array in the top two, its index in the low 14), each
	// run 16 bytes: a pair a draw switches to is cold, and its arrays were a dozen allocations of 8-byte operands, as
	// many cold lines on as many pages (TablePrefetchArrays). Same process, 10 rounds: Stonefang standing render 8.94 ->
	// 8.75 ms, Latria 13.47 -> 13.31.
	struct PackedRun {
		std::array<uint16_t, 3>   address {}; // low, high, offset
		uint16_t                  records = NotBounded;
		int32_t                   immediate = 0;
		uint16_t                  first = 0, count = 0;
		static constexpr uint16_t NotBounded = 0xffff;
	};
	struct PackedStage {
		std::span<const PackedRun> runs;
		std::span<const uint16_t>  level_ends, image_words, buffer_words;
		std::span<const uint8_t>   image_counts;
		std::span<const uint32_t>  constants;
		uint32_t                   user_words = 0;
		bool                       valid      = false; // (else an index or a count past its packed width)
	};
	std::array<PackedStage, 2>      packed;
	std::unique_ptr<std::byte[]>    packed_block;
	const std::byte*                packed_at    = nullptr; // (the block's first 64-byte line)
	uint32_t                        packed_bytes = 0;
	std::vector<TableVariant>                                   variants;
	uint32_t                                                    variant_hint = 0; // the one TableTry found last
	uint64_t                                                    store_frame = 0, store_state = 0; // the last request
	std::unordered_map<uint64_t, std::unique_ptr<TableImageSet>, HashedKey> sets; // by T#/S# words
	std::array<TableImageSet*, Roles>                           last_set {}; // by role
	// The set before the last, by role (TableSet asks it before the map: a pass's draws of one pair alternate between
	// two materials' sets). Cleared with last_set.
	std::array<TableImageSet*, Roles>                           previous_set {};
	// ShaderMapGeneration under which the shaders at vs and ps were proven those the programs were compiled
	// from (the game registers shaders often; code replaced at an address resets the pair), by role.
	std::array<uint64_t, Roles>                                 generation = NoRoles();
	// A dispatch's (TableDispatch): its program's input and pipeline, and the normal path's dispatches seen.
	std::unique_ptr<ShaderComputeInputInfo> compute;
	ShaderProgram                           compute_program;
	const PipelineCache::Pipeline*          pipeline  = nullptr;
	uint32_t                                sightings = 0;
	// Per stage, the buffers whose V#s were seen in another format than the programs decode (bits by buffer): the
	// programs are compiled again with those buffers' formats decoded at run time (`respecialize`, TableReset).
	std::array<uint32_t, 2> portable {};
	bool                    respecialize = false;
	// A translation or a pipeline compile the pair's next store needs (TableStore, TableDispatch): until it is done,
	// its draws keep the native path without asking for a store (a request a frame each kept the store budget).
	std::shared_ptr<const std::atomic<bool>> waiting;
	// A dispatch's: the pair of the table dispatch that came after this one's the last time (the main executor's):
	// TablePrefetchNext loads its lines while this one is made.
	TablePair* next = nullptr;
	// The frame the main executor last looked it up (TableSweepPairs lets one go that none used for a while).
	uint64_t used_frame = 0;
};

struct RenderExecutor::TableXpr {
	using Pair    = TablePair;
	using Variant = TableVariant;
	std::unordered_map<uint64_t, Pair, HashedKey> pairs;      // by shader addresses
	std::unordered_map<uint64_t, Pair, HashedKey> dispatches; // by the compute stage's registers (TableDispatchKey)
	// Pairs no draw or dispatch used for PairKeepFrames leave (TableSweepPairs: every PairSweepFrames, past PairsKept):
	// they are keyed by the shaders' addresses, and a world change loaded its shaders elsewhere (~1000 new pairs, ~8 KB
	// each with their runs). Pairs do not move: a pointer stays valid until then, when every one held is dropped.
	static constexpr uint64_t PairSweepFrames = 600, PairKeepFrames = 3600;
	static constexpr size_t   PairsKept       = 2048;
	uint64_t                  pairs_swept     = 0;
	// The pairs found last, direct-mapped by key.
	static constexpr size_t PairCacheEntries = 4096; // (kyty_local_table_mru bit 1: else its first 256 only)
	std::array<std::pair<uint64_t, Pair*>, PairCacheEntries> pair_cache {};
	// The pair TableTry found last, by its key (consecutive draws mostly keep their shaders: no hash of the key, ~0.6% of
	// the render thread at 1-1 standing, nor lookup then).
	std::array<uint64_t, 3> found_key {};
	Pair*                   found_pair = nullptr;
	Pair*                   previous_dispatch = nullptr; // the pair of the last table dispatch (TablePrefetchNext)
	// The same for TableDispatch's dispatches (the main executor's own: its catalog), by TableDispatchWords.
	std::array<uint64_t, 4> found_dispatch_key {};
	Pair*                   found_dispatch = nullptr;
	uint64_t collected  = 0;                           // the frame unused image sets were last dropped
	// Moves when a pair is reset or a set leaves (sets are replaced): an executor reading this catalog drops its last
	// pair, state and set (RenderExecutor::m_catalog) when it moved since it saw it (seen_epoch, its own table's).
	uint64_t epoch = 0, seen_epoch = 0;
	// The next prepared draw of the pair under this state stores (TableStore); the frame's store requests so far.
	Pair*    store       = nullptr;
	uint64_t store_state = 0, store_frame = 0;
	uint32_t store_requests = 0;
	// (tiron1, KYTY_TIRON1=1) The render thread's time this frame from each granted store request to the end of its
	// TableStore (its normal-path draw and the store), and the time the last one was granted.
	int64_t tiron1_store_us = 0, tiron1_grant_ns = 0;
	// The draw's T#/S# words (the set's key) and pixel stage LOD statistics words; its runs' addresses.
	TableWords                           words;
	std::vector<uint32_t>                lod_stats;
	std::vector<uint64_t>                addresses;
	// 16 KiB guest pages seen mapped (direct-mapped by page), while the mapping epoch is the same.
	std::array<uint64_t, 256> mapped {};
	uint64_t                  mapping_epoch = UINT64_MAX;
	// The last table draw (TableContinue): its pair, the index of its variant and its draw state.
	Pair*                     last_pair    = nullptr;
	size_t                    last_variant = 0;
	const NativeXprDrawState* last_state   = nullptr;
	const TableImageSet*      last_set     = nullptr; // the set the last emission bound
	uint64_t                  images_epoch = 0;       // TextureCache::ImagesEpoch after it
	// The game buffer the last arguments came from, and the indices' by their address (each mesh has its own): used
	// again while no registration changed and the guest wrote none of its region (ObtainBuffer then returns the same
	// buffer; it streams CPU-dirty ranges).
	struct HeldBuffer {
		uint64_t begin = 0, end = 0, cpu_epoch = 0, registration = 0;
		Buffer*  buffer = nullptr;
	};
	HeldBuffer                   args;
	std::array<HeldBuffer, 8192> indices;
	// A V#'s buffer as the shaders read it (direct-mapped by the V#'s words, one cache line each): its device address
	// and size in dwords (0: none; clamped to the mappings), while no buffer covering its tracker regions was registered
	// or unregistered since `registration` (the registration epoch it was last proven at: a region stamped later moved
	// a buffer) and the guest wrote none of its range (the tracker regions' epoch, else the range's dirty pages). One
	// whose proof also needs the image starts and the mappings (formatted: an image may start at it; clamped: a mapping
	// may extend it) holds while `special` is the table's special generation (0: no such need). (Buffer registrations
	// anywhere used to drop them all: 3800 resolves a frame walking at 1-1, 1000 standing.)
	// One holding a copy in the table upload ring of a range the guest wrote (RingCopy, made or shared for the main
	// executor) has `registration` RingCopyMark | the observation epoch it is shared in and `cpu_epoch` the range's
	// tracker regions' epoch then: the use takes it as RingCopy's terms allow, without resolving the V# again.
	struct alignas(64) ResolvedBuffer {
		std::array<uint32_t, 4> words {};
		uint32_t                dwords = 0, special = 0;
		vk::DeviceAddress       device = 0;
		uint64_t                address = 0, size = 0, cpu_epoch = 0, registration = 0;
	};
	static constexpr uint64_t RingCopyMark = uint64_t {1} << 63u;
	static constexpr size_t           ResolvedEntries = 65536;
	std::unique_ptr<ResolvedBuffer[]> resolved = std::make_unique<ResolvedBuffer[]>(ResolvedEntries);
	// A copy in the table upload ring of a small range the guest wrote (TableBlocks reads such a range from a copy,
	// none held: the guest writes its pages unseen), direct-mapped by the range: the main executor's uses share it
	// while no point the guest could learn that GPU work ran at passed (PageManager::ObservationEpoch: until then the
	// guest writes nothing that work reads), in the same submission (`tick`: the ring keeps the copy for it), with
	// the range's tracker regions' epoch unchanged (`cpu_epoch`: it moves when pages become CPU-dirty, when the
	// emulator writes dirty pages, MemoryTracker::NoteHostWrite, and when GPU-written pages are given back) and no
	// page of it GPU-written at the use. A frame's constants are read by many draws: in the Latria spin ~2950 copies,
	// 11.8 MB a frame, before; ~1140 and 4.6 MB with the shared ones.
	struct RingCopy {
		uint64_t          address = 0, size = 0, observation = 0, tick = 0, cpu_epoch = 0;
		vk::DeviceAddress device = 0;
	};
	static constexpr size_t           RingCopies = 4096;
	std::unique_ptr<RingCopy[]>       ring_copies = std::make_unique<RingCopy[]>(RingCopies);
	// (bloques juntos, KYTY_BLOQUES_JUNTOS) Windows of whole 4 KiB pages copied once into the ring: a later range of
	// the same submission and observation epoch inside one, with the window's tracker epoch unchanged, takes the
	// window's copy at its offset (same capture point as RingCopy: translation). Also the census's recent copies.
	struct RingWindow {
		uint64_t          base = 0, size = 0, observation = 0, tick = 0, cpu_epoch = 0;
		vk::DeviceAddress device = 0;
		const uint8_t*    host   = nullptr;
	};
	static constexpr size_t RingWindows = 8;
	std::array<RingWindow, RingWindows> ring_windows {};
	uint32_t                            ring_window_next = 0;
	std::array<RingWindow, RingWindows> censo_windows {};
	uint32_t                            censo_next = 0;
	uint32_t                          special_generation = 1; // moves with the image starts and the mappings
	// Addresses no image starts at, each with the TextureCache::StartEpoch it was proven at (only an address gaining
	// its first image moves that epoch): TableBlocks asks the texture cache (a lock and a map lookup) again only then.
	static constexpr size_t                           NoImageStartEntries = 1024;
	std::array<std::pair<uint64_t, uint64_t>, NoImageStartEntries> no_image_start {};
	uint64_t                          resolved_start = UINT64_MAX, resolved_mapping = UINT64_MAX;
	// Each stage's V#s and their entries (TableGather), gathered for `gathered` (the pair of the draw or dispatch being
	// made, null once TableBlocks took them).
	std::array<std::vector<std::pair<std::array<uint32_t, 4>, ResolvedBuffer*>>, 2> entries;
	const TablePair*                                                                gathered = nullptr;
	// A written V#'s range clamped to the mappings, and (`image_checked`) that no image starts at it, while `special`
	// is the special generation (mappings and image starts as then): obtained as written at every use, its proofs
	// (a memory lock and a texture cache lock and lookup) once (direct-mapped by the words).
	// `memo`: the buffer the range was last obtained as written into (ObtainBuffer; the main executor, no speculation,
	// unformatted), good while the tick, the registration epoch, BufferCache::WrittenObtainEpoch and the range's CPU
	// modification epoch are those of that call: everything ObtainBuffer would do over it then is done (its pages
	// GPU-owned and the range in the GPU-modified set, its feedback invalidated, its write noted for this tick, the
	// readbacks over it drained) or would find the same (the buffer). A run of dispatches of one shader writes the same
	// ~7 V#s each; at the dragon's path those ~10 000 second lookups a frame were 5 % of the render thread.
	struct WrittenRange {
		std::array<uint32_t, 4> words {};
		uint32_t                special = 0;
		bool                    image_checked = false;
		bool                    memo = false;
		uint64_t                address = 0, size = 0;
		vk::DeviceAddress       device = 0;
		BufferId                owner {};
		const Buffer*           owner_buffer = nullptr;
		uint64_t                tick = 0, registration = 0, obtain_epoch = 0, cpu_epoch = 0;
	};
	static constexpr size_t                WrittenEntries = 1024;
	std::array<WrittenRange, WrittenEntries> written;
	// (trad1b, KYTY_TRAD1; local/trad1.h) A written range's last full ObtainBuffer (the main executor, no speculation,
	// unformatted), keyed by the range itself (its clamped address and size: what ObtainBuffer is asked), not by the
	// V#'s words or their entry: kept across the special generation's moves (mappings, image starts) and across V#s
	// whose other words differ. While its registration epoch stands, its buffer is live and no GPU-written byte of its
	// tracker regions was given back since `coverage`, a use in another tick only renotes the write. 2-way by the range,
	// the older way replaced; allocated at the first use with KYTY_TRAD1 on.
	struct Trad1Range {
		uint64_t          address = 0, size = 0;
		vk::DeviceAddress device = 0;
		BufferId          owner {};
		const Buffer*     owner_buffer = nullptr;
		uint64_t          registration = 0, coverage = 0, used = 0;
	};
	static constexpr size_t       Trad1Sets = 4096;
	std::unique_ptr<Trad1Range[]> trad1_ranges;
	uint64_t                      trad1_clock = 0;
	// The entry of [address, address + size): the way that holds it, else the one to replace (address 0 there).
	Trad1Range& Trad1Find(uint64_t address, uint64_t size) {
		if (trad1_ranges == nullptr) trad1_ranges = std::make_unique<Trad1Range[]>(2 * Trad1Sets);
		const auto set = (((address >> 2u) ^ (size * 0x9e3779b97f4a7c15ull)) * 0xbf58476d1ce4e5b9ull) >> 52u;
		auto*      way = &trad1_ranges[2 * set];
		if (way[0].address == address && way[0].size == size) return way[0];
		if (way[1].address == address && way[1].size == size) return way[1];
		return way[0].used <= way[1].used ? way[0] : way[1];
	}
	// The draw's blocks (per stage): its slots (IR::TablePlan::BlockDwords), then its buffers.
	std::array<std::vector<uint32_t>, 2> blocks;
	// An indirect image's candidates for the draw or dispatch being made (per stage), in the order its set binds them
	// (TableEvaluate: the set's key; TableResolveSet), and the enumeration's scratch.
	std::array<std::vector<ShaderRecompiler::IR::DescriptorValue>, 2> indirect;
	ShaderRecompiler::IR::AddressProbeCandidates                      probe;
};

// The pair's programs and what was made for them leave (its shaders changed, or a V#'s format did:
// TablePair::portable); the next draw or dispatch compiles them again.
void RenderExecutor::TableReset(TablePair& pair) {
	auto& table = *m_table_xpr;
	EXIT_IF(Role() != 0);
	TrustedDepth::Forget(); // (trust1: its entries name the pair's programs and variants)
	for (auto& [words, set]: pair.sets) NativeXprRetireSet(set->pool, set->set);
	if (table.store == &pair) table.store = nullptr;
	if (table.last_pair == &pair) table.last_pair = nullptr;
	table.last_set = nullptr;
	++table.epoch;
	// (The key's start instance SGPR too: without it every later draw of the pair failed TableTry's key check.)
	pair = {.vs = pair.vs, .ps = pair.ps, .start_instance = pair.start_instance, .generation = pair.generation,
	        .sightings = pair.sightings, .portable = pair.portable};
}

// Draw states left (NativeXprSweepStates): the last table draw's and the image sets' transits under one are forgotten,
// and the executors reading this catalog drop what they hold of it (epoch).
void RenderExecutor::TableForgetStates() {
	auto* table = m_table_xpr.get();
	if (table == nullptr) return;
	TrustedDepth::Forget(); // (trust1: its entries name draw states)
	table->last_pair  = nullptr;
	table->last_state = nullptr;
	table->last_set   = nullptr;
	++table->epoch;
	for (auto* pairs: {&table->pairs, &table->dispatches})
		for (auto& [key, pair]: *pairs)
			for (auto& [words, set]: pair.sets)
				if (set->transit_state != nullptr) {
					set->transit_state = nullptr;
					set->transit_epoch = UINT64_MAX;
				}
}

// The pairs and dispatches no draw or dispatch looked up for PairKeepFrames leave, their image sets retired. Every
// pointer to a pair goes too: the lookups' (pair_cache, found_*), the last draw's and dispatch's, the store request,
// the gather and the prefetch chain (next). Only the catalog's owner, with no executor reading its catalog, and only at
// the start of a table draw or dispatch (no pair is held then).
void RenderExecutor::TableSweepPairs() {
	auto* table = m_table_xpr.get();
	if (table == nullptr || Role() != 0) return;
	const auto frame = m_context.FrameNumber();
	if (frame - table->pairs_swept < TableXpr::PairSweepFrames) return;
	table->pairs_swept = frame;
	if (m_catalog_readers.load(std::memory_order_relaxed) != 0 ||
	    table->pairs.size() + table->dispatches.size() <= TableXpr::PairsKept)
		return;
	const auto   write  = CatalogWrite();
	const size_t before = table->pairs.size() + table->dispatches.size();
	for (auto* pairs: {&table->pairs, &table->dispatches})
		std::erase_if(*pairs, [&](auto& item) {
			auto& pair = item.second;
			// (A translation or compile it waits for keeps it.)
			if (pair.used_frame + TableXpr::PairKeepFrames >= frame || pair.waiting) return false;
			for (auto& [words, set]: pair.sets) NativeXprRetireSet(set->pool, set->set);
			return true;
		});
	const size_t after = table->pairs.size() + table->dispatches.size();
	if (after == before) return;
	TrustedDepth::Forget(); // (trust1: its entries name pairs)
	table->pair_cache         = {};
	table->found_key          = {};
	table->found_pair         = nullptr;
	table->found_dispatch_key = {};
	table->found_dispatch     = nullptr;
	table->previous_dispatch  = nullptr;
	table->store              = nullptr;
	table->gathered           = nullptr;
	table->last_pair          = nullptr;
	table->last_state         = nullptr;
	table->last_set           = nullptr;
	++table->epoch;
	for (auto* pairs: {&table->pairs, &table->dispatches})
		for (auto& [key, pair]: *pairs) pair.next = nullptr;
	if (SlowLog::HitchThreshold() > 0.0)
		std::printf("[tsc %llu] table pairs %zu -> %zu\n", static_cast<unsigned long long>(__rdtsc()), before, after);
}

RenderExecutor::TableXpr& RenderExecutor::OwnTable() {
	if (!m_table_xpr) m_table_xpr = std::make_shared<TableXpr>();
	return *m_table_xpr;
}

// What this executor held of another's catalog (its last pair, state and set), dropped when the owner reset a pair or
// let a set go since.
void RenderExecutor::DropStaleCatalog() {
	auto* catalog = CatalogTable();
	auto& table   = OwnTable();
	if (Role() == 0 || catalog == nullptr || catalog->epoch == table.seen_epoch) return;
	table.seen_epoch = catalog->epoch;
	table.last_pair  = nullptr;
	table.last_state = nullptr;
	table.last_set   = nullptr;
}

// A table draw's or dispatch's pair: each draw's pair is another, cold, object and its evaluation
// walks a dozen arrays one after the other (runs, operands, V# words, constants, the last image set's words). Their
// lines are asked for once the pair is known (its own lines first: the vectors' headers), so the misses overlap the
// checks before the evaluation instead of following each other (1-1 spin, per call: table draw path 1.78 -> 1.68 us,
// table dispatch path 2.96 -> 2.86 us).
template <typename Pair>
static void TablePrefetchPair(const Pair& pair) {
	for (size_t line = 0; line < sizeof(Pair); line += 64) __builtin_prefetch(reinterpret_cast<const char*>(&pair) + line);
}
template <typename Pair, typename Set>
static void TablePrefetchArrays(const Pair& pair, const Set* set) {
	const auto lines = [](const auto& items) {
		const auto* at    = reinterpret_cast<const char*>(items.data());
		const auto  bytes = std::min<size_t>(items.size() * sizeof(*items.data()), 256);
		for (size_t offset = 0; offset < bytes; offset += 64) __builtin_prefetch(at + offset);
	};
	for (uint32_t offset = 0; offset < std::min<uint32_t>(pair.packed_bytes, 2048); offset += 64)
		__builtin_prefetch(pair.packed_at + offset);
	for (uint32_t st = 0; st < 2 && pair.programs[st] != nullptr; ++st) {
		lines(pair.programs[st]->table_plan.buffers); // (TableBlocks' plan and resource flags of each V#)
		lines(pair.programs[st]->info.buffers);
	}
	if (set != nullptr) {
		__builtin_prefetch(set);
		lines(set->words);
	}
}

// A frame's dispatches come in much the same order frame after frame (the culling chain's ~1000 of a few hundred
// programs): the pair that followed this one's last dispatch is noted, and the pairs of the next two dispatches expected
// load while this one is made, in two steps (each line a load reads is in by then: the next pair's arrays from its own
// lines, which the dispatch before asked for, and the pair after it, whose lines the next one's arrays need). Their
// lines were cold at each use: TablePrefetchPair asks for them only once the pair is found, ~0.1 us before the first
// read. Census per call, 1-1 standing and Latria (2 boots x 2 rounds each): async-queue table dispatches 1.76 -> 1.70
// and 2.06 -> 2.00 us (their evaluation -9%), graphics-queue ones -4%. (For draws, whose successors vary more, the
// same cost +3% a draw.)
template <typename Pair>
static void TablePrefetchNext(Pair*& previous, Pair& pair) {
	if (previous != nullptr && previous != &pair) previous->next = &pair;
	previous = &pair;
	auto* const next = pair.next;
	if (next == nullptr || next == &pair) return;
	using Set = std::remove_pointer_t<std::remove_cvref_t<decltype(next->last_set[0])>>;
	TablePrefetchArrays(*next, static_cast<const Set*>(nullptr));
	__builtin_prefetch(next->last_set[0]);
	if (auto* const after = next->next; after != nullptr && after != &pair && after != next) TablePrefetchPair(*after);
}

RenderExecutor::TableResult RenderExecutor::TableTry(CommandBuffer& buffer, std::span<const uint64_t> commands,
                                                     uint64_t index_base, uint64_t index_bytes,
                                                     vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	auto& table = OwnTable();
	Tiron1::t_no_native_store = false; // (tiron1: set below only for a pair waiting for its store)
	// (cine1) A store request whose granted draw never reached TableStore (the draw left ExecutePreparedDraw early,
	// or drew the flip composite) is dropped here, at the next draw: it no longer waits for a later, unrelated draw's
	// TableStore, which consumed it (the pair and state did not match: the body returned at once) and charged the whole
	// time in between to the store (opening cinematic, logs 20261124-074027-854 / -082011-604: "draw normal del store"
	// 8.7 ms a store, 286 bodies of which 257 returned at once).
	if (Role() == 0 && table.store != nullptr) {
		table.store           = nullptr;
		table.tiron1_grant_ns = 0;
	}
	if (LiveCensus::Mdraw::Active() && Role() == 0) LiveCensus::Mdraw::t_entry = __rdtsc(); // (mdraw C0: the call's clock)
	// (Another executor's catalog is read under its lock.)
	std::shared_lock<std::shared_mutex> catalog_read;
	if (Role() != 0) catalog_read = CatalogRead();
	DropStaleCatalog();
	TableSweepPairs();
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1e, 0); // census: the table path's phases (b)
	// (thread1, KYTY_RENDER_STATS: the pair, variant and draw state lookups, up to TableDraw.)
	std::optional<LiveCensus::StatScope> stat_lookup;
	stat_lookup.emplace(0, 15);
	auto*       catalog = CatalogTable();
	auto&       cache   = *m_native_xpr;
	const auto& sh      = buffer.GetShaders();
	const auto& vs      = sh.GetVs();
	// The vertex program is the ES stage's (ShaderGetInputInfoVS), without tessellation.
	if (vs.es_regs.data_addr == 0 || vs.ls_regs.data_addr != 0 || vs.hs_regs.data_addr != 0)
		return Spec::Refuse("table: stages"), TableResult::Native;
	const auto frame      = m_context.FrameNumber();
	const auto vs_address = vs.es_regs.data_addr, ps_address = sh.GetPs().ps_regs.data_addr;
	const std::array<uint64_t, 3> pair_key {vs_address, ps_address, static_cast<uint32_t>(vs.start_instance_user_sgpr)};
	auto* own   = Role() != 0 ? nullptr : &table.pairs;
	auto* known = own != nullptr && table.found_key == pair_key ? table.found_pair : nullptr;
	if (known == nullptr) {
		const auto pair_hash = XXH3_64bits(pair_key.data(), sizeof(pair_key));
		// (Another executor's catalog: what it learned, none of it made here.)
		const auto found = catalog == nullptr || Role() == 0 ? decltype(catalog->pairs.end()) {} : catalog->pairs.find(pair_hash);
		if (Role() != 0 && (catalog == nullptr || found == catalog->pairs.end())) return Spec::Refuse("table store"), TableResult::Native;
		// (A frame's draws use ~600 pairs at 1-1: 256 entries missed often, each a cold map bucket and node.)
		const auto entries = (kyty_local_table_mru.load(std::memory_order_relaxed) & 1u) != 0 ? TableXpr::PairCacheEntries : 256;
		auto&      cached  = table.pair_cache[pair_hash & (entries - 1)];
		known        = own != nullptr && cached.first == pair_hash ? cached.second : nullptr;
		if (known == nullptr) {
			auto it = own != nullptr ? own->find(pair_hash) : found;
			if (own != nullptr && it == own->end()) {
				const auto write    = CatalogWrite();
				it                  = own->try_emplace(pair_hash).first;
				auto& fresh         = it->second;
				fresh.vs            = vs_address;
				fresh.ps            = ps_address;
				fresh.start_instance = vs.start_instance_user_sgpr;
			}
			known = &it->second;
			if (own != nullptr) cached = {pair_hash, known};
		}
		if (own != nullptr) {
			table.found_key  = pair_key;
			table.found_pair = known;
		}
	}
	auto& pair = *known;
	TablePrefetchPair(pair);
	if (Role() == 0) pair.used_frame = frame;
	if (pair.vs != vs_address || pair.ps != ps_address || pair.start_instance != vs.start_instance_user_sgpr)
		return Spec::Refuse("table: pair key"), TableResult::Native;
	if (const auto generation = ShaderMapGeneration(); generation != pair.generation[Role()]) {
		if (pair.programs[0] != nullptr &&
		    (ShaderHashAt(vs_address) != pair.programs[0]->shader_hash ||
		     (pair.programs[1] != nullptr && ShaderHashAt(ps_address) != pair.programs[1]->shader_hash))) {
			if (Role() != 0) return Spec::Refuse("table: shaders changed"), TableResult::Native;
			const auto write = CatalogWrite();
			pair.portable    = {};
			TableReset(pair);
		}
		pair.generation[Role()] = generation;
	}
	if (pair.respecialize) {
		if (Role() != 0) return Spec::Refuse("table: shaders changed"), TableResult::Native;
		const auto write = CatalogWrite();
		TableReset(pair);
	}
	if (pair.refused) return Spec::Refuse("table: pair refused"), TableResult::Native;
	// KYTY_FRAMEGEN: the frame's first G-buffer draw goes the normal way (or a native record's), whose hook reads the
	// camera from its constants (a speculation's thread leaves them all to the GPU thread's).
	if (FrameGen::Enabled() && pair.programs[0] != nullptr &&
	    FrameGen::IsGbufferVertexShader(pair.programs[0]->shader_hash) && (Role() != 0 || FrameGen::WantsCamera()))
		return Spec::Refuse("table: frame generation camera"), TableResult::Native;
	// A store at most once a frame per pair and draw state, within the frame's budget (kyty_local_table_store_budget):
	// while its programs and pipeline compile, the pair's draws take the normal path (Deferred). (Once a frame per pair
	// kept its other states' draws on the native path, storing native records, until a later frame.) Those draws
	// stored native records too, which the table variant replaced a frame or two later: 100-230 a slow frame walking
	// into new parts of Boletarian Palace 1-3 (each traced and sorted, ~3 ms of the frame).
	// (tiron1, KYTY_TIRON1=1) A pair waiting for its store: its draw takes the native record it has (a miss draws the
	// normal way, as Deferred), and stores none (native-xpr.inc NativeXprTry): the table variant replaces it soon.
	const bool tiron1        = Tiron1::On();
	const auto tiron1_wait   = [&] {
		LiveCounters::Add(LiveCounters::TableStoreDeferred);
		Tiron1::t_no_native_store = true;
		Tiron1::g_native_kept.fetch_add(1, std::memory_order_relaxed);
		return TableResult::Native;
	};
	// (KYTY_STORE_SIN_DRAW) The store made here from the seed a normal draw left (StoreSinDraw, native-xpr.inc):
	// true when done (or waiting for its programs/pipeline, or refused), the draw then takes its native record.
	// (0: no seed, the normal way; 1: done; 2: the frame's time cap reached, the draw waits.)
	const auto seed_store = [&]() -> int {
		static uint64_t cap_frame = 0;
		static int64_t  cap_used  = 0;
		if (cap_frame != frame) cap_frame = frame, cap_used = 0;
		const auto pair_key = StoreSinDraw::PairKey(pair.vs, pair.ps, static_cast<uint64_t>(pair.start_instance));
		auto       seed     = StoreSinDraw::Copy(
            StoreSinDraw::Key(pair.vs, pair.ps, static_cast<uint64_t>(pair.start_instance), cache.state_key));
		if (!seed) {
			StoreSinDraw::g_no_seed.fetch_add(1, std::memory_order_relaxed);
			StoreSinDraw::Classify(pair_key, cache.state_key);
			return 0;
		}
		{
			const auto& states = Catalog().NativeXprState().states;
			if (states.find(cache.state_key) == states.end())
				return StoreSinDraw::g_no_state.fetch_add(1, std::memory_order_relaxed), 0;
		}
		if (cap_used >= StoreSinDraw::CapUs()) return StoreSinDraw::g_capped.fetch_add(1, std::memory_order_relaxed), 2;
		struct Done {
			int64_t  begin;
			int64_t& used;
			~Done() {
				const auto us = (Tiron1::NowNs() - begin) / 1000;
				StoreSinDraw::g_done.fetch_add(1, std::memory_order_relaxed);
				if (us > 0 && us <= 1000000) {
					used += us;
					StoreSinDraw::g_done_us.fetch_add(static_cast<uint64_t>(us), std::memory_order_relaxed);
				}
			}
		} done {Tiron1::NowNs(), cap_used};
		auto&       pipelines = m_context.GetPipelineCache();
		const auto& sh        = buffer.GetShaders();
		const auto& ctx       = buffer.GetRegisters();
		auto&       st        = seed->state;
		std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
		for (uint32_t i = 0; i < st.color_count; i++) mapping[st.color_info[i].target_slot] = st.color_info[i].export_mapping;
		st.programs = pipelines.GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx, buffer.GetUserConfig(),
		                                            mapping, st.ps_active, st.vs_input_info, st.ps_input_info, true, pair.portable);
		if (pipelines.TablePending()) {
			pair.waiting = pipelines.TableWaiting(); // (the pair stores again once it is done)
			return 1;
		}
		const auto* vertex = st.vs_input_info.stage.program;
		const auto* pixel  = st.ps_active ? st.ps_input_info.stage.program : nullptr;
		const auto  refuse = [&](const char* reason) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
				std::printf("Table: VS 0x%llx PS 0x%llx refused: %s (store-sin-draw)\n", static_cast<unsigned long long>(pair.vs),
				            static_cast<unsigned long long>(pair.ps), reason);
			const auto write = CatalogWrite();
			pair.refused     = true;
			return 1;
		};
		if (!st.programs.vertex || vertex == nullptr || !vertex->table_mode) return refuse("no table mode vertex program");
		if (vertex->stage != ShaderType::Vertex) return refuse("not a vertex program");
		if (st.ps_active && (!st.programs.pixel || pixel == nullptr || !pixel->table_mode))
			return refuse("no table mode pixel program");
		const std::array<const ShaderRecompiler::IR::CompiledShaderInfo*, 2> programs {vertex, pixel};
		if (pair.programs[0] == nullptr) {
			const auto write = CatalogWrite();
			pair.programs    = programs;
			TableRuns(pair);
		} else if (pair.programs != programs) {
			const auto write = CatalogWrite();
			const auto found = std::ranges::find(pair.variants, cache.state_key, &TableXpr::Variant::state);
			if (found == pair.variants.end()) pair.variants.push_back({.state = cache.state_key, .refused = true});
			return 1;
		}
		const auto* pipeline = pipelines.TryCreateGraphicsPipeline(
		    std::span {st.color_info, st.color_count}, st.depth_info, st.vs_input_info, buffer,
		    st.ps_active ? &st.ps_input_info : nullptr, seed->topology, seed->restart, st.programs.vertex, st.programs.pixel, true);
		if (pipeline == nullptr) pair.waiting = pipelines.PipelineWaiting();
		{
			const auto write   = CatalogWrite();
			auto       variant = std::ranges::find(pair.variants, cache.state_key, &TableXpr::Variant::state);
			if (variant == pair.variants.end()) variant = pair.variants.insert(variant, {.state = cache.state_key});
			variant->pipeline       = pipeline;
			variant->mapping        = mapping;
			variant->fetch_embedded = st.vs_input_info.fetch_embedded;
		}
		if (pixel != nullptr && FrameGen::RtWaterIsWaterPs(pixel->shader_hash) && st.color_count >= 1)
			FrameGen::RtWaterNoteState(vertex->shader_hash, pixel->shader_hash, st.color_count,
			                           st.depth_info.image_id ? st.depth_info.desc.info.extent.width : 0,
			                           st.depth_info.depth_write_enable, static_cast<int>(seed->topology));
		return 1;
	};
	// (KYTY_STORE_SIN_DRAW) The 5 s line: stores without a draw against the normal ones (tiron1's counters: their cost
	// less the store body is the normal-path draw), in us a frame of the render thread before and after.
	const auto seed_line = [&] {
		static int64_t  last_ns    = 0;
		static uint64_t last_frame = 0;
		static uint64_t last[11]   = {};
		const auto      now        = Tiron1::NowNs();
		if (last_ns == 0) {
			last_ns    = now;
			last_frame = frame;
			return;
		}
		if (now - last_ns < 5000000000LL) return;
		const uint64_t cur[11] = {StoreSinDraw::g_done.load(),    StoreSinDraw::g_done_us.load(),
		                          StoreSinDraw::g_no_seed.load(), StoreSinDraw::g_no_state.load(),
		                          Tiron1::g_stores.load(),        Tiron1::g_store_us.load(),
		                          Tiron1::g_store_body_us.load(), StoreSinDraw::g_miss_new.load(),
		                          StoreSinDraw::g_miss_pair_other_state.load(), StoreSinDraw::g_miss_state_other_pair.load(),
		                          StoreSinDraw::g_capped.load()};
		uint64_t d[11];
		for (int i = 0; i < 11; ++i) d[i] = cur[i] - last[i], last[i] = cur[i];
		const double frames  = static_cast<double>(std::max<uint64_t>(frame - last_frame, 1));
		const double normal  = static_cast<double>(d[4]);
		const double each_us = normal > 0 && d[5] > d[6] ? static_cast<double>(d[5] - d[6]) / normal : 0.0;
		const double now_us  = (static_cast<double>(d[5]) + static_cast<double>(d[1])) / frames;
		const double was_us  = (static_cast<double>(d[5]) + static_cast<double>(d[0]) * each_us) / frames;
		std::printf("[store-sin-draw] 5 s: %.0f fotogramas | stores sin draw %llu (%.2f/fotograma, %.1f us/fotograma) | "
		            "stores por la ruta normal %llu (%.2f/fotograma, draw normal %.0f us cada uno) | caidos a la ruta normal: "
		            "sin semilla %llu (par nunca dibujado %llu, par solo en otros estados %llu, estado solo con otros pares %llu), "
		            "sin estado %llu | esperan por tope de tiempo %llu | hilo de render en stores: antes %.1f us/fotograma, ahora %.1f "
		            "(ahorro %.1f us/fotograma)%s\n",
		            frames, static_cast<unsigned long long>(d[0]), static_cast<double>(d[0]) / frames,
		            static_cast<double>(d[1]) / frames, static_cast<unsigned long long>(d[4]), normal / frames, each_us,
		            static_cast<unsigned long long>(d[2]), static_cast<unsigned long long>(d[7]), static_cast<unsigned long long>(d[8]),
		            static_cast<unsigned long long>(d[9]), static_cast<unsigned long long>(d[3]), static_cast<unsigned long long>(d[10]),
		            was_us, now_us, was_us - now_us,
		            Tiron1::On() ? "" : " (sin KYTY_TIRON1=1 no hay coste de la ruta normal)");
		std::fflush(stdout);
		last_ns    = now;
		last_frame = frame;
	};
	const auto request_store = [&] {
		if (Spec::Current() != nullptr) return Spec::Refuse("table store"), TableResult::Native;
		if (pair.waiting) {
			if (!pair.waiting->load(std::memory_order_acquire)) {
				if (tiron1) return tiron1_wait();
				return LiveCounters::Add(LiveCounters::TableStoreDeferred), TableResult::Deferred; // (programs compiling)
			}
			pair.waiting.reset();
		}
		if (table.store_frame != frame) {
			table.store_frame     = frame;
			table.store_requests  = 0;
			table.tiron1_store_us = 0;
		}
		if (pair.store_frame == frame && pair.store_state == cache.state_key) {
			if (tiron1) return tiron1_wait();
			return LiveCounters::Add(LiveCounters::TableStoreDeferred), TableResult::Deferred;
		}
		// (KYTY_STORE_SIN_DRAW) Made here from the seed: the draw takes its native record (no normal-path draw).
		if (StoreSinDraw::On() && Role() == 0 && !HwScene::Active()) {
			seed_line();
			const int seeded = seed_store();
			if (seeded == 1) {
				pair.store_frame          = frame;
				pair.store_state          = cache.state_key;
				Tiron1::t_no_native_store = true;
				return TableResult::Native;
			}
			if (seeded == 2) {
				if (tiron1) return tiron1_wait();
				return LiveCounters::Add(LiveCounters::TableStoreDeferred), TableResult::Native;
			}
		}
		// (Past the budget, the others draw from native records meanwhile.) (tiron1: past it, more while the stores
		// granted this frame took less than KYTY_TIRON1_STORE_MS of the render thread.)
		bool by_time = false;
		if (const auto budget = kyty_local_table_store_budget.load(std::memory_order_relaxed);
		    budget != 0 && table.store_requests >= budget) {
			if (!tiron1 || table.tiron1_store_us >= Tiron1::StoreBudgetUs()) {
				if (tiron1) return tiron1_wait();
				return LiveCounters::Add(LiveCounters::TableStoreDeferred), TableResult::Native;
			}
			by_time = true;
		}
		if (tiron1) {
			table.tiron1_grant_ns = Tiron1::NowNs();
			Tiron1::g_stores.fetch_add(1, std::memory_order_relaxed);
			if (by_time) Tiron1::g_stores_by_time.fetch_add(1, std::memory_order_relaxed);
		}
		++table.store_requests;
		pair.store_frame  = frame;
		pair.store_state  = cache.state_key;
		table.store       = &pair;
		table.store_state = cache.state_key;
		return TableResult::Store;
	};
	// (The variant found last first: a pair's variants are a cold line each, its draws mostly keep their state.)
	const bool hint    = (kyty_local_table_mru.load(std::memory_order_relaxed) & 2u) != 0 &&
	                  pair.variant_hint < pair.variants.size() && pair.variants[pair.variant_hint].state == cache.state_key;
	const auto variant = hint ? pair.variants.begin() + pair.variant_hint
	                          : std::ranges::find(pair.variants, cache.state_key, &TableXpr::Variant::state);
	if (variant != pair.variants.end()) pair.variant_hint = static_cast<uint32_t>(variant - pair.variants.begin());
	if (variant != pair.variants.end() && variant->refused) return Spec::Refuse("table: variant refused"), TableResult::Native;
	if (pair.programs[0] == nullptr || variant == pair.variants.end() || variant->pipeline == nullptr)
		return LiveCounters::Add(LiveCounters::TableStoreVariant), request_store();
	// The draw state (TableStore captures it).
	if (cache.last_state == nullptr || cache.last_state_key != cache.state_key) {
		const auto& states   = Catalog().NativeXprState().states;
		const auto  state_it = states.find(cache.state_key);
		if (state_it == states.end()) return request_store();
		cache.last_state_key = cache.state_key;
		cache.last_state     = state_it->second.get();
	}
	const auto& draw_state = *cache.last_state;
	TablePrefetchArrays(pair, pair.last_set[Role()]);
	if (!NativeXprTargetsCurrent(draw_state)) return LiveCounters::Add(LiveCounters::TableStoreTargets), request_store();
	stat_lookup.reset();
	return TableDraw(buffer, pair, *variant, draw_state, commands, index_base, index_bytes, index_type, direct);
}

bool RenderExecutor::TableContinue(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
                                   uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	if (kyty_local_table_xpr_mode.load(std::memory_order_relaxed) < 2 || !m_table_xpr || !m_native_xpr) return false;
	if (LiveCensus::Mdraw::Active() && Role() == 0) LiveCensus::Mdraw::t_entry = __rdtsc(); // (mdraw C0: the call's clock)
	std::shared_lock<std::shared_mutex> catalog_read;
	if (Role() != 0) catalog_read = CatalogRead();
	DropStaleCatalog();
	auto& table = OwnTable();
	if (table.last_pair == nullptr || buffer.IsInvalid() || index_base == 0 || index_bytes == 0 ||
	    commands.size() > 64 || table.last_pair->generation[Role()] != ShaderMapGeneration() ||
	    table.last_pair->start_instance != buffer.GetShaders().GetVs().start_instance_user_sgpr)
		return false;
	auto&       variant = table.last_pair->variants[table.last_variant];
	const auto& emit    = m_native_xpr->emit;
	if (emit.command != buffer.Handle() || emit.generation != buffer.GraphicsGeneration() ||
	    emit.state != table.last_state ||
	    (emit.pipeline != variant.pipeline && (variant.hwrt_pipeline == nullptr || emit.pipeline != variant.hwrt_pipeline)))
		return false;
	// (The GPU thread's: a speculation's thread does not take it.)
	std::optional<RenderLockGuard> lock;
	if (Role() == 0) lock.emplace(m_context.GetMutex());
	if (!NativeXprTargetsCurrent(*table.last_state)) return false;
	if (TableDraw(buffer, *table.last_pair, variant, *table.last_state, commands, index_base, index_bytes, index_type,
	              direct) != TableResult::Drawn)
		return false;
	LiveCounters::Add(LiveCounters::TableContinued, std::max<size_t>(commands.size(), 1));
	// (thread1, KYTY_RENDER_STATS: the calls continued, runs and direct draws apart.)
	if (LiveCensus::g_render && LiveCensus::RenderStatsOn()) LiveCensus::g_continued_calls[direct != nullptr ? 1 : 0] += 1;
	return true;
}

// The image set of the T#/S# words (`words`) of a draw (under `variant`) or a dispatch (`variant` null): its
// resources as the normal path evaluates them, which must select the pair's table mode programs (the same
// specializations), and its images and samplers as the normal path resolves and acquires them (PrepareBindingsInto,
// RebindImages), in a descriptor set with the page table and the fault buffer. Refused for another specialization,
// or when an image cannot be held in a set (a target the shader cannot read, or writes as storage).
RenderExecutor::TableImageSet* RenderExecutor::TableResolveSet(CommandBuffer& buffer, TablePair& pair,
                                                               const TableVariant*       variant,
                                                               const NativeXprDrawState* draw_state,
                                                               std::span<const uint32_t> words) {
	if (Spec::Current() != nullptr) return Spec::Refuse("table image set"), nullptr;
	auto       set      = std::make_unique<TableImageSet>();
	auto&      textures = m_context.GetTextureCache();
	// (Before the lookups: a registration they make over a depth image's pages is one IsSampledDepthCurrent sees.)
	const auto epoch    = textures.ResolutionEpoch();
	auto&      buffers  = m_context.GetBufferCache();
	const auto stages   = pair.programs[1] != nullptr ? 2u : 1u;
	const auto& sh      = buffer.GetShaders();
	const auto& ctx     = buffer.GetRegisters();
	set->words.assign(words.begin(), words.end());
	set->validated_frame[0] = m_context.FrameNumber();
	const auto publish = [&] {
		const auto write = CatalogWrite();
		auto&      slot  = pair.sets[XXH3_64bits(words.data(), words.size() * sizeof(uint32_t))];
		if (slot) {
			NativeXprRetireSet(slot->pool, slot->set);
			pair.last_set.fill(nullptr);
			pair.previous_set.fill(nullptr);
			++m_table_xpr->epoch;
		}
		slot = std::move(set);
		return slot.get();
	};
	const auto refuse = [&](const char* reason) {
		static std::atomic<uint32_t> reports[2] {}; // (dispatches, draws)
		if (reports[pair.ps != 0 ? 1 : 0].fetch_add(1, std::memory_order_relaxed) < 32)
			std::printf("Table: VS 0x%llx PS 0x%llx image set refused: %s\n", static_cast<unsigned long long>(pair.vs),
			            static_cast<unsigned long long>(pair.ps), reason);
		set->refused = true;
		return publish();
	};
	ShaderVertexInputInfo  vertex_info;
	ShaderPixelInputInfo   pixel_info;
	ShaderComputeInputInfo compute_info;
	std::array<const ShaderStageRuntime*, 2> runtimes {&vertex_info.stage, &pixel_info.stage};
	if (variant != nullptr) {
		ResetNativeVertexInput(vertex_info);
		ResetNativeStageInput(pixel_info);
		(void)m_context.GetPipelineCache().GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx,
		                                                       buffer.GetUserConfig(), variant->mapping, stages == 2,
		                                                       vertex_info, pixel_info, true, pair.portable);
		if (m_context.GetPipelineCache().TablePending()) return nullptr;
		if (vertex_info.stage.program != pair.programs[0] || (stages == 2 && pixel_info.stage.program != pair.programs[1]))
			return refuse(vertex_info.stage.program != pair.programs[0] ? "another specialization (VS)"
			                                                            : "another specialization (PS)");
	} else {
		ResetNativeStageInput(compute_info);
		compute_info.dispatch_thread_dimensions = pair.compute->dispatch_thread_dimensions;
		auto&      pipelines = m_context.GetPipelineCache();
		const auto handle    = pipelines.GetComputeProgram(sh.GetCs(), ctx.GetShaderRegisters(), compute_info, nullptr, true,
		                                                   pair.portable[0]);
		if (pipelines.TablePending()) return nullptr;
		if (compute_info.stage.program != pair.programs[0]) {
			// Another specialization the table way alike (a particle simulation's texture 2D in one emitter and 3D in
			// another, at Latria 90 programs): the set is made for it, with its pipeline.
			const auto* other = compute_info.stage.program;
			if (!handle || other == nullptr || !TableInterchangeable(*pair.programs[0], *other))
				return refuse("another specialization");
			set->pipeline = pipelines.TryCreateComputePipeline(compute_info, handle);
			if (set->pipeline == nullptr) {
				pair.waiting = pipelines.PipelineWaiting(); // (asked again once it is done)
				return nullptr;
			}
			set->program = other;
		}
		runtimes[0] = &compute_info.stage;
	}
	std::array<PreparedBindings, 2>                prepared;
	for (uint32_t st = 0; st < stages; ++st) PrepareBindingsInto(*runtimes[st], prepared[st]);
	for (uint32_t st = 0; st < stages; ++st) RebindImages(prepared[st]);
	// An address probe's indirect image (a dispatch's): its root's elements in the order of the candidates the set's
	// key holds (TableEvaluate's), each found among the dispatch's own (the normal path's materialization: the same
	// enumeration, in the order the records name them); past them the null images it was padded with.
	if (variant == nullptr) {
		const auto& program = set->program != nullptr ? *set->program : *pair.programs[0];
		for (const auto& entry: program.table_plan.indirect) {
			const auto& elements = program.info.images.at(entry.root).indirect_resources;
			const auto& own      = compute_info.stage.resources.images;
			const auto& wanted   = OwnTable().indirect[0];
			auto&       images   = prepared[0].images;
			if (wanted.size() > elements.size()) return refuse("indirect candidates differ");
			const auto original = images;
			for (size_t j = 0; j < wanted.size(); ++j) {
				const auto found = std::ranges::find_if(elements, [&](uint32_t resource) {
					return resource < own.size() && own[resource] == wanted[j];
				});
				if (found == elements.end() || elements[j] >= images.size() || *found >= original.size())
					return refuse("indirect candidates differ");
				images[elements[j]] = original[*found];
			}
		}
	}
	// Each image's layout and access as the normal path's CommitBindings gives them: a null image (a texture not
	// streamed in yet) as it is bound (not held); a target of the draw state in its attachment layout and access
	// (NativeXprEmitTargets moves it there); one bound both as a texture and as storage (force_general: the deferred
	// decals' G-buffer) wholly in General; storage in General; else read-only.
	std::array<std::vector<TableImageSet::Image>, 2> deps;
	bool                                             ok = true;
	for (uint32_t st = 0; st < stages; ++st) deps[st].reserve(prepared[st].images.size()); // (no regrowth: new sets come in bursts)
	for (uint32_t st = 0; st < stages && ok; ++st)
		for (auto& binding: prepared[st].images) {
			auto&       image   = textures.GetImage(binding.image_id);
			const bool  storage = binding.desc.type == TextureCache::BindingType::Storage;
			const auto  found   = draw_state == nullptr ? decltype(draw_state->targets.end()) {}
			                                                : std::ranges::find(draw_state->targets, binding.image_id,
			                                                                    &NativeXprDrawState::Target::id);
			const auto* target  = draw_state == nullptr || found == draw_state->targets.end() ? nullptr : &*found;
			const auto& view    = binding.desc.view_info;
			auto&       dep     = deps[st].emplace_back();
			dep                 = {binding.image_id, textures.m_slot_images[binding.image_id].serial,
			                       vk::ImageLayout::eUndefined, {view.base_level, view.level_count, view.base_layer,
			                                                     view.layer_count}, binding.desc};
			if (image.info.data.Empty() && !storage && image.backing.state.layout == vk::ImageLayout::eGeneral) {
				binding.layout = vk::ImageLayout::eGeneral;
				continue; // (dep.layout undefined: not held)
			}
			// (A sampled aspect the attachment writes is not shader-readable: the target's access then lacks it. A
			// linear image the shader writes is enrolled for readback at every acquisition: TrackImageDownload.)
			if (image.info.data.Empty() ||
			    (target != nullptr && (storage || !(target->access & vk::AccessFlagBits2::eShaderRead))) ||
			    ((storage || (image.binding.force_general && image.binding.shader_write)) && !image.info.IsTiled())) {
				ok = false;
				break;
			}
			if (target != nullptr) {
				binding.layout = target->layout;
				dep.access     = target->access;
				dep.target     = true;
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				binding.layout = vk::ImageLayout::eGeneral;
				dep.access     = vk::AccessFlagBits2::eShaderRead |
				                 (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlags2 {});
				dep.storage    = image.binding.shader_write;
				dep.whole      = true;
			} else if (storage) {
				binding.layout = vk::ImageLayout::eGeneral;
				dep.access     = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
				dep.storage    = true;
			} else {
				binding.layout = image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                      : vk::ImageLayout::eShaderReadOnlyOptimal;
			}
			dep.layout = binding.layout;
		}
	ResetBindings();
	if (!ok) return refuse("an image no set can hold");
	// The set: per stage its bindings (images, samplers, LOD statistics, page table, fault buffer).
	struct Pending {
		vk::WriteDescriptorSet write;
		size_t                 start = 0;
		bool                   image = false;
	};
	std::vector<Pending>                  pending;
	std::vector<vk::DescriptorBufferInfo> buffer_infos;
	std::vector<vk::DescriptorImageInfo>  image_infos;
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	const auto fixed = [](const Buffer* value) -> vk::DescriptorBufferInfo { return {value->Handle(), 0, value->Size()}; };
	set->bda_generation = BufferCache::BdaTableGeneration(); // (read before the table: BufferCache::GrowBdaTable)
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = set->program != nullptr ? *set->program : *pair.programs[st];
		std::vector<uint32_t> occurrences(prepared[st].images.size(), 0);
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
				for (const auto resource: binding.resources)
					image_infos.emplace_back(prepared[st].samplers.at(resource), nullptr, vk::ImageLayout::eUndefined);
			} else if (item.image) {
				for (const auto resource: binding.resources)
					image_infos.push_back(MakeImageInfo(prepared[st].images.at(resource), occurrences.at(resource)++));
			} else if (binding.kind == Kind::LodStats) {
				buffer_infos.push_back(fixed(buffers.GetLodStatsBuffer()));
			} else if (binding.kind == Kind::BdaPagetable) {
				buffer_infos.push_back(fixed(buffers.GetBdaPageTableBuffer()));
			} else if (binding.kind == Kind::FaultBuffer) {
				buffer_infos.push_back(fixed(buffers.GetFaultBuffer()));
			} else {
				return refuse("a binding table mode does not have");
			}
			pending.push_back(item);
		}
		if (program.stage == ShaderType::Pixel && HwScene::MaterialActive()) {
			set->hwrt_ps_images.clear();
			for (const auto& binding: program.bindings.descriptors) {
				if (ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) != ShaderRecompiler::IR::ImageResourceClass::Sampled) continue;
				for (uint32_t j = 0; j < binding.resources.size(); ++j) {
					const auto& texture = prepared[st].images.at(binding.resources[j]);
					auto*       image   = textures.TryGetImage(texture.image_id);
					if (image == nullptr || texture.image_view == nullptr) continue;
					HwScene::PsImage ps;
					ps.binding          = ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind);
					ps.element          = j;
					ps.image_index      = texture.image_id.index;
					ps.image_generation = texture.image_id.generation;
					ps.serial           = image->serial;
					ps.view             = texture.image_view;
					ps.format           = texture.desc.view_info.format;
					ps.type             = texture.desc.view_info.type;
					set->hwrt_ps_images.push_back(ps);
				}
			}
		}
		if (st == 0) set->images.reserve(deps[0].size() + deps[1].size());
		for (const auto& dep: deps[st])
			if (dep.layout != vk::ImageLayout::eUndefined) set->images.push_back(dep);
	}
	set->set = NativeXprAllocateSet((variant != nullptr      ? variant->pipeline
	                                 : set->pipeline != nullptr ? set->pipeline
	                                                            : pair.pipeline)
	                                    ->descriptor_set_layout,
	                                set->pool);
	std::vector<vk::WriteDescriptorSet> writes;
	for (auto& item: pending) {
		item.write.dstSet = set->set;
		if (item.image) item.write.pImageInfo = image_infos.data() + item.start;
		else item.write.pBufferInfo = buffer_infos.data() + item.start;
		writes.push_back(item.write);
	}
	// Written now, not queued in order with the recorded commands (the recording stream's descriptor writes): a
	// speculation's thread binds the set once it is published, which was before the stream wrote it. No recorded
	// command names the set: a new one, or one retired at a tick completed since.
	if (!writes.empty())
		m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		                                                    , LocalVulkanRecording::DirectDispatch()
#endif
		);
	set->resolution_epoch[0] = epoch;
	LiveCounters::Add(LiveCounters::TableEvaluations);
	return publish();
}

// What FindTexture uploads before an image is used with `desc`: guest bytes the CPU wrote, a GPU buffer write over them,
// or one over its stencil plane (of a sampled view, only into its levels: TextureCache::RefreshPending).
static bool TableRefreshPending(const Image& image, const TextureCache::ImageDesc& desc) {
	return TextureCache::RefreshPending(image, desc);
}

// A set's images are the same objects, registered and current (as NativeXprValidateResources proves a record's):
// once a frame, and again when registrations moved. The main executor checks a use it proved again only when its
// image's validity moved or a registration stamped the view's pages since (1-1: ~1500 sets a frame, ~0.5 ms, nearly
// none changed; the registrations, every frame, one depth surface's memory taken as a color target and back), and
// every use in full once in TouchTicks flips or frames, which touches it (KYTY_TABLE_SET_PROOFS: 0 checks all of
// them in full, 2 does and counts the uses a proof held for that did not pass).
bool RenderExecutor::TableValidateSet(TableImageSet& set) {
	auto&      textures   = m_context.GetTextureCache();
	const auto frame      = m_context.FrameNumber();
	const auto resolution = textures.ResolutionEpoch();
	if (set.bda_generation != BufferCache::BdaTableGeneration()) return false; // (the page table grew since)
	if (set.validated_frame[Role()] == frame && set.resolution_epoch[Role()] == resolution) return true;
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1e, 7);
	const auto mode  = kyty_local_table_set_proofs.load(std::memory_order_relaxed);
	const bool own   = Role() == 0 && Spec::Current() == nullptr && mode != 0;
	const bool full  = !own || textures.GcTick() >= set.touched_tick + TableImageSet::TouchTicks ||
	                  textures.CurrentFrame() >= set.touched_frame + TableImageSet::TouchTicks ||
	                  m_context.GetCommandScheduler().Current().IsInvalid();
	G1::NoteSetCheck(full); // (g1 NP1, KYTY_G1_PROBE: one relaxed load when off)
	// Registrations since: their pages say which proofs they reach (none, when the log of the latest has none over the
	// views' span: the game moves registrations every frame somewhere).
	bool moved = set.resolution_epoch[Role()] != resolution;
	if (moved && own) {
		if (set.span_end == 0) {
			set.span_begin = UINT64_MAX;
			for (const auto& dep: set.images) {
				set.span_begin = std::min(set.span_begin, dep.desc.info.data.address);
				set.span_end   = std::max(set.span_end, dep.desc.info.data.End());
			}
		}
		bool known = false;
		moved      = set.span_end <= set.span_begin ||
		        textures.RegistrationsOverlapSince(set.span_begin, set.span_end - set.span_begin, set.resolution_epoch[0], known) ||
		        !known;
	}
	if (!full)
		for (const auto& dep: set.images) textures.m_slot_images.prefetch(dep.id);
	for (auto& dep: set.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || image->serial != dep.serial) return false;
		const bool held = !full && dep.proven && image->validity == dep.validity &&
		                  (!moved || !textures.RegistrationsSince(dep.desc.info.data.address, dep.desc.info.data.size,
		                                                          set.resolution_epoch[0]));
		G1::NoteSetUse(held); // (g1 NP1)
		if (held && mode != 2) continue;
		const auto refuse = [&] {
			if (held) {
				LiveCounters::Add(LiveCounters::TableProofMismatches);
				static std::atomic<uint32_t> reports {0};
				if (reports.fetch_add(1, std::memory_order_relaxed) < 16)
					std::printf("Table: proven image use failed: addr=0x%llx size=0x%llx serial=%llu validity=%llu\n",
					            static_cast<unsigned long long>(dep.desc.info.data.address),
					            static_cast<unsigned long long>(dep.desc.info.data.size),
					            static_cast<unsigned long long>(dep.serial), static_cast<unsigned long long>(dep.validity));
			}
			return false;
		};
		// The guest rewrote it (a texture streaming in), or a GPU buffer write its bytes or its stencil plane: upload it
		// as the normal path's FindTexture does.
		if (TableRefreshPending(*image, dep.desc)) {
			G1::NoteSetRefresh(); // (g1 NP1)
			if (held) (void)refuse();
			if (Spec::Current() != nullptr) return Spec::Refuse("texture upload"), false;
			if (!image->registered || image->depth_id || image->binding.needs_rebind) return false;
			(void)textures.FindTexture(dep.id, dep.desc);
			image = textures.m_slot_images.try_get(dep.id);
			if (image == nullptr || image->serial != dep.serial || TableRefreshPending(*image, dep.desc)) return false;
		}
		if (dep.target) {
			if (!image->registered || image->binding.needs_rebind) return refuse();
			(void)textures.GetImage(dep.id); // LRU
			if (own) {
				dep.proven   = true;
				dep.validity = image->validity;
			}
			continue;
		}
		// (And its LRU, as FindTexture's. A DCC image written as storage, a downsampling chain's levels: TableTransit
		// checks its pending clears at every use, as a sampled one's.) A depth image sampled as a texture (the scene's
		// depth and stencil read by lighting and post passes: at 1-1 their sets were made again every other frame) is
		// found by other rules than a texture's: the proof is that nothing was registered over its view since the set
		// was last proven (or made).
		bool sampled = textures.IsSampledImageCurrent(dep.id, dep.desc, true);
		bool current = sampled || textures.IsSampledDepthCurrent(dep.id, dep.desc, set.resolution_epoch[Role()]);
		const bool texture = dep.desc.type == TextureCache::BindingType::Texture;
		if (!current && own && (texture || dep.desc.type == TextureCache::BindingType::Storage) &&
		    !m_context.GetCommandScheduler().Current().IsInvalid()) {
			// A surface the game uses as depth and as colour in turn (0x28a010000 at 1-1 and Latria, twice a frame): the
			// form a use leaves is parked (TextureCache::ParkImage), registrations over it move every frame, and the sets
			// using it failed here at every use (made again, 30-70 us each, the dispatch on the normal path meanwhile). The
			// normal path's lookup takes the set's image back (TakeParkedImage: the same id and serial, its views kept, the
			// other form's bytes copied in): the set then binds what that lookup and its acquisition bind. (The lookup is
			// the proof for a depth-tiled view too, which IsSampledImageCurrent declines: the surface's storage use.)
			// (A view the set's lookup moved into a larger image, its first layer in that image (on AMD GPUs a texture in the
			// layers a texture pool's views hold), is no request of its own: looked up again it ended the emulator (issue
			// #102, an RX 9070 XT). Its image is the one the set holds, and the checks below are its proof.)
			auto       desc  = dep.desc;
			const bool taken = !TextureCache::ViewFitsImage(desc) ||
			                   (textures.FindImage(desc) == dep.id && desc.view_info == dep.desc.view_info);
			image            = textures.m_slot_images.try_get(dep.id); // (the lookup may have moved the slots)
			if (taken && image != nullptr && image->serial == dep.serial && image->registered && !image->depth_id &&
			    !image->binding.needs_rebind) {
				if (texture && (dep.desc.info.IsDepth() || image->info.IsDepth())) {
					current = textures.IsSampledDepthCurrent(dep.id, dep.desc, textures.ResolutionEpoch());
				} else {
					(void)textures.FindTexture(dep.id, dep.desc);
					sampled = textures.IsSampledImageCurrent(dep.id, dep.desc, true);
					current = sampled || (!dep.desc.info.HasMetadata() && !dep.desc.info.HasStencil());
				}
			}
		}
		if (!current) return refuse();
		// (A DCC image's surface and a depth image's view are proven otherwise: checked at every validation.)
		if (own) {
			dep.proven   = sampled && dep.desc.info.metadata.kind != ImageMetadataKind::Dcc;
			dep.validity = image->validity;
		}
	}
	// (The epoch before the checks: a registration they made, FindTexture's, stamps pages a later check looks at.)
	set.validated_frame[Role()]  = frame;
	set.resolution_epoch[Role()] = resolution;
	if (own && full) {
		set.touched_tick  = textures.GcTick();
		set.touched_frame = textures.CurrentFrame();
	}
	G1::NoteSetOk(); // (g1 NP1)
	return true;
}

// An operand of the stage's code address (a V# of constants the code embeds), and its word: the low or high dword of
// the pair's code address (the stage's registers named it) plus the operand's offset.
static bool TableCodeOperand(const ShaderRecompiler::IR::TablePlan::Operand& operand) {
	using Kind = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	return operand.kind == Kind::ShaderBaseLow || operand.kind == Kind::ShaderBaseHigh;
}
static uint32_t TableCodeWord(const ShaderRecompiler::IR::TablePlan::Operand& operand, uint64_t code) {
	const auto address = code + operand.value;
	return operand.kind == ShaderRecompiler::IR::TablePlan::Operand::Kind::ShaderBaseLow ? static_cast<uint32_t>(address)
	                                                                                    : static_cast<uint32_t>(address >> 32u);
}

// The slots of the pair's stages, read into their blocks where the plans address them from the user data and lower
// slots (a pointer read from memory may hold anything: what is read through it must be mapped), and the T#/S# words
// they make (TableXpr::words, the set's key) and the pixel stage's LOD statistics words. False: something unmapped.
namespace {
// KYTY_GPUREAD_PROBE: the table reads that waited (> 20 us: a synchronous download of GPU-written memory), per shader and
// run: how often, the time, whether the words changed since the run's previous read (served from the previous frame
// they would be right when they did not), the words before and now, and whether the CPU needs them (to resolve
// resources) or only the shader reads them. Every 10 s.
struct GpuReadProbe {
	uint64_t shader = 0;
	uint32_t stage = 0, run = 0, first = 0, count = 0, level = 0, cpu = 0, gpu = 0;
	uint64_t address = 0, hash = 0, reads = 0, waits = 0, wait_ns = 0, same = 0, changed = 0;
	std::array<uint32_t, 4> before {}, now {};
	bool     used = false;
};
std::array<GpuReadProbe, 1024>        g_gpu_read_probes {};
std::chrono::steady_clock::time_point g_gpu_read_start {};
void GpuReadProbePrint() {
	std::vector<const GpuReadProbe*> list;
	for (const auto& probe: g_gpu_read_probes)
		if (probe.used && probe.waits != 0) list.push_back(&probe);
	std::sort(list.begin(), list.end(), [](auto* a, auto* b) { return a->wait_ns > b->wait_ns; });
	std::printf("[gpuread] 10 s: %zu table runs waited\n", list.size());
	for (size_t i = 0; i < std::min<size_t>(list.size(), 10); ++i) {
		const auto& p = *list[i];
		std::printf("[gpuread]  0x%llx st %u run %u (slots %u+%u lvl %u cpu %u gpu %u) at 0x%llx: %llu reads, %llu waits "
		            "%.1f ms, %llu same %llu changed; before %08x %08x %08x %08x now %08x %08x %08x %08x\n",
		            static_cast<unsigned long long>(p.shader), p.stage, p.run, p.first, p.count, p.level, p.cpu, p.gpu,
		            static_cast<unsigned long long>(p.address), static_cast<unsigned long long>(p.reads),
		            static_cast<unsigned long long>(p.waits), p.wait_ns / 1e6, static_cast<unsigned long long>(p.same),
		            static_cast<unsigned long long>(p.changed), p.before[0], p.before[1], p.before[2], p.before[3], p.now[0],
		            p.now[1], p.now[2], p.now[3]);
	}
	std::fflush(stdout);
	g_gpu_read_probes = {};
}
} // namespace

bool RenderExecutor::TableEvaluate(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	auto&      table     = OwnTable();
	const auto stages    = pair.programs[1] != nullptr ? 2u : 1u;
	auto&      resources = m_context.GetGpuResources();
	if (const auto epoch = resources.MappingEpoch(); epoch != table.mapping_epoch) {
		table.mapped.fill(UINT64_MAX);
		table.mapping_epoch = epoch;
	}
	using Kind  = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto& words = table.words;
	words.clear();
	table.lod_stats.clear();
	NativeXprReader reader(resources);
	const bool probing = SyncStats::ReadProbe();
	// (KYTY_GPUREAD_PROBE) A run read through `reader`, noted when its memory is GPU-written.
	const auto read_run = [&](uint32_t st, uint32_t index, uint64_t address, uint32_t count, uint32_t first,
	                          uint32_t* out) {
		TrustedDepth::NoteRun(st, first, count, address); // (trust1: while an XPR depth draw captures its reads)
		if (!probing) {
			reader.Read(address, count, out);
			if (PortR::Verifying() && PortR::t_capture) PortR::NoteRun(st, address, count, out); // (port-r=2: the run, for TableDraw's comparison)
			return;
		}
		// (TSC cycles, not steady_clock: two clock reads per run were 20% of the render thread at the dragon's path.)
		const auto begin = __rdtsc();
		reader.Read(address, count, out);
		if (PortR::Verifying() && PortR::t_capture) PortR::NoteRun(st, address, count, out); // (port-r=2)
		const auto cycles = __rdtsc() - begin;
		const auto ns     = cycles / 4u; // (~4 GHz TSC: a threshold and a total, not a measurement)
		const uint64_t key   = pair.vs * 31u + st * 7919u + index;
		auto&          probe = g_gpu_read_probes[(key * 0x9e3779b97f4a7c15ull) >> 54u];
		const bool     mine  = probe.used && probe.shader == pair.vs && probe.stage == st && probe.run == index;
		if (!mine && ns < 20000) return; // (only runs that waited once are followed)
		const uint64_t hash = XXH3_64bits(out, uint64_t {count} * 4u);
		std::array<uint32_t, 4> words {};
		std::copy_n(out, std::min<uint32_t>(count, 4u), words.begin());
		if (!mine) {
			const auto& plan = pair.programs[st]->table_plan;
			const auto& run  = pair.runs[st][index];
			probe = {.shader = pair.vs, .stage = st, .run = index, .first = first, .count = count, .level = run.level,
			         .address = address, .hash = hash, .now = words, .used = true};
			for (uint32_t slot = first; slot < first + count && slot < plan.cpu.size(); ++slot) {
				probe.cpu += plan.cpu[slot] != 0;
				probe.gpu += slot < plan.gpu.size() && plan.gpu[slot] != 0;
			}
		} else if (probe.hash == hash) {
			++probe.same;
		} else {
			++probe.changed;
			probe.hash   = hash;
			probe.before = probe.now;
			probe.now    = words;
		}
		++probe.reads;
		if (ns >= 20000) {
			++probe.waits;
			probe.wait_ns += ns;
		}
		if (ns >= 20000) {
			// (The clock only after a wait: rare.)
			const auto end = std::chrono::steady_clock::now();
			if (g_gpu_read_start == std::chrono::steady_clock::time_point {}) g_gpu_read_start = end;
			if (end - g_gpu_read_start >= std::chrono::seconds(10)) {
				GpuReadProbePrint();
				g_gpu_read_start = end;
			}
		}
	};
	// A run's address, its pages proven mapped and its lines loading.
	// (A bounded run, read through a V#: within its range, NUM_RECORDS bytes or NUM_RECORDS strides, as the normal
	// path's evaluation, which fails past it: so does this one.)
	const auto locate = [&](uint64_t low, uint64_t high, uint32_t offset, int32_t immediate, uint32_t count,
	                        uint64_t& out, bool bounded = false, uint32_t records = 0) {
		if (bounded) {
			const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
			const auto size   = stride == 0 ? uint64_t {records} : uint64_t {stride} * records;
			if (immediate < 0 || static_cast<uint64_t>(immediate) + count * 4ull > size) return false;
		}
		const auto address = ((low | high << 32u) & 0x0000fffffffffffcull) + (offset & ~3u) +
		                     static_cast<uint64_t>(int64_t {immediate});
		for (auto page = address >> 14u; page <= (address + count * 4ull - 1) >> 14u; ++page) {
			auto& seen = table.mapped[page & 255u];
			if (seen == page) continue;
			if (!Libs::LibKernel::Memory::IsFullyMapped(page << 14u, 16384)) return false;
			seen = page;
		}
		for (auto line = address & ~63ull; line < address + count * 4ull; line += 64)
			__builtin_prefetch(reinterpret_cast<const void*>(line));
		out = address;
		return true;
	};
	// The LOD statistics fields of the T#s (the game reassigns the counters every frame) reach only the shader data
	// (as RebindBuffers computes it), not the set.
	const auto lod_stats = [&](const std::array<uint32_t, 8>& image) {
		ShaderTextureResource desc;
		std::memcpy(&desc, image.data(), sizeof(desc));
		table.lod_stats.push_back(desc.MipStatsCntEn() ? (0x80000000u | desc.MipStatsCntId() |
		                                                  (uint32_t(desc.BaseLevel()) << 8u) |
		                                                  (uint32_t(desc.MinLodWarn5()) << 12u))
		                                               : 0u);
	};
	// A stage from its plan's operands: for user data the flat form cannot index (fewer words than it reads, or more
	// than 64: one beyond them reads 0 here, the flat form's constant 0 stands for those of 64 and up).
	const auto by_operands = [&](uint32_t st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		auto&       slots   = table.blocks[st];
		const auto  value   = [&](const ShaderRecompiler::IR::TablePlan::Operand& operand) -> uint32_t {
			if (operand.kind == Kind::Immediate) return operand.value;
			if (operand.kind == Kind::Slot) return slots[operand.value];
			if (TableCodeOperand(operand)) return TableCodeWord(operand, st == 0 ? pair.vs : pair.ps);
			const auto index = operand.value - program.user_data_base;
			return index < user_data[st].size() ? user_data[st][index] : 0;
		};
		const auto& runs = pair.runs[st];
		for (size_t begin = 0, end = 0; begin < runs.size(); begin = end) {
			for (end = begin; end < runs.size() && runs[end].level == runs[begin].level; ++end) {
				const auto& run = runs[end];
				if (!locate(value(run.read.low), value(run.read.high), value(run.read.offset), run.read.immediate, run.count,
				            table.addresses[end], run.read.bounded, run.read.bounded ? value(run.read.records) : 0))
					return false;
			}
			for (size_t i = begin; i < end; ++i)
				read_run(st, static_cast<uint32_t>(i), table.addresses[i], runs[i].count, runs[i].first, slots.data() + runs[i].first);
		}
		for (size_t i = 0; i < plan.images.size(); ++i) {
			const auto& descriptor = plan.images[i];
			std::array<uint32_t, 8> image {};
			for (uint32_t w = 0; w < descriptor.count; ++w) {
				image[w] = value(descriptor.words[w]);
				words.push_back(image[w] & ~NativeXprLodStatsMask[w]);
			}
			if (i < program.bindings.lod_stats_count) lod_stats(image);
		}
		for (const auto& descriptor: plan.samplers)
			for (uint32_t w = 0; w < descriptor.count; ++w) words.push_back(value(descriptor.words[w]));
		return true;
	};
	// An address probe's indirect image (IR::TablePlan::IndirectImage): its candidates as resource materialization
	// enumerates them (the same code and readers), in an order of their own (key 0's first, the rest by their words:
	// one set whatever order the records name them in), their count and words in the set's key, the key mapping (by
	// key) in the block. False: the normal path takes it (a table not readable, more candidates or keys than room).
	const auto indirect = [&](uint32_t st) {
		const auto& program = *pair.programs[st];
		auto&       out     = table.indirect[st];
		out.clear();
		if (program.table_plan.indirect.empty()) return true;
		thread_local const auto runtime = [] {
			ShaderRecompiler::IR::SrtRuntime value;
			MaterializationReaders(value);
			return value;
		}();
		const auto& slots   = table.blocks[st];
		const auto  operand = [&](const ShaderRecompiler::IR::TablePlan::Operand& value) -> uint32_t {
			if (value.kind == Kind::Immediate) return value.value;
			if (value.kind == Kind::Slot) return slots[value.value];
			if (TableCodeOperand(value)) return TableCodeWord(value, st == 0 ? pair.vs : pair.ps);
			const auto index = value.value - program.user_data_base;
			return index < user_data[st].size() ? user_data[st][index] : 0;
		};
		const auto address = [&](const ShaderRecompiler::IR::TablePlan::Operand& low,
		                         const ShaderRecompiler::IR::TablePlan::Operand& high) {
			return (uint64_t {operand(low)} | uint64_t {operand(high)} << 32u) & 0x0000ffffffffffffull;
		};
		using Plan = ShaderRecompiler::IR::TablePlan;
		for (const auto& entry: program.table_plan.indirect) {
			auto& found = table.probe;
			if (!ShaderRecompiler::IR::EnumerateAddressProbe(entry.probe, program.info.images[entry.root],
			                                                 address(entry.records_low, entry.records_high),
			                                                 address(entry.heap_low, entry.heap_high), runtime, found))
				return false;
			const auto count = static_cast<uint32_t>(found.descriptors.size());
			if (count == 0 || count > Plan::IndirectCapacity || found.keys.size() > Plan::IndirectMaxKeys ||
			    found.candidates.size() != found.keys.size())
				return false;
			std::array<uint32_t, Plan::IndirectCapacity> order {}, position {};
			for (uint32_t j = 0; j < count; ++j) order[j] = j;
			std::sort(order.begin() + 1, order.begin() + count,
			          [&](uint32_t a, uint32_t b) { return found.descriptors[a].dwords < found.descriptors[b].dwords; });
			for (uint32_t j = 0; j < count; ++j) position[order[j]] = j;
			for (uint32_t j = 0; j < count; ++j) out.push_back(found.descriptors[order[j]]);
			std::array<uint32_t, Plan::IndirectMaxKeys> by_key {};
			const auto keys = static_cast<uint32_t>(found.keys.size());
			for (uint32_t k = 0; k < keys; ++k) by_key[k] = k;
			std::sort(by_key.begin(), by_key.begin() + keys,
			          [&](uint32_t a, uint32_t b) { return found.keys[a] < found.keys[b]; });
			auto* mapping = table.blocks[st].data() + entry.mapping;
			mapping[0]    = keys;
			for (uint32_t k = 0; k < keys; ++k) {
				mapping[1 + 2 * k] = found.keys[by_key[k]];
				mapping[2 + 2 * k] = position[found.candidates[by_key[k]]];
			}
			words.push_back(count);
			for (const auto& candidate: out) words.insert(words.end(), candidate.dwords.begin(), candidate.dwords.end());
		}
		return true;
	};
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& form    = pair.packed[st];
		// (Slots no stage reads keep what an earlier draw left: the block carries them unread.)
		auto& slots = table.blocks[st];
		slots.resize(program.table_plan.BlockDwords());
		if (table.addresses.size() < pair.runs[st].size()) table.addresses.resize(pair.runs[st].size());
		if (!form.valid || form.runs.size() != pair.runs[st].size() || user_data[st].size() < form.user_words ||
		    user_data[st].size() > 64) {
			if (!by_operands(st) || !indirect(st)) return false;
			continue;
		}
		// The flat form (TablePair::packed): every operand one load from one of three arrays. A level's addresses are
		// known together: their lines load at once (the guest's tables are cold).
		const std::array<const uint32_t*, 3> bases {user_data[st].data(), slots.data(), form.constants.data()};
		const auto get = [&](uint16_t operand) { return bases[operand >> 14u][operand & 0x3fffu]; };
		uint32_t   begin = 0;
		for (const auto end: form.level_ends) {
			for (auto i = begin; i < end; ++i) {
				const auto& run     = form.runs[i];
				const bool  bounded = run.records != TablePair::PackedRun::NotBounded;
				if (!locate(get(run.address[0]), get(run.address[1]), get(run.address[2]), run.immediate, run.count,
				            table.addresses[i], bounded, bounded ? get(run.records) : 0))
					return false;
			}
			for (auto i = begin; i < end; ++i)
				read_run(st, i, table.addresses[i], form.runs[i].count, form.runs[i].first, slots.data() + form.runs[i].first);
			begin = end;
		}
		const auto first = words.size();
		words.resize(first + form.image_words.size());
		auto*  out = words.data() + first;
		size_t at  = 0;
		for (size_t i = 0; i < form.image_counts.size(); ++i) {
			std::array<uint32_t, 8> image {};
			const uint32_t          count = form.image_counts[i];
			for (uint32_t w = 0; w < count; ++w) {
				image[w]    = get(form.image_words[at + w]);
				out[at + w] = image[w] & ~NativeXprLodStatsMask[w];
			}
			at += count;
			if (i < program.bindings.lod_stats_count) lod_stats(image);
		}
		for (; at < form.image_words.size(); ++at) out[at] = get(form.image_words[at]);
		if (!indirect(st)) return false;
	}
	return true;
}

// The image set of the words TableEvaluate made: the last one when they are the same, else the pair's set for them
// (`variant` null: a dispatch's), proven current. Null: the normal path takes the draw or dispatch.
RenderExecutor::TableImageSet* RenderExecutor::TableSet(CommandBuffer& buffer, TablePair& pair,
                                                        const TableVariant*       variant,
                                                        const NativeXprDrawState* draw_state) {
	auto&       table = OwnTable();
	const auto& words = table.words;
	const auto  frame = m_context.FrameNumber();
	auto*       set   = pair.last_set[Role()];
	if (set == nullptr || set->words != words) {
		auto* const last     = set;
		auto* const previous = pair.previous_set[Role()];
		// (The set before the last next, without the words' hash or the map's cold bucket and node.)
		if ((kyty_local_table_mru.load(std::memory_order_relaxed) & 4u) != 0 && previous != nullptr && previous != last &&
		    !previous->refused && previous->words == words) {
			set = previous;
		} else {
			const auto found = pair.sets.find(XXH3_64bits(words.data(), words.size() * sizeof(uint32_t)));
			set              = found != pair.sets.end() && found->second->words == words ? found->second.get() : nullptr;
			if (set == nullptr || (set->refused && frame >= set->validated_frame[0] + TableImageSet::RetryFrames))
				set = TableResolveSet(buffer, pair, variant, draw_state, words);
		}
		// (TableResolveSet may have cleared both: a set it replaced.)
		if (last != nullptr && pair.last_set[Role()] == last) pair.previous_set[Role()] = last;
		pair.last_set[Role()] = set;
		if (set == nullptr) return nullptr; // (a program still translating)
	}
	set->used_frame.store(frame, std::memory_order_relaxed);
	if (set->refused) return LiveCounters::Add(LiveCounters::TableRefusedSets), Spec::Refuse("table: refused set"), nullptr;
	if (!TableValidateSet(*set)) {
		if (Spec::Current() != nullptr) return Spec::Refuse("table: set validation"), nullptr;
		// Resolved again at the next draw that uses it.
		const auto write = CatalogWrite();
		if (table.last_set == set) table.last_set = nullptr;
		pair.last_set.fill(nullptr);
		pair.previous_set.fill(nullptr);
		++table.epoch;
		const auto found = pair.sets.find(XXH3_64bits(set->words.data(), set->words.size() * sizeof(uint32_t)));
		NativeXprRetireSet(set->pool, set->set);
		if (found != pair.sets.end() && found->second.get() == set) pair.sets.erase(found);
		return nullptr;
	}
	// Unused image sets leave once a second (the catalog's owner drops them).
	if (Role() == 0 && frame >= table.collected + 60) {
		const auto write = CatalogWrite();
		table.collected  = frame;
		for (auto* pairs: {&table.pairs, &table.dispatches})
			for (auto& [key, held]: *pairs)
				std::erase_if(held.sets, [&](auto& item) {
					if (item.second->used_frame.load(std::memory_order_relaxed) + 60 >= frame) return false;
					held.last_set.fill(nullptr);
					held.previous_set.fill(nullptr);
					if (table.last_set == item.second.get()) table.last_set = nullptr;
					++table.epoch;
					NativeXprRetireSet(item.second->pool, item.second->set);
					return true;
				});
	}
	return set;
}

// The pair's blocks (TableEvaluate read their slots): each stage's buffers' device addresses and sizes in dwords
// after its slots, in the table upload ring. A read-only V# is resolved once (TableXpr::resolved); a written one
// is obtained as written at every use, as the normal path's NativeStorageBuffer (the buffer cache tracks the GPU's
// writes). False: a buffer the shaders cannot take.
// The V#s' words and their resolved entries (TableXpr::entries), as soon as TableEvaluate made the slots they are read
// from: the entries' lines (a 4 MiB direct-mapped table: TLB misses too) load while the image set is found (TableSet).
void RenderExecutor::TableGather(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	using Kind        = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto&      table  = OwnTable();
	const auto stages = pair.programs[1] != nullptr ? 2u : 1u;
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		const auto& block   = table.blocks[st];
		auto&       entries = table.entries[st];
		entries.resize(plan.buffers.size());
		const auto& form = pair.packed[st];
		const bool  flat = form.valid && form.buffer_words.size() == 4 * plan.buffers.size() &&
		                  user_data[st].size() >= form.user_words && user_data[st].size() <= 64;
		const std::array<const uint32_t*, 3> bases {user_data[st].data(), block.data(), form.constants.data()};
		const bool memo = kyty_local_table_gather_memo.load(std::memory_order_relaxed) != 0; // (KYTY_TABLE_GATHER_MEMO)
		uint32_t   memo_hits = 0;
		for (size_t i = 0; i < plan.buffers.size(); ++i) {
			auto& [words4, entry] = entries[i];
			std::array<uint32_t, 4> fresh;
			if (flat) {
				for (uint32_t w = 0; w < 4; ++w) {
					const auto operand = form.buffer_words[4 * i + w];
					fresh[w]           = bases[operand >> 14u][operand & 0x3fffu];
				}
			} else {
				for (uint32_t w = 0; w < 4; ++w) {
					const auto& operand = plan.buffers[i][w];
					fresh[w] = operand.kind == Kind::Immediate ? operand.value
					           : operand.kind == Kind::Slot    ? block[operand.value]
					           : TableCodeOperand(operand)     ? TableCodeWord(operand, st == 0 ? pair.vs : pair.ps)
					           : operand.value - program.user_data_base < user_data[st].size()
					               ? user_data[st][operand.value - program.user_data_base]
					               : 0;
				}
			}
			// (The same words as gathered last at this index hash to the same entry: kept, no hash; the prefetch stays, the
			// line is what TableBlocks reads.)
			if (memo && entry != nullptr && words4 == fresh) {
				++memo_hits;
			} else {
				words4 = fresh;
				entry  = &table.resolved[XXH3_64bits(words4.data(), sizeof(words4)) & (TableXpr::ResolvedEntries - 1)];
			}
			__builtin_prefetch(entry);
		}
		if (memo_hits != 0) LiveCounters::Add(LiveCounters::TableGatherMemoHits, memo_hits);
	}
	table.gathered = &pair;
}

// KYTY_TABLE_WATCH=<hash>:<sgpr>:<offset>[:<limit>],... (or ;): a diagnostic of those programs' table dispatches. The
// V# at the user data pointer s<sgpr>, s<sgpr+1> plus <offset>: its buffer's first dwords as guest memory holds them
// before the blocks are made, printed ("ODD") when dword 2 exceeds <limit> (default 32), and every 4096th use. When the
// block gives the program a copy in the table upload ring, the copy is compared with them ("RING MISMATCH") and again
// once the GPU ran the dispatch ("RING CHANGED": the ring was reused under it); when it gives a cached buffer, a copy of
// its bytes made on the GPU just before the dispatch is compared with them once it ran ("DEVICE MISMATCH": the buffer
// held other bytes than the guest's). (The culling programs' plane loops run dword 2 times: a garbage count hung the
// GPU, 10-13.)
// (bloques juntos / censo bloques) Counters of KYTY_BLOQUES_JUNTOS (ring windows) and KYTY_CENSO_BLOQUES (ring copies
// inside an earlier copy's window; the checks' and the copy's cycles), one '[censo-bloques]' line every 5 s.
namespace BloquesJuntos {
struct State {
	std::atomic<uint64_t> copies {0}, inside {0}, bytes {0}, inside_bytes {0}, total_cycles {0}, copy_cycles {0};
	std::atomic<uint64_t> taken {0}, windows {0}, window_bytes {0}, checked {0}, mismatches {0};
	std::atomic<int64_t>  since_ns {0};
	std::atomic<uint64_t> since_tsc {0};
};
inline State g;
inline void NoteMismatch(uint64_t address, uint64_t size) {
	if (g.mismatches.fetch_add(1, std::memory_order_relaxed) < 16)
		std::printf("[censo-bloques] DIFERENCIA ventana vs copia en 0x%llx (%llu bytes)\n",
		            static_cast<unsigned long long>(address), static_cast<unsigned long long>(size));
}
inline void MaybeReport() {
	const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch()).count();
	int64_t since = g.since_ns.load(std::memory_order_relaxed);
	if (since == 0) {
		g.since_ns.store(now, std::memory_order_relaxed);
		g.since_tsc.store(__rdtsc(), std::memory_order_relaxed);
		return;
	}
	if (now - since < 5000000000ll || !g.since_ns.compare_exchange_strong(since, now)) return;
	const uint64_t tsc  = __rdtsc();
	const double   nspc = static_cast<double>(now - since) /
	                    static_cast<double>(std::max<uint64_t>(1, tsc - g.since_tsc.exchange(tsc)));
	const auto  take   = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
	const auto  copies = take(g.copies), inside = take(g.inside), bytes = take(g.bytes), ib = take(g.inside_bytes);
	const auto  total = take(g.total_cycles), copy = take(g.copy_cycles);
	const auto  taken = take(g.taken), windows = take(g.windows), wbytes = take(g.window_bytes), checked = take(g.checked);
	const double n     = static_cast<double>(std::max<uint64_t>(1, copies));
	std::printf("[censo-bloques] 5 s: copias %llu (%.1f MB) | dentro de ventana previa %.1f%% (%.1f%% bytes) | por copia: "
	            "total %.0f ns = comprobaciones %.0f + Map/lectura/Commit %.0f | juntos: ventanas %llu (%.1f MB) tomadas %llu "
	            "comprobadas %llu diferencias %llu\n",
	            static_cast<unsigned long long>(copies), bytes / 1e6, 100.0 * inside / n,
	            bytes == 0 ? 0.0 : 100.0 * ib / bytes, total * nspc / n, (total - std::min(total, copy)) * nspc / n,
	            copy * nspc / n, static_cast<unsigned long long>(windows), wbytes / 1e6,
	            static_cast<unsigned long long>(taken), static_cast<unsigned long long>(checked),
	            static_cast<unsigned long long>(g.mismatches.load(std::memory_order_relaxed)));
}
inline void NoteCenso(bool inside, uint64_t size, uint64_t total, uint64_t copy) {
	g.copies.fetch_add(1, std::memory_order_relaxed);
	g.bytes.fetch_add(size, std::memory_order_relaxed);
	if (inside) {
		g.inside.fetch_add(1, std::memory_order_relaxed);
		g.inside_bytes.fetch_add(size, std::memory_order_relaxed);
	}
	g.total_cycles.fetch_add(total, std::memory_order_relaxed);
	g.copy_cycles.fetch_add(copy, std::memory_order_relaxed);
	MaybeReport();
}
} // namespace BloquesJuntos

namespace TableWatchState {
struct Watch {
	uint64_t hash;
	uint32_t sgpr, offset, limit;
	uint64_t uses;
	std::atomic<uint64_t> checked {0}, mismatches {0}; // (GPU-side comparisons done, failed)
};
struct Current {
	Watch*                  watch = nullptr;
	uint64_t                table = 0;
	std::array<uint32_t, 4> words {};
	std::array<uint32_t, 8> dwords {};
	bool                    read   = false;
	vk::DeviceAddress       device = 0;
};
// The last watched dispatches as they were recorded: a stall (DeviceFault::ReportStall) compares the stalled ones' block
// and V# bytes in the ring with what the ring holds then ("STALL-WATCH": the hanging dispatch itself, whose own check
// after it ran never comes).
struct Recorded {
	uint64_t                tick = 0, hash = 0;
	vk::DeviceAddress       block_address = 0, device = 0;
	const uint8_t*          block_host = nullptr; // (in the mapped ring)
	std::vector<uint32_t>   block;
	const uint8_t*          ring_host = nullptr; // the V#'s copy in the ring, if it is one
	std::array<uint32_t, 8> ring {}, guest {};
	uint64_t                indirect = 0;     // the arguments' guest address of an indirect dispatch
	bool                    chained  = false; // no barrier before it (CommandBuffer::ComputeChainPending)
};
static std::mutex                g_recorded_mutex;
static std::array<Recorded, 256> g_recorded;
static size_t                    g_recorded_next = 0;
thread_local size_t              g_recorded_last = SIZE_MAX; // (this thread's last, for TableWatchDispatch)
static void StallDump(uint64_t reached) {
	std::lock_guard lock(g_recorded_mutex);
	for (size_t i = g_recorded_next > g_recorded.size() ? g_recorded_next - g_recorded.size() : 0; i < g_recorded_next; ++i) {
		const auto& r = g_recorded[i % g_recorded.size()];
		if (r.tick <= reached || r.tick > reached + 64) continue;
		size_t first = r.block.size();
		for (size_t n = 0; n < r.block.size() && first == r.block.size(); ++n)
			if (std::memcmp(r.block_host + n * 4, &r.block[n], 4) != 0) first = n;
		uint32_t block_now = 0;
		if (first < r.block.size()) std::memcpy(&block_now, r.block_host + first * 4, 4);
		std::array<uint32_t, 8> ring_now {};
		if (r.ring_host != nullptr) std::memcpy(ring_now.data(), r.ring_host, sizeof(ring_now));
		std::printf("STALL-WATCH tick %llu %016llx%s%s: block 0x%llx (%zu dwords) %s dword %zu %08x -> %08x; V# at 0x%llx "
		            "(%s): guest %08x %08x %08x %08x, ring then %08x %08x %08x %08x, now %08x %08x %08x %08x\n",
		            static_cast<unsigned long long>(r.tick), static_cast<unsigned long long>(r.hash),
		            r.indirect != 0 ? " indirect" : "", r.chained ? " chained" : "",
		            static_cast<unsigned long long>(r.block_address), r.block.size(),
		            first < r.block.size() ? "CHANGED at" : "same, last", first < r.block.size() ? first : r.block.size(),
		            first < r.block.size() ? r.block[first] : 0u, block_now, static_cast<unsigned long long>(r.device),
		            r.ring_host != nullptr ? "ring" : "cached buffer", r.guest[0], r.guest[1], r.guest[2], r.guest[3],
		            r.ring[0], r.ring[1], r.ring[2], r.ring[3], ring_now[0], ring_now[1], ring_now[2], ring_now[3]);
	}
	std::fflush(stdout);
}
static std::deque<Watch>& Watches() {
	static std::deque<Watch> watches;
	static const bool        parsed = [] {
		const char* value = std::getenv("KYTY_TABLE_WATCH");
		for (const char* at = value; at != nullptr && *at != '\0';) {
			char*          end  = nullptr;
			const uint64_t hash = std::strtoull(at, &end, 16);
			if (end == nullptr || *end != ':') break;
			const auto sgpr = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
			if (*end != ':') break;
			const auto offset = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
			uint32_t   limit  = 32;
			if (*end == ':') limit = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
			auto& watch  = watches.emplace_back();
			watch.hash   = hash;
			watch.sgpr   = sgpr;
			watch.offset = offset;
			watch.limit  = limit;
			watch.uses   = 0;
			if (*end != ',' && *end != ';') break;
			at = end + 1;
		}
		if (!watches.empty()) DeviceFault::g_stall_dump.store(&StallDump, std::memory_order_release);
		return true;
	}();
	(void)parsed;
	return watches;
}
thread_local Current g_current;
} // namespace TableWatchState

// Before the blocks: the watched V# and its buffer's bytes.
static void TableWatchBegin(const ShaderRecompiler::IR::CompiledShaderInfo& program, std::span<const uint32_t> user_data) {
	auto& current = TableWatchState::g_current;
	current.watch = nullptr;
	auto& watches = TableWatchState::Watches();
	if (watches.empty()) return;
	for (auto& watch : watches) {
		if (watch.hash != program.shader_hash || watch.sgpr + 1 >= user_data.size()) continue;
		current       = {.watch = &watch};
		current.table = user_data[watch.sgpr] | uint64_t {user_data[watch.sgpr + 1] & 0xffffu} << 32u;
		current.read  = Libs::LibKernel::Memory::TryReadBackingToHost(current.table + watch.offset, current.words.data(), 16);
		if (current.read) {
			ShaderBufferResource descriptor;
			std::copy_n(current.words.begin(), 4, descriptor.fields);
			current.read = descriptor.Base48() != 0 &&
			               Libs::LibKernel::Memory::TryReadBackingToHost(descriptor.Base48(), current.dwords.data(),
			                                                             sizeof(current.dwords));
		}
		return;
	}
}

// TableBlocks: the device address a block gives the buffer of these V# words.
static void TableWatchNote(const std::array<uint32_t, 4>& words, vk::DeviceAddress device) {
	auto& current = TableWatchState::g_current;
	if (current.watch != nullptr && current.device == 0 && words == current.words) current.device = device;
}

// After the blocks, before the dispatch is recorded into `vk_buffer`: printed, and what the GPU reads checked once it ran
// the dispatch (`scheduler`: the main executor's own translation, else no check). The program's block (`block`, copied
// to `block_address` in the ring) is checked the same way ("BLOCK CHANGED": the device addresses it reads were
// overwritten before it ran; a ring reuse between the GPU finishing and the check running would show too), and kept
// for a stall (TableWatchState::StallDump). The GPU copy of a cached buffer's bytes needs KYTY_TABLE_WATCH_GPU=1: its
// barrier orders the dispatch after all earlier work, which would hide a missing dependency.
static void TableWatchEnd(const StreamBuffer& upload, BufferCache& cache, GraphicContext& graphics,
                          CommandScheduler* scheduler, vk::CommandBuffer vk_buffer, vk::DeviceAddress block_address,
                          std::span<const uint32_t> block) {
	auto& current = TableWatchState::g_current;
	if (current.watch == nullptr) return;
	if (const auto ring_base = upload.BufferDeviceAddress(), bytes = block.size_bytes();
	    scheduler != nullptr && block_address >= ring_base && bytes != 0 &&
	    block_address - ring_base + bytes <= upload.Mapped().size()) {
		const auto* at = upload.Mapped().data() + (block_address - ring_base);
		if (std::memcmp(at, block.data(), bytes) != 0)
			std::printf("TABLE_WATCH %016llx BLOCK MISMATCH at record: 0x%llx\n",
			            static_cast<unsigned long long>(current.watch->hash),
			            static_cast<unsigned long long>(block_address));
		scheduler->DeferOperation([at, copy = std::vector<uint32_t>(block.begin(), block.end()), hash = current.watch->hash,
		                           block_address] {
			if (std::memcmp(at, copy.data(), copy.size() * sizeof(uint32_t)) == 0) return;
			size_t first = 0;
			while (first < copy.size() && std::memcmp(at + first * 4, &copy[first], 4) == 0) ++first;
			uint32_t now = 0;
			std::memcpy(&now, at + first * 4, 4);
			std::printf("TABLE_WATCH %016llx BLOCK CHANGED before the GPU finished: 0x%llx dword %zu of %zu was %08x now "
			            "%08x\n",
			            static_cast<unsigned long long>(hash), static_cast<unsigned long long>(block_address), first,
			            copy.size(), first < copy.size() ? copy[first] : 0u, now);
		});
	}
	auto& watch   = *current.watch;
	current.watch = nullptr;
	const auto              ring_base = upload.BufferDeviceAddress();
	const auto              mapped    = upload.Mapped();
	const bool              in_ring   = current.device >= ring_base && !mapped.empty() &&
	                                    current.device - ring_base + sizeof(current.dwords) <= mapped.size();
	std::array<uint32_t, 8> ring {};
	if (in_ring) std::memcpy(ring.data(), mapped.data() + (current.device - ring_base), sizeof(ring));
	ShaderBufferResource descriptor;
	std::copy_n(current.words.begin(), 4, descriptor.fields);
	const uint64_t bytes   = descriptor.Stride() != 0 ? uint64_t {descriptor.Stride()} * descriptor.NumRecords()
	                                                  : uint64_t {descriptor.NumRecords()};
	bool           differs = false;
	for (uint64_t n = 0; in_ring && current.read && n < std::min<uint64_t>(bytes / 4, ring.size()); ++n)
		differs |= ring[n] != current.dwords[n];
	if (const auto block_bytes = block.size_bytes(); scheduler != nullptr && block_address >= ring_base && block_bytes != 0 &&
	                                                 block_address - ring_base + block_bytes <= mapped.size()) {
		std::lock_guard lock(TableWatchState::g_recorded_mutex);
		auto& recorded = TableWatchState::g_recorded[TableWatchState::g_recorded_next++ % TableWatchState::g_recorded.size()];
		recorded.tick          = scheduler->CurrentTick();
		recorded.hash          = watch.hash;
		recorded.block_address = block_address;
		recorded.device        = current.device;
		recorded.block_host    = mapped.data() + (block_address - ring_base);
		recorded.block.assign(block.begin(), block.end());
		recorded.ring_host = in_ring ? mapped.data() + (current.device - ring_base) : nullptr;
		recorded.ring      = ring;
		recorded.guest     = current.dwords;
		recorded.indirect  = 0;
		recorded.chained   = false;
		TableWatchState::g_recorded_last = TableWatchState::g_recorded_next - 1;
	}
	const bool odd = !current.read || current.dwords[2] > watch.limit || (in_ring && ring[2] > watch.limit);
	if (odd || differs || (watch.uses & 4095u) == 0)
		std::printf("TABLE_WATCH %016llx use %llu (checked %llu, mismatches %llu): table 0x%llx V# %08x %08x %08x %08x "
		            "bytes %llu %s 0x%llx: %s%s guest %08x %08x %08x %08x %08x %08x %08x %08x ring %08x %08x %08x %08x\n",
		            static_cast<unsigned long long>(watch.hash), static_cast<unsigned long long>(watch.uses),
		            static_cast<unsigned long long>(watch.checked.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(watch.mismatches.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(current.table), current.words[0], current.words[1], current.words[2],
		            current.words[3], static_cast<unsigned long long>(bytes),
		            in_ring ? "ring" : current.device != 0 ? "device" : "unnoted",
		            static_cast<unsigned long long>(current.device), odd ? "ODD" : "ok", differs ? " RING MISMATCH" : "",
		            current.dwords[0], current.dwords[1], current.dwords[2], current.dwords[3], current.dwords[4],
		            current.dwords[5], current.dwords[6], current.dwords[7], ring[0], ring[1], ring[2], ring[3]);
	++watch.uses;
	if (in_ring && scheduler != nullptr) {
		const auto* at = mapped.data() + (current.device - ring_base);
		scheduler->DeferOperation([at, ring, hash = watch.hash, device = current.device] {
			std::array<uint32_t, 8> now {};
			std::memcpy(now.data(), at, sizeof(now));
			if (now != ring)
				std::printf("TABLE_WATCH %016llx RING CHANGED before the GPU finished: 0x%llx was %08x %08x %08x %08x, "
				            "now %08x %08x %08x %08x\n",
				            static_cast<unsigned long long>(hash), static_cast<unsigned long long>(device), ring[0],
				            ring[1], ring[2], ring[3], now[0], now[1], now[2], now[3]);
		});
	}
	static const bool gpu_copy = [] {
		const char* value = std::getenv("KYTY_TABLE_WATCH_GPU");
		return value != nullptr && value[0] == '1';
	}();
	const auto base = descriptor.Base48();
	auto*      owner =
	    gpu_copy && !in_ring && current.device != 0 && current.read && scheduler != nullptr && bytes >= sizeof(current.dwords)
	              ? cache.GetBufferIfLive(cache.RegisteredBuffer(base, sizeof(current.dwords)))
	              : nullptr;
	if (owner == nullptr || owner->BufferDeviceAddress() + owner->Offset(base) != current.device) return;
	static std::unique_ptr<StreamBuffer> readback;
	if (!readback)
		readback = std::make_unique<StreamBuffer>(graphics, *scheduler, MemoryUsage::Download, uint64_t {1} << 20u,
		                                          vk::BufferUsageFlagBits::eTransferDst);
	const auto [host, offset] = readback->Map(sizeof(current.dwords), 16, false);
	if (host == nullptr) return;
	readback->Commit();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
	                          vk::DependencyFlags {}, 1, &before, 0, nullptr, 0, nullptr);
	vk::BufferCopy copy {};
	copy.srcOffset = owner->Offset(base);
	copy.dstOffset = offset;
	copy.size      = sizeof(current.dwords);
	vk_buffer.copyBuffer(owner->Handle(), readback->Handle(), 1, &copy);
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                          vk::DependencyFlags {}, 1, &after, 0, nullptr, 0, nullptr);
	scheduler->DeferOperation([buffer = readback.get(), host, offset, guest = current.dwords, hash = watch.hash, &watch,
	                           device = current.device, base] {
		buffer->Invalidate(offset, sizeof(guest));
		std::array<uint32_t, 8> gpu {};
		std::memcpy(gpu.data(), host, sizeof(gpu));
		watch.checked.fetch_add(1, std::memory_order_relaxed);
		if (gpu != guest) watch.mismatches.fetch_add(1, std::memory_order_relaxed);
		if (gpu != guest)
			std::printf("TABLE_WATCH %016llx DEVICE MISMATCH 0x%llx (guest 0x%llx): guest %08x %08x %08x %08x %08x %08x "
			            "%08x %08x gpu %08x %08x %08x %08x %08x %08x %08x %08x\n",
			            static_cast<unsigned long long>(hash), static_cast<unsigned long long>(device),
			            static_cast<unsigned long long>(base), guest[0], guest[1], guest[2], guest[3], guest[4], guest[5],
			            guest[6], guest[7], gpu[0], gpu[1], gpu[2], gpu[3], gpu[4], gpu[5], gpu[6], gpu[7]);
	});
}

// The dispatch TableWatchEnd recorded last on this thread: indirect (its arguments' address) and chained.
static void TableWatchDispatch(uint64_t indirect, bool chained) {
	auto& last = TableWatchState::g_recorded_last;
	if (last == SIZE_MAX) return;
	std::lock_guard lock(TableWatchState::g_recorded_mutex);
	if (TableWatchState::g_recorded_next - last <= TableWatchState::g_recorded.size()) {
		auto& recorded    = TableWatchState::g_recorded[last % TableWatchState::g_recorded.size()];
		recorded.indirect = indirect;
		recorded.chained  = chained;
	}
	last = SIZE_MAX;
}

bool RenderExecutor::TableBlocks(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data,
                                 std::array<vk::DeviceAddress, 2>& blocks, bool* writes) {
	using Kind         = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto&      table        = OwnTable();
	auto&      buffer_cache = m_context.GetBufferCache();
	auto&      textures     = m_context.GetTextureCache();
	auto&      upload       = buffer_cache.GetTableUploadBuffer();
	const auto stages       = pair.programs[1] != nullptr ? 2u : 1u;
	const auto start_epoch  = textures.StartEpoch(); // (before any question below: a later image start moves it)
	const auto image_starts = [&](uint64_t address) {
		auto& known = table.no_image_start[(address * 0x9e3779b97f4a7c15ull) >> 54u];
		if (known.first == address && known.second == start_epoch) return false;
		if (textures.HasImageStartingAt(address)) return true;
		known = {address, start_epoch};
		return false;
	};
	if (table.resolved_start != start_epoch || table.resolved_mapping != table.mapping_epoch) {
		table.resolved_start   = start_epoch;
		table.resolved_mapping = table.mapping_epoch;
		++table.special_generation;
	}
	// A tracker region's epoch, read once here (the buffers mostly share one: the line is another core's).
	std::array<std::pair<uint64_t, uint64_t>, 4> region_epochs;
	region_epochs.fill({UINT64_MAX, 0});
	const auto cpu_epoch = [&](uint64_t address, uint64_t size) {
		const auto region = address / TRACKER_REGION_SIZE;
		if (region != (address + size - 1) / TRACKER_REGION_SIZE) return buffer_cache.CpuModificationEpoch(address, size);
		auto& known = region_epochs[region & 3u];
		if (known.first != region) known = {region, buffer_cache.CpuModificationEpoch(address, size)};
		return known.second;
	};
	// A range none of whose pages the guest wrote since they were uploaded (with no write protection still queued:
	// a page marked clean can change unseen until it lands).
	const auto range_clean = [&](uint64_t address, uint64_t size) {
		for (auto region = address / TRACKER_REGION_SIZE; region <= (address + size - 1) / TRACKER_REGION_SIZE; ++region)
			if (buffer_cache.DeferredProtectionPending(region * TRACKER_REGION_SIZE)) return false;
		return !buffer_cache.IsRegionCpuModified(address, size);
	};
	// A null buffer's device address: valid memory (the shaders' loads of a buffer are not branched around its bounds:
	// out of bounds they load its first dword).
	const vk::DeviceAddress null_device = upload.BufferDeviceAddress();
	// A V#'s buffer word (IR::BufferWord, as IR::BufferDescriptorWord makes it) and stride. (Each format's class
	// once: the V#s of every draw's buffers asked for it.)
	static const auto format_classes = [] {
		std::array<uint32_t, 128> classes {};
		for (uint32_t format = 0; format < classes.size(); ++format)
			classes[format] = ShaderRecompiler::IR::BufferFormatClass(static_cast<Prospero::BufferFormat>(format));
		return classes;
	}();
	const auto word_of = [](const std::array<uint32_t, 4>& words4) -> std::pair<uint32_t, uint32_t> {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		if (descriptor.Type() != 0) return {0, 0};
		using Word = ShaderRecompiler::IR::BufferWord;
		return {(descriptor.Stride() <= Word::MaxStride ? descriptor.Stride() : 0u) |
		            format_classes[descriptor.RawFormat()] << Word::LayoutShift |
		            descriptor.DstSelXYZW() << Word::DstSelShift,
		        descriptor.Stride()};
	};
	// The bytes a V# names (0: none).
	const auto requested_of = [](const std::array<uint32_t, 4>& words4) -> uint64_t {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		if (descriptor.Type() != 0 || descriptor.Base48() == 0) return 0;
		return descriptor.Stride() != 0 ? uint64_t {descriptor.Stride()} * descriptor.NumRecords()
		                                : uint64_t {descriptor.NumRecords()};
	};
	// As RenderExecutor::FindBuffers: a V# that is no buffer, or of nothing mapped, is a null buffer (size 0).
	const auto range_of = [](const std::array<uint32_t, 4>& words4) -> std::pair<uint64_t, uint64_t> {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		const auto requested = descriptor.Stride() != 0 ? uint64_t {descriptor.Stride()} * descriptor.NumRecords()
		                                                : uint64_t {descriptor.NumRecords()};
		const auto address   = descriptor.Base48();
		return {address, descriptor.Type() != 0 || address == 0 || requested == 0
		                     ? 0
		                     : Libs::LibKernel::Memory::ClampRangeSize(address, requested)};
	};
	// A V# the program was specialized for (a format and dst_sel it decodes, no swizzle or ADD_TID; strides are the
	// block's): another one takes the normal path, which chooses the program for it.
	const auto specialized = [](const ShaderRecompiler::IR::BufferResource& resource,
	                            const std::array<uint32_t, 4>&             words4) {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		if (descriptor.Type() != 0) return true; // no buffer: read as a null one
		return !descriptor.SwizzleEnabled() && !descriptor.AddTid() &&
		       (!resource.formatted || resource.descriptor_format == Prospero::BufferFormat::kInvalid ||
		        (descriptor.Format() == resource.descriptor_format && descriptor.DstSelXYZW() == resource.descriptor_swizzle));
	};
	if (table.gathered != &pair) TableGather(pair, user_data);
	table.gathered = nullptr; // (taken: a later draw or dispatch gathers its own)
	// (render4, KYTY_RENDER_STATS: what the blocks do with the V#s, LiveCensus::g_table_blocks; the render thread's only.)
	const bool block_stats = LiveCensus::g_render && LiveCensus::RenderStatsOn();
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		auto&       block   = table.blocks[st];
		auto&       entries = table.entries[st];
		// The buffers it writes first: a range it also reads through another V# is GPU-written by then, as the
		// normal path's second lookup (RebindBuffers) sees it (not a copy of the guest's bytes in the ring).
		for (size_t n = 0; n < 2 * plan.buffers.size(); ++n) {
			const auto  i         = n < plan.buffers.size() ? n : n - plan.buffers.size();
			const auto& words4    = entries[i].first;
			const auto& resource  = program.info.buffers[i];
			const bool  formatted = resource.formatted;
			if ((resource.written || resource.atomic) != (n < plan.buffers.size())) continue;
			if (!specialized(resource, words4)) {
				if (Role() != 0) return Spec::Refuse("table: another format"), false;
				if (ShaderRecompiler::IR::RuntimeBufferFormat(resource) && i < 32 && ((pair.portable[st] >> i) & 1u) == 0) {
					const auto write = CatalogWrite();
					pair.portable[st] |= 1u << i;
					pair.respecialize = true;
				}
				return false;
			}
			if (resource.written || resource.atomic) {
				// The shader writes it: marked GPU-written now, whatever an earlier use proved. (A write hazard as
				// HasShaderBufferWrites sees one: a V# of a nonempty range.)
				if (writes != nullptr && resource.written && (uint64_t {words4[0]} | (words4[1] & 0xffffu)) != 0 &&
				    words4[2] != 0)
					*writes = true;
				if (block_stats) ++LiveCensus::g_table_blocks[0];
				auto& proof = table.written[XXH3_64bits(words4.data(), sizeof(words4)) & (TableXpr::WrittenEntries - 1)];
				if (proof.special != table.special_generation || proof.words != words4) {
					if (Trad1::Mode() != 0) ++(proof.words != words4 ? Trad1::g.reset_words : Trad1::g.reset_special); // (trad1b)
					const auto [clamped_address, clamped_size] = range_of(words4);
					proof = {.words = words4, .special = table.special_generation, .address = clamped_address, .size = clamped_size};
				}
				const auto address = proof.address, size = proof.size;
				vk::DeviceAddress device = null_device;
				if (size != 0) {
					if (address % 4 != 0) return false;
					if (formatted && !proof.image_checked) {
						if (image_starts(address)) return false;
						proof.image_checked = true;
					}
					const bool own_plain = Role() == 0 && Spec::Current() == nullptr && !formatted &&
					                       kyty_local_table_written_memo.load(std::memory_order_relaxed) != 0 &&
					                       !LaneSync::g_on; // (KYTY_ASYNC_COMPUTE: every write through ObtainBuffer)
					const auto tick      = m_context.GetCommandScheduler().CurrentTick();
					if (proof.memo && own_plain && proof.tick == tick &&
					    proof.registration == buffer_cache.RegistrationEpoch() &&
					    proof.obtain_epoch == buffer_cache.WrittenObtainEpoch() &&
					    cpu_epoch(address, size) == proof.cpu_epoch &&
					    buffer_cache.GetBufferIfLive(proof.owner) == proof.owner_buffer) {
						device = proof.device;
						LiveCounters::Add(LiveCounters::TableWrittenMemo);
						if (Trad1::Mode() != 0) ++Trad1::g.old_memo; // (trad1)
						// KYTY_TABLE_WRITTEN_VERIFY=1: the lookup as well, its result against the memo's.
						static const bool verify = std::getenv("KYTY_TABLE_WRITTEN_VERIFY") != nullptr;
						if (verify) {
							const auto [owner, offset] = buffer_cache.ObtainBuffer(address, size, true, formatted);
							region_epochs.fill({UINT64_MAX, 0});
							const auto found = owner != nullptr ? owner->BufferDeviceAddress() + offset : vk::DeviceAddress {0};
							static std::atomic<uint32_t> mismatches {0};
							if (found != device && mismatches.fetch_add(1, std::memory_order_relaxed) < 16)
								std::printf("Table: written memo mismatch at 0x%llx+0x%llx: memo 0x%llx, lookup 0x%llx\n",
								            static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
								            static_cast<unsigned long long>(device), static_cast<unsigned long long>(found));
						}
					} else {
						// (trad1b, KYTY_TRAD1; local/trad1.h) Another tick, or an epoch the old memo needs moved: with this
						// range's last full obtain still standing on ObtainBuffer's write memo terms (registration, the
						// same live buffer, no GPU-written byte of its regions given back), the write is only noted again
						// (=1), or the old path checked against it (=2). Looked up by the range, not the V#'s entry.
						const uint32_t trad1    = own_plain ? Trad1::Mode() : 0u;
						bool           taken    = false;
						bool           eligible = false;
						auto           why      = Trad1::NoProof;
						TableXpr::Trad1Range* range = nullptr;
						if (trad1 != 0) {
							buffer_cache.Trad1EnableCoverage();
							range = &table.Trad1Find(address, size);
							if (range->address != address || range->size != size) why = Trad1::NoProof;
							else if (range->registration != buffer_cache.RegistrationEpoch()) why = Trad1::Registration;
							else if (buffer_cache.GetBufferIfLive(range->owner) != range->owner_buffer) why = Trad1::BufferGone;
							else if (buffer_cache.Trad1CoverageMoved(address, size, range->coverage)) why = Trad1::Coverage;
							else eligible = true;
							if (eligible && trad1 == 1) {
								const bool     sample = Trad1::SampleFast();
								const uint64_t start  = sample ? __rdtsc() : 0;
								if (buffer_cache.Trad1RenoteWritten(range->owner, range->owner_buffer, address, size,
								                                    range->registration, range->coverage)) {
									taken       = true;
									device      = range->device;
									range->used = ++table.trad1_clock;
									++Trad1::g.hits;
									region_epochs.fill({UINT64_MAX, 0}); // (the drain may have moved them)
									// As after a full obtain: the old memo holds for the rest of the tick.
									const auto epoch   = buffer_cache.CpuModificationEpoch(address, size);
									proof.memo         = epoch != 0;
									proof.device       = range->device;
									proof.owner        = range->owner;
									proof.owner_buffer = range->owner_buffer;
									proof.tick         = tick;
									proof.registration = range->registration;
									proof.obtain_epoch = buffer_cache.WrittenObtainEpoch();
									proof.cpu_epoch    = epoch;
									if (sample) {
										Trad1::g.fast_cycles += __rdtsc() - start;
										++Trad1::g.fast_samples;
									}
								} else {
									eligible = false;
									why      = Trad1::Refused;
								}
							}
							if (!eligible) ++Trad1::g.miss[why];
						}
						if (!taken) {
							const bool verify      = trad1 == 2 && eligible;
							bool       would_write = false;
							vk::DeviceAddress expect_device = 0;
							const Buffer*     expect_buffer = nullptr;
							uint64_t          expect_coverage = 0, expect_registration = 0;
							if (verify) {
								++Trad1::g.hits; // (what =1 would take)
								would_write         = buffer_cache.Trad1SyncWouldUpload(address, size);
								expect_device       = range->device;
								expect_buffer       = range->owner_buffer;
								expect_coverage     = range->coverage;
								expect_registration = range->registration;
							}
							const bool     sample = trad1 != 0 && (trad1 == 1 || verify) && Trad1::SampleFull();
							const uint64_t start  = sample ? __rdtsc() : 0;
							if (block_stats) ++LiveCensus::g_table_blocks[1];
							proof.memo                 = false;
							const auto [owner, offset] = buffer_cache.ObtainBuffer(address, size, true, formatted);
							region_epochs.fill({UINT64_MAX, 0}); // (its uploads may have moved them)
							const auto id = buffer_cache.FindBuffer(address, size);
							if (owner == nullptr || buffer_cache.GetBufferIfLive(id) != owner) return false;
							if (formatted) textures.InvalidateMemoryFromGPU(address, size, "storage-buffer");
							device = owner->BufferDeviceAddress() + offset;
							if (sample) {
								Trad1::g.full_cycles += __rdtsc() - start;
								++Trad1::g.full_samples;
							}
							// (trad1, =2) Only where the shortcut would still have been taken after the drain (the
							// epochs unchanged by the old path's own drain and lookups).
							if (verify && !buffer_cache.Trad1CoverageMoved(address, size, expect_coverage) &&
							    buffer_cache.RegistrationEpoch() == expect_registration) {
								++Trad1::g.checked;
								if (owner != expect_buffer)
									Trad1::NoteDifference("buffer", address, size, reinterpret_cast<uint64_t>(expect_buffer),
									                      reinterpret_cast<uint64_t>(owner));
								else if (device != expect_device)
									Trad1::NoteDifference("device", address, size, expect_device, device);
								if (would_write) {
									++Trad1::g.dirty;
									Trad1::NoteDifference("upload (pages not GPU-owned)", address, size, 0, 1);
								}
							}
							// (The epochs after ObtainBuffer: its uploads and GPU marks are what the next use must find.)
							if (const auto epoch = buffer_cache.CpuModificationEpoch(address, size); own_plain && epoch != 0) {
								proof.memo         = true;
								proof.device       = device;
								proof.owner        = id;
								proof.owner_buffer = owner;
								proof.tick         = tick;
								proof.registration = buffer_cache.RegistrationEpoch();
								proof.obtain_epoch = buffer_cache.WrittenObtainEpoch();
								proof.cpu_epoch    = epoch;
							}
							if (trad1 != 0)
								*range = {.address      = address,
								          .size         = size,
								          .device       = device,
								          .owner        = id,
								          .owner_buffer = owner,
								          .registration = buffer_cache.RegistrationEpoch(),
								          .coverage     = buffer_cache.GpuCoverageEpoch(),
								          .used         = ++table.trad1_clock};
						}
						if (trad1 != 0) Trad1::MaybeReport(LiveControl::g_flips.load(std::memory_order_relaxed));
					}
				}
				const auto [word, stride]      = word_of(words4);
				block[plan.BufferDword(i)]     = static_cast<uint32_t>(device);
				block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(device >> 32u);
				block[plan.BufferDword(i) + 2] = static_cast<uint32_t>(std::min<uint64_t>(size / 4, UINT32_MAX));
				block[plan.BufferDword(i) + 3] = word;
				block[plan.BufferDword(i) + 4] = stride;
				continue;
			}
			if (block_stats) ++LiveCensus::g_table_blocks[2];
			auto& held    = *entries[i].second;
			bool  current = held.words == words4 && (held.special == 0 || held.special == table.special_generation);
			if (current && (held.registration & TableXpr::RingCopyMark) != 0) {
				// Its ring copy (RingCopy's terms: the same observation epoch, the regions' epoch unchanged, no page
				// GPU-written).
				if (Role() == 0 && Spec::Current() == nullptr &&
				    held.registration == (TableXpr::RingCopyMark | PageManager::ObservationEpoch()) &&
				    cpu_epoch(held.address, held.size) == held.cpu_epoch &&
				    !buffer_cache.IsRegionGpuModified(held.address, held.size)) {
					LiveCounters::Add(LiveCounters::TableRingShared);
					LiveCounters::Add(LiveCounters::TableRingSharedBytes, held.size);
					const auto [word, stride]      = word_of(words4);
					block[plan.BufferDword(i)]     = static_cast<uint32_t>(held.device);
					block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(held.device >> 32u);
					block[plan.BufferDword(i) + 2] = held.dwords;
					block[plan.BufferDword(i) + 3] = word;
					block[plan.BufferDword(i) + 4] = stride;
					continue;
				}
				current = false;
			}
			// (A speculation's guest memory writes wait for its commit: such a range is resolved, ObtainBuffer decides.)
			if (const auto* spec = Spec::Current();
			    current && held.size != 0 && spec != nullptr && spec->JournalIntersects(held.address, held.size))
				current = false;
			if (current && held.size != 0 && held.registration != buffer_cache.RegistrationEpoch()) {
				// Buffers were registered or unregistered since: it holds while none covered its tracker regions.
				for (auto region = held.address / TRACKER_REGION_SIZE, last = (held.address + held.size - 1) / TRACKER_REGION_SIZE;
				     current && region <= last; ++region)
					current = buffer_cache.RegionRegistrationEpoch(region * TRACKER_REGION_SIZE) <= held.registration;
				if (current) held.registration = buffer_cache.RegistrationEpoch();
			}
			if (current && held.size != 0) {
				// Another page of its tracker regions was written: the range itself may still be clean.
				if (const auto epoch = cpu_epoch(held.address, held.size); epoch == 0 || epoch != held.cpu_epoch) {
					LiveCensus::Scope reprove(LiveCensus::DrawPhase, 0x7ab1e, 6);
					if (block_stats) ++LiveCensus::g_table_blocks[4];
					current = range_clean(held.address, held.size);
					G1::NoteReprove(current); // (g1 NP1)
					if (current) held.cpu_epoch = epoch;
				}
			}
			if (!current) {
				LiveCensus::Scope resolve(LiveCensus::DrawPhase, 0x7ab1e, 5);
				if (block_stats) ++LiveCensus::g_table_blocks[3];
				// (Kept only once resolved: a refusal leaves nothing a later use could take.)
				held = {};
				TableXpr::ResolvedBuffer fresh {.words = words4, .device = null_device};
				const auto [address, size] = range_of(words4);
				if (formatted || size < requested_of(words4)) fresh.special = table.special_generation;
				if (size != 0) {
					// The shader reads whole dwords from the device address; a buffer an image starts at is made from
					// the image by ObtainBuffer only (SynchronizeBufferFromImage).
					if (address % 4 != 0 || (formatted && image_starts(address))) return false;
					// (A speculation's: the guest's writes, not those its commit makes current; with none of its own
					// guest memory writes pending over the range, read as no GPU work wrote it.)
					auto* const spec        = Spec::Current();
					// (censo bloques, KYTY_CENSO_BLOQUES=1) The checks' and the copy's time per ring copy.
					static const bool censo = [] {
						const char* v = std::getenv("KYTY_CENSO_BLOQUES");
						return v != nullptr && std::atoi(v) != 0;
					}();
					const uint64_t censo_start = censo ? __rdtsc() : 0;
					const bool  cpu_written = buffer_cache.IsRegionCpuDirty(address, size) &&
					                         !buffer_cache.IsRegionGpuModified(address, size);
					if (cpu_written && size <= BufferCache::CACHING_PAGESIZE &&
					    (spec == nullptr || !spec->JournalIntersects(address, size))) {
						// As ObtainBuffer: a small range the guest wrote is read from a copy in the ring (nothing held: the
						// main executor's later uses share it while it holds, TableXpr::RingCopy), not uploaded into its
						// buffer.
						auto* const shared = spec == nullptr && Role() == 0
						                         ? &table.ring_copies[((address ^ (size << 40u)) * 0x9e3779b97f4a7c15ull) >> 52u]
						                         : nullptr;
						const auto observation = PageManager::ObservationEpoch();
						const auto tick        = m_context.GetCommandScheduler().CurrentTick();
						const auto cpu         = buffer_cache.CpuModificationEpoch(address, size); // (before the copy)
						vk::DeviceAddress device = 0;
						// (bloques juntos) =1 windows, =2 windows checked against a plain copy; (censo bloques) =1.
						static const int juntos = [] {
							const char* v = std::getenv("KYTY_BLOQUES_JUNTOS");
							return v == nullptr ? 0 : std::atoi(v);
						}();
						uint64_t censo_copy = 0;
						bool     censo_copied = false;
						vk::DeviceAddress window_device = 0;
						if (juntos != 0 && shared != nullptr && cpu != 0) {
							for (const auto& w: table.ring_windows)
								if (w.device != 0 && w.tick == tick && w.observation == observation && address >= w.base &&
								    address + size <= w.base + w.size &&
								    buffer_cache.CpuModificationEpoch(w.base, w.size) == w.cpu_epoch) {
									window_device = w.device + (address - w.base);
									break;
								}
						}
						if (shared != nullptr && cpu != 0 && shared->address == address && shared->size == size &&
						    shared->observation == observation && shared->tick == tick && shared->cpu_epoch == cpu) {
							device = shared->device;
							LiveCounters::Add(LiveCounters::TableRingShared);
							LiveCounters::Add(LiveCounters::TableRingSharedBytes, size);
						} else if (window_device != 0 && juntos != 2) {
							device = window_device;
							BloquesJuntos::g.taken.fetch_add(1, std::memory_order_relaxed);
							BloquesJuntos::MaybeReport();
							LiveCounters::Add(LiveCounters::TableRingShared);
							LiveCounters::Add(LiveCounters::TableRingSharedBytes, size);
						} else if (const bool windowed = juntos != 0 && shared != nullptr && cpu != 0 && window_device == 0;
						           windowed && [&] {
							           // A window of the whole pages around the range plus the next page.
							           const uint64_t base = address & ~uint64_t {4095};
							           const uint64_t end  = ((address + size + 4095) & ~uint64_t {4095}) + 4096;
							           if (end - base > 16384) return false;
							           const uint64_t wepoch = buffer_cache.CpuModificationEpoch(base, end - base);
							           if (wepoch == 0) return false;
							           const uint64_t t0 = censo ? __rdtsc() : 0;
							           auto [wmapped, woffset] = upload.Map(end - base, 16, false);
							           if (wmapped == nullptr ||
							               !Libs::LibKernel::Memory::TryReadBackingToHost(base, wmapped, end - base))
								           return false;
							           upload.Commit();
							           if (censo) censo_copy = __rdtsc() - t0;
							           censo_copied = true;
							           const auto wdev = upload.BufferDeviceAddress() + woffset;
							           table.ring_windows[table.ring_window_next++ % TableXpr::RingWindows] = {
							               .base = base, .size = end - base, .observation = observation, .tick = tick,
							               .cpu_epoch = wepoch, .device = wdev,
							               .host = reinterpret_cast<const uint8_t*>(wmapped)};
							           device = wdev + (address - base);
							           BloquesJuntos::g.windows.fetch_add(1, std::memory_order_relaxed);
							           BloquesJuntos::g.window_bytes.fetch_add(end - base, std::memory_order_relaxed);
							           return true;
						           }()) {
							LiveCounters::Add(LiveCounters::TableRingCopies);
							LiveCounters::Add(LiveCounters::TableRingBytes, size);
							if (shared != nullptr)
								*shared = {.address = address, .size = size, .observation = observation, .tick = tick,
								           .cpu_epoch = cpu, .device = device};
						} else if (const uint64_t t0 = censo ? __rdtsc() : 0; true)
						if (auto [mapped, offset] = upload.Map(size, 16, false);
						           mapped != nullptr && Libs::LibKernel::Memory::TryReadBackingToHost(address, mapped, size)) {
							upload.Commit();
							if (censo) censo_copy = __rdtsc() - t0;
							censo_copied = true;
							if (juntos == 2 && window_device != 0) {
								// (=2) The window's bytes at the range against the copy just made: reads the ring's mapping.
								BloquesJuntos::g.checked.fetch_add(1, std::memory_order_relaxed);
								for (const auto& w: table.ring_windows)
									if (w.device != 0 && window_device >= w.device && window_device < w.device + w.size) {
										if (w.host != nullptr &&
										    std::memcmp(w.host + (window_device - w.device), mapped, size) != 0)
											BloquesJuntos::NoteMismatch(address, size);
										break;
									}
							}
							if (spec != nullptr) spec->NoteCleanRead(address, size);
							device = upload.BufferDeviceAddress() + offset;
							LiveCounters::Add(LiveCounters::TableRingCopies);
							LiveCounters::Add(LiveCounters::TableRingBytes, size);
							F0Measure::NoteRingCopy(address, size); // (f0)
							if (shared != nullptr && cpu != 0)
								*shared = {.address     = address,
								           .size        = size,
								           .observation = observation,
								           .tick        = tick,
								           .cpu_epoch   = cpu,
								           .device      = device};
						}
						if (censo && censo_copied) {
							// (censo bloques) Inside the 4 KiB page, or the next, of an earlier copy of this submission and
							// observation epoch whose tracker epoch held?
							bool inside = false;
							for (const auto& w: table.censo_windows)
								if (w.size != 0 && w.tick == tick && w.observation == observation && address >= w.base &&
								    address + size <= w.base + w.size &&
								    buffer_cache.CpuModificationEpoch(w.base, w.size) == w.cpu_epoch) {
									inside = true;
									break;
								}
							const uint64_t base = address & ~uint64_t {4095};
							const uint64_t end  = ((address + size + 4095) & ~uint64_t {4095}) + 4096;
							table.censo_windows[table.censo_next++ % TableXpr::RingWindows] = {
							    .base = base, .size = end - base, .observation = observation, .tick = tick,
							    .cpu_epoch = buffer_cache.CpuModificationEpoch(base, end - base)};
							BloquesJuntos::NoteCenso(inside, size, __rdtsc() - censo_start, censo_copy);
						}
						// (The V#'s later uses take it from its held entry.)
						if (device != 0 && shared != nullptr && cpu != 0)
							held = {.words        = words4,
							        .dwords       = static_cast<uint32_t>(size / 4),
							        .special      = fresh.special,
							        .device       = device,
							        .address      = address,
							        .size         = size,
							        .cpu_epoch    = cpu,
							        .registration = TableXpr::RingCopyMark | observation};
						if (device != 0) {
							const auto [word, stride]      = word_of(words4);
							block[plan.BufferDword(i)]     = static_cast<uint32_t>(device);
							block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(device >> 32u);
							block[plan.BufferDword(i) + 2] = static_cast<uint32_t>(size / 4);
							block[plan.BufferDword(i) + 3] = word;
							block[plan.BufferDword(i) + 4] = stride;
							continue;
						}
					}
					if (cpu_written) buffer_cache.EnsureBufferContents(address, size);
					const auto [owner, offset] = buffer_cache.ObtainBuffer(address, size, false, formatted);
					region_epochs.fill({UINT64_MAX, 0});
					// (The epoch before the dirty check: a write after it moves the epoch.)
					fresh.cpu_epoch = buffer_cache.CpuModificationEpoch(address, size);
					if (owner == nullptr || buffer_cache.GetBufferIfLive(buffer_cache.FindBuffer(address, size)) != owner ||
					    buffer_cache.IsRegionCpuModified(address, size))
						return false;
					if (Spec::Current() != nullptr && buffer_cache.IsRegionCpuDirty(address, size)) fresh.cpu_epoch = 0;
					fresh.device       = owner->BufferDeviceAddress() + offset;
					fresh.dwords       = static_cast<uint32_t>(std::min<uint64_t>(size / 4, UINT32_MAX));
					fresh.address      = address;
					fresh.size         = size;
					fresh.registration = buffer_cache.RegistrationEpoch(); // (after ObtainBuffer's own registrations)
				}
				held = fresh;
			}
			// (A speculation's: the buffer of the range its device address is in.)
			if (auto* spec = Spec::Current(); spec != nullptr && held.size != 0) spec->NoteBufferRange(held.address, held.size);
			const auto [word, stride]      = word_of(words4);
			block[plan.BufferDword(i)]     = static_cast<uint32_t>(held.device);
			block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(held.device >> 32u);
			block[plan.BufferDword(i) + 2] = held.dwords;
			block[plan.BufferDword(i) + 3] = word;
			block[plan.BufferDword(i) + 4] = stride;
		}
		if (TableWatchState::g_current.watch != nullptr)
			for (size_t i = 0; i < plan.buffers.size(); ++i)
				TableWatchNote(entries[i].first,
				               block[plan.BufferDword(i)] | uint64_t {block[plan.BufferDword(i) + 1]} << 32u);
		if (plan.BlockDwords() != 0) // (a stage with neither slots nor buffers reads no block)
			blocks[st] = upload.BufferDeviceAddress() + upload.Copy(block.data(), block.size() * sizeof(uint32_t), 16);
	}
	return true;
}

// The page table ranges the pair's flat and global loads read made current, as the normal path's PrepareBdaBindings
// does at every use (the read plans' words from the user data and the blocks' slots); an unbounded plan, all of them.
// (Without it, particle simulations at Stonefang read stale pages: readbacks the game crashed on, and indirect
// arguments that lost the device.)
void RenderExecutor::TablePrepareBda(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	const auto global = [&](uint32_t st) { return pair.programs[st] != nullptr && pair.programs[st]->table_plan.global_memory; };
	if (!global(0) && !global(1)) return;
	auto&                                               table = OwnTable();
	thread_local ShaderRecompiler::IR::ResourceSnapshot snapshot;
	thread_local std::vector<GuestRange>                ranges, stage_ranges;
	ranges.clear();
	BdaRepite::Scope repite; // (bda-repite, KYTY_TECHO: solo mide)
	bool bounded = true;
	for (uint32_t st = 0; st < 2 && bounded; ++st) {
		if (!global(st)) continue;
		const auto& program = *pair.programs[st];
		snapshot.user_data.assign(user_data[st].begin(), user_data[st].end());
		snapshot.flattened_srt.assign(table.blocks[st].begin(), table.blocks[st].begin() + program.table_plan.slots.size());
		if (repite.on) BdaRepite::AddStage(repite.key, st, program.bda_read_plan, snapshot.user_data, snapshot.flattened_srt);
		bounded = ShaderRecompiler::IR::EvaluateBdaReadPlan(program.bda_read_plan, snapshot, stage_ranges);
		ranges.insert(ranges.end(), stage_ranges.begin(), stage_ranges.end());
	}
	repite.Eval(bounded, ranges);
	auto& resources = m_context.GetGpuResources();
	if (!bounded || !resources.PrepareBdaReadRanges(ranges)) resources.PrepareBda();
}

// A set's images in their layouts (other work may have moved them since the set was last bound), a DCC image's
// pending fast clear applied first, as the normal path's FindTexture does (a speculation's refuses: its packet is a
// hole), unless the draw's targets are not those the set was made for (an image held as a texture became a target,
// or the reverse).
bool RenderExecutor::TableTransit(const TableImageSet& set, vk::CommandBuffer vk_buffer,
                                  const NativeXprDrawState* draw_state) {
	auto& textures = m_context.GetTextureCache();
	for (const auto& dep: set.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr) return false;
		// A target of the draw: NativeXprEmitTargets put it in its attachment layout, which the set must have been
		// made for; an image the set holds as a texture must not be one.
		const auto target = draw_state == nullptr
		                        ? decltype(draw_state->targets.end()) {}
		                        : std::ranges::find(draw_state->targets, dep.id, &NativeXprDrawState::Target::id);
		const bool is_target = draw_state != nullptr && target != draw_state->targets.end();
		if (is_target != dep.target || (is_target && (target->layout != dep.layout || target->access != dep.access)))
			return false;
		if (is_target) continue;
		// An image the shader writes is GPU-modified, CPU-clean and current with buffer writes (as the normal path's
		// FindTexture leaves a storage image: it acquires it again otherwise). One written since the set was proven this
		// frame (by the CPU, a GPU buffer write, or into its stencil plane) is uploaded by the normal path's FindTexture.
		if ((dep.storage && !image->SafeToDownload()) || TableRefreshPending(*image, dep.desc)) return false;
		if (dep.desc.info.metadata.kind == ImageMetadataKind::Dcc) {
			if (image->info.IsVolume()) return false;
			const auto address = dep.desc.info.metadata.range.address;
			for (uint32_t layer = dep.range.base_layer; layer < dep.range.base_layer + dep.range.layer_count; ++layer) {
				if (layer >= 32) return false;
				if (!textures.IsMetaCleared(address, layer)) continue;
				// The normal path's FindTexture clears its layers first. A speculation's packet is a hole that does:
				// the work after it finds them cleared (speculation-state.h).
				if (auto* spec = Spec::Current()) {
					for (uint32_t at = layer; at < std::min(dep.range.base_layer + dep.range.layer_count, 32u); ++at)
						if (textures.IsMetaCleared(address, at)) spec->ConsumeMeta(address, at);
					return Spec::Refuse("table: cleared texture"), false;
				}
				// The main executor's: FindTexture, as the normal path's use (it clears what it can, into this command
				// buffer, before the transitions below; what it leaves pending, the normal path binds as well).
				if (Role() != 0 || !image->registered || image->depth_id || image->binding.needs_rebind) return false;
				(void)textures.FindTexture(dep.id, dep.desc);
				image = textures.m_slot_images.try_get(dep.id);
				if (image == nullptr || image->serial != dep.serial ||
				    m_context.GetCommandScheduler().Current().RawHandle() != vk_buffer)
					return false;
				break;
			}
		}
		// A depth image with an HTile clear pending: the normal path's FindTexture clears it first
		// (KYTY_SAMPLE_PENDING_HTILE_CLEAR, upstream 5a65bfb); the draw or dispatch is left to it.
		if (image->info.IsDepth() && TextureCache::SampleHtileClearOn()) {
			uint64_t meta    = 0;
			bool     consume = false;
			if (textures.SampleHtileClearPending(*image, dep.desc, &meta, &consume)) {
				if (auto* spec = Spec::Current()) {
					if (consume) spec->ConsumeMeta(meta, 0);
					return Spec::Refuse("table: depth texture with an HTile clear pending"), false;
				}
				return false;
			}
		}
	}
	// As CommitBindings': the transitions all precede the one draw or dispatch.
	const TransitGroup transit_group;
	for (const auto& dep: set.images)
		if (!dep.target)
			textures.m_slot_images[dep.id].Transit(dep.layout, dep.access, dep.whole ? std::nullopt : std::optional {dep.range},
			                                       vk_buffer);
	return true;
}

RenderExecutor::TableResult RenderExecutor::TableDraw(CommandBuffer& buffer, TablePair& pair, TableVariant& variant,
                                                      const NativeXprDrawState& draw_state,
                                                      std::span<const uint64_t> commands, uint64_t index_base,
                                                      uint64_t index_bytes, vk::IndexType index_type,
                                                      const NativeXprDirectDraw* direct) {
	auto&       table  = OwnTable();
	auto&       cache  = *m_native_xpr;
	const auto& sh     = buffer.GetShaders();
	const auto  stages = pair.programs[1] != nullptr ? 2u : 1u;
	const std::array<std::span<const uint32_t>, 2> user_data {std::span<const uint32_t> {sh.GetVs().gs_user_sgpr.value},
	                                                          std::span<const uint32_t> {sh.GetPs().ps_user_sgpr.value}};
	// (mdraw C0/C1, KYTY_MDRAW_STATS / KYTY_MDRAW_SKIP; local/live-census.h LiveCensus::Mdraw; off: one relaxed load)
	// Is this call a member of a multi-draw group with the table call before it: the tests known before any
	// preparation (the image set is compared at the end, C0 only). C1: a member is neither prepared nor emitted.
	const bool     mdraw       = LiveCensus::Mdraw::Active() && Role() == 0 && Spec::Current() == nullptr;
	const uint64_t mdraw_start = mdraw ? (LiveCensus::Mdraw::t_entry != 0 ? LiveCensus::Mdraw::t_entry : __rdtsc()) : 0;
	uint64_t       mdraw_emit  = 0;
	uint32_t       mdraw_break = LiveCensus::Mdraw::Breaks; // (Breaks: a member)
	uint32_t       mdraw_ud    = 0;                         // a member's per-draw words of the vertex stage that differ
	if (mdraw) {
		LiveCensus::Mdraw::t_entry = 0;
		auto&       last   = LiveCensus::Mdraw::g_last;
		const auto& vertex = *pair.programs[0];
		uint32_t    per_draw = 0; // (the vertex stage's user data words a multi-draw would pass per draw)
		const auto  allow    = [&](int64_t index) {
            if (index >= 0 && index < 32) per_draw |= 1u << static_cast<uint32_t>(index);
		};
		allow(pair.start_instance);
		if (vertex.info.vertex_offset_sgpr >= 0) allow(int64_t {vertex.info.vertex_offset_sgpr} - int64_t {vertex.user_data_base});
		if (vertex.info.instance_offset_sgpr >= 0)
			allow(int64_t {vertex.info.instance_offset_sgpr} - int64_t {vertex.user_data_base});
		const uint64_t command = buffer.IsInvalid() ? uint64_t {0}
		                                            : reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(buffer.RawHandle()));
		if (direct != nullptr || commands.empty() || index_base == 0) {
			mdraw_break = LiveCensus::Mdraw::BDirect;
		} else if (!last.valid) {
			mdraw_break = LiveCensus::Mdraw::BFirst;
		} else if (command == 0 || command != last.command || buffer.GraphicsGeneration() != last.generation) {
			mdraw_break = LiveCensus::Mdraw::BCommand;
		} else if (LiveCounters::Value(LiveCounters::RenderPasses) != last.passes || !buffer.RenderingWith(draw_state.rendering) ||
		           buffer.InCopyRun() || buffer.ComputeChainPending()) {
			mdraw_break = LiveCensus::Mdraw::BPass;
		} else if (reinterpret_cast<uint64_t>(&pair) != last.pair) {
			mdraw_break = LiveCensus::Mdraw::BPair;
		} else if (reinterpret_cast<uint64_t>(&variant) != last.variant || reinterpret_cast<uint64_t>(&draw_state) != last.state) {
			mdraw_break = LiveCensus::Mdraw::BState;
		} else if (last.writes) {
			mdraw_break = LiveCensus::Mdraw::BWrites;
		} else if (index_base != last.index_base || index_bytes != last.index_bytes ||
		           static_cast<uint32_t>(index_type) != last.index_type) {
			mdraw_break = LiveCensus::Mdraw::BIndex;
		} else if (HwScene::Active() || Render1::Mode() == 2 || PortR::Verifying()) {
			mdraw_break = LiveCensus::Mdraw::BExcluded; // (paths that capture or batch each draw: never skipped)
		} else {
			std::array<uint32_t, 2> other {};
			for (uint32_t st = 0; st < 2; ++st)
				for (uint32_t i = 0; i < 32; ++i) {
					const uint32_t now = st < stages && i < user_data[st].size() ? user_data[st][i] : 0u;
					if (now == last.ud[st][i]) continue;
					if (st == 0 && ((per_draw >> i) & 1u) != 0) mdraw_ud |= 1u << i;
					else other[st] |= 1u << i;
				}
			if (other[0] != 0 || other[1] != 0) {
				mdraw_break = other[0] != 0 ? LiveCensus::Mdraw::BUdVs : LiveCensus::Mdraw::BUdPs;
				for (uint32_t st = 0; st < 2; ++st)
					for (uint32_t i = 0; i < 32; ++i)
						if (((other[st] >> i) & 1u) != 0) ++LiveCensus::Mdraw::g_c.ud_break[st][i];
			}
		}
		if (mdraw_break == LiveCensus::Mdraw::Breaks && LiveCensus::Mdraw::SkipNow()) {
			// (C1) Neither prepared nor emitted. What a later table draw reads of this one stays the group's first
			// call's (cache.emit, table.last_*: the same pair, variant, state, pipeline, index buffer and command
			// buffer, which the tests above proved); the comparison base takes this call's user data and arguments.
			auto& c = LiveCensus::Mdraw::g_c;
			++c.skipped;
			c.skipped_draws += commands.size();
			++c.members;
			c.member_draws += commands.size();
			++LiveCensus::Mdraw::g_group_calls;
			for (uint32_t st = 0; st < 2; ++st)
				for (uint32_t i = 0; i < 32; ++i)
					last.ud[st][i] = st < stages && i < user_data[st].size() ? user_data[st][i] : 0u;
			last.last_arg   = commands.back();
			last.args_known = false;
			return TableResult::Drawn;
		}
		last.valid = false; // (valid again only once this call is drawn: a refusal below leaves no group behind)
	}
	// (do2-rapido, KYTY_DO2_RAPIDO; local/do2-rapido.h) A direct draw right after a direct table draw of the same pair,
	// variant, state, set and pipeline, with the same user data, in the same command buffer, rendering and pass, no
	// image / buffer registration / mapping change and no packet but registers in between: what TableEvaluate,
	// TableGather, TableSet, the BDA and the blocks would make is the last draw's (the same inputs), still bound; only
	// the indices, count, instances and vertex offset are this draw's. Off: one static load.
	const bool do2 = Do2Rapido::Mode() != 0 && Role() == 0 && Spec::Current() == nullptr;
	if (do2) {
		auto&      s      = Do2Rapido::g_snap;
		const bool was    = s.valid;
		s.valid           = false; // (valid again only once this call is drawn)
		uint32_t   why    = Do2Rapido::WCount;
		auto&      bcache = m_context.GetBufferCache();
		auto&      emit   = cache.emit;
		const auto vk_now = buffer.IsInvalid() ? uint64_t {0}
		                                       : reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(buffer.RawHandle()));
		if (direct == nullptr) {
			// (not a direct draw: not counted)
		} else if (!was || !direct->indexed) {
			why = Do2Rapido::WNoSnap;
		} else if (Do2Rapido::t_seq != s.seq + 1) {
			why = Do2Rapido::WSeq;
		} else if (s.pair != &pair || s.variant != &variant || s.state != &draw_state || s.programs[0] != pair.programs[0] ||
		           s.programs[1] != pair.programs[1] || s.frame != m_context.FrameNumber() ||
		           s.shader_gen != ShaderMapGeneration() || s.index_type != static_cast<uint32_t>(index_type) ||
		           s.pipeline != variant.pipeline || table.last_pair != &pair || table.last_state != &draw_state ||
		           table.last_set != s.set || table.last_variant != static_cast<size_t>(&variant - pair.variants.data())) {
			why = Do2Rapido::WKey;
		} else if (vk_now == 0 || vk_now != s.command || buffer.GraphicsGeneration() != s.generation ||
		           LiveCounters::Value(LiveCounters::RenderPasses) != s.passes || !buffer.RenderingWith(draw_state.rendering) ||
		           buffer.InCopyRun() || buffer.ComputeChainPending() ||
		           reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(emit.command)) != vk_now ||
		           emit.generation != s.generation || emit.state != &draw_state || emit.pipeline != s.pipeline ||
		           emit.index_base != s.ib_base || emit.index_bytes != s.ib_bytes || emit.index_type != index_type) {
			why = Do2Rapido::WCommand;
		} else if (m_context.GetTextureCache().ImagesEpoch() != s.images_epoch || table.images_epoch != s.images_epoch ||
		           bcache.RegistrationEpoch() != s.registration || m_context.GetGpuResources().MappingEpoch() != s.mapping) {
			why = Do2Rapido::WEpoch;
		} else if (HwScene::Active() || Render1::Mode() != 0 || PortR::Verifying() || LiveCensus::Mdraw::Active() ||
		           TrustedDepth::Mode() != 0 || TrustedDepth::StatsOn() || NativeRender::NoteTargetsWanted() ||
		           (kyty_local_native_xpr_diag.load(std::memory_order_relaxed) & 16u) != 0) {
			why = Do2Rapido::WExcluded;
		} else {
			bool same_ud = true;
			for (uint32_t st = 0; st < 2 && same_ud; ++st) {
				const auto size = st < stages ? static_cast<uint32_t>(user_data[st].size()) : 0u;
				same_ud = size == s.ud_size[st] && size <= 32 &&
				          (size == 0 || std::memcmp(user_data[st].data(), s.ud[st].data(), size * sizeof(uint32_t)) == 0);
			}
			if (!same_ud) {
				why = Do2Rapido::WUd;
			} else if (s.ib_buffer == nullptr || index_base == 0 || index_bytes == 0 || index_base < s.ib_begin ||
			           index_base + index_bytes > s.ib_end || bcache.RegistrationEpoch() != s.ib_registration ||
			           s.ib_cpu_epoch == 0 || bcache.CpuModificationEpoch(s.ib_begin, s.ib_end - s.ib_begin) != s.ib_cpu_epoch) {
				why = Do2Rapido::WIndex;
			} else if (!Do2Rapido::Active()) {
				Do2Rapido::g_would.fetch_add(1, std::memory_order_relaxed);
			} else {
				// The fast draw: the last draw's bindings stand (pipeline, set, targets, images in their layouts).
				const auto  vk_buffer = buffer.Handle();
				auto*       ib        = static_cast<Buffer*>(s.ib_buffer);
				const auto* bound     = static_cast<const PipelineCache::Pipeline*>(s.pipeline);
				if (index_base != emit.index_base || index_bytes != emit.index_bytes) {
					vk_buffer.bindIndexBuffer(ib->Handle(), ib->Offset(index_base), index_type);
					emit.index_base  = index_base;
					emit.index_bytes = index_bytes;
				}
				if (s.push_dwords != 0)
					vk_buffer.pushConstants(bound->pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
					                        0, s.push_dwords * sizeof(uint32_t), s.push.data());
				// The game-side hooks the normal path calls with the (same) blocks.
				const std::span<const uint32_t> slots(table.blocks[0].data(), pair.programs[0]->table_plan.slots.size());
				if (Render5Flags::Capturing()) FrameGen::OnDraw(pair.programs[0]->shader_hash, slots, 't');
				if (PathTrace::Requested()) PathTrace::OnDraw(pair.programs[0]->shader_hash, slots);
				if (Render5Flags::Wanted(Render5Flags::Ssr)) FrameGen::OnSsrDraw(pair.programs[0]->shader_hash, slots);
				if (Render5Flags::Wanted(Render5Flags::Contact)) FrameGen::OnShadowDraw(pair.programs[0]->shader_hash, slots, false);
				if (Render5Flags::Wanted(Render5Flags::RtSun) && draw_state.targets.size() == 1) {
					const auto& t = draw_state.targets[0];
					const auto& r = t.requested;
					if (r.IsDepth() && r.resources.layers >= 4 && r.extent.width == r.extent.height && r.extent.width >= 512)
						FrameGen::OnSunCascadeDraw(pair.programs[0]->shader_hash, slots, t.range.base_layer, r.data.address);
				}
				// This draw's offsets, as the normal path's direct draw gives them.
				const auto& vertex = *pair.programs[0];
				const auto  word   = [&](int32_t sgpr) {
					const auto index = int64_t {sgpr} - vertex.user_data_base;
					return index >= 0 && index < static_cast<int64_t>(user_data[0].size()) ? user_data[0][index] : 0u;
				};
				const uint32_t index_offset   = buffer.GetUserConfig().GetIndexOffset();
				const uint32_t vertex_offset  = variant.fetch_embedded && index_offset == 0 ? word(vertex.info.vertex_offset_sgpr)
				                                                                            : index_offset;
				const uint32_t first_instance = variant.fetch_embedded ? word(vertex.info.instance_offset_sgpr) : 0u;
				vk_buffer.drawIndexed(direct->index_count, direct->instance_count, 0, static_cast<int32_t>(vertex_offset),
				                      first_instance);
				if (Render5Flags::Wanted(Render5Flags::ParticlesDebug) && !draw_state.targets.empty())
					FrameGen::NoteOtherPathDraw(draw_state.targets[0].requested.data.address, 0, false, 1);
				if (Render5Flags::Capturing()) FrameGen::AfterDraw(buffer);
				draw_state.Use(m_context.FrameNumber());
				LiveCounters::Add(LiveCounters::TableDraws, 1);
				s.ib_base  = emit.index_base;
				s.ib_bytes = emit.index_bytes;
				s.seq      = Do2Rapido::t_seq;
				s.valid    = true;
				Do2Rapido::g_taken.fetch_add(1, std::memory_order_relaxed);
				return TableResult::Drawn;
			}
		}
		if (why != Do2Rapido::WCount) {
			Do2Rapido::g_fallback.fetch_add(1, std::memory_order_relaxed);
			Do2Rapido::g_why[why].fetch_add(1, std::memory_order_relaxed);
		}
	}
	// (port-r, KYTY_PORT_R=2; local/port-r.h) A draw of the game's geometry list run by the host (eboot+0xab57d0): its
	// native description (found by the table address its user data carries) is compared with what this draw translates
	// to, after the evaluation (PortR::End below); the clock takes the whole step, every return. Off: one load.
	PortR::DrawClock port_r_clock(PortR::Verifying() && Role() == 0 && Spec::Current() == nullptr &&
	                              PortR::Begin(user_data[0], stages > 1 ? user_data[1] : std::span<const uint32_t> {}));
	// (The index buffer's held entry, a cold line of a 320 KiB table, loads while the tables are evaluated.)
	if (index_base != 0) {
		const auto* held = &table.indices[(index_base * 0x9e3779b97f4a7c15ull) >> 51u];
		__builtin_prefetch(held);
		__builtin_prefetch(reinterpret_cast<const char*>(held + 1) - 1);
	}
	// (trust1) KYTY_DS_TRUSTED_DEPTH / KYTY_RENDER_STATS (local/trusted-depth.h): an XPR depth draw's key (the pair and
	// its programs, the draw state, both stages' user data), its entry in the trusted registry (=1 drawn from it, =2
	// compared with the normal path's result) and its reads, captured while TableEvaluate makes them.
	// (draw1) KYTY_DRAW1: the same registry proven by the runs' words read again (trusted-depth.h), for XPR runs into
	// any target; the measurement (KYTY_RENDER_STATS) stays the depth draws'.
	const int  trust_mode  = TrustedDepth::Mode();
	const bool trust_stats = TrustedDepth::StatsOn();
	const bool trust_bytes = TrustedDepth::ByteProof();
	const bool depth_only  = draw_state.targets.size() == 1 && draw_state.targets[0].requested.IsDepth();
	const bool trust_depth = (trust_mode != 0 || trust_stats) && direct == nullptr && !commands.empty() && Role() == 0 &&
	                         Spec::Current() == nullptr && (depth_only || (trust_mode != 0 && trust_bytes));
	bool                 trust_pair = false, trust_hit = false, trust_sample = false;
	uint64_t             trust_key = 0, trust_mapping = 0, trust_start = 0;
	TrustedDepth::Entry* trust_entry = nullptr;
	// (reuso1) KYTY_REUSO1: the preparation's cycles (key, proof or evaluation+gather, set, BDA, blocks) of the draws
	// the registry follows, hit or not: the [reuso1] line's saving per hit is their difference.
	const uint64_t reuso_start = TrustedDepth::Reuso1Mode() != 0 && trust_depth ? __rdtsc() : 0;
	// (draw1, KYTY_DRAW1 with KYTY_RENDER_STATS) A render list's direct draw's key, for its repetition (measurement only).
	if (direct != nullptr && trust_bytes && trust_stats && Role() == 0 && Spec::Current() == nullptr) {
		const std::array<uint64_t, 3> head {reinterpret_cast<uint64_t>(&pair), variant.state, reinterpret_cast<uint64_t>(&draw_state)};
		uint64_t key = XXH3_64bits_withSeed(user_data[0].data(), user_data[0].size_bytes(), XXH3_64bits(head.data(), sizeof(head)));
		if (stages > 1) key = XXH3_64bits_withSeed(user_data[1].data(), user_data[1].size_bytes(), key);
		TrustedDepth::NoteKindKey(TrustedDepth::KindList, key);
	}
	if (trust_depth) {
		trust_pair = true;
		for (uint32_t st = 0; st < stages && trust_pair; ++st) {
			const auto& program = *pair.programs[st];
			if (!program.table_plan.indirect.empty()) trust_pair = false;
			for (const auto& resource: program.info.buffers)
				if (resource.written || resource.atomic) trust_pair = false;
		}
		const std::array<uint64_t, 6> head {reinterpret_cast<uint64_t>(&pair), reinterpret_cast<uint64_t>(pair.programs[0]),
		                                    reinterpret_cast<uint64_t>(pair.programs[1]), variant.state,
		                                    reinterpret_cast<uint64_t>(&draw_state),
		                                    user_data[0].size() | uint64_t {stages > 1 ? user_data[1].size() : 0} << 32u};
		trust_key = XXH3_64bits_withSeed(user_data[0].data(), user_data[0].size_bytes(), XXH3_64bits(head.data(), sizeof(head)));
		if (stages > 1) trust_key = XXH3_64bits_withSeed(user_data[1].data(), user_data[1].size_bytes(), trust_key);
		if (trust_stats && depth_only) TrustedDepth::NoteKey(trust_key);
		if (trust_stats && !depth_only) TrustedDepth::NoteKindKey(TrustedDepth::KindXprColor, trust_key); // (draw1)
		trust_sample = trust_stats && depth_only && TrustedDepth::Sample();
		if (trust_mode != 0 && !trust_pair) ++TrustedDepth::g_counts[TrustedDepth::CPairIneligible];
		if (trust_mode != 0 && trust_pair) {
			++TrustedDepth::g_counts[TrustedDepth::CEligible];
			if (!DsTrustFeed::Live() && !trust_bytes) {
				++TrustedDepth::g_counts[TrustedDepth::CNoFeed];
			} else if (auto& unstable = TrustedDepth::Entries()[TrustedDepth::EntryIndex(trust_key)];
			           trust_bytes && unstable.changed >= 2 && unstable.key == trust_key && unstable.forget == TrustedDepth::g_forget) {
				// (draw1) A key whose words keep changing (memory rewritten between its draws): neither proven nor stored
				// again, nothing captured; its draws go the normal way.
				++TrustedDepth::g_counts[TrustedDepth::CUnstable];
			} else {
				// (Before any read: what a later proof compares with.)
				trust_mapping = m_context.GetGpuResources().MappingEpoch();
				trust_start   = m_context.GetTextureCache().StartEpoch();
				trust_entry   = &unstable;
				auto& entry   = *trust_entry;
				bool  valid   = entry.valid && entry.key == trust_key && entry.forget == TrustedDepth::g_forget && entry.pair == &pair &&
				             entry.programs[0] == pair.programs[0] && entry.programs[1] == pair.programs[1] &&
				             entry.state_key == variant.state && entry.draw_state == &draw_state;
				for (uint32_t st = 0; st < stages && valid; ++st)
					valid = entry.user[st].size() == user_data[st].size() &&
					        std::equal(user_data[st].begin(), user_data[st].end(), entry.user[st].begin());
				if (!valid) {
					if (entry.valid) ++TrustedDepth::g_counts[TrustedDepth::CInvalidKey];
					entry.valid = false;
				} else {
					if (trust_bytes) {
						// (draw1) The mappings and image starts first (the runs' pages as mapped as when they were read),
						// then the runs' words, read as TableEvaluate reads them (its reader: a page GPU work wrote is read
						// as it would read it), against the entry's: the slots they filled.
						if (entry.mapping != trust_mapping || entry.start_epoch != trust_start) {
							valid = false;
							++TrustedDepth::g_counts[TrustedDepth::CInvalidReg];
						} else {
							// Each run's pages mapped, as TableEvaluate's locate proves them (the same per-page memo, reset
							// with the mapping epoch as there), before anything is read.
							if (const auto epoch = m_context.GetGpuResources().MappingEpoch(); epoch != table.mapping_epoch) {
								table.mapped.fill(UINT64_MAX);
								table.mapping_epoch = epoch;
							}
							for (const auto& run: entry.runs) {
								if (run.count == 0 || !valid) continue;
								for (auto page = run.address >> 14u; page <= (run.address + uint64_t {run.count} * 4u - 1) >> 14u; ++page) {
									auto& seen = table.mapped[page & 255u];
									if (seen == page) continue;
									if (!Libs::LibKernel::Memory::IsFullyMapped(page << 14u, 16384)) {
										valid = false;
										break;
									}
									seen = page;
								}
								for (auto line = run.address & ~63ull; valid && line < run.address + uint64_t {run.count} * 4u; line += 64)
									__builtin_prefetch(reinterpret_cast<const void*>(line));
							}
							NativeXprReader                    reader(m_context.GetGpuResources());
							thread_local std::vector<uint32_t> now;
							for (const auto& run: entry.runs) {
								if (run.count == 0 || !valid) continue;
								const auto& block = entry.block[run.st & 1u];
								if (run.st > 1 || uint64_t {run.first} + run.count > block.size()) {
									valid = false;
									break;
								}
								now.resize(run.count);
								reader.Read(run.address, run.count, now.data());
								if (std::memcmp(now.data(), block.data() + run.first, uint64_t {run.count} * 4u) != 0) {
									// (reuso1) Which words changed, for the next decision (counted only with KYTY_REUSO1).
									if (TrustedDepth::Reuso1Mode() != 0)
										TrustedDepth::NoteWordDiff(now.data(), block.data() + run.first, run.count,
										                           static_cast<uint32_t>(&run - entry.runs.data()));
									valid = false;
									break;
								}
							}
							if (!valid) {
								++TrustedDepth::g_counts[TrustedDepth::CInvalidBytes];
								++entry.changed;
							}
						}
					} else {
						for (const auto& range: entry.ranges)
							if (DsTrustFeed::Generation(range.address, range.size) != range.generation) {
								valid = false;
								break;
							}
						if (!valid) ++TrustedDepth::g_counts[TrustedDepth::CInvalidGen];
					}
					if (valid) {
						const auto& buffers = m_context.GetBufferCache();
						valid = entry.mapping == trust_mapping && entry.start_epoch == trust_start;
						for (const auto& [address, size]: entry.buffers)
							if (valid && buffers.RegisteredSince(address, size, entry.registration)) valid = false;
						if (!valid) {
							++TrustedDepth::g_counts[TrustedDepth::CInvalidReg];
						} else {
							// (The buffers' own tracking too: a write it saw, announced or not, is uploaded by the normal path.)
							for (size_t b = 0; b < entry.buffers.size() && valid; ++b)
								valid = buffers.CpuModificationEpoch(entry.buffers[b].first, entry.buffers[b].second) ==
								        entry.cpu_epochs[b];
							if (!valid) ++TrustedDepth::g_counts[TrustedDepth::CInvalidCpu];
						}
					}
					if (!valid) entry.valid = false;
				}
				// An entry is used (=1) or compared (=2) once a second sighting found the normal path's result the same
				// with no announcement in between (confirmed): a write announced between the read and the generations'
				// snapshot is caught there.
				if (valid && entry.confirmed) {
					if (trust_mode == 1) {
						trust_hit = true;
						for (uint32_t st = 0; st < stages; ++st) {
							table.blocks[st].assign(entry.block[st].begin(), entry.block[st].end());
							table.indirect[st].clear();
						}
						table.words.assign(entry.words.begin(), entry.words.end());
						table.lod_stats.assign(entry.lod.begin(), entry.lod.end());
						table.gathered = nullptr;
						++TrustedDepth::g_counts[TrustedDepth::CHit];
					} else {
						++TrustedDepth::g_counts[TrustedDepth::CWouldHit];
					}
					if (!depth_only) ++TrustedDepth::g_counts[TrustedDepth::CHitColor];
				}
			}
		}
		if (!trust_hit && (trust_entry != nullptr || trust_sample)) {
			TrustedDepth::t_capture.runs.clear();
			TrustedDepth::t_capture.on = true;
		}
	}
	std::optional<LiveCensus::Scope> phase;
	if (!trust_hit) {
		phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 1);
		const bool evaluated          = TableEvaluate(pair, user_data);
		TrustedDepth::t_capture.on = false;
		if (!evaluated)
			return LiveCounters::Add(LiveCounters::TableDrawNative), Spec::Refuse("table: evaluation"), TableResult::Native;
		TableGather(pair, user_data);
		phase.reset();
	}
	if (port_r_clock.on) {
		PortR::Translated port_r;
		port_r.vs            = pair.vs;
		port_r.ps            = pair.ps;
		port_r.state         = variant.state;
		port_r.draw_state    = &draw_state;
		port_r.words         = std::span<const uint32_t> {table.words.data(), table.words.size()};
		port_r.ud_vs         = user_data[0];
		port_r.ud_ps         = stages > 1 ? user_data[1] : std::span<const uint32_t> {};
		port_r.evaluated     = !trust_hit;
		port_r.direct        = direct != nullptr && direct->indexed && direct->instance_count == 1;
		port_r.index_base    = index_base;
		port_r.index_address = direct != nullptr ? direct->index_address : 0;
		port_r.index_count   = direct != nullptr ? direct->index_count : 0;
		port_r.index_size    = index_type == vk::IndexType::eUint16 ? 2u : 4u;
		PortR::End(port_r);
	}
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 2);
	auto* set = TableSet(buffer, pair, &variant, &draw_state);
	if (auto* spec = Spec::Current(); spec != nullptr && set != nullptr)
		spec->NoteSet(reinterpret_cast<uint64_t>(static_cast<VkDescriptorSet>(set->set)));
	if (set == nullptr) return LiveCounters::Add(LiveCounters::TableNoSet), Spec::Refuse("table: no set"), TableResult::Native;

	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 3);
	// Emission, as NativeXprEmit's: indices and arguments are read by the GPU; lookups may restart the
	// scheduler, so the command buffer handle is taken after them.
	auto&              buffer_cache = m_context.GetBufferCache();
	constexpr uint64_t stride       = sizeof(vk::DrawIndexedIndirectCommand);
	// A tracker region's epoch, read once in the draw (its buffers mostly share one: the line is another core's).
	std::array<std::pair<uint64_t, uint64_t>, 4> region_epochs;
	region_epochs.fill({UINT64_MAX, 0});
	const auto cpu_epoch = [&](uint64_t address, uint64_t size) {
		const auto region = address / TRACKER_REGION_SIZE;
		if (region != (address + size - 1) / TRACKER_REGION_SIZE) return buffer_cache.CpuModificationEpoch(address, size);
		auto& known = region_epochs[region & 3u];
		if (known.first != region) known = {region, buffer_cache.CpuModificationEpoch(address, size)};
		return known.second;
	};
	const auto obtain = [&](TableXpr::HeldBuffer& held, uint64_t address, uint64_t size,
	                        auto&& prepare) -> std::pair<Buffer*, uint64_t> {
		const auto* spec = Spec::Current(); // (its guest memory writes wait for its commit: ObtainBuffer decides)
		if (held.buffer != nullptr && address >= held.begin && address + size <= held.end &&
		    held.registration == buffer_cache.RegistrationEpoch() && held.cpu_epoch != 0 &&
		    cpu_epoch(held.begin, held.end - held.begin) == held.cpu_epoch &&
		    (spec == nullptr || !spec->JournalIntersects(address, size))) {
			if (auto* noting = Spec::Current()) noting->NoteBufferRange(address, size);
			return {held.buffer, held.buffer->Offset(address)};
		}
		held = {};
		LiveCensus::Scope miss(LiveCensus::DrawPhase, 0x7ab1e, &held == &table.args ? 10 : 11);
		prepare();
		const auto result = buffer_cache.ObtainBuffer(address, size, false);
		region_epochs.fill({UINT64_MAX, 0}); // (its uploads may have moved them)
		const auto owner  = Spec::Current() != nullptr ? buffer_cache.RegisteredBuffer(address, size)
		                                               : buffer_cache.FindBuffer(address, size);
		if (result.first != nullptr && buffer_cache.GetBufferIfLive(owner) == result.first) {
			// The buffer within the tracker region of the address (one region's epoch proves it) when none of it is
			// CPU-dirty, else the range. (The epoch before the dirty check: a write after it moves the epoch.)
			const auto region = address & ~(TRACKER_REGION_SIZE - 1);
			auto       begin  = std::max(result.first->CpuAddress(), region);
			auto       end    = std::min(result.first->CpuAddress() + result.first->Size(), region + TRACKER_REGION_SIZE);
			if (address + size <= end) {
				const auto epoch = buffer_cache.CpuModificationEpoch(begin, end - begin);
				if (buffer_cache.IsRegionCpuModified(begin, end - begin)) {
					begin = address;
					end   = address + size;
				}
				if (epoch != 0 && !buffer_cache.IsRegionCpuDirty(begin, end - begin))
					held = {begin, end, epoch, buffer_cache.RegistrationEpoch(), result.first};
			}
		}
		return result;
	};
	std::array<vk::DeviceAddress, 2> blocks {};
	bool                             writes = false;
	// (Before the blocks, which it does not depend on (the slots TableEvaluate read): the entries TableGather asked
	// for meanwhile arrive.)
	{
		LiveCensus::StatScope stat_bda(0, 12); // (render3, KYTY_RENDER_STATS: the BDA preparation and its uploads)
		TablePrepareBda(pair, user_data);
	}
	// The blocks take each buffer's device address as it resolves: a later lookup that registers a buffer (a merge
	// replaces an earlier one) leaves an earlier address stale, where the normal path binds after all its lookups
	// (RebindBuffers). Such a draw takes the normal path; the next use finds the buffers registered.
	const auto registration = buffer_cache.RegistrationEpoch();
	bool       blocks_made  = false;
	{
		LiveCensus::StatScope stat_blocks(0, 13); // (render3, KYTY_RENDER_STATS: the blocks)
		if (trust_hit) {
			// (trust1, KYTY_DS_TRUSTED_DEPTH=1) The entry's blocks into the ring, as TableBlocks copies its own.
			auto& upload = buffer_cache.GetTableUploadBuffer();
			for (uint32_t st = 0; st < stages; ++st)
				if (pair.programs[st]->table_plan.BlockDwords() != 0)
					blocks[st] = upload.BufferDeviceAddress() +
					             upload.Copy(table.blocks[st].data(), table.blocks[st].size() * sizeof(uint32_t), 16);
			blocks_made = true;
		} else {
			blocks_made = TableBlocks(pair, user_data, blocks, &writes);
		}
	}
	if (!blocks_made)
		return LiveCounters::Add(LiveCounters::TableDrawNative), Spec::Refuse("table: blocks"), TableResult::Native;
	if (reuso_start != 0 && trust_entry != nullptr) TrustedDepth::NotePrep(trust_hit, __rdtsc() - reuso_start);
	// (trust1) The normal path's result of an XPR depth draw the registry follows (or a sampled one): its reads
	// classified (KYTY_RENDER_STATS), compared with a confirmed entry (=2) or one to confirm, and stored.
	if (trust_depth && !trust_hit && (trust_entry != nullptr || trust_sample)) {
		const auto&                                    captured = TrustedDepth::t_capture.runs;
		thread_local std::vector<TrustedDepth::Range>                ranges;
		thread_local std::vector<std::array<uint32_t, 2>> buffer_dwords;
		ranges.clear();
		buffer_dwords.clear();
		bool ring = false; // (a V# read from a copy in the ring, or not held: its device address is this draw's only)
		for (const auto& run: captured) ranges.push_back({run.address, uint64_t {run.count} * 4u, 0});
		const size_t slot_ranges = ranges.size();
		for (uint32_t st = 0; st < stages; ++st) {
			const auto& program = *pair.programs[st];
			const auto& plan    = program.table_plan;
			for (size_t i = 0; i < plan.buffers.size() && i < table.entries[st].size(); ++i) {
				const auto& resource = program.info.buffers[i];
				if (resource.written || resource.atomic) continue;
				const auto& [words4, held] = table.entries[st][i];
				const auto dword           = plan.BufferDword(i);
				buffer_dwords.push_back({st, dword});
				const vk::DeviceAddress device = table.blocks[st][dword] | uint64_t {table.blocks[st][dword + 1]} << 32u;
				if (held == nullptr || held->words != words4 || (held->registration & TableXpr::RingCopyMark) != 0 ||
				    held->device != device) {
					ring = true;
					continue;
				}
				if (held->size != 0) ranges.push_back({held->address, held->size, 0});
			}
		}
		uint32_t     bad        = TrustedDepth::ReadStatic;
		bool         all_static = true;
		TrustedDepth::ShaderReads* reads      = trust_sample ? TrustedDepth::ReadsOf(pair.programs[0]->shader_hash) : nullptr;
		const auto   classify   = [&] {
			for (size_t r = 0; r < ranges.size(); ++r) {
				const uint32_t read = TrustedDepth::Classify(ranges[r].address, ranges[r].size);
				if (reads != nullptr) reads->bytes[r < slot_ranges ? 0 : 1][read] += ranges[r].size;
				if (read != TrustedDepth::ReadStatic && all_static) {
					all_static = false;
					bad        = read;
				}
			}
		};
		bool classified = false;
		if (reads != nullptr) {
			classify();
			classified = true;
			reads->calls += 1;
			if (all_static && !ring) reads->all_static += 1;
		}
		if (trust_entry != nullptr) {
			auto&          entry  = *trust_entry;
			const uint64_t shader = pair.programs[0]->shader_hash;
			const bool     same_key = entry.valid && entry.key == trust_key && entry.forget == TrustedDepth::g_forget;
			// The normal path's result against the entry's: =2 counts and lists what differs in a confirmed entry; an
			// unconfirmed one is confirmed when nothing does.
			bool differs = false;
			if (same_key) {
				const bool report = entry.confirmed && trust_mode == 2;
				const auto note   = [&](uint32_t st, char what, uint32_t index, uint32_t trusted, uint32_t fresh) {
					differs = true;
					if (report) TrustedDepth::NoteMismatch(shader, st, what, index, trusted, fresh);
				};
				if (entry.runs.size() != captured.size()) {
					note(0, 'n', 0, static_cast<uint32_t>(entry.runs.size()), static_cast<uint32_t>(captured.size()));
				} else {
					for (size_t k = 0; k < captured.size(); ++k) {
						const auto& a = entry.runs[k];
						const auto& b = captured[k];
						if (a.st != b.st || a.first != b.first || a.count != b.count || a.address != b.address)
							note(b.st, 'a', static_cast<uint32_t>(k), static_cast<uint32_t>(a.address), static_cast<uint32_t>(b.address));
						for (uint32_t w = b.first; w < b.first + b.count; ++w)
							if (w < entry.block[b.st].size() && w < table.blocks[b.st].size() &&
							    entry.block[b.st][w] != table.blocks[b.st][w])
								note(b.st, 's', w, entry.block[b.st][w], table.blocks[b.st][w]);
					}
				}
				if (entry.buffer_dwords != buffer_dwords) {
					note(0, 'n', 1, static_cast<uint32_t>(entry.buffer_dwords.size()), static_cast<uint32_t>(buffer_dwords.size()));
				} else {
					for (const auto& [st, dword]: buffer_dwords)
						for (uint32_t w = dword; w < dword + 5; ++w)
							if (w < entry.block[st].size() && w < table.blocks[st].size() && entry.block[st][w] != table.blocks[st][w])
								note(st, 'b', w, entry.block[st][w], table.blocks[st][w]);
				}
				if (entry.words.size() != table.words.size()) {
					note(0, 'n', 2, static_cast<uint32_t>(entry.words.size()), static_cast<uint32_t>(table.words.size()));
				} else {
					for (size_t w = 0; w < table.words.size(); ++w)
						if (entry.words[w] != table.words[w]) note(0, 'w', static_cast<uint32_t>(w), entry.words[w], table.words[w]);
				}
				if (entry.lod.size() != table.lod_stats.size()) {
					note(1, 'n', 3, static_cast<uint32_t>(entry.lod.size()), static_cast<uint32_t>(table.lod_stats.size()));
				} else {
					for (size_t w = 0; w < table.lod_stats.size(); ++w)
						if (entry.lod[w] != table.lod_stats[w]) note(1, 'l', static_cast<uint32_t>(w), entry.lod[w], table.lod_stats[w]);
				}
				if (differs && report) ++TrustedDepth::g_counts[TrustedDepth::CMismatchCalls];
				if (!differs && !entry.confirmed) entry.confirmed = true;
			}
			if (same_key && !differs) {
				// (Confirmed, or compared clean: kept as it is, with its own proofs.)
			} else if (!same_key && (entry.key != trust_key || entry.forget != TrustedDepth::g_forget || entry.sightings == 0)) {
				// First sighting of the key here: only noted (most keys are never seen again: nothing to store).
				entry           = {};
				entry.key       = trust_key;
				entry.forget    = TrustedDepth::g_forget;
				entry.sightings = 1;
				++TrustedDepth::g_counts[TrustedDepth::CFirstSight];
			} else {
				if (!classified && !trust_bytes) classify(); // (draw1: the words are the proof, not the memory's class)
				entry.valid     = false;
				entry.confirmed = false;
				entry.key       = trust_key;
				entry.forget    = TrustedDepth::g_forget;
				++entry.sightings;
				if (ring) {
					++TrustedDepth::g_counts[TrustedDepth::CRefuseRing];
				} else if (ranges.size() > TrustedDepth::MaxRanges) {
					++TrustedDepth::g_counts[TrustedDepth::CRefuseRanges];
				} else if (!all_static && !trust_bytes) {
					++TrustedDepth::g_counts[TrustedDepth::CRefuseClass];
					++TrustedDepth::g_refused_class[bad];
				} else if (std::any_of(ranges.begin() + static_cast<std::ptrdiff_t>(slot_ranges), ranges.end(), [&](const auto& range) {
					           return buffer_cache.CpuModificationEpoch(range.address, range.size) == 0 ||
					                  buffer_cache.IsRegionCpuModified(range.address, range.size);
				           })) {
					++TrustedDepth::g_counts[TrustedDepth::CRefuseDirty];
				} else {
					// (Generations now, after the reads: a write announced in between is caught by the confirmation.)
					if (!trust_bytes)
						for (auto& range: ranges) range.generation = DsTrustFeed::Generation(range.address, range.size);
					entry.pair       = &pair;
					entry.programs   = {pair.programs[0], pair.programs[1]};
					entry.state_key  = variant.state;
					entry.draw_state = &draw_state;
					for (uint32_t st = 0; st < 2; ++st) {
						if (st < stages) {
							entry.user[st].assign(user_data[st].begin(), user_data[st].end());
							entry.block[st].assign(table.blocks[st].begin(), table.blocks[st].end());
						} else {
							entry.user[st].clear();
							entry.block[st].clear();
						}
					}
					entry.words.assign(table.words.begin(), table.words.end());
					entry.lod.assign(table.lod_stats.begin(), table.lod_stats.end());
					entry.runs.assign(captured.begin(), captured.end());
					entry.buffer_dwords.assign(buffer_dwords.begin(), buffer_dwords.end());
					entry.ranges.assign(ranges.begin(), ranges.end());
					entry.buffers.clear();
					entry.cpu_epochs.clear();
					for (size_t r = slot_ranges; r < ranges.size(); ++r) {
						entry.buffers.emplace_back(ranges[r].address, ranges[r].size);
						entry.cpu_epochs.push_back(buffer_cache.CpuModificationEpoch(ranges[r].address, ranges[r].size));
					}
					entry.registration = registration; // (before TableBlocks' lookups)
					entry.mapping      = trust_mapping;
					entry.start_epoch  = trust_start;
					entry.valid        = true;
					++TrustedDepth::g_counts[TrustedDepth::CStored];
				}
			}
		}
	}
	// Frame generation's camera (FrameGen::OnDraw: the G-buffer vertex shaders' view-projection, previous one and
	// jitter), from the block's slots: dword s of the vertex stage's block is flattened SRT slot s, as the normal path's
	// flattened_srt. Without it every frame of a session reported "camera default" (the G-buffer draws take this path).
	// (KYTY_HWRT: the camera from any G-buffer draw of the frame, captured or not)
	if (HwScene::Active() && Spec::Current() == nullptr)
		HwScene::NoteCamera(pair.programs[0]->shader_hash,
		                    std::span<const uint32_t>(table.blocks[0].data(), pair.programs[0]->table_plan.slots.size()));
	// (render5: KYTY_DRAW_FLAGS, Render5Flags in native-xpr.inc, for the five questions below and the two after the draw)
	if (Render5Flags::Capturing() && Spec::Current() == nullptr)
		FrameGen::OnDraw(pair.programs[0]->shader_hash,
		                 std::span<const uint32_t>(table.blocks[0].data(), pair.programs[0]->table_plan.slots.size()), 't');
	// (pt1) BB_PATHTRACING: the sun's light matrix from the draws inside a RenderShadowCascade scope
	if (PathTrace::Requested() && Spec::Current() == nullptr)
		PathTrace::OnDraw(pair.programs[0]->shader_hash,
		                  std::span<const uint32_t>(table.blocks[0].data(), pair.programs[0]->table_plan.slots.size()));
	if (Render5Flags::Wanted(Render5Flags::Ssr) && Spec::Current() == nullptr)
		FrameGen::OnSsrDraw(pair.programs[0]->shader_hash,
		                    std::span<const uint32_t>(table.blocks[0].data(), pair.programs[0]->table_plan.slots.size()));
	if (Render5Flags::Wanted(Render5Flags::Contact) && Spec::Current() == nullptr)
		FrameGen::OnShadowDraw(pair.programs[0]->shader_hash,
		                       std::span<const uint32_t>(table.blocks[0].data(), pair.programs[0]->table_plan.slots.size()), false);
	// (KYTY_RT_SUN: a sun cascade's draw, a depth-only draw into a layer of a square layered depth image: its layer and
	// light matrix)
	if (Render5Flags::Wanted(Render5Flags::RtSun) && Spec::Current() == nullptr && draw_state.targets.size() == 1) {
		const auto& t = draw_state.targets[0];
		const auto& r = t.requested;
		if (r.IsDepth() && r.resources.layers >= 4 && r.extent.width == r.extent.height && r.extent.width >= 512)
			FrameGen::OnSunCascadeDraw(pair.programs[0]->shader_hash,
			                           std::span<const uint32_t>(table.blocks[0].data(), pair.programs[0]->table_plan.slots.size()),
			                           t.range.base_layer, r.data.address);
	}
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 9);
	std::array<std::pair<Buffer*, uint64_t>, 64> arguments;
	std::array<uint32_t, 64>                     group_sizes;
	uint32_t                                     groups = 0;
	for (size_t first = 0; first < commands.size();) {
		size_t count = 1;
		while (first + count < commands.size() && commands[first + count] == commands[first] + count * stride) ++count;
		const auto [owner, offset] = obtain(table.args, commands[first], count * stride, [] {});
		if (owner == nullptr) return TableResult::Native;
		arguments[groups]   = {owner, offset};
		group_sizes[groups] = static_cast<uint32_t>(count);
		++groups;
		first += count;
	}
	// (render1 F2c) What the draw's shaders read of its own depth target (its set's images: the program's bindings).
	if (Render1::Mode() == 2 && Render1::t_current.native) {
		auto& read = Render1::t_read;
		read       = {};
		read.known = true;
		for (const auto& target: draw_state.targets) {
			if (!target.requested.IsDepth()) continue;
			for (const auto& image: set->images) {
				++read.images;
				if (!(image.id == target.id)) continue;
				read.reads_depth = true;
				read.base_layer  = image.range.base_layer;
				read.layers      = image.range.layer_count;
				read.base_level  = image.range.base_level;
				read.levels      = image.range.level_count;
			}
			break;
		}
	}
	// (render1 F2, KYTY_RENDER1=2: a p0/p1 depth draw into the native batch; before anything is bound or compared with
	// what is bound: a new batch starts by copying the depth and invalidates the graphics state)
	const bool render1 = Render1::Mode() == 2 && Render1::t_current.native &&
	                     Render1Begin(buffer, draw_state,
	                                  direct != nullptr || commands.empty() ? Render1::RDirect
	                                  : writes                              ? Render1::RWrites
	                                  : variant.hwrt_scene && HwScene::Active() ? Render1::RHwrt
	                                                                            : Render1::Reasons);
	auto&      emit       = cache.emit;
	const bool indexed    = direct == nullptr || direct->indexed; // (a draw of vertices binds no indices)
	const bool same_index = !indexed || (emit.command == buffer.Handle() && emit.generation == buffer.GraphicsGeneration() &&
	                                     emit.index_base == index_base && emit.index_bytes == index_bytes &&
	                                     emit.index_type == index_type);
	std::pair<Buffer*, uint64_t> indices {};
	if (!same_index) {
		indices = obtain(table.indices[(index_base * 0x9e3779b97f4a7c15ull) >> 51u], index_base, index_bytes, [&] {
			if (index_bytes <= BufferCache::CACHING_PAGESIZE && buffer_cache.IsRegionCpuModified(index_base, index_bytes) &&
			    !buffer_cache.IsRegionGpuModified(index_base, index_bytes))
				buffer_cache.EnsureBufferContents(index_base, index_bytes);
		});
		if (indices.first == nullptr) return TableResult::Native;
	}
	if (buffer_cache.RegistrationEpoch() != registration)
		return LiveCounters::Add(LiveCounters::TableRegistered), TableResult::Native;
	// (A speculation's lookups and preparation refuse without failing: nothing is recorded for a refused draw.)
	if (const auto* spec = Spec::Current(); spec != nullptr && spec->refused != nullptr) return TableResult::Native;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 4);
	if (mdraw) mdraw_emit = __rdtsc(); // (mdraw C0: the emission from here)
	if (buffer.IsInvalid()) return Spec::Refuse("table: buffer"), TableResult::Native;
	const auto vk_buffer    = buffer.Handle();
	const bool same_command = emit.command == vk_buffer && emit.generation == buffer.GraphicsGeneration();
	// The targets and the set's images as the last table draw in this command buffer left them (the same state and
	// set), no image's state, registration, metadata or bytes changed since: in place, nothing to check or transit.
	auto&      texture_cache = m_context.GetTextureCache();
	const auto images_epoch  = texture_cache.ImagesEpoch();
	const bool own           = Spec::Current() == nullptr;
	// (f5b) A verified pass's targets (KYTY_NATIVE_PASSES ..:2; one load otherwise).
	if (own && NativeRender::NoteTargetsWanted()) // (pt1: BB_PATHTRACING's GBuffer / ResolveLighting targets too)
		for (const auto& target: draw_state.targets)
			NativeRender::NoteTarget(target.id.index, target.id.generation, target.range.base_level, target.range.base_layer,
			                         target.requested.IsDepth());
	const bool in_place      = same_command && emit.state == &draw_state && table.last_set == set &&
	                      table.images_epoch == images_epoch && own;
	if (!in_place) {
		// The same for the targets apart, as this draw state's last emission left them, and the set's images, as its
		// last transit under this state did (the passes alternate materials: another set or state in between).
		if ((!own || draw_state.emitted_epoch != images_epoch) && !NativeXprEmitTargets(draw_state, vk_buffer))
			return LiveCounters::Add(LiveCounters::TableDrawNative), Spec::Refuse("table: targets"), TableResult::Native;
		// The set's images in their layouts (a DCC image's pending fast clear applied first).
		if ((!own || set->transit_epoch != images_epoch || set->transit_state != &draw_state) &&
		    !TableTransit(*set, vk_buffer, &draw_state))
			return LiveCounters::Add(LiveCounters::TableDrawNative), Spec::Refuse("table: transit"), TableResult::Native;
		// (Neither moves what the other placed: a target is no image the set holds as a texture.)
		if (own) {
			draw_state.emitted_epoch = set->transit_epoch = texture_cache.ImagesEpoch();
			set->transit_state       = &draw_state;
		}
	}
	m_context.GetCommandScheduler().BeginRendering(draw_state.rendering);
	if (same_command && emit.state != &draw_state && emit.state != nullptr &&
	    emit.state->dynamic_hash == draw_state.dynamic_hash &&
	    std::memcmp(&emit.state->dynamic, &draw_state.dynamic, sizeof(draw_state.dynamic)) == 0)
		emit.state = &draw_state;
	if (!same_command || emit.state != &draw_state) NativeXprEmitDynamic(cache, vk_buffer, draw_state, same_command);
	// KYTY_HWRT (local/hwrt-scene.h): a G-buffer state's draws take its capturing pipeline (same layout), on this
	// render thread only; the first record of each draw call goes in push constant dword HwScene::CursorDword.
	bool hwrt_capture = false;
	// (pt1) BB_PATHTRACING: only inside the game's capture scopes (GBuffer, VirtualGeo: the frame map); CaptureHere is
	// true when it is off.
	if (variant.hwrt_scene && HwScene::Active() && PathTrace::CaptureHere()) {
		const uint32_t draws = direct != nullptr ? 1u : static_cast<uint32_t>(commands.size());
		if (Role() != 0 || Spec::Current() != nullptr) {
			HwScene::NoteSkip(HwScene::Skip::Speculation, draws);
		} else {
			if (variant.hwrt_pipeline == nullptr && variant.hwrt_make) variant.hwrt_pipeline = (*variant.hwrt_make)(buffer);
			if (variant.hwrt_pipeline == nullptr || variant.hwrt_pipeline->pipeline_layout != variant.pipeline->pipeline_layout)
				HwScene::NoteSkip(HwScene::Skip::NoVariant, draws);
			else
				hwrt_capture = true;
		}
	}
	if (hwrt_capture) HwScene::NoteCamera(pair.programs[0]->shader_hash,
	                                      std::span<const uint32_t>(table.blocks[0].data(), pair.programs[0]->table_plan.slots.size()));
	if (hwrt_capture) {
		// (A program's push data reaching the cursor's dword: drawn uncaptured.)
		uint32_t reach = 0;
		for (uint32_t st = 0; st < stages; ++st) {
			const auto& bindings = pair.programs[st]->bindings;
			if (bindings.UsesPushData()) reach = std::max(reach, bindings.push_data_start_dword + bindings.ShaderDataDwords());
		}
		if (reach > HwScene::CursorDword) {
			HwScene::NoteSkip(HwScene::Skip::PushData, direct != nullptr ? 1u : static_cast<uint32_t>(commands.size()));
			hwrt_capture = false;
		}
	}
	// The call's meshes noted before the pipeline is chosen: a call none of whose draws is captured this frame (a cached
	// mesh: most of them) takes the game's own pipeline (10-26: the capture variant on every G-buffer draw, reading its
	// record per vertex, was most of a 6 fps frame).
	uint32_t                 hwrt_direct_cursor = 0;
	std::array<uint32_t, 64> hwrt_cursors {};
	if (hwrt_capture) {
		const uint64_t hwrt_index_size = index_type == vk::IndexType::eUint32 ? 4u : 2u;
		const auto     hwrt_note       = [&](uint32_t record, uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                                    int32_t vertex_offset) -> bool {
			HwScene::Draw d;
			d.vs             = pair.programs[0]->shader_hash;
			d.ps             = variant.hwrt_ps;
			if (HwScene::MaterialActive()) {
				d.ps_module = variant.hwrt_ps_module;
				d.ps_images = &set->hwrt_ps_images;
				d.cache     = &texture_cache;
			} else if (PathTrace::Requested()) {
				d.ps_module = variant.hwrt_ps_module; // (pt1: whether it discards - alpha-tested meshes stay out of the rays)
			}
			d.index_size     = static_cast<uint32_t>(hwrt_index_size);
			d.index_count    = index_count;
			d.instance_count = instance_count;
			d.vertex_offset  = vertex_offset;
			const uint64_t begin = index_base + uint64_t {first_index} * hwrt_index_size;
			if (indexed && buffer_cache.IsRegionGpuModified(begin, uint64_t {index_count} * hwrt_index_size)) {
				HwScene::NoteSkip(HwScene::Skip::GpuWritten);
				HwScene::NoteGpuWrittenKind(false); // (pt1e: the indices)
				return false;
			}
			d.index_address = indexed && index_base != 0 && begin + uint64_t {index_count} * hwrt_index_size <= index_base + index_bytes
			                      ? begin
			                      : 0;
			return HwScene::Note(record, d);
		};
		bool any = false;
		if (direct != nullptr) {
			hwrt_direct_cursor = HwScene::Reserve(1);
			const auto& vertex = *pair.programs[0];
			const auto  word   = [&](int32_t sgpr) {
                const auto index = int64_t {sgpr} - vertex.user_data_base;
                return index >= 0 && index < static_cast<int64_t>(user_data[0].size()) ? user_data[0][index] : 0u;
			};
			const uint32_t index_offset  = buffer.GetUserConfig().GetIndexOffset();
			const uint32_t vertex_offset = variant.fetch_embedded && index_offset == 0 ? word(vertex.info.vertex_offset_sgpr) : index_offset;
			if (indexed) any |= hwrt_note(hwrt_direct_cursor, direct->index_count, direct->instance_count, 0, static_cast<int32_t>(vertex_offset));
			else HwScene::NoteSkip(HwScene::Skip::Topology);
		}
		for (uint32_t group = 0, first_command = 0; group < groups; first_command += group_sizes[group], ++group) {
			// Each indirect draw's arguments as the game's CPU wrote them (guest memory): the mesh's counts and offsets.
			hwrt_cursors[group] = HwScene::Reserve(group_sizes[group]);
			for (uint32_t i = 0; i < group_sizes[group]; ++i) {
				if (buffer_cache.IsRegionGpuModified(commands[first_command + i], stride)) {
					HwScene::NoteSkip(HwScene::Skip::GpuWritten);
					HwScene::NoteGpuWrittenKind(true); // (pt1e: the indirect arguments)
					continue;
				}
				const auto* args = reinterpret_cast<const vk::DrawIndexedIndirectCommand*>(commands[first_command + i]);
				any |= hwrt_note(hwrt_cursors[group] + i, args->indexCount, args->instanceCount, args->firstIndex, args->vertexOffset);
			}
		}
		hwrt_capture = any;
	}
	const auto& pipeline = hwrt_capture ? *variant.hwrt_pipeline : *variant.pipeline;
	const PipelineCache::Pipeline* bound_pipeline = &pipeline;
	if (!same_command || emit.pipeline != &pipeline)
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, BindPipelineHandle(pipeline));
	if (!same_command || emit.pipeline != &pipeline || table.last_set != set)
		vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.pipeline_layout, 0, 1, &set->set, 0,
		                             nullptr);
	// Each stage's user data and the pixel stage's LOD statistics words, where its push data starts.
	ShaderRecompiler::IR::PushData push_data;
	uint32_t                       push_dwords = 0; // the stages' shader data end there: nothing after it is pushed
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program  = *pair.programs[st];
		const auto& bindings = program.bindings;
		if (!bindings.UsesPushData()) continue;
		push_dwords = std::max(push_dwords, bindings.push_data_start_dword + bindings.ShaderDataDwords());
		for (size_t i = 0; i < bindings.user_data_registers.size(); ++i) {
			const auto index = bindings.user_data_registers[i] - program.user_data_base;
			push_data.dwords[bindings.push_data_start_dword + i] = index < user_data[st].size() ? user_data[st][index] : 0;
		}
		if (program.stage == ShaderType::Pixel)
			for (size_t i = 0; i < table.lod_stats.size(); ++i)
				push_data.dwords[bindings.push_data_start_dword + bindings.LodStatsDword() + i] = table.lod_stats[i];
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword()]     = static_cast<uint32_t>(blocks[st]);
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword() + 1] = static_cast<uint32_t>(blocks[st] >> 32u);
	}
	if (push_dwords != 0)
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
		                        push_dwords * sizeof(uint32_t), push_data.dwords.data());
	if (!same_index) vk_buffer.bindIndexBuffer(indices.first->Handle(), indices.second, index_type);
	const auto push_cursor = [&](uint32_t cursor) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
		                        HwScene::CursorDword * sizeof(uint32_t), sizeof(uint32_t), &cursor);
	};
	if (hwrt_capture && direct != nullptr) push_cursor(hwrt_direct_cursor);
	if (direct != nullptr) {
		// This draw's offsets, as the normal path's ResolveVertexOffset/ResolveInstanceOffset give them: the index
		// offset register, else (a vertex program that fetches) the user data its fetch adds.
		const auto& vertex = *pair.programs[0];
		const auto  word   = [&](int32_t sgpr) {
			const auto index = int64_t {sgpr} - vertex.user_data_base;
			return index >= 0 && index < static_cast<int64_t>(user_data[0].size()) ? user_data[0][index] : 0u;
		};
		const uint32_t index_offset   = buffer.GetUserConfig().GetIndexOffset();
		const uint32_t vertex_offset  = variant.fetch_embedded && index_offset == 0 ? word(vertex.info.vertex_offset_sgpr)
		                                                                            : index_offset;
		const uint32_t first_instance = variant.fetch_embedded ? word(vertex.info.instance_offset_sgpr) : 0u;
		if (indexed)
			vk_buffer.drawIndexed(direct->index_count, direct->instance_count, 0, static_cast<int32_t>(vertex_offset),
			                      first_instance);
		else
			vk_buffer.draw(direct->index_count, direct->instance_count, vertex_offset, first_instance);
	}
	for (uint32_t group = 0, first_command = 0; group < groups; first_command += group_sizes[group], ++group) {
		if (hwrt_capture) push_cursor(hwrt_cursors[group]);
		vk_buffer.drawIndexedIndirect(arguments[group].first->Handle(), arguments[group].second, group_sizes[group],
		                              static_cast<uint32_t>(stride));
	}
	if (render1 && !hwrt_capture)
		Render1Add(buffer, draw_state, pipeline, set->set, {}, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
		           {push_data.dwords.data(), push_dwords}, !same_index, !same_index ? indices.first->Handle() : vk::Buffer {},
		           !same_index ? indices.second : 0, index_base, index_bytes, index_type, {arguments.data(), groups},
		           {group_sizes.data(), groups}, commands);
	if (Render5Flags::Wanted(Render5Flags::ParticlesDebug) && !draw_state.targets.empty()) // (rtparticles census: this path's draws by target)
		FrameGen::NoteOtherPathDraw(draw_state.targets[0].requested.data.address, 0, false, 1);
	if (Render5Flags::Capturing()) FrameGen::AfterDraw(buffer); // (frame generation's HUD-less image)
	// As the normal path's: shaders that write buffers end the rendering, and a barrier orders their writes.
	if (writes) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader);
	}
	emit.command     = vk_buffer;
	emit.generation  = buffer.GraphicsGeneration();
	emit.state       = &draw_state;
	emit.pipeline    = bound_pipeline;
	draw_state.Use(m_context.FrameNumber());
	if (indexed) {
		emit.index_base  = index_base;
		emit.index_bytes = index_bytes;
		emit.index_type  = index_type;
	}
	table.last_pair    = &pair;
	table.last_variant = static_cast<size_t>(&variant - pair.variants.data());
	table.last_state   = &draw_state;
	table.last_set     = set;
	table.images_epoch = texture_cache.ImagesEpoch();
	LiveCounters::Add(LiveCounters::TableDraws, std::max<size_t>(commands.size(), 1));
	// (do2-rapido) This direct draw, drawn the normal way, as the base of the next one's fast draw.
	if (do2 && direct != nullptr && direct->indexed && !writes && !hwrt_capture && !render1 && bound_pipeline == variant.pipeline &&
	    push_dwords <= 32 && user_data[0].size() <= 32 && (stages < 2 || user_data[1].size() <= 32)) {
		auto& s = Do2Rapido::g_snap;
		// The index buffer bound: this draw's lookup, else the one the last snapshot held for the same base.
		auto&   held = table.indices[(emit.index_base * 0x9e3779b97f4a7c15ull) >> 51u];
		Buffer* ib   = !same_index ? indices.first
		               : (s.ib_buffer != nullptr && s.ib_base == emit.index_base && s.ib_bytes == emit.index_bytes)
		                   ? static_cast<Buffer*>(s.ib_buffer)
		                   : nullptr;
		if (ib != nullptr && held.buffer == ib && held.cpu_epoch != 0 && emit.index_base >= held.begin &&
		    emit.index_base + emit.index_bytes <= held.end) {
			s.pair         = &pair;
			s.variant      = &variant;
			s.state        = &draw_state;
			s.programs[0]  = pair.programs[0];
			s.programs[1]  = pair.programs[1];
			s.set          = set;
			s.pipeline     = bound_pipeline;
			s.frame        = m_context.FrameNumber();
			s.shader_gen   = ShaderMapGeneration();
			s.command      = reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(vk_buffer));
			s.generation   = buffer.GraphicsGeneration();
			s.passes       = LiveCounters::Value(LiveCounters::RenderPasses);
			s.images_epoch = texture_cache.ImagesEpoch();
			s.registration = buffer_cache.RegistrationEpoch();
			s.mapping      = m_context.GetGpuResources().MappingEpoch();
			s.index_type   = static_cast<uint32_t>(index_type);
			for (uint32_t st = 0; st < 2; ++st) {
				s.ud_size[st] = st < stages ? static_cast<uint32_t>(user_data[st].size()) : 0u;
				if (s.ud_size[st] != 0) std::memcpy(s.ud[st].data(), user_data[st].data(), s.ud_size[st] * sizeof(uint32_t));
			}
			s.push_dwords = push_dwords;
			if (push_dwords != 0) std::memcpy(s.push.data(), push_data.dwords.data(), push_dwords * sizeof(uint32_t));
			s.ib_buffer       = ib;
			s.ib_begin        = held.begin;
			s.ib_end          = held.end;
			s.ib_cpu_epoch    = held.cpu_epoch;
			s.ib_registration = held.registration;
			s.ib_base         = emit.index_base;
			s.ib_bytes        = emit.index_bytes;
			s.seq             = Do2Rapido::t_seq;
			s.valid           = true;
		}
	}
	// (thread1, KYTY_RENDER_STATS: the draws by what they draw into, LiveCensus::g_draw_targets; nothing otherwise.)
	if (LiveCensus::g_render && LiveCensus::RenderStatsOn()) {
		uint32_t drawn_into = LiveCensus::DrawColor;
		if (draw_state.targets.size() == 1 && draw_state.targets[0].requested.IsDepth())
			drawn_into = draw_state.targets[0].requested.resources.layers >= 4 ? LiveCensus::DrawDepthLayered
			                                                                  : LiveCensus::DrawDepthOnly;
		LiveCensus::g_draw_targets[drawn_into][0] += 1;
		LiveCensus::g_draw_targets[drawn_into][1] += std::max<size_t>(commands.size(), 1);
		// (trust1: the call's class, TrustedDepth::ClassClock, and the calls by depth image.)
		TrustedDepth::t_target = drawn_into == LiveCensus::DrawColor ? 2u : 1u;
		if (drawn_into != LiveCensus::DrawColor)
			TrustedDepth::NoteDepthImage(draw_state.targets[0].requested.data.address,
			                             draw_state.targets[0].requested.resources.layers,
			                             std::max<size_t>(commands.size(), 1), direct == nullptr);
	}
	// (mdraw C0) This call measured and compared with the last (its image set, its arguments, its blocks), the group
	// it opens or joins, and the comparison base for the next one. (The end's clock first: the bookkeeping is not
	// counted.)
	if (mdraw) {
		const uint64_t mdraw_end = __rdtsc();
		auto&          c         = LiveCensus::Mdraw::g_c;
		auto&          last      = LiveCensus::Mdraw::g_last;
		const uint64_t draws     = std::max<size_t>(commands.size(), 1);
		const uint64_t emit_at   = mdraw_emit != 0 ? mdraw_emit : mdraw_end;
		// (C1 on: no arguments read nor blocks copied, so that the OFF window's members cost what the ON window's do.)
		const bool     detail    = LiveCensus::Mdraw::SkipMode() == 0;
		if (mdraw_break == LiveCensus::Mdraw::Breaks && reinterpret_cast<uint64_t>(set) != last.set)
			mdraw_break = LiveCensus::Mdraw::BSet;
		++c.calls;
		c.draws += draws;
		c.all_cycles += mdraw_end - mdraw_start;
		c.all_emit += mdraw_end - emit_at;
		const auto read_args = [&](uint64_t address, std::array<uint32_t, 5>& out) {
			if (buffer_cache.IsRegionGpuModified(address, stride)) return false;
			std::memcpy(out.data(), reinterpret_cast<const void*>(address), sizeof(out));
			return true;
		};
		if (mdraw_break == LiveCensus::Mdraw::Breaks) {
			++LiveCensus::Mdraw::g_group_calls;
			++c.members;
			c.member_draws += draws;
			c.member_cycles += mdraw_end - mdraw_start;
			c.member_emit += mdraw_end - emit_at;
			c.member_prep += emit_at - mdraw_start;
			for (uint32_t i = 0; i < 32; ++i)
				if (((mdraw_ud >> i) & 1u) != 0) ++c.ud_member[0][i];
			++c.fields[commands[0] == last.last_arg + stride ? LiveCensus::Mdraw::FArgsContig : LiveCensus::Mdraw::FArgsJump];
			std::array<uint32_t, 5> now {};
			if (!detail) {
				// (not compared)
			} else if (!last.args_known || !read_args(commands[0], now)) {
				++c.fields[LiveCensus::Mdraw::FArgsGpu];
			} else {
				for (uint32_t i = 0; i < 5; ++i)
					if (now[i] != last.args[i]) ++c.fields[LiveCensus::Mdraw::FIndexCount + i];
			}
			bool blocks_differ = false;
			for (uint32_t st = 0; st < stages && detail; ++st)
				if (table.blocks[st] != last.blocks[st]) blocks_differ = true;
			if (blocks_differ) ++c.fields[LiveCensus::Mdraw::FBlocks];
		} else {
			++c.breaks[mdraw_break];
			LiveCensus::Mdraw::Close();
			LiveCensus::Mdraw::g_group_calls = 1;
		}
		last.valid      = direct == nullptr && !commands.empty() && index_base != 0;
		last.writes     = writes;
		last.pair       = reinterpret_cast<uint64_t>(&pair);
		last.variant    = reinterpret_cast<uint64_t>(&variant);
		last.state      = reinterpret_cast<uint64_t>(&draw_state);
		last.set        = reinterpret_cast<uint64_t>(set);
		last.command    = reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(vk_buffer));
		last.generation = buffer.GraphicsGeneration();
		last.passes     = LiveCounters::Value(LiveCounters::RenderPasses);
		last.index_base = index_base;
		last.index_bytes = index_bytes;
		last.index_type = static_cast<uint32_t>(index_type);
		for (uint32_t st = 0; st < 2; ++st)
			for (uint32_t i = 0; i < 32; ++i) last.ud[st][i] = st < stages && i < user_data[st].size() ? user_data[st][i] : 0u;
		last.last_arg   = commands.empty() ? 0 : commands.back();
		last.args_known = detail && !commands.empty() && read_args(commands.back(), last.args);
		for (uint32_t st = 0; st < 2 && detail; ++st) {
			if (st < stages) last.blocks[st].assign(table.blocks[st].begin(), table.blocks[st].end());
			else last.blocks[st].clear();
		}
	}
	return TableResult::Drawn;
}

// The pair's runs (TablePair::Run): the slots its stages evaluate, in reads of consecutive slots, by level.
void RenderExecutor::TableRuns(TablePair& pair) {
	using Kind  = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto& table = *m_table_xpr;
	std::array<TableFlatStage, 2> flat_forms; // (packed into the pair: TablePack)
	for (uint32_t st = 0; st < 2; ++st) {
		if (pair.programs[st] == nullptr) continue;
		const auto&           plan = pair.programs[st]->table_plan;
		std::vector<uint32_t> levels(plan.slots.size(), 0);
		const auto            same = [](const ShaderRecompiler::IR::TablePlan::Operand& a,
		                     const ShaderRecompiler::IR::TablePlan::Operand& b) {
			return a.kind == b.kind && a.value == b.value;
		};
		pair.runs[st].clear();
		for (uint32_t first = 0; first < plan.slots.size();) {
			if (plan.cpu[first] == 0) {
				++first;
				continue;
			}
			const auto& slot  = plan.slots[first];
			uint32_t    count = 1;
			while (first + count < plan.slots.size() && plan.cpu[first + count] != 0 &&
			       same(plan.slots[first + count].low, slot.low) && same(plan.slots[first + count].high, slot.high) &&
			       same(plan.slots[first + count].offset, slot.offset) &&
			       plan.slots[first + count].bounded == slot.bounded && same(plan.slots[first + count].records, slot.records) &&
			       plan.slots[first + count].immediate == slot.immediate + static_cast<int32_t>(4 * count))
				++count;
			uint32_t level = 0;
			for (const auto* operand: {&slot.low, &slot.high, &slot.offset, &slot.records})
				if (operand->kind == Kind::Slot) level = std::max(level, levels[operand->value] + 1);
			std::fill_n(levels.begin() + first, count, level);
			pair.runs[st].push_back({slot, first, count, level});
			first += count;
		}
		std::ranges::stable_sort(pair.runs[st], {}, &TablePair::Run::level);
		table.addresses.resize(std::max(table.addresses.size(), pair.runs[st].size()));
		// The flat form: a user data register by its index (TableEvaluate takes it where the stage's user data hold it,
		// value() reads one beyond them as 0: one of 64 and up is the constant 0), a slot by its index, an immediate
		// from the constants (the first is 0).
		const auto& program = *pair.programs[st];
		auto&       flat    = flat_forms[st];
		flat.constants.push_back(0);
		const auto flat_of = [&](const ShaderRecompiler::IR::TablePlan::Operand& operand) -> TableFlatStage::Operand {
			if (operand.kind == Kind::Slot) return {operand.value, 1};
			if (TableCodeOperand(operand)) {
				flat.constants.push_back(TableCodeWord(operand, st == 0 ? pair.vs : pair.ps));
				return {static_cast<uint32_t>(flat.constants.size() - 1), 2};
			}
			if (operand.kind == Kind::UserData) {
				const auto index = operand.value - program.user_data_base;
				if (index >= 64) return {0, 2};
				flat.user_words = std::max(flat.user_words, index + 1);
				return {index, 0};
			}
			flat.constants.push_back(operand.value);
			return {static_cast<uint32_t>(flat.constants.size() - 1), 2};
		};
		const auto& runs = pair.runs[st];
		for (size_t i = 0; i < runs.size(); ++i) {
			const auto& run = runs[i];
			flat.runs.push_back({{flat_of(run.read.low), flat_of(run.read.high), flat_of(run.read.offset)},
			                     run.read.immediate, run.first, run.count, flat_of(run.read.records), run.read.bounded});
			if (i + 1 == runs.size() || runs[i + 1].level != run.level) flat.level_ends.push_back(static_cast<uint32_t>(i + 1));
		}
		for (const auto& descriptor: plan.images) {
			flat.image_counts.push_back(descriptor.count);
			for (uint32_t w = 0; w < descriptor.count; ++w) flat.image_words.push_back(flat_of(descriptor.words[w]));
		}
		for (const auto& descriptor: plan.samplers)
			for (uint32_t w = 0; w < descriptor.count; ++w) flat.image_words.push_back(flat_of(descriptor.words[w]));
		for (const auto& words: plan.buffers)
			for (const auto& word: words) flat.buffer_words.push_back(flat_of(word));
	}
	TablePack(pair, flat_forms);
}

// The pair's flat forms packed into one block (TablePair::packed): a stage whose operand indices, counts or levels do not
// fit their packed widths is left invalid (evaluated by its plan's operands).
void RenderExecutor::TablePack(TablePair& pair, const std::array<TableFlatStage, 2>& flat_forms) {
	using Pair = TablePair;
	pair.packed       = {};
	pair.packed_block = nullptr;
	pair.packed_at    = nullptr;
	pair.packed_bytes = 0;
	const auto operand = [](const TableFlatStage::Operand& from, bool& fits) -> uint16_t {
		if (from.base > 2 || from.index >= 0x4000u) fits = false;
		return static_cast<uint16_t>(from.base << 14u | (from.index & 0x3fffu));
	};
	const auto align = [](size_t at, size_t to) { return (at + to - 1) / to * to; };
	// (Layout per stage: runs, constants, level ends, image words, buffer words, image counts.)
	std::array<std::array<size_t, 6>, 2> offsets {};
	size_t                               bytes = 0;
	std::array<bool, 2>                  fits {};
	for (uint32_t st = 0; st < 2; ++st) {
		if (pair.programs[st] == nullptr) continue;
		const auto& flat = flat_forms[st];
		fits[st]         = flat.runs.size() < 0x10000 && flat.image_words.size() < 0x10000;
		for (const auto end: flat.level_ends) fits[st] = fits[st] && end < 0x10000;
		for (const auto count: flat.image_counts) fits[st] = fits[st] && count < 0x100;
		for (const auto& run: flat.runs) fits[st] = fits[st] && run.first < 0x10000 && run.count < 0x10000;
		if (!fits[st]) continue;
		bytes          = align(bytes, 64);
		offsets[st][0] = bytes;
		bytes += flat.runs.size() * sizeof(Pair::PackedRun);
		offsets[st][1] = bytes;
		bytes += flat.constants.size() * sizeof(uint32_t);
		offsets[st][2] = bytes;
		bytes += flat.level_ends.size() * sizeof(uint16_t);
		offsets[st][3] = bytes;
		bytes += flat.image_words.size() * sizeof(uint16_t);
		offsets[st][4] = bytes;
		bytes += flat.buffer_words.size() * sizeof(uint16_t);
		offsets[st][5] = bytes;
		bytes += flat.image_counts.size();
	}
	if (bytes == 0) return;
	pair.packed_block = std::make_unique<std::byte[]>(bytes + 64);
	auto* base        = reinterpret_cast<std::byte*>(align(reinterpret_cast<uintptr_t>(pair.packed_block.get()), 64));
	pair.packed_at    = base;
	pair.packed_bytes = static_cast<uint32_t>(bytes);
	for (uint32_t st = 0; st < 2; ++st) {
		if (!fits[st]) continue;
		const auto& flat    = flat_forms[st];
		auto&       packed  = pair.packed[st];
		bool        operand_fits = true;
		auto*       runs    = reinterpret_cast<Pair::PackedRun*>(base + offsets[st][0]);
		for (size_t i = 0; i < flat.runs.size(); ++i) {
			const auto& run = flat.runs[i];
			runs[i]         = {.address   = {operand(run.address[0], operand_fits), operand(run.address[1], operand_fits),
			                                 operand(run.address[2], operand_fits)},
			                   .records   = run.bounded ? operand(run.records, operand_fits) : Pair::PackedRun::NotBounded,
			                   .immediate = run.immediate,
			                   .first     = static_cast<uint16_t>(run.first),
			                   .count     = static_cast<uint16_t>(run.count)};
			if (run.bounded && runs[i].records == Pair::PackedRun::NotBounded) operand_fits = false;
		}
		auto* constants = reinterpret_cast<uint32_t*>(base + offsets[st][1]);
		std::copy(flat.constants.begin(), flat.constants.end(), constants);
		auto* level_ends = reinterpret_cast<uint16_t*>(base + offsets[st][2]);
		for (size_t i = 0; i < flat.level_ends.size(); ++i) level_ends[i] = static_cast<uint16_t>(flat.level_ends[i]);
		auto* image_words = reinterpret_cast<uint16_t*>(base + offsets[st][3]);
		for (size_t i = 0; i < flat.image_words.size(); ++i) image_words[i] = operand(flat.image_words[i], operand_fits);
		auto* buffer_words = reinterpret_cast<uint16_t*>(base + offsets[st][4]);
		for (size_t i = 0; i < flat.buffer_words.size(); ++i) buffer_words[i] = operand(flat.buffer_words[i], operand_fits);
		auto* image_counts = reinterpret_cast<uint8_t*>(base + offsets[st][5]);
		for (size_t i = 0; i < flat.image_counts.size(); ++i) image_counts[i] = static_cast<uint8_t>(flat.image_counts[i]);
		packed = {.runs         = {runs, flat.runs.size()},
		          .level_ends   = {level_ends, flat.level_ends.size()},
		          .image_words  = {image_words, flat.image_words.size()},
		          .buffer_words = {buffer_words, flat.buffer_words.size()},
		          .image_counts = {image_counts, flat.image_counts.size()},
		          .constants    = {constants, flat.constants.size()},
		          .user_words   = flat.user_words,
		          .valid        = operand_fits};
	}
}

// After a prepared draw (`storable`: a draw without vertex buffers) of a pair the table path asked to
// store: the pair's table mode programs (with this draw's specializations), the pipeline for this draw state,
// compiled by a worker, and the draw state.
void RenderExecutor::TableStore(CommandBuffer& buffer, const DrawRenderState& state, vk::PrimitiveTopology topology,
                                bool primitive_restart_enable, const RenderState& rendering, bool storable) {
	auto&       table = *m_table_xpr;
	auto* const pair  = std::exchange(table.store, nullptr);
	// (KYTY_STORE_SIN_DRAW) Every normal-path draw leaves its seed (once a pair state): a later store of the same
	// pair and state - one deferred by the budget, or waiting - is then made without a normal-path draw.
	if (StoreSinDraw::On() && storable && Role() == 0) {
		const auto& sinsh = buffer.GetShaders();
		const auto  sinvs = sinsh.GetVs().es_regs.data_addr, sinps = sinsh.GetPs().ps_regs.data_addr;
		const auto  sinsi = static_cast<uint64_t>(sinsh.GetVs().start_instance_user_sgpr);
		const auto  sinst = NativeXprStateKey(buffer);
		StoreSinDraw::Note(StoreSinDraw::PairKey(sinvs, sinps, sinsi), sinst, StoreSinDraw::Key(sinvs, sinps, sinsi, sinst), state,
		                   topology, primitive_restart_enable, true);
	}
	if (!storable || pair == nullptr) return;
	// (tiron1) The request's time, from its grant (TableTry) to here, whatever the store's way out.
	struct Tiron1StoreTime {
		TableXpr& table;
		~Tiron1StoreTime() {
			if (table.tiron1_grant_ns == 0) return;
			const auto us = (Tiron1::NowNs() - std::exchange(table.tiron1_grant_ns, 0)) / 1000;
			if (us <= 0 || us > 1000000) return;
			table.tiron1_store_us += us;
			Tiron1::g_store_us.fetch_add(static_cast<uint64_t>(us), std::memory_order_relaxed);
		}
	};
	std::optional<Tiron1StoreTime> tiron1_time;
	if (Tiron1::On()) tiron1_time.emplace(table);
	// (tablas1) The body's own time (the rest of g_store_us is the granted draw's normal-path draw).
	struct Tiron1BodyTime {
		int64_t begin = Tiron1::NowNs();
		~Tiron1BodyTime() {
			const auto us = (Tiron1::NowNs() - begin) / 1000;
			Tiron1::g_store_bodies.fetch_add(1, std::memory_order_relaxed);
			if (us > 0 && us <= 1000000) Tiron1::g_store_body_us.fetch_add(static_cast<uint64_t>(us), std::memory_order_relaxed);
		}
	};
	std::optional<Tiron1BodyTime> tiron1_body;
	if (tiron1_time) tiron1_body.emplace();
	const auto& sh = buffer.GetShaders();
	// (A request whose packet never reached a prepared draw must not store for another state.)
	if (sh.GetVs().es_regs.data_addr != pair->vs || sh.GetPs().ps_regs.data_addr != pair->ps ||
	    NativeXprStateKey(buffer) != table.store_state)
		return;
	auto&           pipelines   = m_context.GetPipelineCache();
	const auto&     ctx         = buffer.GetRegisters();
	DrawRenderState table_state = state;
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
	for (uint32_t i = 0; i < state.color_count; i++)
		mapping[state.color_info[i].target_slot] = state.color_info[i].export_mapping;
	const int64_t programs_begin = tiron1_body ? Tiron1::NowNs() : 0;
	table_state.programs = pipelines.GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx,
	                                                     buffer.GetUserConfig(), mapping, state.ps_active,
	                                                     table_state.vs_input_info, table_state.ps_input_info, true,
	                                                     pair->portable);
	if (tiron1_body) {
		const auto us = (Tiron1::NowNs() - programs_begin) / 1000;
		if (us > 0 && us <= 1000000) Tiron1::g_store_programs_us.fetch_add(static_cast<uint64_t>(us), std::memory_order_relaxed);
	}
	if (pipelines.TablePending()) {
		pair->waiting = pipelines.TableWaiting(); // (the pair stores again once it is done)
		return;
	}
	const auto* vertex = table_state.vs_input_info.stage.program;
	const auto* pixel  = state.ps_active ? table_state.ps_input_info.stage.program : nullptr;
	const auto  refuse = [&](const char* reason) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
			std::printf("Table: VS 0x%llx PS 0x%llx refused: %s\n", static_cast<unsigned long long>(pair->vs),
			            static_cast<unsigned long long>(pair->ps), reason);
		const auto write = CatalogWrite();
		pair->refused    = true;
	};
	if (!table_state.programs.vertex || vertex == nullptr || !vertex->table_mode)
		return refuse("no table mode vertex program");
	if (vertex->stage != ShaderType::Vertex) return refuse("not a vertex program");
	if (state.ps_active && (!table_state.programs.pixel || pixel == nullptr || !pixel->table_mode))
		return refuse("no table mode pixel program");
	const std::array<const ShaderRecompiler::IR::CompiledShaderInfo*, 2> programs {vertex, pixel};
	if (pair->programs[0] == nullptr) {
		{
			const auto write = CatalogWrite();
			pair->programs   = programs;
			TableRuns(*pair);
		}
		const auto count = [](const ShaderRecompiler::IR::CompiledShaderInfo* program, bool cpu) -> size_t {
			if (program == nullptr) return 0;
			const auto& plan = program->table_plan;
			return static_cast<size_t>(std::ranges::count(cpu ? plan.cpu : plan.gpu, uint8_t {1}));
		};
		std::printf("Table: VS 0x%llx (0x%016llx) PS 0x%llx (0x%016llx): reads %zu+%zu slots (evaluates %zu+%zu), "
		            "buffers %zu+%zu, images %zu, samplers %zu\n",
		            static_cast<unsigned long long>(pair->vs), static_cast<unsigned long long>(vertex->shader_hash),
		            static_cast<unsigned long long>(pair->ps),
		            static_cast<unsigned long long>(pixel != nullptr ? pixel->shader_hash : 0), count(vertex, false),
		            count(pixel, false), count(vertex, true), count(pixel, true), vertex->table_plan.buffers.size(),
		            pixel != nullptr ? pixel->table_plan.buffers.size() : 0,
		            vertex->table_plan.images.size() + (pixel != nullptr ? pixel->table_plan.images.size() : 0),
		            vertex->table_plan.samplers.size() + (pixel != nullptr ? pixel->table_plan.samplers.size() : 0));
	} else if (pair->programs != programs) {
		// Another specialization (the pixel program's export of another draw state's targets): this state's draws
		// keep the native path.
		const auto write = CatalogWrite();
		const auto found = std::ranges::find(pair->variants, table.store_state, &TableXpr::Variant::state);
		if (found == pair->variants.end()) pair->variants.push_back({.state = table.store_state, .refused = true});
		return;
	}
	const auto* pipeline = pipelines.TryCreateGraphicsPipeline(
	    std::span {state.color_info, state.color_count}, state.depth_info, table_state.vs_input_info, buffer,
	    state.ps_active ? &table_state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    table_state.programs.vertex, table_state.programs.pixel, true);
	if (pipeline == nullptr) pair->waiting = pipelines.PipelineWaiting();
	{
		const auto write   = CatalogWrite();
		auto       variant = std::ranges::find(pair->variants, table.store_state, &TableXpr::Variant::state);
		if (variant == pair->variants.end()) variant = pair->variants.insert(variant, {.state = table.store_state});
		variant->pipeline       = pipeline;
		variant->mapping        = mapping;
		variant->fetch_embedded = table_state.vs_input_info.fetch_embedded;
	}
	// KYTY_HWRT (local/hwrt-scene.h): a G-buffer draw state (>= 3 colour targets, a single-layer depth target >= 1280 px
	// wide it writes, a triangle list) gets the capturing variant of its pipeline too, compiled by a worker.
	// (10-25 combo25: the G-buffer pass draws after the depth prepass with depth writes off - a depth-equal test - so
	// depth writes alone found 5 states: a state that writes depth, tests it for equality, or is a known G-buffer
	// vertex shader's qualifies; decals test without writing and are not equal tests.)
	const bool hwrt_gbuffer_vs = vertex != nullptr && FrameGen::IsGbufferVertexShader(vertex->shader_hash);
	const bool hwrt_depth      = state.depth_info.depth_write_enable ||
	                        (state.depth_info.depth_test_enable && state.depth_info.depth_compare_op == vk::CompareOp::eEqual) ||
	                        hwrt_gbuffer_vs;
	if (HwScene::Active() && Role() == 0 && state.color_count >= 3 && state.depth_info.image_id &&
	    state.depth_info.desc.info.resources.layers == 1 && state.depth_info.desc.info.extent.width >= 1280) {
		static std::atomic<uint32_t> hwrt_reports {0};
		if (hwrt_reports.fetch_add(1, std::memory_order_relaxed) < 48) {
			std::printf("HWRT: G-buffer-like state VS %016llx PS %016llx: %u colours, depth %ux%u test %d write %d op %d, topology %d "
			            "restart %d, known G-buffer VS %d -> %s\n",
			            static_cast<unsigned long long>(vertex != nullptr ? vertex->shader_hash : 0),
			            static_cast<unsigned long long>(pixel != nullptr ? pixel->shader_hash : 0), state.color_count,
			            state.depth_info.desc.info.extent.width, state.depth_info.desc.info.extent.height,
			            state.depth_info.depth_test_enable ? 1 : 0, state.depth_info.depth_write_enable ? 1 : 0,
			            static_cast<int>(state.depth_info.depth_compare_op), static_cast<int>(topology), primitive_restart_enable ? 1 : 0,
			            hwrt_gbuffer_vs ? 1 : 0,
			            hwrt_depth && topology == vk::PrimitiveTopology::eTriangleList && !primitive_restart_enable ? "captured" : "not");
			std::fflush(stdout);
		}
	}
	// (KYTY_HWRT_WATER, rt-water.cpp: off by default) the water surface's draws (its pixel shader, drawn after the
	// lighting, forward, maybe without depth writes) are captured too, into their own TLAS (hwrt-scene.cpp).
	const bool hwrt_water = pixel != nullptr && FrameGen::RtWaterIsWaterPs(pixel->shader_hash) && state.color_count >= 1;
	if (hwrt_water) FrameGen::RtWaterNoteState(vertex != nullptr ? vertex->shader_hash : 0, pixel->shader_hash, state.color_count,
	                                           state.depth_info.image_id ? state.depth_info.desc.info.extent.width : 0,
	                                           state.depth_info.depth_write_enable, static_cast<int>(topology));
	if (HwScene::Active() && Role() == 0 && (state.color_count >= 3 || hwrt_water) && state.depth_info.image_id &&
	    (hwrt_depth || hwrt_water) && state.depth_info.desc.info.resources.layers == 1 &&
	    state.depth_info.desc.info.extent.width >= 1280 && topology == vk::PrimitiveTopology::eTriangleList &&
	    !primitive_restart_enable) {
		const auto module = HwScene::CaptureModule(m_context.GetGraphics(), table_state.programs.vertex.module);
		HwScene::NoteSceneState(module != nullptr);
		if (module) {
			const ShaderProgram capture_program {HwScene::VariantId(table_state.programs.vertex.id), module};
			auto make = std::make_shared<std::function<const PipelineCache::Pipeline*(CommandBuffer&)>>(
			    [this, s = table_state, topology, primitive_restart_enable, capture_program](CommandBuffer& command) {
				    return static_cast<const PipelineCache::Pipeline*>(m_context.GetPipelineCache().TryCreateGraphicsPipeline(
				        std::span<const RenderColorInfo> {s.color_info, s.color_count}, s.depth_info, s.vs_input_info, command,
				        s.ps_active ? &s.ps_input_info : nullptr, topology, primitive_restart_enable, capture_program,
				        s.programs.pixel, true));
			    });
			const auto* made  = (*make)(buffer);
			const auto  write = CatalogWrite();
			auto variant = std::ranges::find(pair->variants, table.store_state, &TableXpr::Variant::state);
			if (variant != pair->variants.end()) {
				variant->hwrt_scene    = true;
				variant->hwrt_ps       = pixel != nullptr ? pixel->shader_hash : 0;
				variant->hwrt_ps_module = table_state.programs.pixel.module;
				variant->hwrt_make     = std::move(make);
				variant->hwrt_pipeline = made;
			}
		}
	}
	if (tiron1_body && pipeline != nullptr) Tiron1::g_store_captured.fetch_add(1, std::memory_order_relaxed);
	NativeXprCaptureState(buffer, state, rendering, table.store_state);
}

// --- Table dispatches ---------------------------------------------------------------------------------------------
//
// KYTY_TABLE_DISPATCH=1: a compute shader the normal path dispatched twice (its resource tables evaluated in full),
// whose program allows table mode, is dispatched as table draws are drawn: its slots in a block, its read-only V#s
// resolved once, its written ones obtained as written at every dispatch (the buffer cache tracks the GPU's writes),
// its images and samplers in a set; the barriers and the compute chain as the normal path's DispatchDirect. Shaders
// a special path of the normal path took (linear copies, clears), indirect thread-dimension dispatches, frame
// capture and frame generation stay on the normal path.

// The compute stage's registers that select the program (ShaderGetStaticInputInfoCS reads them), and whether the
// dispatch counts threads (another program: ShaderComputeInputInfo::dispatch_thread_dimensions).
static std::array<uint64_t, 4> TableDispatchWords(const HW::ComputeShaderInfo& cs, bool thread_dimensions) {
	const auto& r = cs.cs_regs;
	return {r.data_addr, uint64_t {r.num_thread_x} | uint64_t {r.num_thread_y} << 32u,
	        uint64_t {r.num_thread_z} | uint64_t {r.lds_size} << 32u,
	        uint64_t {r.wave_size} | uint64_t {r.user_sgpr} << 8u | uint64_t {r.tidig_comp_cnt} << 16u |
	            uint64_t {r.tgid_x_en} << 24u | uint64_t {r.tgid_y_en} << 25u | uint64_t {r.tgid_z_en} << 26u |
	            uint64_t {r.tg_size_en} << 27u | uint64_t {r.scratch_en} << 28u | uint64_t {thread_dimensions} << 29u};
}
static uint64_t TableDispatchKey(const std::array<uint64_t, 4>& words) {
	return XXH3_64bits(words.data(), sizeof(words));
}
static uint64_t TableDispatchKey(const HW::ComputeShaderInfo& cs, bool thread_dimensions) {
	return TableDispatchKey(TableDispatchWords(cs, thread_dimensions));
}

// The normal path dispatched (`special`: one of its special paths took the dispatch).
void RenderExecutor::TableDispatchSeen(const HW::ComputeShaderInfo& cs, const char* special, bool thread_dimensions) {
	if (kyty_local_table_dispatch_mode.load(std::memory_order_relaxed) == 0 || Role() != 0) return;
	auto&      dispatches = OwnTable().dispatches;
	const auto key        = TableDispatchKey(cs, thread_dimensions);
	auto       it         = dispatches.find(key);
	if (it == dispatches.end() || (special != nullptr && !it->second.refused)) {
		const auto write = CatalogWrite();
		it               = dispatches.try_emplace(key).first;
		it->second.vs    = cs.cs_regs.data_addr;
		if (special != nullptr) {
			it->second.refused = true;
			it->second.special = special;
		}
	}
	++it->second.sightings;
	it->second.used_frame = m_context.FrameNumber();
}

bool RenderExecutor::TableDispatch(CommandBuffer& buffer, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
                                   uint64_t indirect_args, bool thread_dimensions) {
	auto* catalog = CatalogTable();
	m_dispatch_declined = "dispatch: nothing learned";
	// (Frame generation, DLSS-G or FSR 3, needs the normal path for one dispatch a frame, the TAA pass's, refused
	// below; with every dispatch refused under DLSS-G, as before, the game ran at ~30 real fps where FSR 3 ran at 57.)
	if (catalog == nullptr || (FrameCapture::Active() || SkipShader::g_any.load(std::memory_order_relaxed)) || XprCapture::Enabled()) return false;
	std::shared_lock<std::shared_mutex> catalog_read;
	if (Role() != 0) catalog_read = CatalogRead();
	DropStaleCatalog();
	TableSweepPairs();
	const auto& sh    = buffer.GetShaders();
	const auto& cs    = sh.GetCs();
	const auto  words = TableDispatchWords(cs, thread_dimensions);
	// (The pair found last again without its hash or lookup: runs of dispatches keep their program. A pointer stays
	// valid until TableSweepPairs, which drops this one; the main executor's own catalog only.)
	const bool own   = Role() == 0 && catalog == m_table_xpr.get();
	auto*      known = own && catalog->found_dispatch_key == words ? catalog->found_dispatch : nullptr;
	if (known == nullptr) {
		const auto found = catalog->dispatches.find(TableDispatchKey(words));
		// (Another executor's: a pair with its programs and pipeline had its sightings.)
		if (found == catalog->dispatches.end()) return m_dispatch_declined = "dispatch: not seen", false;
		known = &found->second;
		if (own) {
			catalog->found_dispatch_key = words;
			catalog->found_dispatch     = known;
		}
	}
	if (known->refused)
		return m_dispatch_declined = known->special != nullptr ? known->special : "dispatch: refused", false;
	if ((Role() == 0 && known->sightings < 2) || known->vs != cs.cs_regs.data_addr) return false;
	auto& pair = *known;
	// (render4, KYTY_RENDER_STATS: the dispatch's shader for the [render] line's "cs" part; the dispatch's scope reads it.)
	if (pair.programs[0] != nullptr) LiveCensus::SetDispatchShader(pair.programs[0]->shader_hash);
	// (The game's 3D view; a dispatch declined below goes the normal way, which notes it too.)
	if (Role() == 0 && pair.programs[0] != nullptr && FrameGen::IsTaaShader(pair.programs[0]->shader_hash))
		FrameGen::NoteScene();
	// Frame generation reads the TAA pass's depth and motion vectors from the normal path's bindings
	// (RenderExecutor::DispatchDirect, FrameGen::OnDispatch): one dispatch a frame.
	if (Render5Flags::Capturing() && pair.programs[0] != nullptr && FrameGen::IsTaaShader(pair.programs[0]->shader_hash))
		return m_dispatch_declined = "table: frame generation (TAA)", false;
	// (KYTY_HWRT: the TAA's dispatch is where the scene's acceleration structures are built, on the normal path.)
	if (HwScene::Active() && pair.programs[0] != nullptr && FrameGen::IsTaaShader(pair.programs[0]->shader_hash))
		return m_dispatch_declined = "table: hardware ray tracing (TAA)", false;
	// (pt1) BB_PATHTRACING reads the images of the dispatches in a few scopes (ResolveSunShadows, RenderLights,
	// ResolveLighting, RenderSSR: ~45 a frame): the normal path, whose bindings reach PathTrace::AfterDispatch.
	if (PathTrace::Requested() && Role() == 0 && PathTrace::WantDispatchHook())
		return m_dispatch_declined = "table: path tracing (scope images)", false;
	TablePrefetchPair(pair);
	if (Role() == 0) pair.used_frame = m_context.FrameNumber();
	if (Role() == 0) TablePrefetchNext(OwnTable().previous_dispatch, pair);
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1f, 0); // census: the dispatches' phases (b)
	// The dispatch executor's sets retire here too (NativeXprTry collects only the draw executor's): without it they
	// were never reused or freed, a new pool every 1024.
	NativeXprCollect();
	// (Another executor's pair: its programs and pipeline as its owner made them, or the normal path's dispatch.)
	if (Role() != 0 && (pair.programs[0] == nullptr || pair.pipeline == nullptr || pair.respecialize))
		return Spec::Refuse("table program"), false;
	if (const auto generation = ShaderMapGeneration(); generation != pair.generation[Role()]) {
		if (pair.programs[0] != nullptr && ShaderHashAt(pair.vs) != pair.programs[0]->shader_hash) {
			if (Role() != 0) return Spec::Refuse("table: shaders changed"), false;
			const auto write = CatalogWrite();
			pair.portable    = {};
			TableReset(pair);
		}
		pair.generation[Role()] = generation;
	}
	if (pair.respecialize) {
		const auto write = CatalogWrite();
		TableReset(pair);
	}
	if (pair.waiting && Role() == 0) {
		if (!pair.waiting->load(std::memory_order_acquire)) return false;
		pair.waiting.reset();
	}
	TablePrefetchArrays(pair, pair.last_set[Role()]);
	auto& pipelines = m_context.GetPipelineCache();
	if (Spec::Current() != nullptr && (pair.programs[0] == nullptr || pair.pipeline == nullptr))
		return Spec::Refuse("table program"), false;
	if (pair.programs[0] == nullptr) {
		// The table mode program of the shader the normal path prepared.
		pair.compute = std::make_unique<ShaderComputeInputInfo>();
		ResetNativeStageInput(*pair.compute);
		pair.compute->dispatch_thread_dimensions = thread_dimensions;
		pair.compute_program = pipelines.GetComputeProgram(cs, buffer.GetRegisters().GetShaderRegisters(), *pair.compute,
		                                                   nullptr, true, pair.portable[0]);
		const auto* program = pair.compute->stage.program;
		if (pipelines.TablePending()) {
			pair.waiting = pipelines.TableWaiting(); // (asked again once it is done)
			return false;
		}
		if (!pair.compute_program || program == nullptr || !program->table_mode) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
				std::printf("Table: CS 0x%llx refused: no table mode program\n", static_cast<unsigned long long>(pair.vs));
			const auto write = CatalogWrite();
			pair.refused     = true;
			return false;
		}
		{
			const auto write = CatalogWrite();
			pair.programs[0] = program;
			TableRuns(pair);
		}
		std::printf("Table: CS 0x%llx (0x%016llx): reads %zu slots (evaluates %zu), buffers %zu (written %zu), "
		            "images %zu, samplers %zu\n",
		            static_cast<unsigned long long>(pair.vs), static_cast<unsigned long long>(program->shader_hash),
		            static_cast<size_t>(std::ranges::count(program->table_plan.gpu, uint8_t {1})),
		            static_cast<size_t>(std::ranges::count(program->table_plan.cpu, uint8_t {1})),
		            program->table_plan.buffers.size(),
		            static_cast<size_t>(std::ranges::count_if(program->info.buffers, [](const auto& item) {
			            return item.written || item.atomic;
		            })),
		            program->table_plan.images.size(), program->table_plan.samplers.size());
	}
	// KYTY_TABLE_DENY=<hash>,... (or ;): those programs' dispatches take the normal path (a bisection of the table
	// dispatches: the culling chain's GPU hangs at 2-2, 10-13).
	static const std::vector<uint64_t> denied = [] {
		std::vector<uint64_t> list;
		for (const char* at = std::getenv("KYTY_TABLE_DENY"); at != nullptr && *at != '\0';) {
			char* end = nullptr;
			list.push_back(std::strtoull(at, &end, 16));
			if (end == nullptr || (*end != ',' && *end != ';')) break;
			at = end + 1;
		}
		return list;
	}();
	if (!denied.empty() && std::ranges::find(denied, pair.programs[0]->shader_hash) != denied.end())
		return m_dispatch_declined = "table: denied", false;
	// KYTY_TABLE_DENY_RANGE=<lo>-<hi> (hex): the programs whose hash is in [lo, hi] too (a bisection by hash).
	static const std::pair<uint64_t, uint64_t> denied_range = [] {
		const char* value = std::getenv("KYTY_TABLE_DENY_RANGE");
		if (value == nullptr || *value == '\0') return std::pair<uint64_t, uint64_t> {1, 0};
		char*          end = nullptr;
		const uint64_t lo  = std::strtoull(value, &end, 16);
		return std::pair<uint64_t, uint64_t> {lo, end != nullptr && *end == '-' ? std::strtoull(end + 1, nullptr, 16) : lo};
	}();
	if (const auto hash = pair.programs[0]->shader_hash; hash >= denied_range.first && hash <= denied_range.second)
		return m_dispatch_declined = "table: denied", false;
	if (pair.pipeline == nullptr) {
		const auto* pipeline = pipelines.TryCreateComputePipeline(*pair.compute, pair.compute_program);
		{
			const auto write = CatalogWrite();
			pair.pipeline    = pipeline;
		}
		if (pair.pipeline == nullptr) {
			pair.waiting = pipelines.PipelineWaiting();
			return false;
		}
	}
	const std::array<std::span<const uint32_t>, 2> user_data {std::span<const uint32_t> {cs.cs_user_sgpr.value},
	                                                          std::span<const uint32_t> {}};
	std::optional<LiveCensus::Scope> phase;
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 1);
	if (kyty_local_table_indirect_mode.load(std::memory_order_relaxed) == 0 &&
	    (!pair.programs[0]->table_plan.indirect.empty() || pair.programs[0]->table_plan.dynamic_reads))
		return m_dispatch_declined = "table: indirect off", false;
	// (draw1, KYTY_DRAW1 with KYTY_RENDER_STATS) The dispatch's key (its pair and user data), for its repetition
	// (measurement only).
	if (TrustedDepth::ByteProof() && TrustedDepth::StatsOn() && Role() == 0 && Spec::Current() == nullptr) {
		const uint64_t head = reinterpret_cast<uint64_t>(&pair);
		TrustedDepth::NoteKindKey(TrustedDepth::KindDispatch,
		                          XXH3_64bits_withSeed(user_data[0].data(), user_data[0].size_bytes(), XXH3_64bits(&head, sizeof(head))));
	}
	if (!TableEvaluate(pair, user_data)) return m_dispatch_declined = "table: tables unmapped", false;
	TableGather(pair, user_data);
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 2);
	auto* set = TableSet(buffer, pair, nullptr, nullptr);
	if (set == nullptr) return false;
	if (auto* spec = Spec::Current()) spec->NoteSet(reinterpret_cast<uint64_t>(static_cast<VkDescriptorSet>(set->set)));
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 3);
	std::array<vk::DeviceAddress, 2> blocks {};
	bool                             storage_writes = false;
	auto&                            cache          = m_context.GetBufferCache();
	{
		LiveCensus::StatScope stat_bda(1, 5); // (render3, KYTY_RENDER_STATS: the BDA preparation and its uploads)
		TablePrepareBda(pair, user_data); // (as TableDraw's: before the blocks)
	}
	const auto                       registration   = cache.RegistrationEpoch(); // (as TableDraw's)
	TableWatchBegin(*pair.programs[0], user_data[0]);
	bool blocks_made = false;
	{
		LiveCensus::StatScope stat_blocks(1, 6); // (render3, KYTY_RENDER_STATS: the blocks)
		blocks_made = TableBlocks(pair, user_data, blocks, &storage_writes);
	}
	if (!blocks_made) {
		TableWatchState::g_current.watch = nullptr;
		return m_dispatch_declined = "table: blocks", false;
	}
	vk::Buffer indirect_buffer = nullptr;
	uint64_t   indirect_offset = 0;
	if (indirect_args != 0) {
		LiveCensus::StatScope stat_args(1, 7); // (render3, KYTY_RENDER_STATS: the indirect arguments)
		// As DispatchDirect's: the arguments' owner, made current.
		cache.EnsureBufferContents(indirect_args, 3u * sizeof(uint32_t));
		const auto id = cache.FindBuffer(indirect_args, 3u * sizeof(uint32_t));
		if (Spec::Current() != nullptr && Spec::Current()->refused != nullptr) return false;
		auto& owner     = cache.GetBuffer(id);
		indirect_buffer = owner.Handle();
		indirect_offset = owner.Offset(indirect_args);
	}
	if (cache.RegistrationEpoch() != registration)
		return LiveCounters::Add(LiveCounters::TableRegistered), m_dispatch_declined = "table: buffers registered meanwhile", false;
	// (A speculation's lookups and preparation refuse without failing: nothing is recorded for a refused dispatch.)
	if (const auto* spec = Spec::Current(); spec != nullptr && spec->refused != nullptr) return false;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 4);
	// Emission, as DispatchDirect's.
	buffer.EndRendering();
	const bool chain     = DemonsSouls::ComputeChains(pair.programs[0]->shader_hash);
	const auto vk_buffer = chain ? buffer.ChainHandle() : buffer.Handle();
	TableWatchEnd(cache.GetTableUploadBuffer(), cache, m_context.GetGraphics(),
	              Role() == 0 && Spec::Current() == nullptr ? &m_context.GetCommandScheduler() : nullptr, vk_buffer,
	              blocks[0], pair.programs[0]->table_plan.BlockDwords() != 0 ? std::span<const uint32_t> {OwnTable().blocks[0]}
	                                                                         : std::span<const uint32_t> {});
	// (As a draw's: the set's images as its last dispatch's transit left them, while no image changed since.)
	auto&      dispatch_textures = m_context.GetTextureCache();
	const bool own_transit       = Role() == 0 && Spec::Current() == nullptr;
	if ((!own_transit || set->transit_epoch != dispatch_textures.ImagesEpoch() || set->transit_state != nullptr) &&
	    !TableTransit(*set, vk_buffer, nullptr))
		return m_dispatch_declined = "table: transit", false;
	if (own_transit) {
		set->transit_epoch = dispatch_textures.ImagesEpoch();
		set->transit_state = nullptr;
	}
	// (A set made for another specialization: its program's pipeline and shader data.)
	const auto& pipeline = set->pipeline != nullptr ? *set->pipeline : *pair.pipeline;
	const auto& program  = set->program != nullptr ? *set->program : *pair.programs[0];
	const auto& bindings = program.bindings;
	const Envio1::Scope envio1_record(Envio1::Record); // (envio1, KYTY_ENVIO1=1: to the end of the dispatch)
	vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.pipeline_layout, 0, 1, &set->set, 0, nullptr);
	if (bindings.UsesPushData()) {
		ShaderRecompiler::IR::PushData push_data;
		for (size_t i = 0; i < bindings.user_data_registers.size(); ++i) {
			const auto index = bindings.user_data_registers[i] - program.user_data_base;
			push_data.dwords[bindings.push_data_start_dword + i] = index < user_data[0].size() ? user_data[0][index] : 0;
		}
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword()]     = static_cast<uint32_t>(blocks[0]);
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword() + 1] = static_cast<uint32_t>(blocks[0] >> 32u);
		vk_buffer.pushConstants(pipeline.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
		                        (bindings.push_data_start_dword + bindings.ShaderDataDwords()) * sizeof(uint32_t),
		                        push_data.dwords.data());
	}
	const bool continues_chain = chain && buffer.ComputeChainPending();
	TableWatchDispatch(indirect_args, continues_chain);
	if (storage_writes && !continues_chain) ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	// (p7b, C1 precise: local/p7-compat.h) As DispatchDirect's: the hazards against the chain's earlier dispatches,
	// from the guest ranges of its V#s (the words TableBlocks resolved); one barrier only where there is one.
	if (chain && P7::Tracking()) {
		const auto&                     shader  = *pair.programs[0];
		const auto&                     entries = OwnTable().entries[0];
		std::array<P7::ChainAccess, 16> accesses {};
		size_t                          count   = 0;
		bool                            unknown = shader.info.uses_dma || shader.table_plan.global_memory ||
		                                          shader.table_plan.dynamic_reads;
		for (size_t i = 0; i < entries.size() && i < shader.info.buffers.size(); ++i) {
			ShaderBufferResource descriptor;
			std::copy_n(entries[i].first.begin(), 4, descriptor.fields);
			const auto address = descriptor.Base48();
			const auto bytes   = descriptor.Stride() != 0 ? uint64_t {descriptor.Stride()} * descriptor.NumRecords()
			                                              : uint64_t {descriptor.NumRecords()};
			if (descriptor.Type() != 0 || address == 0 || bytes == 0) continue;
			if (count == accesses.size()) {
				unknown = true;
				break;
			}
			const auto& resource = shader.info.buffers[i];
			accesses[count++]    = {.begin  = address,
			                        .end    = address + bytes,
			                        .read   = resource.read,
			                        .write  = resource.written,
			                        .atomic = resource.atomic};
		}
		const auto hazard = P7::ChainStep(&buffer, continues_chain, indirect_args, accesses.data(), count, unknown);
		if (hazard != 0 && P7::ChainPrecise()) {
			VulkanMemoryBarrier hazard_barrier {}; // (the alias: <windows.h> defines MemoryBarrier)
			hazard_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
			hazard_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
			                               vk::AccessFlagBits::eIndirectCommandRead;
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
			                          vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eDrawIndirect,
			                          vk::DependencyFlags {}, 1, &hazard_barrier, 0, nullptr, 0, nullptr);
			P7::Add(P7::ChainBarriers);
		}
	}
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, BindPipelineHandle(pipeline));
	if (indirect_args != 0) {
		vk::BufferMemoryBarrier args_barrier {};
		args_barrier.srcAccessMask       = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite |
		                                   vk::AccessFlagBits::eMemoryWrite;
		args_barrier.dstAccessMask       = vk::AccessFlagBits::eIndirectCommandRead;
		args_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		args_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		args_barrier.buffer              = indirect_buffer;
		args_barrier.offset              = indirect_offset;
		args_barrier.size                = 3u * sizeof(uint32_t);
		// (KYTY_INDIRECT_ARGS_BARRIER=1: also within a chain of dispatches - a test of whether the arguments a dispatch
		// before wrote were read before they landed: the culling chain's GPU hangs, 10-13.)
		static const bool args_barrier_always = [] {
			const char* value = std::getenv("KYTY_INDIRECT_ARGS_BARRIER");
			return value != nullptr && value[0] == '1';
		}();
		// (p7, C1: also inside a chain, local/p7-compat.h; live.)
		const bool args_barrier_p7 = continues_chain && !args_barrier_always && P7::ArgsBarrierInChain();
		if (args_barrier_p7) P7::Add(P7::ArgsBarriers);
		if (!continues_chain || args_barrier_always || args_barrier_p7)
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eDrawIndirect,
			                          vk::DependencyFlags {}, 0, nullptr, 1, &args_barrier, 0, nullptr);
		vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
		DeviceFault::NoteDispatch(m_context.GetCommandScheduler().CurrentTick(), program.shader_hash, 0, 0, 0, indirect_args);
	} else if (thread_dimensions) {
		// Thread counts: the groups that cover them, as the normal path's.
		const auto& r      = cs.cs_regs;
		const auto  groups = [](uint32_t threads, uint32_t size) { return (threads + std::max(size, 1u) - 1u) / std::max(size, 1u); };
		vk_buffer.dispatch(groups(groups_x, r.num_thread_x), groups(groups_y, r.num_thread_y), groups(groups_z, r.num_thread_z));
		DeviceFault::NoteDispatch(m_context.GetCommandScheduler().CurrentTick(), program.shader_hash,
		                          groups(groups_x, r.num_thread_x), groups(groups_y, r.num_thread_y),
		                          groups(groups_z, r.num_thread_z), 0);
	} else {
		vk_buffer.dispatch(groups_x, groups_y, groups_z);
		DeviceFault::NoteDispatch(m_context.GetCommandScheduler().CurrentTick(), program.shader_hash, groups_x, groups_y,
		                          groups_z, 0);
	}
	if (chain) buffer.ContinueComputeChain();
	else ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	m_context.GetCommandScheduler().CompleteDispatch();
	// (A later table draw binds its pipeline and set again: the graphics state is untouched.)
	LiveCounters::Add(LiveCounters::TableDispatches);
	return true;
}
