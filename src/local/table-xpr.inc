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
};

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
	};
	std::vector<uint32_t> words; // without the LOD statistics fields (NativeXprLodStatsMask)
	vk::DescriptorSet     set  = nullptr;
	vk::DescriptorPool    pool = nullptr;
	std::vector<Image>    images;
	// Proven current in that frame under those registrations, by role (RenderExecutor::Role: [0] also the frame it was
	// made or refused in); used in that frame (by any executor).
	std::array<uint64_t, Roles> validated_frame = NoRoles();
	std::array<uint64_t, Roles> resolution_epoch {};
	std::atomic<uint64_t>       used_frame {0};
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
	// TableEvaluate's form of a stage's plan (TableRuns): every operand an index into one of three arrays (the stage's
	// user data, its block's slots, the stage's constants), so a draw's evaluation branches on no operand kind.
	struct FlatOperand {
		uint32_t index = 0;
		uint32_t base  = 2; // 0: user data, 1: slots, 2: constants
	};
	struct FlatRun {
		std::array<FlatOperand, 3> address; // low, high, offset
		int32_t                    immediate = 0;
		uint32_t                   first = 0, count = 0;
	};
	struct FlatStage {
		std::vector<FlatRun>     runs;        // the runs in level order
		std::vector<uint32_t>    level_ends;  // the run after each level
		std::vector<FlatOperand> image_words; // every image's words, then every sampler's
		std::vector<uint32_t>    image_counts;
		std::vector<FlatOperand> buffer_words; // every buffer's four V# words (TableBlocks)
		std::vector<uint32_t>    constants;
		uint32_t                 user_words = 0; // the user data it reads: below this
	};
	std::array<FlatStage, 2> flat;
	std::vector<TableVariant>                                   variants;
	uint64_t                                                    store_frame = 0;
	std::unordered_map<uint64_t, std::unique_ptr<TableImageSet>, HashedKey> sets; // by T#/S# words
	std::array<TableImageSet*, Roles>                           last_set {}; // by role
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
};

struct RenderExecutor::TableXpr {
	using Pair    = TablePair;
	using Variant = TableVariant;
	std::unordered_map<uint64_t, Pair, HashedKey> pairs;      // by shader addresses
	std::unordered_map<uint64_t, Pair, HashedKey> dispatches; // by the compute stage's registers (TableDispatchKey)
	// The pairs found last, direct-mapped by key (pairs are never erased and do not move: a pointer stays valid).
	static constexpr size_t PairCacheEntries = 256;
	std::array<std::pair<uint64_t, Pair*>, PairCacheEntries> pair_cache {};
	uint64_t collected  = 0;                           // the frame unused image sets were last dropped
	// Moves when a pair is reset or a set leaves (sets are replaced): an executor reading this catalog drops its last
	// pair, state and set (RenderExecutor::m_catalog) when it moved since it saw it (seen_epoch, its own table's).
	uint64_t epoch = 0, seen_epoch = 0;
	// The next prepared draw of the pair under this state stores (TableStore); the frame's store requests so far.
	Pair*    store       = nullptr;
	uint64_t store_state = 0, store_frame = 0;
	uint32_t store_requests = 0;
	// The draw's T#/S# words (the set's key) and pixel stage LOD statistics words; its runs' addresses.
	std::vector<uint32_t>                words;
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
	struct alignas(64) ResolvedBuffer {
		std::array<uint32_t, 4> words {};
		uint32_t                dwords = 0, special = 0;
		vk::DeviceAddress       device = 0;
		uint64_t                address = 0, size = 0, cpu_epoch = 0, registration = 0;
	};
	static constexpr size_t           ResolvedEntries = 65536;
	std::unique_ptr<ResolvedBuffer[]> resolved = std::make_unique<ResolvedBuffer[]>(ResolvedEntries);
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
	struct WrittenRange {
		std::array<uint32_t, 4> words {};
		uint32_t                special = 0;
		bool                    image_checked = false;
		uint64_t                address = 0, size = 0;
	};
	static constexpr size_t                WrittenEntries = 1024;
	std::array<WrittenRange, WrittenEntries> written;
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
	for (auto& [words, set]: pair.sets) NativeXprRetireSet(set->pool, set->set);
	if (table.store == &pair) table.store = nullptr;
	if (table.last_pair == &pair) table.last_pair = nullptr;
	table.last_set = nullptr;
	++table.epoch;
	// (The key's start instance SGPR too: without it every later draw of the pair failed TableTry's key check.)
	pair = {.vs = pair.vs, .ps = pair.ps, .start_instance = pair.start_instance, .generation = pair.generation,
	        .sightings = pair.sightings, .portable = pair.portable};
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
	for (uint32_t st = 0; st < 2 && pair.programs[st] != nullptr; ++st) {
		const auto& form = pair.flat[st];
		lines(form.runs);
		lines(form.level_ends);
		lines(form.image_words);
		lines(form.image_counts);
		lines(form.buffer_words);
		lines(form.constants);
		lines(pair.programs[st]->table_plan.buffers); // (TableBlocks' plan and resource flags of each V#)
		lines(pair.programs[st]->info.buffers);
	}
	if (set != nullptr) {
		__builtin_prefetch(set);
		lines(set->words);
	}
}

RenderExecutor::TableResult RenderExecutor::TableTry(CommandBuffer& buffer, std::span<const uint64_t> commands,
                                                     uint64_t index_base, uint64_t index_bytes,
                                                     vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	auto& table = OwnTable();
	// (Another executor's catalog is read under its lock.)
	std::shared_lock<std::shared_mutex> catalog_read;
	if (Role() != 0) catalog_read = CatalogRead();
	DropStaleCatalog();
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1e, 0); // census: the table path's phases (b)
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
	const auto                    pair_hash = XXH3_64bits(pair_key.data(), sizeof(pair_key));
	// (Another executor's catalog: what it learned, none of it made here.)
	const auto found = catalog == nullptr || Role() == 0 ? decltype(catalog->pairs.end()) {} : catalog->pairs.find(pair_hash);
	if (Role() != 0 && (catalog == nullptr || found == catalog->pairs.end())) return Spec::Refuse("table store"), TableResult::Native;
	auto* own    = Role() != 0 ? nullptr : &table.pairs;
	auto& cached = table.pair_cache[pair_hash & (TableXpr::PairCacheEntries - 1)];
	auto* known  = own != nullptr && cached.first == pair_hash ? cached.second : nullptr;
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
	auto& pair = *known;
	TablePrefetchPair(pair);
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
	// A store at most once a frame per pair, within the frame's budget (kyty_local_table_store_budget): while its
	// programs and pipeline compile, the pair's draws keep the native path.
	const auto request_store = [&] {
		if (Spec::Current() != nullptr) return Spec::Refuse("table store"), TableResult::Native;
		if (pair.waiting) {
			if (!pair.waiting->load(std::memory_order_acquire)) return TableResult::Native;
			pair.waiting.reset();
		}
		if (table.store_frame != frame) {
			table.store_frame    = frame;
			table.store_requests = 0;
		}
		const auto budget = kyty_local_table_store_budget.load(std::memory_order_relaxed);
		if (pair.store_frame == frame || (budget != 0 && table.store_requests >= budget)) return TableResult::Native;
		++table.store_requests;
		pair.store_frame  = frame;
		table.store       = &pair;
		table.store_state = cache.state_key;
		return TableResult::Store;
	};
	const auto variant = std::ranges::find(pair.variants, cache.state_key, &TableXpr::Variant::state);
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
	return TableDraw(buffer, pair, *variant, draw_state, commands, index_base, index_bytes, index_type, direct);
}

bool RenderExecutor::TableContinue(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
                                   uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	if (kyty_local_table_xpr_mode.load(std::memory_order_relaxed) < 2 || !m_table_xpr || !m_native_xpr) return false;
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
	    emit.state != table.last_state || emit.pipeline != variant.pipeline)
		return false;
	// (The GPU thread's: a speculation's thread does not take it.)
	std::optional<RenderLockGuard> lock;
	if (Role() == 0) lock.emplace(m_context.GetMutex());
	if (!NativeXprTargetsCurrent(*table.last_state)) return false;
	if (TableDraw(buffer, *table.last_pair, variant, *table.last_state, commands, index_base, index_bytes, index_type,
	              direct) != TableResult::Drawn)
		return false;
	LiveCounters::Add(LiveCounters::TableContinued, std::max<size_t>(commands.size(), 1));
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

// What FindTexture's RefreshImage uploads before an image is used: guest bytes the CPU wrote, a GPU buffer write over
// them, or one over its stencil plane.
static bool TableRefreshPending(const Image& image) {
	return image.IsCpuDirty() || image.IsBufferModified() || image.IsStencilModified();
}

// A set's images are the same objects, registered and current (as NativeXprValidateResources proves a record's):
// once a frame, and again when registrations moved.
bool RenderExecutor::TableValidateSet(TableImageSet& set) {
	auto&      textures = m_context.GetTextureCache();
	const auto frame    = m_context.FrameNumber();
	if (set.validated_frame[Role()] == frame && set.resolution_epoch[Role()] == textures.ResolutionEpoch()) return true;
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1e, 7);
	for (const auto& dep: set.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || image->serial != dep.serial) return false;
		// The guest rewrote it (a texture streaming in), or a GPU buffer write its bytes or its stencil plane: upload it
		// as the normal path's FindTexture does.
		if (TableRefreshPending(*image)) {
			if (Spec::Current() != nullptr) return Spec::Refuse("texture upload"), false;
			if (!image->registered || image->depth_id || image->binding.needs_rebind) return false;
			(void)textures.FindTexture(dep.id, dep.desc);
			image = textures.m_slot_images.try_get(dep.id);
			if (image == nullptr || image->serial != dep.serial || TableRefreshPending(*image)) return false;
		}
		if (dep.target) {
			if (!image->registered || image->binding.needs_rebind) return false;
			(void)textures.GetImage(dep.id); // LRU
			continue;
		}
		// (And its LRU, as FindTexture's. A DCC image written as storage, a downsampling chain's levels: TableTransit
		// checks its pending clears at every use, as a sampled one's.) A depth image sampled as a texture (the scene's
		// depth and stencil read by lighting and post passes: at 1-1 their sets were made again every other frame) is
		// found by other rules than a texture's: the proof is that nothing was registered over its view since the set
		// was last proven (or made).
		if (!textures.IsSampledImageCurrent(dep.id, dep.desc, true) &&
		    !textures.IsSampledDepthCurrent(dep.id, dep.desc, set.resolution_epoch[Role()]))
			return false;
	}
	set.validated_frame[Role()]  = frame;
	set.resolution_epoch[Role()] = textures.ResolutionEpoch();
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
	// A run's address, its pages proven mapped and its lines loading.
	const auto locate = [&](uint64_t low, uint64_t high, uint32_t offset, int32_t immediate, uint32_t count,
	                        uint64_t& out) {
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
				            table.addresses[end]))
					return false;
			}
			for (size_t i = begin; i < end; ++i) reader.Read(table.addresses[i], runs[i].count, slots.data() + runs[i].first);
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
		const auto& form    = pair.flat[st];
		// (Slots no stage reads keep what an earlier draw left: the block carries them unread.)
		auto& slots = table.blocks[st];
		slots.resize(program.table_plan.BlockDwords());
		if (table.addresses.size() < pair.runs[st].size()) table.addresses.resize(pair.runs[st].size());
		if (form.runs.size() != pair.runs[st].size() || user_data[st].size() < form.user_words || user_data[st].size() > 64) {
			if (!by_operands(st) || !indirect(st)) return false;
			continue;
		}
		// The flat form (TablePair::flat): every operand one load from one of three arrays. A level's addresses are
		// known together: their lines load at once (the guest's tables are cold).
		const std::array<const uint32_t*, 3> bases {user_data[st].data(), slots.data(), form.constants.data()};
		const auto get = [&](const TablePair::FlatOperand& operand) { return bases[operand.base][operand.index]; };
		uint32_t   begin = 0;
		for (const auto end: form.level_ends) {
			for (auto i = begin; i < end; ++i) {
				const auto& run = form.runs[i];
				if (!locate(get(run.address[0]), get(run.address[1]), get(run.address[2]), run.immediate, run.count,
				            table.addresses[i]))
					return false;
			}
			for (auto i = begin; i < end; ++i) reader.Read(table.addresses[i], form.runs[i].count, slots.data() + form.runs[i].first);
			begin = end;
		}
		const auto first = words.size();
		words.resize(first + form.image_words.size());
		auto*  out = words.data() + first;
		size_t at  = 0;
		for (size_t i = 0; i < form.image_counts.size(); ++i) {
			std::array<uint32_t, 8> image {};
			const auto              count = form.image_counts[i];
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
		const auto found = pair.sets.find(XXH3_64bits(words.data(), words.size() * sizeof(uint32_t)));
		set              = found != pair.sets.end() && found->second->words == words ? found->second.get() : nullptr;
		if (set == nullptr || (set->refused && frame >= set->validated_frame[0] + TableImageSet::RetryFrames))
			set = TableResolveSet(buffer, pair, variant, draw_state, words);
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
		const auto& form = pair.flat[st];
		const bool  flat = form.buffer_words.size() == 4 * plan.buffers.size() && user_data[st].size() >= form.user_words &&
		                  user_data[st].size() <= 64;
		const std::array<const uint32_t*, 3> bases {user_data[st].data(), block.data(), form.constants.data()};
		for (size_t i = 0; i < plan.buffers.size(); ++i) {
			auto& [words4, entry] = entries[i];
			if (flat) {
				for (uint32_t w = 0; w < 4; ++w) {
					const auto& operand = form.buffer_words[4 * i + w];
					words4[w]           = bases[operand.base][operand.index];
				}
			} else {
				for (uint32_t w = 0; w < 4; ++w) {
					const auto& operand = plan.buffers[i][w];
					words4[w] = operand.kind == Kind::Immediate ? operand.value
					            : operand.kind == Kind::Slot    ? block[operand.value]
					            : TableCodeOperand(operand)     ? TableCodeWord(operand, st == 0 ? pair.vs : pair.ps)
					            : operand.value - program.user_data_base < user_data[st].size()
					                ? user_data[st][operand.value - program.user_data_base]
					                : 0;
				}
			}
			entry = &table.resolved[XXH3_64bits(words4.data(), sizeof(words4)) & (TableXpr::ResolvedEntries - 1)];
			__builtin_prefetch(entry);
		}
	}
	table.gathered = &pair;
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
				auto& proof = table.written[XXH3_64bits(words4.data(), sizeof(words4)) & (TableXpr::WrittenEntries - 1)];
				if (proof.special != table.special_generation || proof.words != words4) {
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
					const auto [owner, offset] = buffer_cache.ObtainBuffer(address, size, true, formatted);
					region_epochs.fill({UINT64_MAX, 0}); // (its uploads may have moved them)
					if (owner == nullptr || buffer_cache.GetBufferIfLive(buffer_cache.FindBuffer(address, size)) != owner)
						return false;
					if (formatted) textures.InvalidateMemoryFromGPU(address, size, "storage-buffer");
					device = owner->BufferDeviceAddress() + offset;
				}
				const auto [word, stride]      = word_of(words4);
				block[plan.BufferDword(i)]     = static_cast<uint32_t>(device);
				block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(device >> 32u);
				block[plan.BufferDword(i) + 2] = static_cast<uint32_t>(std::min<uint64_t>(size / 4, UINT32_MAX));
				block[plan.BufferDword(i) + 3] = word;
				block[plan.BufferDword(i) + 4] = stride;
				continue;
			}
			auto& held    = *entries[i].second;
			bool  current = held.words == words4 && (held.special == 0 || held.special == table.special_generation);
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
					current = range_clean(held.address, held.size);
					if (current) held.cpu_epoch = epoch;
				}
			}
			if (!current) {
				LiveCensus::Scope resolve(LiveCensus::DrawPhase, 0x7ab1e, 5);
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
					const bool  cpu_written = buffer_cache.IsRegionCpuDirty(address, size) &&
					                         !buffer_cache.IsRegionGpuModified(address, size);
					if (cpu_written && size <= BufferCache::CACHING_PAGESIZE &&
					    (spec == nullptr || !spec->JournalIntersects(address, size))) {
						// As ObtainBuffer: a small range the guest wrote is read from a copy in the ring (this use only,
						// nothing held), not uploaded into its buffer.
						auto [mapped, offset] = upload.Map(size, 16, false);
						if (mapped != nullptr && Libs::LibKernel::Memory::TryReadBackingToHost(address, mapped, size)) {
							upload.Commit();
							if (spec != nullptr) spec->NoteCleanRead(address, size);
							const auto [word, stride]      = word_of(words4);
							block[plan.BufferDword(i)]     = static_cast<uint32_t>(upload.BufferDeviceAddress() + offset);
							block[plan.BufferDword(i) + 1] = static_cast<uint32_t>((upload.BufferDeviceAddress() + offset) >> 32u);
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
	bool bounded = true;
	for (uint32_t st = 0; st < 2 && bounded; ++st) {
		if (!global(st)) continue;
		const auto& program = *pair.programs[st];
		snapshot.user_data.assign(user_data[st].begin(), user_data[st].end());
		snapshot.flattened_srt.assign(table.blocks[st].begin(), table.blocks[st].begin() + program.table_plan.slots.size());
		bounded = ShaderRecompiler::IR::EvaluateBdaReadPlan(program.bda_read_plan, snapshot, stage_ranges);
		ranges.insert(ranges.end(), stage_ranges.begin(), stage_ranges.end());
	}
	auto& resources = m_context.GetGpuResources();
	if (!bounded || !resources.PrepareBdaReadRanges(ranges)) resources.PrepareBda();
}

// A set's images in their layouts (other work may have moved them since the set was last bound), unless a DCC image
// has a fast clear pending (the normal path, whose FindTexture applies it first, takes the draw or dispatch) or the
// draw's targets are not those the set was made for (an image held as a texture became a target, or the reverse).
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
		if ((dep.storage && !image->SafeToDownload()) || TableRefreshPending(*image)) return false;
		if (dep.desc.info.metadata.kind == ImageMetadataKind::Dcc) {
			if (image->info.IsVolume()) return false;
			const auto address = dep.desc.info.metadata.range.address;
			for (uint32_t layer = dep.range.base_layer; layer < dep.range.base_layer + dep.range.layer_count; ++layer) {
				if (layer >= 32) return false;
				if (!textures.IsMetaCleared(address, layer)) continue;
				// The normal path's FindTexture clears its layers first. A speculation's packet is a hole that does:
				// the work after it finds them cleared (speculation-state.h).
				if (auto* spec = Spec::Current())
					for (uint32_t at = layer; at < std::min(dep.range.base_layer + dep.range.layer_count, 32u); ++at)
						if (textures.IsMetaCleared(address, at)) spec->ConsumeMeta(address, at);
				return Spec::Refuse("table: cleared texture"), false;
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
	// (The index buffer's held entry, a cold line of a 320 KiB table, loads while the tables are evaluated.)
	if (index_base != 0) {
		const auto* held = &table.indices[(index_base * 0x9e3779b97f4a7c15ull) >> 51u];
		__builtin_prefetch(held);
		__builtin_prefetch(reinterpret_cast<const char*>(held + 1) - 1);
	}
	std::optional<LiveCensus::Scope> phase;
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 1);
	if (!TableEvaluate(pair, user_data)) return Spec::Refuse("table: evaluation"), TableResult::Native;
	TableGather(pair, user_data);
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 2);
	const auto* set = TableSet(buffer, pair, &variant, &draw_state);
	if (auto* spec = Spec::Current(); spec != nullptr && set != nullptr)
		spec->NoteSet(reinterpret_cast<uint64_t>(static_cast<VkDescriptorSet>(set->set)));
	if (set == nullptr) return Spec::Refuse("table: no set"), TableResult::Native;

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
	TablePrepareBda(pair, user_data);
	// The blocks take each buffer's device address as it resolves: a later lookup that registers a buffer (a merge
	// replaces an earlier one) leaves an earlier address stale, where the normal path binds after all its lookups
	// (RebindBuffers). Such a draw takes the normal path; the next use finds the buffers registered.
	const auto registration = buffer_cache.RegistrationEpoch();
	if (!TableBlocks(pair, user_data, blocks, &writes)) return Spec::Refuse("table: blocks"), TableResult::Native;
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
	if (buffer.IsInvalid()) return Spec::Refuse("table: buffer"), TableResult::Native;
	const auto vk_buffer    = buffer.Handle();
	const bool same_command = emit.command == vk_buffer && emit.generation == buffer.GraphicsGeneration();
	// The targets and the set's images as the last table draw in this command buffer left them (the same state and
	// set), no image's state, registration, metadata or bytes changed since: in place, nothing to check or transit.
	auto&      texture_cache = m_context.GetTextureCache();
	const bool in_place      = same_command && emit.state == &draw_state && table.last_set == set &&
	                      table.images_epoch == texture_cache.ImagesEpoch() && Spec::Current() == nullptr;
	if (!in_place) {
		if (!NativeXprEmitTargets(draw_state, vk_buffer)) return Spec::Refuse("table: targets"), TableResult::Native;
		// The set's images in their layouts (a DCC image with a fast clear pending takes the normal path, whose
		// FindTexture applies it first).
		if (!TableTransit(*set, vk_buffer, &draw_state)) return Spec::Refuse("table: transit"), TableResult::Native;
	}
	m_context.GetCommandScheduler().BeginRendering(draw_state.rendering);
	if (same_command && emit.state != &draw_state && emit.state != nullptr &&
	    emit.state->dynamic_hash == draw_state.dynamic_hash &&
	    std::memcmp(&emit.state->dynamic, &draw_state.dynamic, sizeof(draw_state.dynamic)) == 0)
		emit.state = &draw_state;
	if (!same_command || emit.state != &draw_state) NativeXprEmitDynamic(cache, vk_buffer, draw_state, same_command);
	const auto& pipeline = *variant.pipeline;
	if (!same_command || emit.pipeline != &pipeline)
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
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
	for (uint32_t group = 0; group < groups; ++group)
		vk_buffer.drawIndexedIndirect(arguments[group].first->Handle(), arguments[group].second, group_sizes[group],
		                              static_cast<uint32_t>(stride));
	// As the normal path's: shaders that write buffers end the rendering, and a barrier orders their writes.
	if (writes) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader);
	}
	emit.command     = vk_buffer;
	emit.generation  = buffer.GraphicsGeneration();
	emit.state       = &draw_state;
	emit.pipeline    = &pipeline;
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
	return TableResult::Drawn;
}

// The pair's runs (TablePair::Run): the slots its stages evaluate, in reads of consecutive slots, by level.
void RenderExecutor::TableRuns(TablePair& pair) {
	using Kind  = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto& table = *m_table_xpr;
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
			       plan.slots[first + count].immediate == slot.immediate + static_cast<int32_t>(4 * count))
				++count;
			uint32_t level = 0;
			for (const auto* operand: {&slot.low, &slot.high, &slot.offset})
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
		auto&       flat    = pair.flat[st];
		flat                = {};
		flat.constants.push_back(0);
		const auto flat_of = [&](const ShaderRecompiler::IR::TablePlan::Operand& operand) -> TablePair::FlatOperand {
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
			                     run.read.immediate, run.first, run.count});
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
}

// After a prepared draw (`storable`: a draw without vertex buffers) of a pair the table path asked to
// store: the pair's table mode programs (with this draw's specializations), the pipeline for this draw state,
// compiled by a worker, and the draw state.
void RenderExecutor::TableStore(CommandBuffer& buffer, const DrawRenderState& state, vk::PrimitiveTopology topology,
                                bool primitive_restart_enable, const RenderState& rendering, bool storable) {
	auto&       table = *m_table_xpr;
	auto* const pair  = std::exchange(table.store, nullptr);
	if (!storable || pair == nullptr) return;
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
	table_state.programs = pipelines.GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx,
	                                                     buffer.GetUserConfig(), mapping, state.ps_active,
	                                                     table_state.vs_input_info, table_state.ps_input_info, true,
	                                                     pair->portable);
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
static uint64_t TableDispatchKey(const HW::ComputeShaderInfo& cs, bool thread_dimensions) {
	const auto& r = cs.cs_regs;
	const std::array<uint64_t, 4> key {r.data_addr, uint64_t {r.num_thread_x} | uint64_t {r.num_thread_y} << 32u,
	                                   uint64_t {r.num_thread_z} | uint64_t {r.lds_size} << 32u,
	                                   uint64_t {r.wave_size} | uint64_t {r.user_sgpr} << 8u |
	                                       uint64_t {r.tidig_comp_cnt} << 16u | uint64_t {r.tgid_x_en} << 24u |
	                                       uint64_t {r.tgid_y_en} << 25u | uint64_t {r.tgid_z_en} << 26u |
	                                       uint64_t {r.tg_size_en} << 27u | uint64_t {r.scratch_en} << 28u |
	                                       uint64_t {thread_dimensions} << 29u};
	return XXH3_64bits(key.data(), sizeof(key));
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
}

bool RenderExecutor::TableDispatch(CommandBuffer& buffer, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
                                   uint64_t indirect_args, bool thread_dimensions) {
	auto* catalog = CatalogTable();
	m_dispatch_declined = "dispatch: nothing learned";
	if (catalog == nullptr || FrameCapture::Active() || XprCapture::Enabled() || FrameGen::Enabled()) return false;
	std::shared_lock<std::shared_mutex> catalog_read;
	if (Role() != 0) catalog_read = CatalogRead();
	DropStaleCatalog();
	const auto& sh    = buffer.GetShaders();
	const auto& cs    = sh.GetCs();
	const auto  found = catalog->dispatches.find(TableDispatchKey(cs, thread_dimensions));
	// (Another executor's: a pair with its programs and pipeline had its sightings.)
	if (found == catalog->dispatches.end()) return m_dispatch_declined = "dispatch: not seen", false;
	if (found->second.refused)
		return m_dispatch_declined = found->second.special != nullptr ? found->second.special : "dispatch: refused", false;
	if ((Role() == 0 && found->second.sightings < 2) || found->second.vs != cs.cs_regs.data_addr) return false;
	auto& pair = found->second;
	TablePrefetchPair(pair);
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
	if (!TableEvaluate(pair, user_data)) return m_dispatch_declined = "table: tables unmapped", false;
	TableGather(pair, user_data);
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 2);
	const auto* set = TableSet(buffer, pair, nullptr, nullptr);
	if (set == nullptr) return false;
	if (auto* spec = Spec::Current()) spec->NoteSet(reinterpret_cast<uint64_t>(static_cast<VkDescriptorSet>(set->set)));
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 3);
	std::array<vk::DeviceAddress, 2> blocks {};
	bool                             storage_writes = false;
	auto&                            cache          = m_context.GetBufferCache();
	TablePrepareBda(pair, user_data); // (as TableDraw's: before the blocks)
	const auto                       registration   = cache.RegistrationEpoch(); // (as TableDraw's)
	if (!TableBlocks(pair, user_data, blocks, &storage_writes)) return m_dispatch_declined = "table: blocks", false;
	vk::Buffer indirect_buffer = nullptr;
	uint64_t   indirect_offset = 0;
	if (indirect_args != 0) {
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
	const bool chain     = DemonsSouls::IsSupportedGame();
	const auto vk_buffer = chain ? buffer.ChainHandle() : buffer.Handle();
	if (!TableTransit(*set, vk_buffer, nullptr)) return m_dispatch_declined = "table: transit", false;
	// (A set made for another specialization: its program's pipeline and shader data.)
	const auto& pipeline = set->pipeline != nullptr ? *set->pipeline : *pair.pipeline;
	const auto& program  = set->program != nullptr ? *set->program : *pair.programs[0];
	const auto& bindings = program.bindings;
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
	if (storage_writes && !continues_chain) ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
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
		if (!continues_chain)
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eDrawIndirect,
			                          vk::DependencyFlags {}, 0, nullptr, 1, &args_barrier, 0, nullptr);
		vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
	} else if (thread_dimensions) {
		// Thread counts: the groups that cover them, as the normal path's.
		const auto& r      = cs.cs_regs;
		const auto  groups = [](uint32_t threads, uint32_t size) { return (threads + std::max(size, 1u) - 1u) / std::max(size, 1u); };
		vk_buffer.dispatch(groups(groups_x, r.num_thread_x), groups(groups_y, r.num_thread_y), groups(groups_z, r.num_thread_z));
	} else {
		vk_buffer.dispatch(groups_x, groups_y, groups_z);
	}
	if (chain) buffer.ContinueComputeChain();
	else ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	m_context.GetCommandScheduler().CompleteDispatch();
	// (A later table draw binds its pipeline and set again: the graphics state is untouched.)
	LiveCounters::Add(LiveCounters::TableDispatches);
	return true;
}
