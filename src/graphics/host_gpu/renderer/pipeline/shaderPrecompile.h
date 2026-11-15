#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERPRECOMPILE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERPRECOMPILE_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

// Marvel's Wolverine's shader and pipeline precompile, Phase A (side-session-notes/precompile-port-plan.md): the Ghost
// of Yotei fork's record -> hints -> import -> precompile (side-session-notes/precompile-yotei-handoff.md), on lean.
//
// A run records every permutation the GPU thread translates (a stage record: every input of its translation) and every
// graphics pipeline made of recorded stages (a pipeline record) in the title's journal. A precompile translates each
// record as the GPU thread did and creates its pipeline into the driver cache, so the game's own create is a hit.
// Hints are a journal without the game's code: each code block becomes its size and hashes, and an import finds the
// blocks in the user's own game files by the shader footer.
namespace Libs::Graphics::Precompile {

using InputInfo = std::variant<ShaderVertexInputInfo, ShaderPixelInputInfo, ShaderComputeInputInfo>;

// One permutation: every input ProgramCache::Get translates and compiles it from.
struct StageRecord {
	ShaderType                                   stage           = ShaderType::Unknown;
	uint64_t                                     shader_hash     = 0;
	uint32_t                                     user_data_count = 0; // the translation reads the count only
	std::vector<uint32_t>                        code;
	std::vector<uint32_t>                        back_code; // a merged GS's back half
	// A shader with calls: each call site's whole callee shader, in call order (the linker reads the leaf from it).
	std::vector<std::vector<uint32_t>>           callees;
	InputInfo                                    input;     // its stage runtime cleared
	uint32_t                                     push_data_cursor = 0; // as Get received it
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	uint64_t                                     spirv_hash = 0;
	uint64_t                                     origin     = 0; // Origin() of the run that recorded it

	// Stage, shader hash, user-data count, code size, static key, push cursor and specialization: the permutation;
	// with callees, also their code hashes.
	[[nodiscard]] uint64_t Key() const;
};

// One graphics pipeline, by its stages' record keys and its state. A compute record is its own pipeline.
struct PipelineRecord {
	std::array<uint64_t, 3>  vertex {}; // VS, mesh or LS; then HS and DS when tessellated
	uint64_t                 pixel = 0; // 0 without a pixel stage
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;

	[[nodiscard]] uint64_t Key() const;
};

// The draw a graphics pipeline was recorded at, without its pixel shader (Phase B: seeds for pixel shaders no route
// recorded, side-session-notes/12-phaseB-plan.md): what PreparePixelStage and MakeGraphicsKey read from the draw, so
// another pixel shader can be prepared and keyed as this draw would have done it. The register blocks are raw bytes
// with their size: a record of another layout is rejected alone, and the schema of journals and hints stays as it is.
struct TemplateRecord {
	struct Color {
		uint32_t slot = 0, format = 0, guest_format = 0, samples = 1;
		uint8_t  mapping = 0;
	};
	uint64_t                pixel    = 0; // the stage record and pipeline record the draw made
	uint64_t                pipeline = 0;
	std::vector<uint8_t>    context, user_config, pixel_regs; // HW::Context, HW::UserConfig, HW::PixelShaderInfo
	std::array<uint8_t, 8>  export_mapping {};                 // DrawPixelExports, packed
	std::vector<Color>      colors;                            // as ResolveRenderColorTarget gave them
	uint32_t                depth_format = 0, depth_samples = 1;
	bool                    depth_bound = false, depth_test = false;
	uint32_t                topology = 0;
	bool                    primitive_restart = false;

	[[nodiscard]] uint64_t Key() const;
};

// XXH3 of the code, as the shader map hashes it; a merged GS hashes its two halves' hashes.
uint64_t CodeHash(std::span<const uint32_t> code, std::span<const uint32_t> back_code);
// The input layouts every file is written in: a file of another layout is not read.
uint64_t SchemaFingerprint();
// What the SPIR-V of a record depends on besides its inputs: the build and the switches that change translation
// (an allow-list). Records keep it, and the replay check and the warm list compare it; record keys do not hold it.
uint64_t Origin();
// vendorID:deviceID: a journal records formats and limits of one GPU.
uint64_t DeviceFingerprint(uint32_t vendor_id, uint32_t device_id);
// The stage input with its runtime pointers cleared, for a record.
InputInfo RecordedInput(const ShaderVertexInputInfo& input);
InputInfo RecordedInput(const ShaderPixelInputInfo& input);
InputInfo RecordedInput(const ShaderComputeInputInfo& input);

struct Journal {
	std::vector<StageRecord>    stages;    // in file order: the order a run first needed them
	std::vector<PipelineRecord> pipelines; // in file order
	std::vector<TemplateRecord> templates; // in file order; not in hints
	uint32_t                    rejected = 0; // records whose hash did not match
	bool                        torn     = false; // a size ran past the end: a killed run's last record
	bool                        found    = false; // the file exists
	bool                        foreign  = false; // another layout or GPU

	// Drops duplicate keys, keeping the first.
	void Dedupe();
};

// Reads a journal; device 0 accepts any GPU.
Journal ReadJournal(const std::filesystem::path& path, uint64_t device);
bool    WriteJournal(const std::filesystem::path& path, uint64_t device, const Journal& journal);
// The records in the buffer table variant this run draws with; pipeline and template records follow their stages'
// keys. Run it after every record is in (the shipped hints are recorded in the normal variant). True if anything
// changed.
bool    MigrateBufferTable(Journal& journal);

// Appends records to a journal as they come, each flushed at once: a crash loses the record being written at most.
class JournalWriter {
public:
	// Rewrites the file from journal when rewrite, else appends (writing the header to a new file).
	bool Open(const std::filesystem::path& path, uint64_t device, const Journal& journal, bool rewrite);
	void Append(const StageRecord& record);
	void Append(const PipelineRecord& record);
	void Append(const TemplateRecord& record);
	[[nodiscard]] bool IsOpen() const { return m_file.is_open(); }

private:
	void          Write(std::span<const uint32_t> payload);
	std::mutex    m_mutex;
	std::ofstream m_file;
};

// Hints: a journal without code. Each code block keeps {bytes, footer hash, XXH3}; the import finds it in the game
// files by its footer. Every Wolverine shader ends in a 48-byte footer that starts with "barefoot".
struct HintsInfo {
	std::string title_id;
	std::string app_version;
};
struct ExportResult {
	bool     ok        = false;
	uint32_t stages    = 0;
	uint32_t pipelines = 0;
	uint32_t no_footer = 0; // stages whose code does not end in its footer: programs, found by the footer after them
};
ExportResult ExportHints(const Journal& journal, const HintsInfo& info, const std::filesystem::path& path);
struct ImportResult {
	bool        ok = false;
	std::string reason; // other_emulator_version, other_game, unreadable
	uint32_t    stages = 0, pipelines = 0, missing = 0;
	uint64_t    bytes_read = 0;
	double      seconds    = 0;
	Journal     journal; // the restored records
};
// Finds the hints' code blocks in the files under game_dir (eboot.bin first, then the material archives, then the
// rest) and restores the records. adjust fixes each record's host fields for this GPU.
ImportResult ImportHints(const std::filesystem::path& path, const std::filesystem::path& game_dir, const HintsInfo& info,
                         const std::function<void(StageRecord&)>& adjust,
                         const std::function<void(uint64_t done, uint64_t total)>& progress = {});
// The hints the emulator ships, hints/<title_id>.kyhints in its own folder (the launcher's install), or empty.
std::filesystem::path ShippedHints(const std::string& title_id);
// A hints file's size and time, "" when it is missing: a change means another import.
std::string HintsStamp(const std::filesystem::path& path);
// The hint of one code block: 0 footer hash without the footer.
uint64_t FooterHash(std::span<const uint8_t> block);

// Which recorded keys the saved driver cache already holds, per build: version 2 of the Yotei fork's warm list, the
// running build's section first and the two saved before it.
class WarmList {
public:
	// Drops the list when the file is missing, damaged, of another driver, or the cache shrank since its save.
	void Load(const std::filesystem::path& path, const std::string& signature, uint64_t cache_bytes);
	[[nodiscard]] bool Contains(uint64_t key, bool graphics) const;
	void               Mark(uint64_t key, bool graphics);
	// Before the driver cache's data is taken: the marks since the last save.
	std::vector<uint64_t> TakeMarked();
	// After that data was written: the marks are trusted, and the list is rewritten.
	void Commit(std::vector<uint64_t> entries, bool written, uint64_t cache_bytes);
	[[nodiscard]] size_t Size() const;

	static uint64_t Entry(uint64_t key, bool graphics);

private:
	mutable std::mutex           m_mutex;
	std::filesystem::path        m_path;
	std::string                  m_signature;
	std::unordered_set<uint64_t> m_saved;  // this build's
	std::vector<std::pair<uint64_t, std::vector<uint64_t>>> m_older; // up to two earlier builds, as read
	std::vector<uint64_t>        m_marked;
};

// A stage the precompile translated: its module and what pipeline creation reads.
struct StageModule {
	vk::ShaderModule                         module     = nullptr;
	uint64_t                                 spirv_hash = 0;
	ShaderRecompiler::IR::CompiledShaderInfo info;
	ShaderRecompiler::IR::ResourceSnapshot   resources; // empty: layouts read the program only
	InputInfo                                input;     // the record's, its runtime pointing at info
};

// The pipeline cache's side of a precompile.
class Compiler {
public:
	virtual ~Compiler() = default;
	// Translates and compiles a record as ProgramCache::Get would; false when it cannot (calls, gave up). The SPIR-V
	// and program also go to the reuse store when keep.
	virtual bool CompileStage(const StageRecord& record, bool keep, StageModule& out) = 0;
	virtual void DestroyStage(StageModule& stage) = 0;
	// keep false: never kept for the GPU thread (a paired or predicted pipeline, which no run recorded).
	virtual bool CreateGraphics(const PipelineRecord& record, std::span<StageModule* const> vertex,
	                            StageModule* pixel, bool keep) = 0;
	virtual bool CreateCompute(uint64_t key, StageModule& stage) = 0; // key: the compute record's
};

struct Progress {
	std::atomic<bool>     started {false}, done {false};
	std::atomic<uint32_t> threads {0};
	std::atomic<uint64_t> total {0}, finished {0}, compute {0}, compute_finished {0}, graphics {0},
	    graphics_finished {0}, compiled {0}, warm {0}, suspects {0}, mismatches {0}, failed {0}, dropped {0},
	    paired {0}, recursored {0}; // recursored: table jobs whose stage keys moved to the draw's push cursors (Run)
};

struct Options {
	uint32_t              threads   = 1;
	bool                  only      = false; // --precompile-only: every pipeline is created
	bool                  reuse     = false; // keep translations for the GPU thread (the reuse store)
	bool                  hand_over = false; // keep created pipelines for the GPU thread: warm ones are created too
	std::filesystem::path inflight; // the running jobs' keys, for suspects after a crash
	std::unordered_set<uint64_t> suspects; // jobs dropped (LoadSuspects)
	// Vertex and mesh shaders (by hash) whose recorded pipelines are also created with each other vertex record set the
	// journal holds for the shader in the same state: pipelines no run recorded, made of recorded parts only, queued
	// after every recorded job and never handed over.
	std::unordered_set<uint64_t> pair_vertex_shaders;
	// Pixel records whose last buffer entry is empty and which have a plain 2D image also get the record of the same
	// shader with that image 3D and the buffer filled (PredictVolumeForm), over each recorded pipeline of the record:
	// pipelines no run recorded, of a stage no run recorded, queued after every recorded job and never handed over.
	bool predict_volume_pixels = false;
};

// The precompile: workers below normal priority translate the journal's records and create their pipelines into
// the driver cache. The GPU thread never waits for them except on a pipeline a worker is creating right now.
class Precompiler {
public:
	Precompiler(Compiler& compiler, Options options, Journal journal, const WarmList* warm);
	~Precompiler();
	KYTY_CLASS_NO_COPY(Precompiler);

	void Start();
	// Lets the running jobs finish and keeps the rest queued; joins the workers.
	void Stop();
	// Until every job is done or Stop was called.
	void Wait();
	// No worker runs: every job is done, or Stop was called, or none was started.
	[[nodiscard]] bool Finished();
	[[nodiscard]] const Progress& GetProgress() const { return m_progress; }
	[[nodiscard]] bool            Empty() const { return m_jobs.empty(); }

	// The GPU thread, before it creates a pipeline (a compute record's key or a pipeline record's): a queued job for
	// it is dropped; one being created is waited for, its worker raised to normal priority. True when a worker created
	// it: the driver cache holds it.
	bool BeforeCreate(uint64_t key);
	// A worker's created pipeline, for the warm list.
	std::function<void(uint64_t key, bool graphics)> on_created;
	// The game needs shaders now (the GPU thread compiled one itself while jobs remain): the workers run at normal
	// priority from here on. Once; any thread.
	void Hurry();

private:
	enum class State : uint8_t { Queued, Running, Done, Dropped };
	struct Stage {
		StageRecord*                 record = nullptr; // its code is dropped once translated
		std::unique_ptr<StageModule> module;
		uint32_t                     users  = 0; // jobs still to use it
		bool                         busy   = false, failed = false, ready = false;
		// A table job's stage keeps its code until its last job is done: Run may translate a copy at another cursor.
		bool                         keep_code = false;
	};
	struct Job {
		uint64_t              key      = 0;
		bool                  graphics = false;
		bool                  create   = true; // false: a warm pipeline, translated only for the reuse store
		const PipelineRecord* pipeline = nullptr;
		std::array<uint64_t, 4> stages {}; // its stage keys, 0 for none
		bool                  paired   = false; // made by pairing or prediction (Options), not recorded
		bool                  table    = false; // a stage of it is in the buffer table variant
		State                 state    = State::Queued;
		bool                  creating = false; // its stages are ready and its pipeline is being created
		bool                  created  = false; // into the driver cache
		uint32_t              worker   = 0;
	};

	void Worker(uint32_t index);
	void Run(Job& job, uint32_t index);
	StageModule* AcquireStage(uint64_t key);
	StageModule* AcquireAt(uint64_t key, uint32_t cursor, uint64_t& draw_key, std::vector<uint64_t>& extra);
	void         ReleaseStageLocked(uint64_t key);
	void         ReleaseStagesLocked(const Job& job);
	void         NoteInflightLocked() { m_inflight_version++; }
	void         WriteInflight();

	Compiler&                               m_compiler;
	Options                                 m_options;
	Journal                                 m_journal;
	std::unordered_map<uint64_t, Stage>     m_stages;
	std::vector<Job>                        m_jobs;
	std::deque<PipelineRecord>              m_paired; // the paired and predicted jobs' records
	std::deque<StageRecord>                 m_predicted_stages; // Options::predict_volume_pixels, and AcquireAt's copies
	std::unordered_map<uint64_t, size_t>    m_by_key; // job index
	size_t                                  m_next = 0;
	std::unordered_set<uint64_t>            m_running_keys;
	uint64_t                                m_inflight_version = 0, m_inflight_written = 0; // under m_mutex
	std::mutex                              m_inflight_mutex; // one writer of the in-flight file at a time
	std::mutex                              m_mutex;
	std::condition_variable                 m_changed;
	std::vector<std::thread>                m_workers;
	std::vector<void*>                      m_worker_handles; // to raise one the GPU thread waits on
	bool                                    m_stop = false;
	std::atomic<bool>                       m_hurry {false};
	uint32_t                                m_live = 0;
	Progress                                m_progress;
	std::chrono::steady_clock::time_point   m_start;
};

// Lowers the calling thread below normal priority; raises a worker back to normal.
void  LowerCurrentThreadPriority();
void* CurrentThreadHandle();
void  SetThreadPriorityNormal(void* handle, bool normal);
void  CloseThreadHandle(void* handle);

// The pipelines handed over to the GPU thread (PipelineCache's ready map) and their keys, oldest first: past max
// entries the oldest go to destroy and leave the map. Keys the GPU thread has taken since are skipped.
template <typename Map, typename Destroy>
void TrimOldest(Map& map, std::deque<uint64_t>& order, size_t max, Destroy&& destroy) {
	while (map.size() > max && !order.empty()) {
		const auto found = map.find(order.front());
		order.pop_front();
		if (found == map.end()) continue;
		destroy(found->second);
		map.erase(found);
	}
}

// Suspects: keys left in the in-flight file by a run that crashed on them, for this origin only.
std::unordered_set<uint64_t> LoadSuspects(const std::filesystem::path& inflight, const std::filesystem::path& suspects);
// Clears the in-flight file on a clean quit (std::quick_exit runs no destructors).
void RegisterInflightCleanup(const std::filesystem::path& inflight);

} // namespace Libs::Graphics::Precompile

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERPRECOMPILE_H_
