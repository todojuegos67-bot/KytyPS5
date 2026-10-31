#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_SPECULATION_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_SPECULATION_H_

#include "common/common.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "speculation-state.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

extern "C" {
// 1: graphics command buffers are translated speculatively and committed; 3: the graphics queue's command buffers
// queued after one translated ahead; 4: ahead on the speculation's threads, from when they are submitted
// (KYTY_SPECULATE).
extern volatile std::atomic_uint32_t kyty_local_speculate_mode;
// KYTY_SPECULATE=4: its threads (KYTY_SPECULATE_THREADS).
extern volatile std::atomic_uint32_t kyty_local_speculate_threads;
}

namespace Libs::Graphics {

class StreamBuffer;

// Speculative translation of graphics command buffers (KYTY_SPECULATE, docs/PARALLEL-TRANSLATION.md): a command
// buffer is translated into command buffers of its own, by its own processor and executors, without changing what
// other translations see (src/local/speculation-state.h), then committed in guest order: the uploads, page-table
// preparation and image transitions its work needs before it, its command buffers, then the effects it noted after.
// A draw, dispatch, end-of-pipe event or wait it cannot translate is a hole, translated the normal way at commit
// between the work before it and the work after it (a wait not satisfied then blocks the commit); another packet it
// cannot translate stops it, and the command buffer is translated on from there the normal way. A wait on memory only
// the CPU writes stops it too: what the guest writes before the label is for the commands after it
// (bdaDirtyRegions.h). The speculation's executors draw with what the graphics queue's learned
// (RenderExecutor::ReadCatalogOf); its holes and the rest of the command buffer are translated by those.
class Speculation {
public:
	explicit Speculation(RenderContext& renderer);
	~Speculation();
	KYTY_CLASS_NO_COPY(Speculation);

	// Blocked: committed up to a wait hole not satisfied yet (the submission waits; Resume goes on from there). Chunk:
	// committed up to the end of its chunk (mode 4: the execution is where the next begins).
	enum class Outcome { Whole, Partly, None, Blocked, Chunk };
	// The command buffer translated speculatively from `cp`'s state, from where `execution` is (its start, or a wait the
	// normal translation stopped at), and committed: whole (`cp` then has its state after it), partly or not at all
	// (the rest is translated the normal way from `execution`, where `cp` has the state; nothing committed: as it was).
	[[nodiscard]] Outcome TranslateAndCommit(CommandProcessor& cp, uint64_t serial, std::span<const uint32_t> commands,
	                                         Pm4Execution& execution);
	// The commit blocked at a wait of the command buffer with that serial, gone on with.
	[[nodiscard]] bool    Blocked(uint64_t serial) const noexcept { return m_blocked && m_blocked->serial == serial; }
	[[nodiscard]] Outcome Resume(CommandProcessor& cp, Pm4Execution& execution);
	// A submission of the graphics queue: its place in the queue, a graphics command buffer a speculation takes (its
	// commands), a flip preparation (between them: the processor state goes on), or another (ends a run ahead); whether
	// the processor is reset before it.
	struct Job {
		enum class Kind { Graphics, Flip, Other };
		uint64_t                  serial = 0;
		std::span<const uint32_t> commands;
		bool                      reset = false;
		Kind                      kind  = Kind::Graphics;
	};
	// KYTY_SPECULATE=3 (run-ahead): the command buffer committed from its start as TranslateAndCommit does, from the
	// speculation translated ahead for it when that started from the state `cp` has, and the graphics queue's command
	// buffers queued after it (`queued`, in order) translated ahead, each from the state the one before leaves (and with
	// its pending guest memory writes: Spec::State::previous).
	[[nodiscard]] Outcome CommitAhead(CommandProcessor& cp, const Job& job, std::span<const Job> queued,
	                                  Pm4Execution& execution);
	// KYTY_SPECULATE=4: the graphics queue's submissions, as they are submitted (Push), pass a state pass (in order:
	// the processor state each leaves, CommandProcessor::SetStateOnly), which cuts them into chunks (ChunkWork draws
	// and dispatches) whose start states and positions it gives, and the speculation's threads translate the chunks,
	// each from the state the pass gave it (reading through the pending writes of the chunk before: its translation, or
	// the pass's record of it). The GPU thread commits them in order (CommitReady, chunk by chunk: None when there is
	// none for it, then it translates that chunk the normal way, up to `end`: where the next begins, none: the end, or
	// it did not start from the state `cp` has), waiting only for one being translated, and gives the state after each
	// submission (Done): a state pass that could not follow one goes on from there.
	void Push(const Job& job);
	[[nodiscard]] Outcome CommitReady(CommandProcessor& cp, const Job& job, uint32_t chunk, Pm4Execution& execution,
	                                  std::optional<Pm4Execution>& end);
	void                  Done(uint64_t serial, const CommandProcessor& cp);
	static constexpr uint32_t ChunkWork = 128;
	// The graphics queue's time in a slice of a command buffer (the report's total).
	void NoteQueueTime(uint64_t ns) noexcept { m_queue_ns += ns; }
	// What the normal translation does while a speculation is not committed (Spec::t_effects of the thread that
	// commits).
	[[nodiscard]] Spec::EffectsLog& Log() noexcept { return m_log; }
	[[nodiscard]] static uint64_t NowNs() noexcept;

private:
	// A packet left to the commit (Spec::State::closed: it follows the segment of its index), why, and the
	// speculation's execution past it.
	struct Hole {
		const uint32_t* packet = nullptr;
		uint32_t        dwords = 0;
		const char*     reason = nullptr;
		Pm4Execution    after;
	};
	enum class Progress { Complete, Stopped, Blocked };
	enum class HoleRun { Done, Changed, Blocked };
	// What the speculation's work assumed of the caches: no buffer or image registered or released, no mapping changed
	// since (and the surface metadata it found not cleared still not: Spec::State::meta_facts).
	struct Epochs {
		uint64_t buffers = 0, images = 0, mapping = 0;
		bool     operator==(const Epochs&) const = default;
	};
	struct Translator;
	// A command buffer's speculative translation, until it is committed: its segments' work (Spec::State, recorded
	// into its recorder), the holes between them and the processor state each is translated from, the processor state
	// at its end or stop; the translator whose it is.
	struct Result {
		Translator*                                    owner  = nullptr;
		uint64_t                                       serial = 0; // its command buffer's (Job::serial)
		uint32_t                                       chunk  = 0; // (mode 4)
		uint64_t                                       pass   = 0; // the state pass it started from (mode 4)
		Spec::State                                    state;
		std::vector<Hole>                              holes;
		std::vector<std::unique_ptr<CommandProcessor>> hole_states; // (more than holes: kept for reuse)
		std::unique_ptr<CommandProcessor>              start_state, end_state;
		std::unique_ptr<CommandScheduler::Recorder>    recorder;
		Epochs                                         epochs;
		uint64_t                                       effects_from = 0; // the effects log's end when it began
		// Why it stopped before the end (null: it did not; stop_execution is at that packet), whether nothing it
		// recorded can be committed, whether it translated anything.
		const char*  stop = nullptr;
		Pm4Execution stop_execution;
		Pm4Execution start_at; // where its translation began
		bool         failed = false, progressed = false;
		bool         blind  = false; // translated without the one before it while that was not done (mode 4)
		size_t       submitted = 0, next = 0; // recorded command buffers submitted, segments committed
		uint64_t     last_tick = 0;           // the tick the last of them was submitted with (0: none)
		size_t       blocked   = SIZE_MAX;    // the wait hole the commit waits at
		uint64_t     retired   = 0;           // when Finish kept it for reuse (m_clock; 0: in use)
		// When its translation started and ended, when each segment's ended.
		uint64_t              start_ns = 0, end_ns = 0;
		std::vector<uint64_t> segment_ends;
	};
	// What translates: a processor, executors that read the graphics queue's catalog, upload rings, its finished
	// results (for reuse) and what the commits released of them (pending tick, tick: applied before its next
	// translation, on its thread); its thread (KYTY_SPECULATE=4; the first translates on the GPU thread in modes 1 and
	// 3), the serial it translates, its latest result (the commits unlink it), when the translation going on began
	// (m_clock; 0: none: it walks no speculation before it).
	struct Translator {
		explicit Translator(RenderContext& renderer, uint32_t number);
		uint32_t                                   index = 0;
		std::unique_ptr<RenderExecutor>            draw, compute;
		RenderContext::Executors                   executors;
		std::unique_ptr<CommandProcessor>          cp;
		std::unique_ptr<StreamBuffer>              table_upload, shader_upload;
		std::vector<std::unique_ptr<Result>>       free;     // (under m_mutex)
		std::vector<std::pair<uint64_t, uint64_t>> resolved; // (under m_mutex)
		std::thread                                thread;
		uint64_t                                   translating = 0; // (under m_mutex: serial, chunk)
		uint32_t                                   translating_chunk = 0;
		Result*                                    current     = nullptr;
		uint64_t                                   since       = 0;
	};
	// A chunk of a submission (mode 4), once the state pass passed it: the processor state and position it starts at,
	// where it ends (where the next begins; `last`: the submission's end), the pass's record of it (Spec::t_record),
	// whether a translation took it.
	struct Chunk {
		std::unique_ptr<CommandProcessor> start;
		Pm4Execution                      at, end;
		bool                              last  = false;
		bool                              taken = false;
		std::unique_ptr<Spec::State>      record;
	};
	// A submission (mode 4), until the GPU thread did it: its chunks so far and which pass gave them.
	struct Entry {
		Job               job;
		std::deque<Chunk> chunks;
		uint64_t          pass      = 0;
		uint64_t          pushed_ns = 0; // (the report's)
	};
	// (From `previous`'s state when it is a speculation not committed yet, else from `record` while it is the record of
	// a submission not done: Spec::State::previous.)
	// (Up to `stop` when it is given: a chunk's end.)
	[[nodiscard]] std::unique_ptr<Result> Translate(Translator& translator, const CommandProcessor& cp,
	                                                std::span<const uint32_t> commands, const Pm4Execution& execution,
	                                                const Result* previous = nullptr, const Spec::State* record = nullptr,
	                                                const Pm4Execution* stop = nullptr);
	// The result committed (cp and execution as TranslateAndCommit leaves them), from where it is.
	[[nodiscard]] Outcome Commit(std::unique_ptr<Result> result, CommandProcessor& cp, Pm4Execution& execution);
	[[nodiscard]] Outcome Continue(std::unique_ptr<Result> result, CommandProcessor& cp, Pm4Execution& execution);
	[[nodiscard]] Epochs  CurrentEpochs() const;
	// Why the work of the segment no longer holds (a buffer or image it used registered over, a mapping changed), or
	// null.
	[[nodiscard]] const char* Changed(const Result& result, size_t index) const;
	// The segments committed, the holes after them run: all, or up to what the work after it no longer holds
	// (Result::next: the segments committed).
	[[nodiscard]] Progress CommitSegments(Result& result, CommandProcessor& cp);
	// The segment before its work and after it.
	void Before(Result& result, size_t index);
	void After(Result& result, size_t index);
	// The hole translated the normal way: changed what the work after it assumed, or a wait not satisfied yet.
	[[nodiscard]] HoleRun RunHole(Result& result, CommandProcessor& cp, size_t index);
	// KYTY_SPECULATE=4: a speculation thread; the state pass of a submission (the state it starts from, null: none to
	// translate, or the pass could not follow it); the state pass on from the GPU thread's last submission when it
	// cannot follow or is behind (under m_mutex).
	void AheadRun(Translator& translator);
	// The state pass of a submission (its chunks given to its entry as it passes them, while the pass is `which`):
	// false: it could not follow it.
	[[nodiscard]] bool StatePass(const Job& job, uint64_t which);
	void               ResyncStatePass();
	// (Under m_mutex.) The submission's entry (null: done, or not pushed), the chunk's record (null: none any more); what
	// links to `from` links to `to` (results, the translations going on, records); when the oldest translation going on
	// began (m_clock; UINT64_MAX: none): what was retired before is reached by none; a record for a state pass (a
	// retired one reused), a record retired; whether the GPU thread took the chunk the normal way (or passed it).
	[[nodiscard]] Entry*                       EntryOf(uint64_t serial);
	[[nodiscard]] const Spec::State*           RecordOf(uint64_t serial, uint32_t chunk);
	[[nodiscard]] bool                         Taken(uint64_t serial, uint32_t chunk) const noexcept {
		return serial < m_taken || (serial == m_taken && chunk <= m_taken_chunk);
	}
	void                                       Unlink(const Spec::State* from, const Spec::State* to);
	[[nodiscard]] uint64_t                     Horizon() const;
	[[nodiscard]] std::unique_ptr<Spec::State> TakeRecord();
	void                                       Retire(std::unique_ptr<Spec::State> record);
	// What a result fenced with `pending` (its rings' and executors') gets `tick` (on the thread that translates).
	void ResolvePending(Translator& translator, uint64_t pending, uint64_t tick);
	// The segments from `first` on dropped (none of them submitted), and what the result fenced given the tick of the
	// scheduler's open command buffer (after all it submitted); the result kept for reuse.
	void Finish(std::unique_ptr<Result> result, size_t first);
	void Report();

	RenderContext&                           m_renderer;
	std::vector<std::unique_ptr<Translator>> m_translators;
	std::deque<std::unique_ptr<Result>>      m_ahead;     // translated ahead (under m_mutex in mode 4)
	std::unique_ptr<Result>                  m_blocked;   // a commit waiting at a wait hole
	std::unique_ptr<CommandProcessor>        m_chain;     // mode 3: the state the next one ahead starts from
	static constexpr size_t                  MaxAhead = 8;
	std::atomic<uint64_t>                    m_pending {CommandScheduler::PendingTick}; // the latest pending tick
	// What the normal translation did, and what of it was logged since the result being committed began.
	Spec::EffectsLog                     m_log;
	Spec::Effects                        m_effects;
	// Outcomes since the last report: command buffers committed whole, partly, not at all; segments and holes
	// committed; why packets became holes, why speculations or commits stopped.
	uint64_t m_whole = 0, m_partly = 0, m_none = 0, m_segments = 0, m_holes_run = 0, m_waits_blocked = 0;
	std::unordered_map<const char*, uint64_t> m_hole_reasons, m_stop_reasons;
	// Time (ns) translating the segments committed and the others, committing (holes apart), in holes, in the graphics
	// queue; the frame of the last report.
	uint64_t m_kept_ns = 0, m_lost_ns = 0, m_commit_ns = 0, m_hole_ns = 0, m_queue_ns = 0, m_report_frame = 0;
	std::mutex m_mutex;
	bool       m_quit  = false;
	uint64_t   m_clock = 0; // (under m_mutex) orders translations beginning and results retired
	// KYTY_SPECULATE=4 (under m_mutex): the submissions the GPU thread has not done; the state pass's processor, the
	// last submission it passed, which pass (a pass that went wrong starts another: what began from it is dropped),
	// whether a thread runs it and whether it cannot follow the submission broken_at; the last submission the GPU
	// thread took the normal way and the last it did, with its state after it.
	std::condition_variable           m_ahead_wake;
	std::deque<Entry>                 m_jobs;
	std::unique_ptr<CommandProcessor> m_state;
	uint64_t                          m_state_serial = 0, m_pass = 1, m_broken_at = 0;
	bool                              m_state_busy = false, m_broken = false;
	uint64_t                          m_taken = 0, m_done_serial = 0; // (taken: with m_taken_chunk)
	uint32_t                          m_taken_chunk = 0;
	std::unique_ptr<CommandProcessor> m_done_state;
	Spec::State*                      m_pass_record = nullptr; // the record the state pass fills
	std::vector<std::pair<uint64_t, std::unique_ptr<Spec::State>>> m_retired_records; // (when: m_clock)
	uint64_t                          m_waits_for_ahead = 0, m_ahead_misses = 0; // (the report's)
	uint64_t                          m_blind = 0, m_stopped = 0, m_blind_stopped = 0; // (the report's)
	// (The report's: the GPU thread's time waiting for a translation; from a push to its translation's start; the
	// translations by duration: < 0.5, 1, 2, 4, 8 ms and longer, and the longest.)
	uint64_t                          m_wait_ns = 0, m_start_delay_ns = 0, m_started = 0, m_longest_ns = 0;
	std::array<uint64_t, 6>           m_durations {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_SPECULATION_H_
