#include "graphics/presentation/videoOut.h"
#include "live-counters.h"
#include "live-trace.h"
#include "time-census.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/renderDoc.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "loader/systemContent.h"
#include "local/fsr3-fg.h"
#include "local/frame-gen.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <thread>
#include <vector>

namespace Libs::Graphics {
struct GraphicContext;
} // namespace Libs::Graphics

namespace Libs::VideoOut {

LIB_NAME("VideoOut", "VideoOut");

namespace EventQueue = LibKernel::EventQueue;

constexpr int      VIDEO_OUT_EVENT_FLIP                                 = 0;
constexpr int      VIDEO_OUT_EVENT_VBLANK                               = 1;
constexpr int      VIDEO_OUT_EVENT_PRE_VBLANK_START                     = 2;
constexpr int      VIDEO_OUT_EVENT_SET_MODE                             = 8;
constexpr int      VIDEO_OUT_TRUE                                       = 1;
constexpr int      VIDEO_OUT_FALSE                                      = 0;
constexpr int      VIDEO_OUT_BUS_TYPE_MAIN                              = 0;
constexpr int      VIDEO_OUT_BUS_TYPE_OVERLAY                           = 1;
constexpr int      VIDEO_OUT_BUS_TYPE_SUB                               = 2;
constexpr int      VIDEO_OUT_FLIP_MODE_VSYNC                            = 1;
constexpr int      VIDEO_OUT_FLIP_MODE_VSYNC_MULTI                      = 4;
constexpr int      VIDEO_OUT_BUFFER_INDEX_BLACK                         = -2;
constexpr int      VIDEO_OUT_BUFFER_INDEX_BLANK                         = -1;
constexpr int      VIDEO_OUT_BUFFER_NUM_MAX                             = 16;
constexpr size_t   VIDEO_OUT_FLIP_QUEUE_CAPACITY                        = 16;
constexpr int      VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX                   = 4;
constexpr uint64_t VIDEO_OUT_OUTPUT_MODE_DEFAULT                        = 0x0000000000000001ULL;
constexpr uint64_t VIDEO_OUT_OUTPUT_MODE_119_88HZ                       = 0x000000000000000FULL;
constexpr uint64_t VIDEO_OUT_REFRESH_RATE_59_94HZ                       = 3;
constexpr uint64_t VIDEO_OUT_REFRESH_RATE_119_88HZ                      = 13;
constexpr int      VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_UNCOMPRESSED     = 0;
constexpr int      VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_COMPRESSED       = 1;
constexpr uint64_t VIDEO_OUT_BUFFER_ATTRIBUTE_OPTION_STRICT_COLORIMETRY = 8;

enum class VideoOutEventKind : uintptr_t {
	Flip           = VIDEO_OUT_EVENT_FLIP,
	Vblank         = VIDEO_OUT_EVENT_VBLANK,
	PreVblankStart = VIDEO_OUT_EVENT_PRE_VBLANK_START,
	OutputMode     = VIDEO_OUT_EVENT_SET_MODE,
};

enum class FlipRequestSource { Cpu, GpuEop };

struct VideoOutEventState;

struct VideoOutEventRegistration {
	EventQueue::KernelEqueue            handle = EventQueue::KERNEL_EQUEUE_INVALID;
	std::shared_ptr<VideoOutEventState> state;
	uint64_t                            generation = 0;
	VideoOutEventKind                   kind       = VideoOutEventKind::Flip;
};

using VideoOutEventRegistrationRef = std::shared_ptr<VideoOutEventRegistration>;
using VideoOutEventQueues          = std::vector<VideoOutEventRegistrationRef>;

struct VideoOutEventState {
	Common::Mutex       mutex;
	VideoOutEventQueues flip;
	VideoOutEventQueues pre_vblank;
	VideoOutEventQueues vblank;
	VideoOutEventQueues output_mode;
};

struct VideoOutBufferAttribute2 {
	uint32_t reserved0;
	uint32_t tiling_mode;
	uint32_t aspect_ratio;
	uint32_t width;
	uint32_t height;
	uint32_t pitch_in_pixel;
	uint64_t option;
	uint64_t pixel_format;
	uint64_t dcc_cb_register_clear_color;
	uint32_t dcc_control;
	uint32_t pad0;
	uint64_t reserved1[3];
};

// PS5 layout
struct VideoOutFlipStatus {
	uint64_t count                    = 0;
	uint64_t processTime              = 0;
	uint64_t reserved0                = 0;
	int64_t  flipArg                  = 0;
	uint64_t reserved1                = 0;
	uint64_t processTimeCounter       = 0;
	int32_t  gcQueueNum               = 0;
	int32_t  flipPendingNum           = 0;
	int32_t  currentBuffer            = 0;
	uint32_t reserved2                = 0;
	uint64_t submitProcessTimeCounter = 0;
	uint64_t reserved3[7]             = {};
};

// PS5 layout
struct VideoOutVblankStatus {
	uint64_t count              = 0;
	uint64_t processTime        = 0;
	uint64_t reserved           = 0;
	uint64_t processTimeCounter = 0;
	uint8_t  flags              = 0;
	uint8_t  phase              = 0;
	uint8_t  pad1[6]            = {};
};

struct VideoOutOutputStatus {
	uint32_t resolution   = 0;
	uint32_t dynamicRange = 0;
	uint64_t refreshRate  = 0;
	uint64_t flags        = 0;
	uint64_t reserved[3]  = {};
};

struct VideoOutOutputOptions {
	uint32_t internalData[16] = {};
};

struct VideoOutColorSettings {
	float    gamma       = 1.0f;
	uint32_t reserved[3] = {};
};

struct VideoOutBuffers {
	const void* data;
	const void* metadata;
	const void* reserved[2];
};

struct VideoOutBuffer {
	int      group_index      = -1;
	uint64_t data_address     = 0;
	uint64_t metadata_address = 0;

	[[nodiscard]] bool Occupied() const noexcept { return group_index >= 0; }
};

struct BufferAttributeGroup {
	VideoOutBufferAttribute2 attribute {};
	int                      category = VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_UNCOMPRESSED;
	bool                     occupied = false;

	[[nodiscard]] Graphics::ImageInfo ImageInfo(const VideoOutBuffer& buffer) const;
};

struct VideoOutConfig {
	Common::Mutex                       mutex;
	Common::CondVar                     vblank_cond;
	std::shared_ptr<VideoOutEventState> events      = std::make_shared<VideoOutEventState>();
	uint32_t                            width       = 0;
	uint32_t                            height      = 0;
	uint64_t                            generation  = 0;
	bool                                opened      = false;
	bool                                closing     = false;
	int                                 flip_rate   = 0;
	uint64_t                            last_flip_vblank = 0; // vblank_status.count at the last flip
	uint64_t                            output_mode = VIDEO_OUT_OUTPUT_MODE_DEFAULT;
	float                               gamma       = 1.0f;
	VideoOutFlipStatus                  flip_status;
	VideoOutVblankStatus                pre_vblank_status;
	VideoOutVblankStatus                vblank_status;
	std::array<VideoOutBuffer, VIDEO_OUT_BUFFER_NUM_MAX>                 buffers;
	std::array<BufferAttributeGroup, VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX> groups;
};

class FlipQueue {
public:
	explicit FlipQueue(Graphics::Presenter& presenter): m_presenter(presenter) {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	}
	~FlipQueue();
	KYTY_CLASS_NO_COPY(FlipQueue);

	bool Reserve(VideoOutConfig& cfg, int index, int64_t flip_arg, FlipRequestSource source,
	             uint64_t& request_id);
	void Cancel(VideoOutConfig& cfg);
	void Prepare(uint64_t request_id, Graphics::CommandBuffer& buffer);
	void Complete(uint64_t request_id);
	void WaitForSubmitSlot();
	bool Flip(uint32_t micros);
	void GetFlipStatus(VideoOutConfig& cfg, VideoOutFlipStatus& out);
	void Wait(VideoOutConfig& cfg, int index);

private:
	enum class RequestState { Reserved, Recording, Ready, Presenting };

	struct Request {
		uint64_t                    id;
		VideoOutConfig*             cfg;
		uint64_t                    generation;
		int                         index;
		int64_t                     flip_arg;
		uint64_t                    submit_ptc;
		FlipRequestSource           source;
		RequestState                state;
		Graphics::Presenter::Frame* frame;
	};

	Graphics::Presenter& m_presenter;
	Common::Mutex        m_mutex;
	Common::CondVar      m_submit_cond_var;
	Common::CondVar      m_submit_slot_cond_var;
	Common::CondVar      m_done_cond_var;
	std::list<Request>   m_requests;
	std::list<Request>   m_cpu_requests;
	std::list<Request>   m_cancelled_requests;
	bool                 m_processing      = false;
	uint64_t             m_next_request_id = 1;
};

struct VideoOutDriver::Impl {
public:
	static constexpr int VIDEO_OUT_NUM_MAX = 4;

	Impl(uint32_t width, uint32_t height, Graphics::Presenter& presenter)
	    : m_renderer(presenter.Renderer()), m_presenter(presenter), m_flip_queue(presenter) {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
		Init(width, height);
		m_present_thread = std::jthread([this](std::stop_token token) { PresentThread(token); });
	}
	~Impl();
	KYTY_CLASS_NO_COPY(Impl);

	int             Open(int bus_type);
	bool            Close(int handle);
	VideoOutConfig* Get(int handle);
	VideoOutConfig* Get(int handle, uint64_t& generation);
	bool            IsOpened(int handle);

	void                     Init(uint32_t width, uint32_t height);
	FlipQueue&               GetFlipQueue() { return m_flip_queue; }
	Graphics::RenderContext& Renderer() const noexcept { return m_renderer; }
	bool                     IsHdrSupported() const noexcept { return m_presenter.IsHdrSupported(); }

	void VblankBegin();
	void VblankEnd();
	void PresentThread(std::stop_token token);

private:
	Common::Mutex            m_mutex;
	VideoOutConfig           m_video_out_ctx[VIDEO_OUT_NUM_MAX];
	Graphics::RenderContext& m_renderer;
	Graphics::Presenter&     m_presenter;
	FlipQueue                m_flip_queue;
	std::jthread             m_present_thread;
};

static std::unique_ptr<VideoOutDriver> g_video_out_driver;

static VideoOutDriver::Impl& DriverState() {
	EXIT_IF(g_video_out_driver == nullptr);
	return g_video_out_driver->State();
}

static uintptr_t VideoOutEventId(VideoOutEventKind kind) {
	return static_cast<uintptr_t>(kind);
}

static VideoOutEventQueues& VideoOutEventQueuesFor(VideoOutEventState& state,
                                                   VideoOutEventKind   kind) {
	switch (kind) {
		case VideoOutEventKind::Flip: return state.flip;
		case VideoOutEventKind::Vblank: return state.vblank;
		case VideoOutEventKind::PreVblankStart: return state.pre_vblank;
		case VideoOutEventKind::OutputMode: return state.output_mode;
	}
	EXIT("unsupported video-out event kind\n");
	return state.flip;
}

static intptr_t MakeVideoOutEventData(intptr_t current_data, void* trigger_data) {
	const uint64_t old_data = static_cast<uint64_t>(current_data);
	uint64_t       counter  = (old_data >> 12u) & 0xfu;
	if (counter != 0xfu) {
		counter++;
	}

	const uint64_t time    = LibKernel::KernelReadTsc() & 0xfffu;
	const uint64_t payload = static_cast<uint64_t>(reinterpret_cast<intptr_t>(trigger_data));

	return static_cast<intptr_t>(time | (counter << 12u) |
	                             ((payload & 0x0000ffffffffffffULL) << 16u));
}

static void ResetVideoOutEvent(EventQueue::KernelEqueueEvent* event) {
	EXIT_IF(event == nullptr);
	event->triggered    = false;
	event->event.fflags = 0;
	event->event.data   = 0;
}

static void TriggerVideoOutEvent(EventQueue::KernelEqueueEvent* event, void* trigger_data) {
	EXIT_IF(event == nullptr);

	auto triggered_event = event->event;
	triggered_event.fflags =
	    triggered_event.fflags < 0xfu ? triggered_event.fflags + 1u : triggered_event.fflags;
	triggered_event.data = MakeVideoOutEventData(triggered_event.data, trigger_data);
	if (event->triggered) {
		event->pending_events.push_back(triggered_event);
		return;
	}
	event->event     = triggered_event;
	event->triggered = true;
}

static void RemoveVideoOutEventQueue(EventQueue::KernelEqueue       eq,
                                     EventQueue::KernelEqueueEvent* event) {
	if (event == nullptr || event->filter.data == nullptr) {
		return;
	}

	auto* registration = static_cast<VideoOutEventRegistration*>(event->filter.data);
	auto  state        = registration->state;
	if (registration->handle != eq || !state) {
		return;
	}
	auto&             queues = VideoOutEventQueuesFor(*state, registration->kind);
	Common::LockGuard lock(state->mutex);
	const auto        entry =
	    std::find_if(queues.begin(), queues.end(), [registration](const auto& candidate) {
		    return candidate.get() == registration;
	    });
	if (entry != queues.end()) {
		queues.erase(entry);
	}
}

static void TriggerVideoOutEvents(VideoOutConfig& video_out, VideoOutEventKind kind,
                                  void* trigger_data) {
	VideoOutEventQueues queues;
	{
		Common::LockGuard lock(video_out.events->mutex);
		queues = VideoOutEventQueuesFor(*video_out.events, kind);
	}
	for (const auto& registration: queues) {
		if (!registration || registration->generation != video_out.generation) {
			continue;
		}
		const auto result =
		    EventQueue::KernelTriggerEvent(registration->handle, VideoOutEventId(kind),
		                                   EventQueue::KERNEL_EVFILT_VIDEO_OUT, trigger_data);
		EXIT_NOT_IMPLEMENTED(result != OK && result != LibKernel::KERNEL_ERROR_EBADF &&
		                     result != LibKernel::KERNEL_ERROR_ENOENT);
	}
}

static void DeleteVideoOutEvents(const VideoOutEventQueues& queues, VideoOutEventKind kind) {
	for (const auto& registration: queues) {
		if (!registration) {
			continue;
		}
		const auto result = EventQueue::KernelDeleteEvent(
		    registration->handle, VideoOutEventId(kind), EventQueue::KERNEL_EVFILT_VIDEO_OUT);
		EXIT_NOT_IMPLEMENTED(result != OK && result != LibKernel::KERNEL_ERROR_EBADF &&
		                     result != LibKernel::KERNEL_ERROR_ENOENT);
	}
}

static int RegisterVideoOutEvent(int handle, EventQueue::KernelEqueue eq, VideoOutEventKind kind,
                                 void* udata) {
	uint64_t generation = 0;
	auto*    video_out  = DriverState().Get(handle, generation);
	if (video_out == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	Common::LockGuard lock(video_out->mutex);
	if (!video_out->opened || video_out->closing || video_out->generation != generation) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (kind == VideoOutEventKind::OutputMode) {
		LOGF("\t eq     = 0x%016" PRIx64 "\n"
		     "\t handle = %d\n"
		     "\t udata  = 0x%016" PRIx64 "\n",
		     static_cast<uint64_t>(eq), handle, reinterpret_cast<uint64_t>(udata));
	}
	if (eq == EventQueue::KERNEL_EQUEUE_INVALID) {
		return VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
	}
	if (!EventQueue::KernelPinEqueue(eq)) {
		return VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
	}
	auto        event_state         = video_out->events;
	auto&       queues              = VideoOutEventQueuesFor(*event_state, kind);
	const bool  initially_triggered = kind == VideoOutEventKind::OutputMode;
	void* const initial_trigger_data =
	    initially_triggered ? reinterpret_cast<void*>(video_out->output_mode) : nullptr;

	EventQueue::KernelEqueueEvent event {};
	event.triggered    = initially_triggered;
	event.event.ident  = VideoOutEventId(kind);
	event.event.filter = EventQueue::KERNEL_EVFILT_VIDEO_OUT;
	event.event.udata  = udata;
	event.event.fflags = initially_triggered ? 1u : 0u;
	event.event.data   = initially_triggered ? MakeVideoOutEventData(0, initial_trigger_data) : 0;
	event.filter.delete_event_func = RemoveVideoOutEventQueue;
	event.filter.reset_func        = ResetVideoOutEvent;
	event.filter.trigger_func      = TriggerVideoOutEvent;

	VideoOutEventRegistrationRef registration;
	bool                         add_queue = false;
	{
		Common::LockGuard event_lock(event_state->mutex);
		const auto        existing =
		    std::find_if(queues.begin(), queues.end(), [&](const auto& candidate) {
			    return candidate->handle == eq && candidate->generation == generation;
		    });
		if (existing != queues.end()) {
			registration = *existing;
		} else {
			registration = std::make_shared<VideoOutEventRegistration>(VideoOutEventRegistration {
			    .handle = eq, .state = event_state, .generation = generation, .kind = kind});
			queues.push_back(registration);
			add_queue = true;
		}
	}
	event.filter.data  = registration.get();
	event.filter.owner = registration;
	const int result   = EventQueue::KernelAddEvent(eq, event);
	if (result != OK && add_queue) {
		Common::LockGuard event_lock(event_state->mutex);
		const auto        added = std::find(queues.begin(), queues.end(), registration);
		if (added != queues.end()) {
			queues.erase(added);
		}
	}
	return result == LibKernel::KERNEL_ERROR_EBADF ? VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE : result;
}

static int DeleteVideoOutEvent(int handle, EventQueue::KernelEqueue eq, VideoOutEventKind kind) {
	uint64_t generation = 0;
	auto*    video_out  = DriverState().Get(handle, generation);
	if (video_out == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	Common::LockGuard lock(video_out->mutex);
	if (!video_out->opened || video_out->closing || video_out->generation != generation) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (!EventQueue::KernelPinEqueue(eq)) {
		return VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
	}
	const int result = EventQueue::KernelDeleteEvent(eq, VideoOutEventId(kind),
	                                                 EventQueue::KERNEL_EVFILT_VIDEO_OUT);
	if (result == LibKernel::KERNEL_ERROR_EBADF) {
		return VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
	}
	return result == LibKernel::KERNEL_ERROR_ENOENT ? OK : result;
}

// KYTY_FLIP_RATE=1 or 2: at least 2 or 3 vblanks between flips (30 or 20 fps at 60 Hz), a steady
// frame rate, e.g. for recordings with frame generation. The vblanks keep their rate, and so does
// the game's clock in play. Unlike the game's own sceVideoOutSetFlipRate (flips only on every 2nd
// vblank), a late frame waits for the next vblank, not for the next even one (50 ms, not 67).
// Movies advance a frame per flip (the intro played at half speed): with frame generation only
// frames of the game's 3D view are capped (Presenter::FlipRateApplies).
static int FlipRateFloor() {
	static const int floor = [] {
		const char* value = std::getenv("KYTY_FLIP_RATE");
		return std::clamp(value != nullptr ? std::atoi(value) : 0, 0, 2);
	}();
	return floor;
}

// Above 60 Hz (-Fps120): at least vblank/60 vblanks between the flips of frames without the game's 3D view
// (Presenter::IsSceneFrame: movies, menus, loading screens), the console's 60 Hz. The game's Bink movies advance a
// frame per flip (the intro played in 54 s instead of 113 at 120 Hz, ahead of its music); menus gain nothing.
static int SceneFreeFloor() {
	static const int floor =
	    std::max(0, static_cast<int>(std::lround(static_cast<double>(Config::GetVblankFrequency()) / 60.0)) - 1);
	return floor;
}

// Frame generation above 60 Hz: the game's frames too keep 60 (KYTY_FRAMEGEN_BASE=<Hz>: another rate, 0: none). DLSS-G
// paces the frames it generates from the intervals between the game's own; on the 120 Hz vblanks a game between 60 and
// 120 fps flipped after one vblank or two by turns (8.3 or 16.7 ms, 3-25 ms with late vblanks; at 60 Hz 16.7 +- 0.3 ms),
// and its frames stuttered (issue #29).
static int FrameGenFloor() {
	static const int floor = [] {
		const char* value = std::getenv("KYTY_FRAMEGEN_BASE");
		const int   base  = value != nullptr ? std::max(0, std::atoi(value)) : 60;
		if (base == 0) return 0;
		return std::max(0, static_cast<int>(std::lround(static_cast<double>(Config::GetVblankFrequency()) / base)) - 1);
	}();
	return floor;
}

// Movies decoded through the system's decoders (VideoDec2, AvPlayer) note each frame: frame generation stays off while
// one plays (frame-gen.cpp, fsr3-fg.cpp).
static std::atomic<int64_t> g_movie_frame_ns {0};
void NoteMovieFrame() noexcept {
	g_movie_frame_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_relaxed);
}
// The world gate (10-18): frame generation and the launcher's frame rates only once the first world loaded. Before it
// (the logos, the title, the menus) 60 fps without frame generation. The first world: after the player's first button,
// a loading screen (frames without the game's 3D view for 1.5 s), then the 3D view again; or 90 s of the 3D view after
// the button (in case no loading screen is seen).
static std::atomic<bool>    g_player_input {false};
static std::atomic<int64_t> g_input_ns {0};
static std::atomic<bool>    g_world_ready {false};
static std::atomic<bool>    g_world_file {false};
void NoteWorldFile(const char* name) noexcept {
	if (g_player_input.load(std::memory_order_relaxed) && !g_world_file.exchange(true, std::memory_order_relaxed)) {
		std::printf("World: the game loads a world (%s)\n", name);
		std::fflush(stdout);
	}
}
void NotePlayerInput() noexcept {
	if (!g_player_input.exchange(true, std::memory_order_relaxed)) {
		g_input_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_relaxed);
		std::printf("Input: the player's first button (frame generation waits for the first world)\n");
		std::fflush(stdout);
	}
}
static void NoteFrameForWorld(bool scene) {
	if (g_world_ready.load(std::memory_order_relaxed) || !g_player_input.load(std::memory_order_relaxed)) return;
	using namespace std::chrono;
	static int64_t nonscene_since = 0;
	static bool    loading_seen   = false;
	const int64_t  now            = steady_clock::now().time_since_epoch().count();
	const auto     ns             = [](auto d) { return duration_cast<steady_clock::duration>(d).count(); };
	if (!scene) {
		if (nonscene_since == 0) nonscene_since = now;
		// (A long one: the menus' short black screens (the title to the main menu, 1.5-3 s) are not the world's load,
		// the fog's. KYTY_WORLD_LOADING_MS: how long, default 4000.)
		static const int64_t loading_ms = [] {
			const char* value = std::getenv("KYTY_WORLD_LOADING_MS");
			return value != nullptr ? std::max<int64_t>(500, std::atoll(value)) : int64_t {4000};
		}();
		if (now - nonscene_since >= ns(milliseconds(loading_ms))) loading_seen = true;
		return;
	}
	if (nonscene_since != 0 && now - nonscene_since >= ns(milliseconds(500))) {
		std::printf("World: a screen without the 3D view for %lld ms (%s)\n",
		            static_cast<long long>(duration_cast<milliseconds>(steady_clock::duration(now - nonscene_since)).count()),
		            loading_seen ? "the world's load" : "shorter than the world's load");
		std::fflush(stdout);
	}
	nonscene_since = 0;
	const bool timeout = now - g_input_ns.load(std::memory_order_relaxed) >= ns(seconds(90));
	// The world: its files read (levels/...) after the first button, then the 3D view (the loading screens of 1.9-4 s
	// measured, menus too: their length tells nothing); else 90 s of the 3D view.
	const bool world_file = g_world_file.load(std::memory_order_relaxed);
	if (world_file || timeout) {
		g_world_ready.store(true, std::memory_order_relaxed);
		std::printf("World: the first world loaded (%s): frame generation and frames above 60 allowed from here\n",
		            world_file ? "its files read" : loading_seen ? "after a loading screen" : "90 s after the first button");
		std::fflush(stdout);
	}
}
bool PlayerInputSeen() noexcept {
	return g_world_ready.load(std::memory_order_relaxed);
}
static std::atomic<int64_t> g_cinematic_ns {0};
void NoteCinematic(const char* name) noexcept {
	const auto now  = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto last = g_cinematic_ns.exchange(now, std::memory_order_relaxed);
	if (name != nullptr && now - last > std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::seconds(5)).count()) {
		std::printf("Cinematic: %s (its frames flipped at 60)\n", name);
		std::fflush(stdout);
	}
}
// An in-engine cinematic's files read within KYTY_CINEMATIC_HOLD_MS (default 2000): its frames are flipped at 60 (the
// unlocked mode sped them up). Frame generation stays on through them: the game's real frames are 60 there anyway, and
// turning FSR 3's generation off right after it came on lost the device (10-20, the Boletaria intro).
bool CinematicPlaying() noexcept {
	static const int64_t hold = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::milliseconds([] {
		const char* value = std::getenv("KYTY_CINEMATIC_HOLD_MS");
		return value != nullptr ? std::atoll(value) : 2000ll;
	}())).count();
	const auto cinematic = g_cinematic_ns.load(std::memory_order_relaxed);
	return cinematic != 0 && std::chrono::steady_clock::now().time_since_epoch().count() - cinematic < hold;
}
bool MoviePlaying() noexcept {
	const auto last = g_movie_frame_ns.load(std::memory_order_relaxed);
	if (last == 0) return false;
	const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
	return now - last < std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::milliseconds(500)).count();
}

// KYTY_FLIP_FPS=<fps>: the game's 3D view is flipped at most that many times a second, a whole number of vblanks apart
// (the launcher's Smooth Souls: 40 on a 120 Hz vblank, 3 vblanks). Menus, loading screens and movies keep their rate.
static int FlipFpsFloor() {
	static const int floor = [] {
		const char* value = std::getenv("KYTY_FLIP_FPS");
		const int   fps   = value != nullptr ? std::atoi(value) : 0;
		if (fps <= 0) return 0;
		const int result = std::max(0, static_cast<int>(std::lround(static_cast<double>(Config::GetVblankFrequency()) / fps)) - 1);
		std::printf("Flips: the game's 3D view held to %u fps (KYTY_FLIP_FPS=%d, %d vblank(s) apart)\n",
		            Config::GetVblankFrequency() / static_cast<uint32_t>(result + 1), fps, result + 1);
		std::fflush(stdout);
		return result;
	}();
	return floor;
}

// `floor`: at least floor + 1 vblanks since the last flip (beyond the game's own flip rate).
static bool IsFlipDueLocked(const VideoOutConfig& cfg, uint64_t generation, int floor) {
	if (!cfg.opened || cfg.closing || cfg.generation != generation) {
		return false;
	}
	if (floor > cfg.flip_rate) {
		return cfg.vblank_status.count - cfg.last_flip_vblank >= static_cast<uint64_t>(floor) + 1u;
	}
	const int interval = cfg.flip_rate + 1;

	return interval <= 1 || (cfg.vblank_status.count % static_cast<uint64_t>(interval)) == 0;
}

static bool IsValidBufferIndex(int index) {
	return index >= VIDEO_OUT_BUFFER_INDEX_BLACK && index < VIDEO_OUT_BUFFER_NUM_MAX;
}

static bool IsSpecialBufferIndex(int index) {
	return index == VIDEO_OUT_BUFFER_INDEX_BLANK || index == VIDEO_OUT_BUFFER_INDEX_BLACK;
}

static bool IsValidFlipMode(int mode) {
	return mode >= VIDEO_OUT_FLIP_MODE_VSYNC && mode <= VIDEO_OUT_FLIP_MODE_VSYNC_MULTI;
}

static int ReserveFlipRequest(VideoOutDriver::Impl& driver, int handle, int index, int flip_mode,
                              int64_t flip_arg, FlipRequestSource source, uint64_t& request_id) {
	auto* video_out = driver.Get(handle);
	if (video_out == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (!IsValidFlipMode(flip_mode)) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}
	if (!IsValidBufferIndex(index)) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}

	Common::LockGuard lock(video_out->mutex);
	if (video_out->closing ||
	    (!IsSpecialBufferIndex(index) && !video_out->buffers[index].Occupied())) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}
	if (!driver.GetFlipQueue().Reserve(*video_out, index, flip_arg, source, request_id)) {
		return VIDEO_OUT_ERROR_FLIP_QUEUE_FULL;
	}
	return OK;
}

Graphics::ImageInfo BufferAttributeGroup::ImageInfo(const VideoOutBuffer& buffer) const {
	const auto compression = Graphics::ClassifyVideoOutCompression(
	    category == VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_COMPRESSED, buffer.metadata_address,
	    attribute.dcc_control, attribute.dcc_cb_register_clear_color);
	if (attribute.reserved0 != 0 || attribute.aspect_ratio != 0 || attribute.width == 0 ||
	    attribute.height == 0 || attribute.width > 16384 || attribute.height > 16384 ||
	    attribute.pitch_in_pixel != 0 ||
	    (attribute.option != 0 &&
	     attribute.option != VIDEO_OUT_BUFFER_ATTRIBUTE_OPTION_STRICT_COLORIMETRY) ||
	    attribute.tiling_mode != 0 || attribute.pad0 != 0 || attribute.reserved1[0] != 0 ||
	    attribute.reserved1[1] != 0 || attribute.reserved1[2] != 0 || buffer.data_address == 0 ||
	    compression == Graphics::VideoOutCompression::Unsupported) {
		EXIT("unsupported or invalid video-out surface attributes\n");
	}
	Graphics::VideoOutPixelFormatInfo pixel_format {};
	if (!Graphics::DecodeVideoOutPixelFormat(attribute.pixel_format, pixel_format)) {
		EXIT("unsupported video-out pixel format: 0x%016" PRIx64 "\n", attribute.pixel_format);
	}
	const auto tile_mode = Graphics::Prospero::TileMode::kRenderTarget;
	const auto pitch =
	    Graphics::TileGetTexturePitch(pixel_format.guest_format, attribute.width, tile_mode);
	Graphics::TileSizeAlign total {};
	if (!Graphics::TileGetTextureTotalSize(pixel_format.guest_format, attribute.width,
	                                       attribute.height, 1, 1, tile_mode, false, total) ||
	    total.size == 0 || total.align != 65536 ||
	    (buffer.data_address & (total.align - 1u)) != 0) {
		EXIT("invalid video-out surface footprint or alignment\n");
	}
	Graphics::ImageInfo info {};
	info.data            = {buffer.data_address, total.size};
	info.pixel_format    = pixel_format.format;
	info.guest_format    = pixel_format.guest_format;
	info.type            = Graphics::Prospero::ImageType::kColor2D;
	info.extent          = {attribute.width, attribute.height, 1};
	info.resources       = {1, 1};
	info.pitch           = pitch;
	info.bytes_per_block = pixel_format.bytes_per_element;
	info.samples         = 1;
	info.tile_mode       = tile_mode;
	info.bgra16          = pixel_format.bgra16;
	info.mip_layout[0]   = {0, total.size, pitch, attribute.height};
	if (compression != Graphics::VideoOutCompression::Uncompressed) {
		info.metadata.range       = {buffer.metadata_address, 0};
		info.metadata.kind        = Graphics::ImageMetadataKind::Dcc;
		info.metadata.control     = attribute.dcc_control;
		info.metadata.compression = compression;
	}
	Graphics::ImageOps::Validate(info);
	if (!Graphics::IsSupportedVideoOutFormat(info)) {
		EXIT("unsupported normalized video-out format\n");
	}
	return info;
}

VideoOutDriver::VideoOutDriver(uint32_t width, uint32_t height, Graphics::Presenter& presenter)
    : m_impl(std::make_unique<Impl>(width, height, presenter)) {}

VideoOutDriver::~VideoOutDriver() = default;

VideoOutDriver::Impl& VideoOutDriver::State() noexcept {
	return *m_impl;
}

VideoOutDriver& VideoOutInit(uint32_t width, uint32_t height, Graphics::Presenter& presenter) {
	EXIT_IF(g_video_out_driver != nullptr);
	g_video_out_driver = std::make_unique<VideoOutDriver>(width, height, presenter);
	return *g_video_out_driver;
}

void VideoOutShutdown() {
	g_video_out_driver.reset();
}

VideoOutDriver::Impl::~Impl() {
	if (m_present_thread.joinable()) {
		m_present_thread.request_stop();
		m_present_thread.join();
	}
	for (int handle = 1; handle < VIDEO_OUT_NUM_MAX; handle++) {
		(void)Close(handle);
	}
}

void VideoOutDriver::Impl::Init(uint32_t width, uint32_t height) {
	for (auto& ctx: m_video_out_ctx) {
		ctx.width  = width;
		ctx.height = height;
	}
}

int VideoOutDriver::Impl::Open(int bus_type) {
	Common::LockGuard lock(m_mutex);

	const int handle = bus_type + 1;
	if (m_video_out_ctx[handle].opened) {
		return -1;
	}
	auto&             config = m_video_out_ctx[handle];
	Common::LockGuard config_lock(config.mutex);

	{
		Common::LockGuard event_lock(config.events->mutex);
		EXIT_IF(!config.events->flip.empty());
		EXIT_IF(!config.events->pre_vblank.empty());
		EXIT_IF(!config.events->vblank.empty());
		EXIT_IF(!config.events->output_mode.empty());
	}
	EXIT_IF(config.flip_rate != 0);
	for (const auto& buffer: config.buffers) {
		EXIT_IF(buffer.Occupied());
	}
	for (const auto& group: config.groups) {
		EXIT_IF(group.occupied);
	}

	config.closing = false;
	config.opened  = true;
	if (++config.generation == 0) {
		EXIT("video-out port generation wrapped\n");
	}
	config.output_mode               = VIDEO_OUT_OUTPUT_MODE_DEFAULT;
	config.flip_status               = VideoOutFlipStatus();
	config.flip_status.flipArg       = -1;
	config.flip_status.currentBuffer = -1;
	config.flip_status.count         = 0;
	config.pre_vblank_status         = VideoOutVblankStatus();
	config.vblank_status             = VideoOutVblankStatus();
	config.last_flip_vblank          = 0;

	return handle;
}

bool VideoOutDriver::Impl::Close(int handle) {
	Common::LockGuard lock(m_mutex);

	if (handle <= 0 || handle >= VIDEO_OUT_NUM_MAX || !m_video_out_ctx[handle].opened) {
		return false;
	}

	auto&               config = m_video_out_ctx[handle];
	VideoOutEventQueues flip_events;
	VideoOutEventQueues pre_vblank_events;
	VideoOutEventQueues vblank_events;
	VideoOutEventQueues output_mode_events;
	{
		Common::LockGuard config_lock(config.mutex);
		if (config.closing) {
			return false;
		}
		config.opened  = false;
		config.closing = true;
		if (++config.generation == 0) {
			EXIT("video-out port generation wrapped\n");
		}
		{
			Common::LockGuard event_lock(config.events->mutex);
			flip_events        = std::move(config.events->flip);
			pre_vblank_events  = std::move(config.events->pre_vblank);
			vblank_events      = std::move(config.events->vblank);
			output_mode_events = std::move(config.events->output_mode);
		}
		config.flip_rate = 0;

		for (const auto& buffer: config.buffers) {
			if (buffer.Occupied() &&
			    (buffer.group_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX ||
			     buffer.data_address == 0 || !config.groups[buffer.group_index].occupied)) {
				EXIT("inconsistent registered video-out buffer state\n");
			}
		}
		for (auto& buffer: config.buffers) {
			buffer = VideoOutBuffer {};
		}
		for (auto& group: config.groups) {
			group = BufferAttributeGroup {};
		}
		config.vblank_cond.SignalAll();
	}

	m_flip_queue.Cancel(config);
	DeleteVideoOutEvents(flip_events, VideoOutEventKind::Flip);
	DeleteVideoOutEvents(pre_vblank_events, VideoOutEventKind::PreVblankStart);
	DeleteVideoOutEvents(vblank_events, VideoOutEventKind::Vblank);
	DeleteVideoOutEvents(output_mode_events, VideoOutEventKind::OutputMode);
	return true;
}

VideoOutConfig* VideoOutDriver::Impl::Get(int handle) {
	Common::LockGuard lock(m_mutex);
	if (handle <= 0 || handle >= VIDEO_OUT_NUM_MAX || !m_video_out_ctx[handle].opened) {
		return nullptr;
	}

	return m_video_out_ctx + handle;
}

VideoOutConfig* VideoOutDriver::Impl::Get(int handle, uint64_t& generation) {
	Common::LockGuard lock(m_mutex);
	if (handle <= 0 || handle >= VIDEO_OUT_NUM_MAX || !m_video_out_ctx[handle].opened) {
		return nullptr;
	}

	auto*             config = m_video_out_ctx + handle;
	Common::LockGuard config_lock(config->mutex);
	if (!config->opened || config->closing) {
		return nullptr;
	}
	generation = config->generation;
	return config;
}

bool VideoOutDriver::Impl::IsOpened(int handle) {
	Common::LockGuard lock(m_mutex);

	return handle > 0 && handle < VIDEO_OUT_NUM_MAX && m_video_out_ctx[handle].opened;
}

void VideoOutDriver::Impl::VblankBegin() {
	Common::LockGuard lock(m_mutex);

	for (int i = 1; i < VIDEO_OUT_NUM_MAX; i++) {
		auto& ctx = m_video_out_ctx[i];
		if (ctx.opened) {
			ctx.mutex.Lock();
			ctx.pre_vblank_status.count++;
			ctx.pre_vblank_status.processTime        = LibKernel::KernelGetProcessTime();
			ctx.pre_vblank_status.reserved           = LibKernel::KernelReadTsc();
			ctx.pre_vblank_status.processTimeCounter = LibKernel::KernelGetProcessTimeCounter();

			TriggerVideoOutEvents(ctx, VideoOutEventKind::PreVblankStart,
			                      reinterpret_cast<void*>(ctx.pre_vblank_status.count));
			ctx.mutex.Unlock();
		}
	}
}

void VideoOutDriver::Impl::VblankEnd() {
	Common::LockGuard lock(m_mutex);
	LiveCounters::Add(LiveCounters::Vblanks);

	for (int i = 1; i < VIDEO_OUT_NUM_MAX; i++) {
		auto& ctx = m_video_out_ctx[i];
		if (ctx.opened) {
			ctx.mutex.Lock();
			ctx.vblank_status.count++;
			ctx.vblank_status.processTime        = LibKernel::KernelGetProcessTime();
			ctx.vblank_status.reserved           = LibKernel::KernelReadTsc();
			ctx.vblank_status.processTimeCounter = LibKernel::KernelGetProcessTimeCounter();

			TriggerVideoOutEvents(ctx, VideoOutEventKind::Vblank,
			                      reinterpret_cast<void*>(ctx.vblank_status.count));
			ctx.vblank_cond.SignalAll();
			ctx.mutex.Unlock();
		}
	}
}

void VideoOutDriver::Impl::PresentThread(std::stop_token token) {
	const auto frequency = Common::Timer::QueryPerformanceFrequency();
	EXIT_IF(frequency == 0);

	int64_t total_wait = 0;
	// VSync on a display refreshing at a whole multiple of the vblank rate: each vblank waits for `display_per_tick` of
	// the display's (Presenter::DisplayVblanksPerTick, looked at again every 120 vblanks) instead of the timer. A display
	// whose vblanks miss that rate keeps the timer for good: with a variable (or dynamic) refresh rate they follow the
	// presents, and vblanks waiting for them would wait for the game.
	uint32_t display_per_tick = 0;
	uint64_t display_seen     = 0;
	uint32_t display_check    = 0; // vblanks until DisplayVblanksPerTick
	bool     display_refused  = false;
	uint32_t display_ticks    = 0; // the vblanks timed since display_since (their average period checked every 120)
	uint64_t display_since    = 0;
	const auto refuse_display = [&](const char* what, double ms, double expected_ms) {
		std::printf("VSync: %s %.3f ms, not %.3f ms (a variable refresh rate?): the vblanks keep their own clock\n", what, ms,
		            expected_ms);
		std::fflush(stdout);
		display_refused  = true;
		display_per_tick = 0;
	};
	while (!token.stop_requested()) {
		const auto refresh = std::max(Config::GetVblankFrequency(), 1u);
		const auto period  = std::max(frequency / refresh, uint64_t {1});

		if (display_check-- == 0) {
			display_check           = 120;
			const uint32_t per_tick = display_refused ? 0 : m_presenter.DisplayVblanksPerTick(refresh);
			if (per_tick != display_per_tick) {
				if (per_tick != 0) {
					std::printf("VSync: the %u Hz vblanks keep to the display's (%u to one)\n", refresh, per_tick);
				} else {
					std::printf("VSync: the display changed: the vblanks keep their own clock\n");
				}
				std::fflush(stdout);
				display_per_tick = per_tick;
				display_seen     = 0;
				display_ticks    = 0;
			}
		}

		const auto sleep_begin = Common::Timer::QueryPerformanceCounter();
		bool       waited      = false; // for the display's vblanks
		if (display_per_tick != 0) {
			waited = true;
			// Half a display refresh late: it has none at this rate. (The first wait: the display's thread may still be
			// starting.)
			const uint32_t interval_us = 1000000u / (refresh * display_per_tick);
			bool           ok          = true;
			uint64_t       last        = sleep_begin;
			for (uint32_t i = 0; i < display_per_tick && ok; i++) {
				const bool first = display_seen == 0;
				last             = Common::Timer::QueryPerformanceCounter();
				ok = m_presenter.WaitForDisplayVblank(display_seen, first ? 100000u : interval_us * 3 / 2);
			}
			const auto now = Common::Timer::QueryPerformanceCounter();
			const auto ms  = [&](uint64_t ticks) { return static_cast<double>(ticks) * 1000.0 / static_cast<double>(frequency); };
			if (!ok) {
				refuse_display("no vblank from the display in", ms(now - last), interval_us / 1000.0);
			} else if (display_ticks++ == 0) {
				display_since = now;
			} else if (display_ticks == 121) {
				// (Within 1%: a 59.94 Hz display keeps 60 Hz vblanks, 0.1% slower; with a variable refresh rate the
				// display waits for each present after a vblank, 3% or more.)
				const double average  = ms(now - display_since) / 120.0;
				const double expected = 1000.0 / refresh;
				if (std::abs(average - expected) > expected * 0.01) {
					refuse_display("a vblank kept to the display's took", average, expected);
				}
				display_ticks = 0;
			}
			total_wait = 0;
		}
		if (!waited && total_wait > 0) {
			const auto remaining_us =
			    (static_cast<uint64_t>(total_wait) * 1000000u + frequency - 1) / frequency;
			Common::Thread::SleepMicro(static_cast<uint32_t>(
			    std::clamp<uint64_t>(remaining_us, 1, std::numeric_limits<uint32_t>::max())));
		}
		if (token.stop_requested()) {
			break;
		}
		const auto frame_begin = Common::Timer::QueryPerformanceCounter();
		if (!waited) total_wait -= static_cast<int64_t>(frame_begin - sleep_begin);

		// The frames go to the presenter's thread (Presenter::Submit): a present waiting for the display never holds
		// up a vblank. Its idle refreshes (the last frame again, a blank one) only when it has nothing else.
		if (m_presenter.IsGuestPaused()) {
			if (m_presenter.PresentIdle()) {
				if (auto* frame = m_presenter.PrepareLastFrame(); frame != nullptr) {
					m_presenter.Submit(*frame, true);
				}
			}
			const auto frame_end = Common::Timer::QueryPerformanceCounter();
			total_wait +=
			    static_cast<int64_t>(period) - static_cast<int64_t>(frame_end - frame_begin);
			total_wait = std::max(total_wait, -static_cast<int64_t>(period));
			continue;
		}

		VblankBegin();
		bool presented = m_flip_queue.Flip(0);
		if (!presented && m_presenter.NeedsSystemOverlayRefresh() && m_presenter.PresentIdle()) {
			if (auto* frame = m_presenter.PrepareLastFrame(); frame != nullptr) {
				m_presenter.Submit(*frame, true);
			} else {
				uint32_t width  = 0;
				uint32_t height = 0;
				{
					Common::LockGuard lock(m_mutex);
					width  = m_video_out_ctx[0].width;
					height = m_video_out_ctx[0].height;
				}
				m_presenter.SubmitBlank(width, height, true);
			}
			presented = true;
		}
		if (!presented && total_wait < 0 && m_presenter.PresentIdle()) {
			bool     any_open = false;
			uint32_t width    = 0;
			uint32_t height   = 0;
			{
				Common::LockGuard lock(m_mutex);
				width  = m_video_out_ctx[0].width;
				height = m_video_out_ctx[0].height;
				for (int handle = 1; handle < VIDEO_OUT_NUM_MAX; handle++) {
					any_open |= m_video_out_ctx[handle].opened;
				}
			}
			if (!any_open) {
				m_presenter.SubmitBlank(width, height, true);
			}
		}
		VblankEnd();

		const auto frame_end = Common::Timer::QueryPerformanceCounter();
		total_wait += static_cast<int64_t>(period) - static_cast<int64_t>(frame_end - frame_begin);
		// A vblank late by more than a period (the thread held up) is not made up for with vblanks back to back: a burst of
		// them flipped the frames queued meanwhile at once (presents bunched), where a console's vblanks keep their rate.
		total_wait = std::max(total_wait, -static_cast<int64_t>(period));
	}
}

bool FlipQueue::Reserve(VideoOutConfig& cfg, int index, int64_t flip_arg, FlipRequestSource source,
                        uint64_t& request_id) {
	Common::LockGuard lock(m_mutex);

	if (m_requests.size() + m_cpu_requests.size() >= VIDEO_OUT_FLIP_QUEUE_CAPACITY) {
		return false;
	}
	auto& pending = source == FlipRequestSource::GpuEop ? m_requests : m_cpu_requests;

	Request r {};
	r.id         = m_next_request_id++;
	r.cfg        = &cfg;
	r.generation = cfg.generation;
	r.index      = index;
	r.flip_arg   = flip_arg;
	r.submit_ptc = LibKernel::KernelGetProcessTimeCounter();
	r.source     = source;
	r.state      = RequestState::Reserved;

	pending.push_back(r);
	request_id = r.id;

	cfg.flip_status.flipPendingNum = static_cast<int>(m_requests.size() + m_cpu_requests.size());
	cfg.flip_status.submitProcessTimeCounter = r.submit_ptc;
	if (source == FlipRequestSource::GpuEop) {
		cfg.flip_status.gcQueueNum++;
	}

	return true;
}

FlipQueue::~FlipQueue() {
	for (auto* queue: {&m_requests, &m_cpu_requests, &m_cancelled_requests}) {
		for (auto& request: *queue) {
			if (request.frame != nullptr) {
				m_presenter.Discard(*request.frame);
			}
		}
	}
}

void FlipQueue::Cancel(VideoOutConfig& cfg) {
	std::vector<Graphics::Presenter::Frame*> frames;
	m_mutex.Lock();
	while (m_processing && !m_requests.empty() && m_requests.front().cfg == &cfg) {
		m_done_cond_var.Wait(&m_mutex);
	}
	for (auto* queue: {&m_requests, &m_cpu_requests}) {
		for (auto it = queue->begin(); it != queue->end();) {
			if (it->cfg != &cfg) {
				++it;
				continue;
			}
			if (it->state == RequestState::Reserved || it->state == RequestState::Recording) {
				auto cancelled = it++;
				m_cancelled_requests.splice(m_cancelled_requests.end(), *queue, cancelled);
				continue;
			}
			if (it->state == RequestState::Presenting) {
				EXIT("video-out cancellation retained a presenting request\n");
			}
			if (it->frame != nullptr) {
				frames.push_back(it->frame);
			}
			it = queue->erase(it);
		}
	}
	m_done_cond_var.SignalAll();
	m_submit_slot_cond_var.SignalAll();
	m_submit_cond_var.SignalAll();
	m_mutex.Unlock();
	for (auto* frame: frames) {
		m_presenter.Discard(*frame);
	}
	Common::LockGuard lock(cfg.mutex);
	cfg.flip_status.flipPendingNum = 0;
	cfg.flip_status.gcQueueNum     = 0;
}

// KYTY_LOW_LATENCY=1 (off by default): a frame's flip is prepared only once the frames before it that wait for a vblank
// were shown. A game faster than the display otherwise keeps about two frames queued (~30 ms from its flip to the
// present at 60 Hz); held to one, input shows a frame sooner, but a frame late for its vblank is no longer covered by
// a queued one. (At most 50 ms: a paused or closing port does not hold the GPU thread.)
static bool LowLatency() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_LOW_LATENCY");
		return value != nullptr && std::atoi(value) != 0;
	}();
	return on;
}

void FlipQueue::Prepare(uint64_t request_id, Graphics::CommandBuffer& buffer) {
	VideoOutConfig* cfg        = nullptr;
	uint64_t        generation = 0;
	int             index      = 0;
	{
		Common::LockGuard lock(m_mutex);
		if (LowLatency()) {
			const auto waiting = [&] {
				return std::any_of(m_requests.begin(), m_requests.end(), [&](const auto& r) {
					return r.id < request_id && (r.state == RequestState::Ready || r.state == RequestState::Presenting);
				});
			};
			const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
			while (waiting()) {
				const auto now = std::chrono::steady_clock::now();
				if (now >= until) break;
				m_done_cond_var.WaitFor(&m_mutex,
				                        static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(until - now).count()));
			}
		}
		auto request = std::find_if(m_requests.begin(), m_requests.end(),
		                            [request_id](const auto& r) { return r.id == request_id; });
		if (request == m_requests.end()) {
			auto pending = std::find_if(m_cpu_requests.begin(), m_cpu_requests.end(),
			                            [request_id](const auto& r) { return r.id == request_id; });
			if (pending == m_cpu_requests.end()) {
				auto cancelled =
				    std::find_if(m_cancelled_requests.begin(), m_cancelled_requests.end(),
				                 [request_id](const auto& r) { return r.id == request_id; });
				if (cancelled == m_cancelled_requests.end() ||
				    cancelled->state != RequestState::Reserved) {
					EXIT("cannot prepare video-out request id=%" PRIu64 "\n", request_id);
				}
				cancelled->state = RequestState::Recording;
				return;
			}
			request = m_requests.insert(m_requests.end(), *pending);
			m_cpu_requests.erase(pending);
		}
		if (request->state != RequestState::Reserved) {
			EXIT("cannot prepare video-out request id=%" PRIu64 "\n", request_id);
		}
		request->state = RequestState::Recording;
		cfg            = request->cfg;
		generation     = request->generation;
		index          = request->index;
	}

	const bool          special = IsSpecialBufferIndex(index);
	Graphics::ImageInfo source_info;
	uint32_t            width   = 0;
	uint32_t            height  = 0;
	bool                current = false;
	bool                hdr     = false;
	{
		Common::LockGuard lock(cfg->mutex);
		current = cfg->opened && !cfg->closing && cfg->generation == generation;
		if (current) {
			if (special) {
				width  = cfg->width;
				height = cfg->height;
			} else {
				const auto& surface = cfg->buffers[index];
				if (!surface.Occupied()) {
					EXIT("cannot prepare flip from an unregistered surface, id=%" PRIu64
					     " index=%d\n",
					     request_id, index);
				}
				if (surface.group_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX ||
				    !cfg->groups[surface.group_index].occupied) {
					EXIT("video-out surface references an unavailable attribute group, id=%" PRIu64
					     " index=%d group=%d\n",
					     request_id, index, surface.group_index);
				}
				const auto& group = cfg->groups[surface.group_index];
				source_info       = group.ImageInfo(surface);
				hdr               = Graphics::IsHdrVideoOutPixelFormat(group.attribute.pixel_format);
			}
		}
	}
	if (!current) {
		Common::LockGuard lock(m_mutex);
		const auto        request =
		    std::find_if(m_requests.begin(), m_requests.end(),
		                 [request_id](const auto& r) { return r.id == request_id; });
		if (request != m_requests.end()) {
			m_cancelled_requests.splice(m_cancelled_requests.end(), m_requests, request);
		}
		return;
	}
	static std::atomic_flag hdr_warned;
	if (hdr && !m_presenter.IsHdrSupported() && !hdr_warned.test_and_set(std::memory_order_relaxed)) {
		std::printf("warning: the game outputs HDR (PQ) frames without an HDR10 swapchain; shown unconverted\n");
		std::fflush(stdout);
	}
	Graphics::Presenter::Frame* frame = nullptr;
	if (special) {
		frame = &m_presenter.PrepareBlankFrame(width, height, index == VIDEO_OUT_BUFFER_INDEX_BLACK,
		                                       &buffer);
	} else {
		frame = &m_presenter.PrepareFrame(buffer, source_info, hdr);
	}

	Common::LockGuard lock(m_mutex);
	Request*          prepared = nullptr;
	if (const auto request =
	        std::find_if(m_requests.begin(), m_requests.end(),
	                     [request_id](const auto& r) { return r.id == request_id; });
	    request != m_requests.end()) {
		prepared = &*request;
	} else if (const auto cancelled =
	               std::find_if(m_cancelled_requests.begin(), m_cancelled_requests.end(),
	                            [request_id](const auto& r) { return r.id == request_id; });
	           cancelled != m_cancelled_requests.end()) {
		prepared = &*cancelled;
	}
	if (prepared == nullptr || prepared->state != RequestState::Recording ||
	    prepared->frame != nullptr) {
		EXIT("video-out request changed while recording, id=%" PRIu64 "\n", request_id);
	}
	prepared->frame = frame;
}

void FlipQueue::Complete(uint64_t request_id) {
	m_mutex.Lock();
	auto request = std::find_if(m_requests.begin(), m_requests.end(),
	                            [request_id](const auto& r) { return r.id == request_id; });
	if (request != m_requests.end()) {
		if (request->state != RequestState::Recording || request->frame == nullptr) {
			m_mutex.Unlock();
			EXIT("completed GPU flip has no prepared recording, id=%" PRIu64 "\n", request_id);
		}
		request->state = RequestState::Ready;
		m_submit_cond_var.Signal();
		m_mutex.Unlock();
		return;
	}
	auto cancelled = std::find_if(m_cancelled_requests.begin(), m_cancelled_requests.end(),
	                              [request_id](const auto& r) { return r.id == request_id; });
	if (cancelled == m_cancelled_requests.end() || cancelled->state != RequestState::Recording) {
		m_mutex.Unlock();
		EXIT("completed GPU flip has no prepared recording, id=%" PRIu64 "\n", request_id);
	}
	auto* frame = cancelled->frame;
	m_cancelled_requests.erase(cancelled);
	m_done_cond_var.SignalAll();
	m_mutex.Unlock();
	if (frame != nullptr) {
		m_presenter.Discard(*frame);
	}
}

void FlipQueue::WaitForSubmitSlot() {
	Common::LockGuard lock(m_mutex);
	while (m_requests.size() + m_cpu_requests.size() >= VIDEO_OUT_FLIP_QUEUE_CAPACITY) {
		if (m_requests.empty()) {
			EXIT("video-out queue is saturated by CPU flips queued behind the current EOP\n");
		}
		m_submit_slot_cond_var.Wait(&m_mutex);
	}
}

void FlipQueue::Wait(VideoOutConfig& cfg, int index) {
	Common::LockGuard lock(m_mutex);

	auto has_request = [this, &cfg, index] {
		auto matches = [&cfg, index](const auto& r) { return r.cfg == &cfg && r.index == index; };
		return std::any_of(m_requests.begin(), m_requests.end(), matches) ||
		       std::any_of(m_cpu_requests.begin(), m_cpu_requests.end(), matches);
	};
	while (has_request()) {
		m_done_cond_var.Wait(&m_mutex);
	}
}

bool FlipQueue::Flip(uint32_t micros) {
	KYTY_PROFILER_BLOCK("FlipQueue::Flip");

	m_mutex.Lock();
	if (m_requests.empty()) {
		m_submit_cond_var.WaitFor(&m_mutex, micros);

		if (m_requests.empty()) {
			m_mutex.Unlock();
			return false;
		}
	}
	if (m_processing) {
		EXIT("video-out flip queue processing is already active\n");
	}
	if (m_requests.front().state != RequestState::Ready) {
		m_mutex.Unlock();
		return false;
	}
	m_processing = true;
	auto r       = m_requests.front();
	m_mutex.Unlock();

	r.cfg->mutex.Lock();
	int  floor      = 0;
	bool scene_free = true;
	bool fg_off     = true;
	if (r.frame != nullptr) {
		if (Graphics::Presenter::FlipRateApplies(*r.frame)) floor = FlipRateFloor();
		// (Also a movie playing, and with frame generation a frame without the game's camera: the PlayStation Studios
		// logo at the start ran at 103-120 fps above 60 Hz, 10-18; the game draws something with its TAA behind it.)
		NoteFrameForWorld(Graphics::Presenter::IsSceneFrame(*r.frame) && !MoviePlaying());
		scene_free = !Graphics::Presenter::IsSceneFrame(*r.frame) || MoviePlaying() || CinematicPlaying() || !PlayerInputSeen() ||
		             (Graphics::Presenter::GeneratesFrames() && !Graphics::Presenter::IsGameView(*r.frame));
		// Before the first world and during movies frame generation is off for certain: 60, not the mode's base rate
		// (Smooth 120 held the intro to 40, 10-18). In the world the floors as they were (no new pacing there).
		fg_off = MoviePlaying() || !PlayerInputSeen();
		if (scene_free) floor = std::max(floor, SceneFreeFloor());
		if (!fg_off) {
			if (Graphics::Presenter::GeneratesFrames()) floor = std::max(floor, FrameGenFloor());
			if (Graphics::Presenter::IsSceneFrame(*r.frame)) floor = std::max(floor, FlipFpsFloor());
		}
	}
	// FSR 3 under the G-SYNC cap (Fsr3::RealFrameInterval: 1/56 s at x4 on 240 Hz, KYTY_VRR_CAP_FPS): the game's own
	// flips keep to that grid, as Reflex holds the game for DLSS-G. Held only at the present, the game made 60 and 4
	// a second were dropped (a stutter). Half a vblank early is due (the flips land on the nearest vblank).
	static std::chrono::steady_clock::time_point s_paced_next {};
	const double paced_interval = r.frame != nullptr && Graphics::Presenter::IsSceneFrame(*r.frame) && !MoviePlaying() && !fg_off
	                                  ? std::max(Graphics::Fsr3::RealFrameInterval(), Graphics::FrameGen::RealFrameInterval())
	                                  : 0.0;
	const auto   paced_now      = std::chrono::steady_clock::now();
	const auto   paced_step     = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
	                                  std::chrono::duration<double>(paced_interval));
	const auto   paced_slack    = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
	                                  std::chrono::duration<double>(0.5 / std::max(Config::GetVblankFrequency(), 1u)));
	const bool   paced_wait     = paced_interval > 0.0 && paced_now + paced_slack < s_paced_next;
	if (paced_wait || !IsFlipDueLocked(*r.cfg, r.generation, floor)) {
		r.cfg->mutex.Unlock();
		Common::LockGuard queue_lock(m_mutex);
		m_processing = false;
		m_done_cond_var.SignalAll();
		return false;
	}

	if (paced_interval > 0.0) s_paced_next = std::max(s_paced_next, paced_now - paced_step) + paced_step;
	m_mutex.Lock();
	if (m_requests.empty() || m_requests.front().id != r.id ||
	    m_requests.front().state != RequestState::Ready || !m_processing) {
		EXIT("video-out request changed before presentation, id=%" PRIu64 "\n", r.id);
	}
	m_requests.front().state = RequestState::Presenting;
	m_mutex.Unlock();
	// The port is not locked while the frame goes to the presenter's thread (Presenter::Submit; the present itself
	// waits for no vblank of the guest's): the GPU thread reserves the next flip under it. The presented frame is its
	// own copy; a port closed meanwhile takes no status or event.
	r.cfg->mutex.Unlock();

	m_presenter.Submit(*r.frame);

	r.cfg->mutex.Lock();
	m_mutex.Lock();
	if (m_requests.empty() || m_requests.front().id != r.id ||
	    m_requests.front().state != RequestState::Presenting) {
		EXIT("video-out flip queue changed while processing its front request\n");
	}
	m_requests.pop_front();

	if (r.cfg->opened && !r.cfg->closing && r.cfg->generation == r.generation) {
		r.cfg->flip_status.count++;
		r.cfg->last_flip_vblank                     = r.cfg->vblank_status.count;
		r.cfg->flip_status.processTime              = LibKernel::KernelGetProcessTime();
		r.cfg->flip_status.processTimeCounter       = LibKernel::KernelGetProcessTimeCounter();
		r.cfg->flip_status.submitProcessTimeCounter = r.submit_ptc;
		r.cfg->flip_status.flipArg                  = r.flip_arg;
		r.cfg->flip_status.currentBuffer            = r.index;
		r.cfg->flip_status.flipPendingNum = static_cast<int>(m_requests.size() + m_cpu_requests.size());
		if (r.source == FlipRequestSource::GpuEop && r.cfg->flip_status.gcQueueNum > 0) {
			r.cfg->flip_status.gcQueueNum--;
		}
		TriggerVideoOutEvents(*r.cfg, VideoOutEventKind::Flip, reinterpret_cast<void*>(r.flip_arg));
	}

	m_processing = false;
	m_done_cond_var.SignalAll();
	m_submit_slot_cond_var.Signal();
	m_mutex.Unlock();
	r.cfg->mutex.Unlock();

	Graphics::RenderDocOnGuestFlip(m_presenter.Renderer());

	if (Config::GraphicsDebugDumpEnabled() &&
	    Config::GetPrintfDirection() != Config::OutputDirection::Silent) {
		LOGF("Flip done: %d\n", r.index);
	}

	return true;
}

void FlipQueue::GetFlipStatus(VideoOutConfig& cfg, VideoOutFlipStatus& out) {
	Common::LockGuard lock(cfg.mutex);

	out = cfg.flip_status;
}

KYTY_SYSV_ABI int VideoOutOpen(int user_id, int bus_type, int index, const void* param) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(user_id != 255 && user_id != 0);
	if (bus_type != VIDEO_OUT_BUS_TYPE_MAIN && bus_type != VIDEO_OUT_BUS_TYPE_OVERLAY &&
	    bus_type != VIDEO_OUT_BUS_TYPE_SUB) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}
	EXIT_NOT_IMPLEMENTED(index != 0);

	LOGF("\t param = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(param));

	int handle = DriverState().Open(bus_type);

	if (handle < 0) {
		return VIDEO_OUT_ERROR_RESOURCE_BUSY;
	}

	return handle;
}

KYTY_SYSV_ABI int VideoOutClose(int handle) {
	PRINT_NAME();

	return DriverState().Close(handle) ? OK : VIDEO_OUT_ERROR_INVALID_HANDLE;
}

KYTY_SYSV_ABI void VideoOutSetBufferAttribute2(VideoOutBufferAttribute2* attribute,
                                               uint64_t pixel_format, uint32_t tiling_mode,
                                               uint32_t width, uint32_t height, uint64_t option,
                                               uint32_t dcc_control,
                                               uint64_t dcc_cb_register_clear_color) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(attribute == nullptr);

	LOGF("\t pixel_format                = %016" PRIx64 "\n"
	     "\t tiling_mode                 = %" PRIu32 "\n"
	     "\t width                       = %" PRIu32 "\n"
	     "\t height                      = %" PRIu32 "\n"
	     "\t option                      = %016" PRIx64 "\n"
	     "\t dcc_control                 = %08" PRIx32 "\n"
	     "\t dcc_cb_register_clear_color = %016" PRIx64 "\n",
	     pixel_format, tiling_mode, width, height, option, dcc_control,
	     dcc_cb_register_clear_color);

	memset(attribute, 0, sizeof(VideoOutBufferAttribute2));

	attribute->tiling_mode                 = tiling_mode;
	attribute->aspect_ratio                = 0;
	attribute->width                       = width;
	attribute->height                      = height;
	attribute->pitch_in_pixel              = 0;
	attribute->option                      = option;
	attribute->pixel_format                = pixel_format;
	attribute->dcc_cb_register_clear_color = dcc_cb_register_clear_color;
	attribute->dcc_control                 = dcc_control;
}

KYTY_SYSV_ABI int VideoOutSetFlipRate(int handle, int rate) {
	PRINT_NAME();

	LOGF("\trate = %d\n", rate);

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	if (rate < 0 || rate > 2) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

	Common::LockGuard lock(ctx->mutex);
	ctx->flip_rate = rate;

	return OK;
}

KYTY_SYSV_ABI int VideoOutDeleteFlipEvent(EventQueue::KernelEqueue eq, int handle) {
	PRINT_NAME();
	return DeleteVideoOutEvent(handle, eq, VideoOutEventKind::Flip);
}

KYTY_SYSV_ABI int VideoOutAddFlipEvent(EventQueue::KernelEqueue eq, int handle, void* udata) {
	PRINT_NAME();
	return RegisterVideoOutEvent(handle, eq, VideoOutEventKind::Flip, udata);
}

KYTY_SYSV_ABI int VideoOutDeleteVblankEvent(EventQueue::KernelEqueue eq, int handle) {
	PRINT_NAME();
	return DeleteVideoOutEvent(handle, eq, VideoOutEventKind::Vblank);
}

KYTY_SYSV_ABI int VideoOutDeletePreVblankStartEvent(EventQueue::KernelEqueue eq, int handle) {
	PRINT_NAME();
	return DeleteVideoOutEvent(handle, eq, VideoOutEventKind::PreVblankStart);
}

KYTY_SYSV_ABI int VideoOutAddVblankEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                         void* udata) {
	PRINT_NAME();
	return RegisterVideoOutEvent(handle, eq, VideoOutEventKind::Vblank, udata);
}

KYTY_SYSV_ABI int VideoOutAddPreVblankStartEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                                 void* udata) {
	PRINT_NAME();
	return RegisterVideoOutEvent(handle, eq, VideoOutEventKind::PreVblankStart, udata);
}

KYTY_SYSV_ABI int VideoOutAddOutputModeEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                             void* udata) {
	PRINT_NAME();
	return RegisterVideoOutEvent(handle, eq, VideoOutEventKind::OutputMode, udata);
}

KYTY_SYSV_ABI int VideoOutRegisterBuffers2(int handle, int set_index, int buffer_index_start,
                                           const VideoOutBuffers* buffers, int buffer_num,
                                           const VideoOutBufferAttribute2* attribute, int category,
                                           void* option) {
	PRINT_NAME();

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	if (buffers == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (attribute == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_OPTION;
	}

	if (set_index < 0 || set_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX ||
	    buffer_index_start < 0 || buffer_index_start >= VIDEO_OUT_BUFFER_NUM_MAX ||
	    buffer_num < 1 || buffer_num > VIDEO_OUT_BUFFER_NUM_MAX ||
	    buffer_index_start + buffer_num > VIDEO_OUT_BUFFER_NUM_MAX) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

	LOGF("\t start_index    = %d\n"
	     "\t buffer_num     = %d\n"
	     "\t set_index      = %d\n"
	     "\t pixel_format   = 0x%016" PRIx64 "\n"
	     "\t tiling_mode    = %" PRIu32 "\n"
	     "\t aspect_ratio   = %" PRIu32 "\n"
	     "\t width          = %" PRIu32 "\n"
	     "\t height         = %" PRIu32 "\n"
	     "\t pitch_in_pixel = %" PRIu32 "\n"
	     "\t option         = %" PRIu64 "\n"
	     "\t category       = %d\n",
	     buffer_index_start, buffer_num, set_index, attribute->pixel_format, attribute->tiling_mode,
	     attribute->aspect_ratio, attribute->width, attribute->height, attribute->pitch_in_pixel,
	     attribute->option, category);

	if (option != nullptr) {
		return VIDEO_OUT_ERROR_INVALID_OPTION;
	}
	if (category != VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_UNCOMPRESSED &&
	    category != VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_COMPRESSED) {
		return VIDEO_OUT_ERROR_INVALID_CATEGORY;
	}

	BufferAttributeGroup group {
	    .attribute = *attribute,
	    .category  = category,
	    .occupied  = true,
	};
	std::vector<VideoOutBuffer> registrations;
	registrations.reserve(static_cast<size_t>(buffer_num));

	for (int i = 0; i < buffer_num; i++) {
		LOGF("\t buffers[%d]: data=%p metadata=%p\n", i, buffers[i].data, buffers[i].metadata);
		if (buffers[i].reserved[0] != nullptr || buffers[i].reserved[1] != nullptr) {
			LOGF("\t buffers[%d]: ignoring reserved fields {%p, %p}\n", i, buffers[i].reserved[0],
			     buffers[i].reserved[1]);
		}
		const auto data_address     = reinterpret_cast<uint64_t>(buffers[i].data);
		const auto metadata_address = reinterpret_cast<uint64_t>(buffers[i].metadata);
		registrations.push_back({
		    .group_index      = set_index,
		    .data_address     = data_address,
		    .metadata_address = metadata_address,
		});
		(void)group.ImageInfo(registrations.back());
	}

	Common::LockGuard lock(ctx->mutex);
	if (ctx->closing) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (ctx->groups[set_index].occupied) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}
	for (int i = 0; i < buffer_num; i++) {
		if (ctx->buffers[buffer_index_start + i].Occupied()) {
			return VIDEO_OUT_ERROR_SLOT_OCCUPIED;
		}
	}

	ctx->groups[set_index] = group;
	for (int i = 0; i < buffer_num; i++) {
		ctx->buffers[buffer_index_start + i] = registrations[static_cast<size_t>(i)];
		const auto& buffer                   = registrations[static_cast<size_t>(i)];
		LOGF("\tbuffers[%d] = %016" PRIx64 " metadata = %016" PRIx64 " dcc = %08" PRIx32 "\n",
		     buffer_index_start + i, buffer.data_address, buffer.metadata_address,
		     attribute->dcc_control);
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutSubmitChangeBufferAttribute2(int handle, int set_index,
                                                       const VideoOutBufferAttribute2* attribute,
                                                       void*                           option) {
	PRINT_NAME();

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	if (attribute == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_OPTION;
	}
	if (set_index < 0 || set_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}

	if (option != nullptr) {
		return VIDEO_OUT_ERROR_INVALID_OPTION;
	}

	Common::LockGuard lock(ctx->mutex);
	const auto&       current = ctx->groups[set_index];
	if (ctx->closing || !current.occupied) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}

	BufferAttributeGroup replacement {
	    .attribute = *attribute,
	    .category  = current.category,
	    .occupied  = true,
	};
	for (const auto& buffer: ctx->buffers) {
		if (buffer.group_index == set_index) {
			(void)replacement.ImageInfo(buffer);
		}
	}
	ctx->groups[set_index] = replacement;

	return OK;
}

KYTY_SYSV_ABI int VideoOutUnregisterBuffers(int handle, int set_index) {
	PRINT_NAME();

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (set_index < 0 || set_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}

	Common::LockGuard lock(ctx->mutex);
	if (ctx->closing) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (!ctx->groups[set_index].occupied) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}
	ctx->groups[set_index] = BufferAttributeGroup {};
	for (auto& buffer: ctx->buffers) {
		if (buffer.group_index == set_index) {
			buffer = VideoOutBuffer {};
		}
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutSubmitFlip(int handle, int index, int flip_mode, int64_t flip_arg) {
	PRINT_NAME();

	uint64_t  request_id = 0;
	const int result     = ReserveFlipRequest(DriverState(), handle, index, flip_mode, flip_arg,
	                                          FlipRequestSource::Cpu, request_id);
	if (result == VIDEO_OUT_ERROR_INVALID_VALUE) {
		LOGF("\t unsupported flip_mode = %d\n", flip_mode);
	}
	if (result != OK) {
		return result;
	}
	g_video_out_driver->SubmitFlipPreparation(request_id);

	return OK;
}

int VideoOutDriver::SubmitFlipFromGpu(Graphics::CommandBuffer& buffer, int handle, int index,
                                      int flip_mode, int64_t flip_arg, uint64_t& request_id) {
	EXIT_IF(buffer.IsInvalid());

	const int result = ReserveFlipRequest(*m_impl, handle, index, flip_mode, flip_arg,
	                                      FlipRequestSource::GpuEop, request_id);
	if (result != OK) {
		return result;
	}
	m_impl->GetFlipQueue().Prepare(request_id, buffer);

	return OK;
}

void VideoOutDriver::SubmitFlipPreparation(uint64_t request_id) {
	m_impl->Renderer().GetGpu().SubmitFlipPreparation(request_id);
}

void VideoOutDriver::PrepareFlip(uint64_t request_id, Graphics::CommandBuffer& buffer) {
	m_impl->GetFlipQueue().Prepare(request_id, buffer);
}

void VideoOutDriver::CompleteFlip(uint64_t request_id) {
	LiveTrace::Event(LiveTrace::FlipComplete, request_id);
	m_impl->GetFlipQueue().Complete(request_id);
}

void VideoOutDriver::WaitForSubmitSlot() {
	m_impl->GetFlipQueue().WaitForSubmitSlot();
}

void VideoOutDriver::WaitFlipDone(int handle, int index) {
	auto* ctx = m_impl->Get(handle);
	EXIT_IF(ctx == nullptr);

	EXIT_NOT_IMPLEMENTED(!IsValidBufferIndex(index));
	m_impl->GetFlipQueue().Wait(*ctx, index);
}

KYTY_SYSV_ABI int VideoOutGetFlipStatus(int handle, VideoOutFlipStatus* status) {
	PRINT_NAME();
	KYTY_TIME_CENSUS(FlipStatus, 0);

	if (status == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	DriverState().GetFlipQueue().GetFlipStatus(*ctx, *status);

	LOGF("\t count = %" PRIu64 "\n"
	     "\t processTime = %" PRIu64 "\n"
	     "\t processTimeCounter = %" PRIu64 "\n"
	     "\t submitProcessTimeCounter = %" PRIu64 "\n"
	     "\t flipArg = %" PRId64 "\n"
	     "\t gcQueueNum = %d\n"
	     "\t flipPendingNum = %d\n"
	     "\t currentBuffer = %d\n",
	     status->count, status->processTime, status->processTimeCounter,
	     status->submitProcessTimeCounter, status->flipArg, status->gcQueueNum,
	     status->flipPendingNum, status->currentBuffer);

	return OK;
}

KYTY_SYSV_ABI int VideoOutIsFlipPending(int handle) {
	PRINT_NAME();

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	VideoOutFlipStatus status {};
	DriverState().GetFlipQueue().GetFlipStatus(*ctx, status);

	LOGF("\t flipPendingNum = %d\n", status.flipPendingNum);

	return status.flipPendingNum;
}

KYTY_SYSV_ABI int VideoOutGetVblankStatus(int handle, VideoOutVblankStatus* status) {
	PRINT_NAME();
	KYTY_TIME_CENSUS(VblankStatus, 0);

	if (status == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	ctx->mutex.Lock();
	*status = ctx->vblank_status;
	ctx->mutex.Unlock();

	LOGF("\t count = %" PRIu64 "\n"
	     "\t processTime = %" PRIu64 "\n"
	     "\t processTimeCounter = %" PRIu64 "\n",
	     status->count, status->processTime, status->processTimeCounter);

	return OK;
}

KYTY_SYSV_ABI int VideoOutGetEventId(const EventQueue::KernelEvent* ev) {
	PRINT_NAME();

	if (ev == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (ev->filter != EventQueue::KERNEL_EVFILT_VIDEO_OUT) {
		return VIDEO_OUT_ERROR_INVALID_EVENT;
	}

	switch (ev->ident) {
		case VIDEO_OUT_EVENT_FLIP:
		case VIDEO_OUT_EVENT_VBLANK:
		case VIDEO_OUT_EVENT_PRE_VBLANK_START:
		case VIDEO_OUT_EVENT_SET_MODE: return static_cast<int>(ev->ident);
		default: return VIDEO_OUT_ERROR_INVALID_EVENT;
	}
}

KYTY_SYSV_ABI int VideoOutGetEventData(const EventQueue::KernelEvent* ev, int64_t* data) {
	PRINT_NAME();

	if (ev == nullptr || data == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (ev->filter != EventQueue::KERNEL_EVFILT_VIDEO_OUT) {
		return VIDEO_OUT_ERROR_INVALID_EVENT;
	}

	uint64_t event_data = static_cast<uint64_t>(ev->data) >> 16u;
	if (ev->ident == VIDEO_OUT_EVENT_FLIP &&
	    (static_cast<uint64_t>(ev->data) & 0x8000000000000000ULL) != 0) {
		event_data |= 0xffff000000000000ULL;
	}

	*data = static_cast<int64_t>(event_data);

	return OK;
}

KYTY_SYSV_ABI int VideoOutGetEventCount(const EventQueue::KernelEvent* ev) {
	PRINT_NAME();

	if (ev == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (ev->filter != EventQueue::KERNEL_EVFILT_VIDEO_OUT) {
		return VIDEO_OUT_ERROR_INVALID_EVENT;
	}

	return static_cast<int>((static_cast<uint64_t>(ev->data) >> 12u) & 0xfu);
}

KYTY_SYSV_ABI int VideoOutWaitVblank(int handle) {
	PRINT_NAME();
	KYTY_TIME_CENSUS(WaitVblank, 0);

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	Common::LockGuard lock(ctx->mutex);
	const auto        count = ctx->vblank_status.count;
	while (ctx->opened && ctx->vblank_status.count == count) {
		ctx->vblank_cond.Wait(&ctx->mutex);
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutGetOutputStatus(int handle, VideoOutOutputStatus* status) {
	PRINT_NAME();

	if (status == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	// HDR when the title declares it (param ATTRIBUTE bit 29) and --hdr found an HDR10 display:
	// the game then flips BT.2100 PQ buffers, which the HDR10 swapchain shows as they are.
	int32_t    attribute   = 0;
	const bool hdr_allowed = !Loader::SystemContentParamSfoGetInt("ATTRIBUTE", &attribute) ||
	                         (attribute & 0x20000000) != 0;
	ctx->mutex.Lock();
	status->resolution   = (ctx->width >= 3840 || ctx->height >= 2160 ? 2u : 1u);
	status->dynamicRange = hdr_allowed && DriverState().IsHdrSupported() ? 2u : 1u;
	status->refreshRate =
	    (ctx->output_mode == VIDEO_OUT_OUTPUT_MODE_119_88HZ || Config::GetVblankFrequency() >= 119
	         ? VIDEO_OUT_REFRESH_RATE_119_88HZ
	         : VIDEO_OUT_REFRESH_RATE_59_94HZ);
	status->flags       = 0;
	status->reserved[0] = 0;
	status->reserved[1] = 0;
	status->reserved[2] = 0;
	ctx->mutex.Unlock();

	return OK;
}

static int ValidateOutputConfig(int handle, uint64_t mode, const VideoOutOutputOptions* options,
                                void* reserved_ptr, uint64_t reserved) {
	if (!DriverState().IsOpened(handle)) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	if (reserved_ptr != nullptr || reserved != 0) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

	if (options != nullptr) {
		for (auto v: options->internalData) {
			if (v != 0) {
				return VIDEO_OUT_ERROR_INVALID_OPTION;
			}
		}
	}

	if (mode != VIDEO_OUT_OUTPUT_MODE_DEFAULT && mode != VIDEO_OUT_OUTPUT_MODE_119_88HZ) {
		return VIDEO_OUT_ERROR_UNSUPPORTED_OUTPUT_MODE;
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutInitializeOutputOptions(VideoOutOutputOptions* options) {
	PRINT_NAME();

	if (options == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	memset(options, 0, sizeof(VideoOutOutputOptions));

	return OK;
}

KYTY_SYSV_ABI int VideoOutIsOutputSupported(int handle, uint64_t mode,
                                            const VideoOutOutputOptions* options,
                                            void* reserved_ptr, uint64_t reserved) {
	PRINT_NAME();

	LOGF("\t mode = 0x%016" PRIx64 "\n", mode);

	int result = ValidateOutputConfig(handle, mode, options, reserved_ptr, reserved);
	if (result != OK) {
		return result;
	}

	if (mode == VIDEO_OUT_OUTPUT_MODE_119_88HZ) {
		return (Config::GetVblankFrequency() >= 119 ? VIDEO_OUT_TRUE : VIDEO_OUT_FALSE);
	}

	return VIDEO_OUT_TRUE;
}

KYTY_SYSV_ABI int VideoOutConfigureOutput(int handle, uint64_t mode,
                                          const VideoOutOutputOptions* options, void* reserved_ptr,
                                          uint64_t reserved) {
	PRINT_NAME();

	LOGF("\t mode = 0x%016" PRIx64 "\n", mode);

	int result = VideoOutIsOutputSupported(handle, mode, options, reserved_ptr, reserved);
	if (result < 0) {
		return result;
	}
	if (result == VIDEO_OUT_FALSE) {
		return VIDEO_OUT_ERROR_UNAVAILABLE_OUTPUT_MODE;
	}

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	ctx->mutex.Lock();
	ctx->output_mode = mode;
	TriggerVideoOutEvents(*ctx, VideoOutEventKind::OutputMode,
	                      reinterpret_cast<void*>(ctx->output_mode));
	ctx->mutex.Unlock();

	return OK;
}

KYTY_SYSV_ABI int VideoOutSetWindowModeMargins(int handle, int top, int bottom) {
	PRINT_NAME();

	[[maybe_unused]] auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	LOGF("\t top    = %d\n"
	     "\t bottom = %d\n",
	     top, bottom);

	return OK;
}

KYTY_SYSV_ABI int VideoOutLatencyControlWaitBeforeInput(int handle) {
	PRINT_NAME();

	[[maybe_unused]] auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutLatencyMeasureSetStartPoint(int handle, uint32_t point) {
	PRINT_NAME();

	[[maybe_unused]] auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	LOGF("\t point = %" PRIu32 "\n", point);

	return OK;
}

KYTY_SYSV_ABI int VideoOutColorSettingsSetGamma(VideoOutColorSettings* settings, float gamma) {
	PRINT_NAME();

	if (settings == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (gamma < 0.1f || gamma > 2.0f) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

	settings->gamma = gamma;
	return OK;
}

KYTY_SYSV_ABI int VideoOutAdjustColor(int handle, const VideoOutColorSettings* settings) {
	PRINT_NAME();

	if (settings == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (!DriverState().IsOpened(handle)) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	ctx->mutex.Lock();
	ctx->gamma = settings->gamma;
	ctx->mutex.Unlock();

	return OK;
}

} // namespace Libs::VideoOut
