#ifndef EMULATOR_SRC_LOCAL_LIVE_CENSUS_H_
#define EMULATOR_SRC_LOCAL_LIVE_CENSUS_H_
// The one piece of the todojuegos67-bot fork's live census the frame generation port uses: whether this thread is
// the renderer thread (Kyty.Gpu), which records the guest's command buffers (set where that thread starts).
namespace LiveCensus {
inline thread_local bool g_render = false;
} // namespace LiveCensus
#endif // EMULATOR_SRC_LOCAL_LIVE_CENSUS_H_
