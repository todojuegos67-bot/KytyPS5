#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_THREADSAMPLER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_THREADSAMPLER_H_

#include <cstdio>

namespace Libs::Graphics {

// Debugging aid (Windows): with KYTY_DEBUG_SAMPLE_GPU=<period in microseconds> (values below 100
// select 500), a helper thread samples the calling thread's call stack and writes each 10-second
// window's stacks to sample-<name>-<window>.txt in the working directory, as module+offset
// frames (innermost first) for llvm-symbolizer. The sampled thread stops only while its
// registers and the top of its stack are copied. Call it from the thread to sample.
void StartThreadSampler(const char* name);

// Debugging aid (Windows): with KYTY_DEBUG_DUMP_THREADS=<seconds>, a helper thread writes the call
// stacks of all the process's threads to thread-dump-<n>.txt every <seconds>, one "# tid=... name=..."
// line and one sampler-format stack per thread, to see what a stalled game waits on.
void StartThreadDumper();

// Windows: the call stack of every other thread of the process, in the dumper's format, each line
// starting with `prefix` (the stall watch, kernel/stallWatch.h). Elsewhere nothing.
void DumpAllThreadStacks(FILE* file, const char* prefix);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_THREADSAMPLER_H_
