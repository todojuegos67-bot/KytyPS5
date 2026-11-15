# The precompile's environment guard (session 12): every KYTY_ switch the shader, pipeline, buffer-cache and
# gpu_format code reads must be in Origin()'s fingerprint list (shaderPrecompile.cpp, the switches that change SPIR-V
# or pipelines) or in the output-only list below. A new switch in neither fails here: decide which it is, or a warm
# list and the replay check would trust results made with another setting.
# usage: cmake -DSRC=<src dir> -P PrecompileEnvGuard.cmake
cmake_minimum_required(VERSION 3.22)
set(output_only
	KYTY_STAGING_MB          # the upload ring's size; what is uploaded is the same
	KYTY_SPARSE_GEOMETRY     # which game buffers are sparse; what they hold is the same
	KYTY_RT                  # runs ray-tracing dispatches that are skipped otherwise; the SPIR-V of the others is the same
	KYTY_TARGET_MEMO         # whether a render target's resolve repeats from its memo (no shader change)
	KYTY_CHECK_SCALAR_READS  # checks scalar reads against a draw's writes; stops or warns, changes no draw
	KYTY_PRECOMPILE          # the precompile on start and the recording
	KYTY_PRECOMPILE_THREADS
	KYTY_PRECOMPILE_REUSE
	KYTY_PRECOMPILE_HAND_OVER # hands created pipelines to the GPU thread; the SPIR-V is the same
	KYTY_PRECOMPILE_READY_MAX # how many of them it keeps; the SPIR-V is the same
	KYTY_PRECOMPILE_PAIR      # adds paired jobs; their SPIR-V and pipelines are what a draw would make
	KYTY_PRECOMPILE_PREDICT   # adds predicted jobs; their SPIR-V and pipelines are what a draw of that record would make
)
file(READ "${SRC}/graphics/host_gpu/renderer/pipeline/shaderPrecompile.cpp" origin)
# Only Origin()'s own list counts: a name merely mentioned elsewhere in the file is not fingerprinted.
string(FIND "${origin}" "uint64_t Origin()" origin_begin)
if(origin_begin EQUAL -1)
	message(FATAL_ERROR "no Origin() in shaderPrecompile.cpp")
endif()
string(SUBSTRING "${origin}" ${origin_begin} -1 origin)
string(FIND "${origin}" "return origin;" origin_end)
string(SUBSTRING "${origin}" 0 ${origin_end} origin)
string(REGEX MATCHALL "\"KYTY_[A-Z0-9_]+\"" fingerprint "${origin}")
string(REPLACE "\"" "" fingerprint "${fingerprint}")
if(NOT fingerprint)
	message(FATAL_ERROR "no KYTY_ names found in shaderPrecompile.cpp's Origin()")
endif()
file(GLOB_RECURSE files
	"${SRC}/graphics/shader/*.cpp" "${SRC}/graphics/shader/*.h"
	"${SRC}/graphics/host_gpu/renderer/pipeline/*.cpp" "${SRC}/graphics/host_gpu/renderer/pipeline/*.h"
	"${SRC}/graphics/host_gpu/renderer/cache/bufferCache.cpp" "${SRC}/graphics/guest_gpu/gpu_format.cpp")
set(unknown "")
foreach(file IN LISTS files)
	file(READ "${file}" text)
	string(REGEX MATCHALL "getenv\\(\"KYTY_[A-Z0-9_]+\"" reads "${text}")
	foreach(read IN LISTS reads)
		string(REGEX REPLACE "getenv\\(\"(KYTY_[A-Z0-9_]+)\"" "\\1" name "${read}")
		if(NOT name IN_LIST fingerprint AND NOT name IN_LIST output_only)
			list(APPEND unknown "${name} (${file})")
		endif()
	endforeach()
endforeach()
if(unknown)
	message(FATAL_ERROR "KYTY_ switches neither in Origin()'s fingerprint nor output-only: ${unknown}")
endif()
message(STATUS "precompile env guard: ${fingerprint} fingerprinted")
