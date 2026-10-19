// Offline repro: translate (and compile) one dumped guest shader, as the pipeline cache does for a CS.
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>
#include <cstdlib>
#include <cstring>
using namespace Libs::Graphics;
int main(int argc, char** argv) {
	if (argc < 2) { std::fprintf(stderr, "usage: %s shader.bin [wave]\n", argv[0]); return 2; }
	std::ifstream f(argv[1], std::ios::binary);
	std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
	std::vector<uint32_t> code(bytes.size() / 4);
	std::memcpy(code.data(), bytes.data(), code.size() * 4);
	ShaderRecompiler::CompileOptions options;
	options.stage = ShaderType::Compute;
	options.maintenance8_enabled = true;
	ShaderComputeInputInfo info {};
	info.wave_size = argc > 2 ? std::atoi(argv[2]) : 32;
	info.host_subgroup_size = 32;
	info.threads_num[0] = 64; info.threads_num[1] = 1; info.threads_num[2] = 1;
	options.wave_size = info.wave_size;
	options.input_info.compute = &info;
	static uint32_t user_data[16] {};
	options.user_data = user_data;
	std::printf("translate %zu dwords wave %u\n", code.size(), info.wave_size);
	std::fflush(stdout);
	auto translated = ShaderRecompiler::TranslateProgram(code, options);
	std::printf("translate done: %zu blocks\n", translated.program.blocks.size());
	return 0;
}
