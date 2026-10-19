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
	info.host_subgroup_size = argc > 3 ? std::atoi(argv[3]) : 32;
	info.threads_num[0] = 64; info.threads_num[1] = 1; info.threads_num[2] = 1;
	options.wave_size = info.wave_size;
	options.input_info.compute = &info;
	static uint32_t user_data[16] {};
	options.user_data = user_data;
	std::printf("translate %zu dwords wave %u\n", code.size(), info.wave_size);
	std::fflush(stdout);
	auto translated = ShaderRecompiler::TranslateProgram(code, options);
	std::printf("translate done: %zu blocks\n", translated.program.blocks.size());
	if (argc > 4) {
		ShaderRecompiler::IR::ResourceSpecialization spec {};
		const auto& pi = translated.program.info;
		spec.buffers.resize(pi.buffers.size());
		spec.images.resize(pi.images.size());
		for (auto& im: spec.images) { im.numeric_class = Prospero::TextureNumericClass::Float; im.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D; }
		spec.samplers.resize(pi.samplers.size());
		std::printf("resources: buffers %zu images %zu samplers %zu\n", pi.buffers.size(), pi.images.size(), pi.samplers.size());
		auto compiled = ShaderRecompiler::CompileProgram(std::move(translated), options, spec, 0);
		std::printf("spirv words: %zu\n", compiled.spirv.size());
		std::FILE* o = std::fopen(argv[4], "wb");
		std::fwrite(compiled.spirv.data(), 4, compiled.spirv.size(), o);
		std::fclose(o);
	}
	return 0;
}
