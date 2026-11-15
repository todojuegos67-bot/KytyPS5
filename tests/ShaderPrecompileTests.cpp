#include "common/emulatorConfig.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"
#include "graphics/shader/shaderCompiler.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

namespace Precompile = Libs::Graphics::Precompile;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "ShaderPrecompileTests: failed: %s\n", message);
		std::abort();
	}
}

// A shader block as the game stores it: code that ends in the 48-byte footer starting with "barefoot".
std::vector<uint32_t> MakeCode(uint32_t seed, uint32_t words) {
	std::vector<uint32_t> code(words);
	for (uint32_t i = 0; i < words; i++) code[i] = seed * 2654435761u + i * 40503u;
	std::memcpy(code.data() + words - 12u, "barefoot", 8);
	return code;
}

// A shader binary as the game stores it since upstream ae67e2db's view: code (code_bytes, not a multiple of 16), its
// literal buffer at 16 bytes, metadata at 16, then the footer giving their sizes. The shader map hashes and translates
// the program: code and literal buffer.
struct Binary {
	std::vector<uint32_t> whole, program;
};
Binary MakeBinary(uint32_t seed, uint32_t code_bytes, uint32_t literal_bytes, uint32_t metadata_bytes) {
	const auto align16 = [](uint32_t value) { return (value + 15u) & ~15u; };
	const auto program = align16(code_bytes) + literal_bytes;
	const auto whole   = align16(program) + align16(metadata_bytes) + 48u;
	Binary     binary;
	binary.whole.resize(whole / 4u);
	for (uint32_t i = 0; i < whole / 4u; i++) binary.whole[i] = seed * 2246822519u + i * 3266489917u;
	auto* footer = binary.whole.data() + (whole - 48u) / 4u;
	std::memcpy(footer, "barefoot", 8);
	footer[4] = 1;
	footer[5] = code_bytes;
	footer[6] = literal_bytes;
	footer[7] = metadata_bytes;
	binary.program.assign(binary.whole.begin(), binary.whole.begin() + program / 4u);
	return binary;
}

Precompile::StageRecord PixelRecord(uint32_t seed) {
	Precompile::StageRecord record;
	record.stage           = Libs::Graphics::ShaderType::Pixel;
	record.code            = MakeCode(seed, 64);
	record.shader_hash     = Precompile::CodeHash(record.code, {});
	record.user_data_count = 4;
	Libs::Graphics::ShaderPixelInputInfo pixel {};
	pixel.input_num                  = seed % 7u;
	pixel.interpolator_settings[0]   = seed;
	record.input                     = pixel;
	record.push_data_cursor          = 12;
	record.specialization.images.push_back({.mip_count = seed % 5u + 1u, .guest_heap = true});
	record.specialization.buffers.push_back({.packed_stride = 16, .zero_stride_oob = true});
	record.specialization.samplers.push_back({.indirect_root = 3});
	record.spirv_hash = seed * 77u;
	record.origin     = Precompile::Origin();
	return record;
}

Precompile::StageRecord ComputeRecord(uint32_t seed) {
	Precompile::StageRecord record;
	record.stage       = Libs::Graphics::ShaderType::Compute;
	record.code        = MakeCode(seed, 96);
	record.shader_hash = Precompile::CodeHash(record.code, {});
	Libs::Graphics::ShaderComputeInputInfo compute {};
	compute.threads_num[0]     = 64;
	compute.host_subgroup_size = 32;
	record.input               = compute;
	return record;
}

Precompile::PipelineRecord Pipeline(uint64_t vertex, uint64_t pixel) {
	Precompile::PipelineRecord record;
	record.vertex[0]                        = vertex;
	record.pixel                            = pixel;
	record.rendering.color_count            = 1;
	record.rendering.color_formats[0]       = vk::Format::eR8G8B8A8Unorm;
	record.vertex_input.binding_count       = 1;
	record.vertex_input.bindings[0].stride  = 32;
	record.vertex_input.attribute_count     = 1;
	record.static_params.blend_enable[0]    = true;
	record.static_params.color_srcblend[0]  = 4;
	return record;
}

// A stage with buffers moves to the variant this run draws with (the table), its pipeline follows its new key, a
// stage without buffers stays, and a second migration changes nothing.
void TestBufferTableMigration() {
	constexpr bool table = true;
	Precompile::Journal journal;
	journal.stages = {PixelRecord(5), ComputeRecord(6)};
	journal.stages[0].specialization.buffer_table = !table;
	const auto compute_key = journal.stages[1].Key();
	journal.pipelines = {Pipeline(compute_key, journal.stages[0].Key())};
	Check(Precompile::MigrateBufferTable(journal), "a stage of the other variant migrates");
	Check(journal.stages.size() == 2 && journal.pipelines.size() == 1, "migration keeps one record each");
	Check(journal.stages[0].specialization.buffer_table == table && journal.stages[0].origin == 0 &&
	          journal.stages[0].spirv_hash == 0,
	      "the migrated stage takes the drawn variant and drops its recorded translation");
	Check(journal.stages[1].Key() == compute_key, "a stage without buffers stays");
	Check(journal.pipelines[0].pixel == journal.stages[0].Key() && journal.pipelines[0].vertex[0] == compute_key,
	      "the pipeline follows its stage's new key");
	Check(!Precompile::MigrateBufferTable(journal), "a second migration changes nothing");
	journal.stages.push_back(journal.stages[0]);
	journal.stages.back().specialization.buffer_table = !table;
	Check(Precompile::MigrateBufferTable(journal) && journal.stages.size() == 2, "a record of each variant becomes one");
}

void TestJournal(const std::filesystem::path& dir) {
	const auto path   = dir / "t.journal";
	const auto device = Precompile::DeviceFingerprint(0x10de, 0x2b85);
	Precompile::Journal journal;
	journal.stages    = {PixelRecord(1), ComputeRecord(2)};
	journal.pipelines = {Pipeline(journal.stages[1].Key(), journal.stages[0].Key())};
	{
		Precompile::JournalWriter writer;
		Check(writer.Open(path, device, journal, true), "a journal opens for writing");
		writer.Append(PixelRecord(3));
	}
	auto read = Precompile::ReadJournal(path, device);
	Check(read.found && !read.foreign && !read.torn && read.rejected == 0, "the journal reads clean");
	Check(read.stages.size() == 3 && read.pipelines.size() == 1, "every record reads back");
	Check(read.stages[0].Key() == journal.stages[0].Key() && read.stages[1].Key() == journal.stages[1].Key(),
	      "keys survive the round trip");
	Check(read.stages[0].code == journal.stages[0].code && read.stages[0].specialization == journal.stages[0].specialization,
	      "code and specialization survive");
	Check(std::get<Libs::Graphics::ShaderPixelInputInfo>(read.stages[0].input).interpolator_settings[0] == 1u,
	      "the stage input survives");
	Check(read.pipelines[0].Key() == journal.pipelines[0].Key(), "a pipeline record survives");
	Check(Precompile::ReadJournal(path, Precompile::DeviceFingerprint(0x1002, 1)).foreign, "another GPU's journal is foreign");

	// A different specialization or push cursor is another permutation; the origin is not in the key.
	auto other = PixelRecord(1);
	other.origin++;
	Check(other.Key() == journal.stages[0].Key(), "the origin is not in the key");
	other.push_data_cursor++;
	Check(other.Key() != journal.stages[0].Key(), "the push cursor is in the key");

	// A killed run's torn tail: the records before it stay.
	const auto size = std::filesystem::file_size(path);
	std::filesystem::resize_file(path, size - 10);
	read = Precompile::ReadJournal(path, device);
	Check(read.torn && read.stages.size() == 2 && read.pipelines.size() == 1, "a torn tail drops the last record only");

	// A damaged word: that record is rejected, the rest stay.
	Check(Precompile::WriteJournal(path, device, journal), "the journal rewrites");
	{
		std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
		file.seekp(100);
		file.put('\x5a');
	}
	read = Precompile::ReadJournal(path, device);
	Check(read.rejected == 1 && read.stages.size() + read.pipelines.size() == 2, "a damaged record is rejected alone");

	// Templates (Phase B): their register blocks and targets survive, and a block of another size is rejected alone.
	{
		Precompile::Journal with;
		with.stages = {PixelRecord(9)};
		Precompile::TemplateRecord tmpl;
		tmpl.pixel       = with.stages[0].Key();
		tmpl.pipeline    = 77;
		tmpl.context     = std::vector<uint8_t>(1001, 7);
		tmpl.user_config = {1, 2, 3};
		tmpl.pixel_regs  = std::vector<uint8_t>(64, 9);
		tmpl.export_mapping[3] = 0x1b;
		tmpl.colors      = {{.slot = 1, .format = 37, .guest_format = 10, .samples = 1, .mapping = 0xe4}};
		tmpl.depth_bound = true;
		tmpl.depth_format = 130;
		tmpl.topology    = 3;
		with.templates   = {tmpl, tmpl};
		Check(Precompile::WriteJournal(path, device, with), "a journal with templates writes");
		auto back = Precompile::ReadJournal(path, device);
		Check(back.rejected == 0 && back.templates.size() == 2, "templates read back");
		back.Dedupe();
		const auto& t = back.templates.at(0);
		Check(back.templates.size() == 1 && t.context == tmpl.context && t.user_config == tmpl.user_config &&
		          t.export_mapping == tmpl.export_mapping && t.colors.size() == 1 && t.colors[0].format == 37 &&
		          t.depth_bound && t.depth_format == 130 && t.topology == 3 && t.Key() == tmpl.Key(),
		      "a template survives the round trip and dedupes");
		Precompile::Journal hints_in;
		hints_in.stages    = with.stages;
		hints_in.templates = with.templates;
		Check(Precompile::ExportHints(hints_in, {}, dir / "t2.kyhints").ok, "hints export with templates in the journal");
	}

	Precompile::Journal twice;
	twice.stages = {PixelRecord(1), PixelRecord(1), ComputeRecord(2)};
	twice.Dedupe();
	Check(twice.stages.size() == 2, "duplicates are dropped");
}

void TestHints(const std::filesystem::path& dir) {
	Precompile::Journal journal;
	journal.stages = {PixelRecord(5), ComputeRecord(6), PixelRecord(7)};
	journal.stages[2].code[52] = 0; // no footer marker: exported as a program, and not found in the game
	journal.pipelines = {Pipeline(journal.stages[1].Key(), journal.stages[0].Key()),
	                     Pipeline(journal.stages[1].Key(), journal.stages[2].Key())};
	const auto hints  = dir / "t.kyhints";
	const auto result = Precompile::ExportHints(journal, {.title_id = "PPSA03671", .app_version = "01.005"}, hints);
	Check(result.ok && result.stages == 3 && result.pipelines == 2 && result.no_footer == 1, "the export keeps blocks without a footer");
	{
		std::ifstream file(hints, std::ios::binary);
		const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		Check(bytes.find("barefoot") == std::string::npos, "hints hold no code");
	}

	// The game folder: the blocks somewhere inside other bytes, one across a file's end.
	const auto game = dir / "game";
	std::filesystem::create_directories(game / "d" / "0");
	{
		std::ofstream eboot(game / "eboot.bin", std::ios::binary);
		const std::string noise(1000, 'x');
		eboot << noise;
		eboot.write(reinterpret_cast<const char*>(journal.stages[0].code.data()), journal.stages[0].code.size() * 4);
		eboot << noise;
		std::ofstream material(game / "d" / "0" / "material", std::ios::binary);
		material << noise << noise;
		material.write(reinterpret_cast<const char*>(journal.stages[1].code.data()), journal.stages[1].code.size() * 4);
	}
	const auto imported = Precompile::ImportHints(hints, game, {.title_id = "PPSA03671"}, {});
	Check(imported.ok && imported.stages == 2 && imported.pipelines == 1 && imported.missing == 1,
	      "the import restores every block the game has");
	Check(imported.journal.stages[0].code == journal.stages[0].code, "the restored code is the game's");
	Check(imported.journal.stages[0].Key() == journal.stages[0].Key(), "the restored record keeps its key");
	Check(imported.journal.pipelines[0].Key() == journal.pipelines[0].Key(), "the restored pipeline keeps its key");
	Check(Precompile::ImportHints(hints, game, {.title_id = "PPSA00000"}, {}).reason == "other_game", "another game is refused");

	// adjust changes a record's host fields, and with them its key: its pipelines follow.
	const auto adjusted = Precompile::ImportHints(hints, game, {}, [](Precompile::StageRecord& record) {
		if (auto* compute = std::get_if<Libs::Graphics::ShaderComputeInputInfo>(&record.input)) compute->host_subgroup_size = 64;
	});
	Check(adjusted.journal.pipelines.size() == 1 && adjusted.journal.pipelines[0].vertex[0] != journal.stages[1].Key() &&
	          adjusted.journal.pipelines[0].vertex[0] == adjusted.journal.stages[1].Key(),
	      "an adjusted stage's pipelines take its new key");

	// The precompile on start imports the shipped hints again only when their stamp changes.
	const auto stamp = Precompile::HintsStamp(hints);
	Check(!stamp.empty() && stamp == Precompile::HintsStamp(hints), "a hints file has a stable stamp");
	Check(Precompile::HintsStamp(dir / "none.kyhints").empty(), "a missing hints file has no stamp");
	std::ofstream(hints, std::ios::app) << 'x';
	Check(Precompile::HintsStamp(hints) != stamp, "a changed hints file has another stamp");
	Check(Precompile::ShippedHints("").empty(), "no title, no shipped hints");
}

// The shader map's records hold programs, without the footer: the import finds each by the footer after its metadata.
void TestProgramHints(const std::filesystem::path& dir) {
	const std::array binaries {MakeBinary(11, 4264, 0, 374), MakeBinary(12, 1000, 52, 518), MakeBinary(13, 6560, 16, 16)};
	Precompile::Journal journal;
	for (uint32_t i = 0; i < binaries.size(); i++) {
		auto record        = i == 1 ? ComputeRecord(20 + i) : PixelRecord(20 + i);
		record.code        = binaries[i].program;
		record.shader_hash = Precompile::CodeHash(record.code, {});
		journal.stages.push_back(record);
	}
	journal.pipelines = {Pipeline(journal.stages[1].Key(), journal.stages[0].Key())};
	const auto hints  = dir / "p.kyhints";
	const auto result = Precompile::ExportHints(journal, {.title_id = "PPSA03671"}, hints);
	Check(result.ok && result.stages == 3 && result.pipelines == 1 && result.no_footer == 3, "programs are exported");
	const auto game = dir / "pgame";
	std::filesystem::create_directories(game);
	{
		std::ofstream eboot(game / "eboot.bin", std::ios::binary);
		const std::string noise(333, 'y');
		for (const auto& binary: binaries) {
			eboot << noise;
			eboot.write(reinterpret_cast<const char*>(binary.whole.data()), static_cast<std::streamsize>(binary.whole.size() * 4));
		}
	}
	const auto imported = Precompile::ImportHints(hints, game, {.title_id = "PPSA03671"}, {});
	Check(imported.ok && imported.stages == 3 && imported.pipelines == 1 && imported.missing == 0, "every program is found");
	for (uint32_t i = 0; i < binaries.size(); i++) {
		Check(imported.journal.stages[i].code == binaries[i].program, "the restored code is the program");
		Check(imported.journal.stages[i].Key() == journal.stages[i].Key(), "a restored program keeps its key");
	}
}

void TestWarm(const std::filesystem::path& dir) {
	const auto        path = dir / "t.warm";
	Precompile::WarmList warm;
	warm.Load(path, "sig", 100);
	warm.Mark(1, true);
	warm.Mark(2, false);
	Check(!warm.Contains(1, true), "a mark counts only after a save");
	warm.Commit(warm.TakeMarked(), true, 100);
	Check(warm.Contains(1, true) && warm.Contains(2, false) && !warm.Contains(2, true), "committed marks are warm");
	Precompile::WarmList again;
	again.Load(path, "sig", 150);
	Check(again.Contains(1, true) && again.Size() == 2, "the list survives a restart");
	again.Load(path, "sig", 50);
	Check(again.Size() == 0, "a cache that shrank drops the list");
	again.Load(path, "other driver", 150);
	Check(again.Size() == 0, "another driver drops the list");
	Precompile::WarmList failed;
	failed.Load(path, "sig", 150);
	failed.Mark(9, true);
	failed.Commit(failed.TakeMarked(), false, 0);
	Check(!failed.Contains(9, true) && failed.TakeMarked().size() == 1, "a failed save keeps the marks for the next");
}

// A compiler that translates and creates nothing: the pipelines asked for, and whether each may be handed over.
class RecordingCompiler: public Precompile::Compiler {
public:
	bool CompileStage(const Precompile::StageRecord& record, bool /*keep*/, Precompile::StageModule& out) override {
		out.spirv_hash = record.spirv_hash;
		out.input      = record.input;
		return true;
	}
	void DestroyStage(Precompile::StageModule& /*stage*/) override {}
	bool CreateGraphics(const Precompile::PipelineRecord& record, std::span<Precompile::StageModule* const> /*vertex*/,
	                    Precompile::StageModule* /*pixel*/, bool keep) override {
		std::lock_guard lock(mutex);
		created.emplace_back(record.Key(), keep);
		return true;
	}
	bool CreateCompute(uint64_t /*key*/, Precompile::StageModule& /*stage*/) override { return true; }

	std::mutex                                 mutex;
	std::vector<std::pair<uint64_t, bool>>     created;
};

// Holds the worker in its next translation or create until released, to catch a job in that state.
class LatchCompiler: public RecordingCompiler {
public:
	bool CompileStage(const Precompile::StageRecord& record, bool keep, Precompile::StageModule& out) override {
		Hold(hold_stage);
		return RecordingCompiler::CompileStage(record, keep, out);
	}
	bool CreateGraphics(const Precompile::PipelineRecord& record, std::span<Precompile::StageModule* const> vertex,
	                    Precompile::StageModule* pixel, bool keep) override {
		{
			std::lock_guard lock(latch);
			creating_key = record.Key();
		}
		Hold(hold_create);
		return RecordingCompiler::CreateGraphics(record, vertex, pixel, keep);
	}
	uint64_t WaitHeld() {
		std::unique_lock lock(latch);
		changed.wait(lock, [&] { return held; });
		return creating_key;
	}
	void Release() {
		std::lock_guard lock(latch);
		released = true;
		changed.notify_all();
	}

	bool hold_stage = false, hold_create = false;

private:
	void Hold(bool on) {
		if (!on) return;
		std::unique_lock lock(latch);
		held = true;
		changed.notify_all();
		changed.wait(lock, [&] { return released; });
	}
	std::mutex              latch;
	std::condition_variable changed;
	bool                    held = false, released = false;
	uint64_t                creating_key = 0;
};

Precompile::StageRecord VertexRecord(uint32_t push) {
	Precompile::StageRecord record;
	record.stage            = Libs::Graphics::ShaderType::Vertex;
	record.code             = MakeCode(7, 64);
	record.shader_hash      = Precompile::CodeHash(record.code, {});
	record.input            = Libs::Graphics::ShaderVertexInputInfo {};
	record.push_data_cursor = push; // another permutation of the same program
	return record;
}

void TestPairing() {
	Precompile::Journal journal;
	journal.stages = {VertexRecord(0), VertexRecord(10), PixelRecord(1), PixelRecord(2), PixelRecord(3)};
	const auto v1 = journal.stages[0].Key(), v2 = journal.stages[1].Key();
	const auto p1 = journal.stages[2].Key(), p2 = journal.stages[3].Key(), p3 = journal.stages[4].Key();
	auto       other_state = Pipeline(v1, p3);
	other_state.rendering.color_formats[0] = vk::Format::eR16G16B16A16Sfloat;
	journal.pipelines = {Pipeline(v1, p1), Pipeline(v2, p2), other_state};
	const auto run = [&](bool pair) {
		RecordingCompiler   compiler;
		Precompile::Options options {.threads = 1, .only = true};
		if (pair) options.pair_vertex_shaders = {journal.stages[0].shader_hash};
		Precompile::Precompiler precompiler(compiler, std::move(options), journal, nullptr);
		const auto              paired = precompiler.GetProgress().paired.load();
		precompiler.Start();
		precompiler.Wait();
		return std::make_pair(paired, compiler.created);
	};
	const auto [none, recorded] = run(false);
	Check(none == 0 && recorded.size() == 3, "without pairing, the recorded pipelines only");
	const auto [paired, created] = run(true);
	Check(paired == 2 && created.size() == 5, "each pixel record is paired with the other vertex record of its state");
	const auto made = [&](uint64_t vertex, uint64_t pixel, bool keep) {
		const auto key = Pipeline(vertex, pixel).Key();
		return std::ranges::find(created, std::make_pair(key, keep)) != created.end();
	};
	Check(made(v2, p1, false) && made(v1, p2, false), "the swapped pipelines are made, never to be handed over");
	Check(made(v1, p1, true) && made(v2, p2, true), "the recorded pipelines keep the hand-over");
	auto unpaired = other_state;
	unpaired.vertex[0] = v2;
	Check(std::ranges::find(created, std::make_pair(unpaired.Key(), false)) == created.end(),
	      "no vertex record is paired into a state it was not recorded in");
	Check(created.back().second == false && created[created.size() - 2].second == false, "paired jobs run last");
}

// A pixel record in the form the prediction takes: plain 2D images, the last buffer entry empty.
Precompile::StageRecord VolumeFormRecord(uint32_t seed) {
	auto record = PixelRecord(seed);
	using Image = decltype(record.specialization.images)::value_type;
	Image plain;
	plain.numeric_class = decltype(Image::numeric_class)::Float;
	plain.dimension     = decltype(Image::dimension)::Dim2D;
	Image volume        = plain;
	volume.dimension    = decltype(Image::dimension)::Dim3D;
	record.specialization.images = {plain, plain, volume};
	record.specialization.buffers = {{.packed_stride = 16}, {.packed_stride = 0, .zero_stride_oob = true, .scalar_empty = true}};
	return record;
}

void TestPrediction() {
	Precompile::Journal journal;
	journal.stages = {VertexRecord(0), PixelRecord(1), VolumeFormRecord(2), VolumeFormRecord(3)};
	const auto vertex = journal.stages[0].Key(), other = journal.stages[1].Key();
	const auto form = journal.stages[2].Key(), recorded_form = journal.stages[3].Key();
	const auto predict = [](Precompile::StageRecord record) {
		record.specialization.images[1].dimension     = record.specialization.images[2].dimension;
		record.specialization.buffers[1].packed_stride   = 4;
		record.specialization.buffers[1].zero_stride_oob = false;
		record.specialization.buffers[1].scalar_empty    = false;
		return record;
	};
	const auto guess = predict(journal.stages[2]);
	journal.stages.push_back(predict(journal.stages[3])); // a run met this one's prediction
	const auto met = journal.stages.back().Key();
	auto       other_state = Pipeline(vertex, form);
	other_state.rendering.color_formats[0] = vk::Format::eR16G16B16A16Sfloat;
	journal.pipelines = {Pipeline(vertex, other), Pipeline(vertex, form), other_state, Pipeline(vertex, recorded_form), Pipeline(vertex, met)};
	const auto run = [&](bool on) {
		RecordingCompiler   compiler;
		Precompile::Options options {.threads = 1, .only = true};
		options.predict_volume_pixels = on;
		Precompile::Precompiler precompiler(compiler, std::move(options), journal, nullptr);
		const auto              jobs = precompiler.GetProgress().paired.load();
		precompiler.Start();
		precompiler.Wait();
		Check(precompiler.GetProgress().mismatches.load() == 0, "a predicted stage has no recorded translation to differ from");
		return std::make_pair(jobs, compiler.created);
	};
	const auto [none, recorded] = run(false);
	Check(none == 0 && recorded.size() == 5, "without prediction, the recorded pipelines only");
	const auto [jobs, created] = run(true);
	Check(jobs == 2 && created.size() == 7, "each recorded pipeline of a predicted record once, none over a recorded one");
	const auto made = [&](const Precompile::PipelineRecord& record, bool keep) {
		return std::ranges::find(created, std::make_pair(record.Key(), keep)) != created.end();
	};
	auto guess_state  = other_state;
	guess_state.pixel = guess.Key();
	Check(made(Pipeline(vertex, guess.Key()), false) && made(guess_state, false), "the predicted pipelines are made, never to be handed over");
	Check(made(Pipeline(vertex, met), true), "a pipeline a run recorded stays recorded");
	Check(created.back().second == false && created[created.size() - 2].second == false, "predicted jobs run after the recorded ones");
}

// A compiler whose stages have the push data size given per shader, placed where the emitter places it.
class SizedCompiler: public RecordingCompiler {
public:
	bool CompileStage(const Precompile::StageRecord& record, bool keep, Precompile::StageModule& out) override {
		using Libs::Graphics::ShaderRecompiler::IR::PushData;
		RecordingCompiler::CompileStage(record, keep, out);
		auto& bindings                 = out.info.bindings;
		bindings.memory_offset_dword   = dwords.at(record.shader_hash);
		bindings.push_data_start_dword = PushData::StartFor(record.push_data_cursor, bindings.ShaderDataDwords());
		return true;
	}
	std::unordered_map<uint64_t, uint32_t> dwords;
};

// The buffer table grows a pixel stage's push data; the vertex stage the migration left at the old cursor is built
// where the draw puts it, and the pipeline is registered under the key the draw records.
void TestTableCursors() {
	using Libs::Graphics::ShaderRecompiler::IR::PushData;
	const auto table_pixel = [](uint32_t seed) {
		auto record                          = PixelRecord(seed);
		record.push_data_cursor              = PushData::MeshDrawDwordCount;
		record.specialization.buffer_table   = true;
		return record;
	};
	Precompile::Journal journal;
	// Recorded in the descriptor variant: the pixel stage's push data was 4 dwords, so the vertex stage's started at 9.
	journal.stages = {VertexRecord(PushData::MeshDrawDwordCount + 4), table_pixel(1), table_pixel(2), PixelRecord(3)};
	const auto& vertex = journal.stages[0];
	const auto  grown = journal.stages[1].Key(), overflow = journal.stages[2].Key(), plain = journal.stages[3].Key();
	journal.pipelines = {Pipeline(vertex.Key(), grown), Pipeline(vertex.Key(), overflow), Pipeline(vertex.Key(), plain)};
	SizedCompiler compiler;
	compiler.dwords = {{vertex.shader_hash, 3}, {journal.stages[1].shader_hash, 6}, {journal.stages[2].shader_hash, 30},
	                   {journal.stages[3].shader_hash, 4}};
	Precompile::Precompiler precompiler(compiler, Precompile::Options {.threads = 1, .only = true}, journal, nullptr);
	precompiler.Start();
	precompiler.Wait();
	const auto at = [](Precompile::StageRecord record, uint32_t cursor) {
		record.push_data_cursor = cursor;
		return record.Key();
	};
	// 6 dwords from 5: the vertex stage at 11.
	const auto moved = Pipeline(at(vertex, 11), grown).Key();
	// 30 dwords do not fit from 5: the pixel stage has no push data (keyed NoStart), the vertex stage starts at the draw's 0.
	const auto spilled = Pipeline(at(vertex, 0), at(journal.stages[2], PushData::NoStart)).Key();
	const auto made    = [&](uint64_t key) {
		return std::ranges::find(compiler.created, std::make_pair(key, true)) != compiler.created.end();
	};
	Check(compiler.created.size() == 3 && made(moved) && made(spilled), "a table job's stages take the draw's push cursors");
	Check(made(Pipeline(vertex.Key(), plain).Key()), "a job without a table stage stays as recorded");
	Check(precompiler.GetProgress().recursored.load() == 2, "both table jobs moved");
	Check(precompiler.BeforeCreate(moved) && precompiler.BeforeCreate(spilled), "the GPU thread finds them by its keys");
}

} // namespace

// What the GPU thread's own build of a pipeline does to a job in each state.
void TestBeforeCreate() {
	Precompile::Journal journal;
	journal.stages = {VertexRecord(0), PixelRecord(1), PixelRecord(2)};
	journal.pipelines = {Pipeline(journal.stages[0].Key(), journal.stages[1].Key()),
	                     Pipeline(journal.stages[0].Key(), journal.stages[2].Key())};
	const auto first = journal.pipelines[0].Key(), second = journal.pipelines[1].Key();
	Check(first != second, "two pipelines to tell apart");
	{
		LatchCompiler compiler;
		compiler.hold_stage = true;
		Precompile::Precompiler precompiler(compiler, Precompile::Options {.threads = 1, .only = true}, journal, nullptr);
		precompiler.Start();
		compiler.WaitHeld();
		// One job is translating, the other queued.
		Check(!precompiler.BeforeCreate(first) && !precompiler.BeforeCreate(second),
		      "the GPU thread creates what no worker is creating");
		Check(precompiler.GetProgress().dropped.load() == 1, "only the queued job is dropped");
		compiler.Release();
		precompiler.Wait();
		Check(compiler.created.empty(), "neither a dropped job nor one left translating creates its pipeline");
		Check(!precompiler.BeforeCreate(0x1234), "a pipeline with no job is the GPU thread's");
	}
	{
		LatchCompiler compiler;
		compiler.hold_create = true;
		Precompile::Precompiler precompiler(compiler, Precompile::Options {.threads = 1, .only = true}, journal, nullptr);
		precompiler.Start();
		const auto        creating = compiler.WaitHeld();
		std::atomic<bool> returned {false}, created {false};
		std::thread       gpu([&] {
            created  = precompiler.BeforeCreate(creating);
            returned = true;
        });
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		Check(!returned.load(), "the GPU thread waits for a pipeline the driver is creating");
		compiler.Release();
		gpu.join();
		Check(created.load(), "and then takes it");
		precompiler.Wait();
		Check(compiler.created.size() == 2, "both pipelines are created");
	}
}

// The hand-over map's bound: the oldest go first, keys taken since are skipped, the newest stay.
void TestTrimOldest() {
	std::unordered_map<uint64_t, int> map {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};
	std::deque<uint64_t>              order {1, 2, 3, 4, 5};
	map.erase(2); // taken by the GPU thread
	std::vector<int> destroyed;
	Precompile::TrimOldest(map, order, 2, [&](int& value) { destroyed.push_back(value); });
	Check(map.size() == 2 && map.contains(4) && map.contains(5), "the two newest stay");
	Check(destroyed == std::vector<int> {1, 3}, "the oldest are destroyed, a taken key is skipped");
	Check(order.size() == 2 && order.front() == 4, "the order keeps the kept keys");
	Precompile::TrimOldest(map, order, 2, [&](int&) { Check(false, "nothing past the bound"); });
}

void TestStop() {
	Precompile::Journal journal;
	journal.stages = {VertexRecord(0), PixelRecord(1), PixelRecord(2)};
	journal.pipelines = {Pipeline(journal.stages[0].Key(), journal.stages[1].Key()),
	                     Pipeline(journal.stages[0].Key(), journal.stages[2].Key())};
	LatchCompiler compiler;
	compiler.hold_stage = true;
	Precompile::Precompiler precompiler(compiler, Precompile::Options {.threads = 1, .only = true}, journal, nullptr);
	precompiler.Start();
	compiler.WaitHeld();
	std::thread stopper([&] { precompiler.Stop(); });
	while (!precompiler.StopRequested()) std::this_thread::yield(); // Stop has asked the worker to end
	compiler.Release();
	stopper.join();
	Check(compiler.created.size() <= 1 && !precompiler.GetProgress().done.load(),
	      "Stop leaves the jobs not started undone, and returns");
}

void TestSuspectStage() {
	Precompile::Journal journal;
	journal.stages = {VertexRecord(0), PixelRecord(1), PixelRecord(2)};
	journal.pipelines = {Pipeline(journal.stages[0].Key(), journal.stages[1].Key()),
	                     Pipeline(journal.stages[0].Key(), journal.stages[2].Key())};
	RecordingCompiler   compiler;
	Precompile::Options options {.threads = 1, .only = true};
	options.suspects = {journal.stages[1].Key()};
	Precompile::Precompiler precompiler(compiler, std::move(options), journal, nullptr);
	Check(precompiler.GetProgress().suspects.load() == 1, "a job with a suspect stage is dropped");
	precompiler.Start();
	precompiler.Wait();
	Check(compiler.created.size() == 1 && compiler.created[0].first == journal.pipelines[1].Key(),
	      "the job without it is still built");
}

void TestSuspectLines(const std::filesystem::path& dir) {
	const auto inflight = dir / "t.inflight", suspects = dir / "t.suspects";
	const auto crash    = [&](uint64_t origin, uint64_t key) {
		std::ofstream(inflight) << std::hex << origin << ' ' << key << '\n';
		return Precompile::LoadSuspects(inflight, suspects);
	};
	const auto here = Precompile::Origin(), other = here + 1;
	Check(crash(here, 0xa).empty(), "one crash blames nothing yet");
	Check(!std::filesystem::exists(inflight), "the in-flight file is taken");
	Check(crash(here, 0xb).empty(), "another key's first crash blames nothing either");
	Check(crash(here, 0xa) == std::unordered_set<uint64_t> {0xa}, "the same key at a second crash is a suspect");
	Check(crash(other, 0xc) == std::unordered_set<uint64_t> {0xa} && crash(other, 0xc) == std::unordered_set<uint64_t> {0xa},
	      "a key of another origin is no suspect, however often it was in flight");
	Check(Precompile::LoadSuspects(inflight, suspects) == std::unordered_set<uint64_t> {0xa},
	      "suspects stay for later runs");
}

// StageKey (the Ahead workers' lookup of a stage store record) is StageRecord::Key() for a record without callees.
void TestStageKey() {
	for (const auto& record: {PixelRecord(5), ComputeRecord(6)}) {
		std::vector<uint32_t> static_key;
		std::visit([&](const auto& info) { Libs::Graphics::BuildStageStaticKey(info, static_key); }, record.input);
		Check(Precompile::StageKey(record.stage, record.shader_hash, record.user_data_count, record.code.size(),
		                           record.back_code.size(), static_key, record.push_data_cursor,
		                           record.specialization) == record.Key(),
		      "StageKey is the record's key");
	}
	auto back = PixelRecord(7);
	back.back_code = MakeCode(8, 32);
	std::vector<uint32_t> static_key;
	std::visit([&](const auto& info) { Libs::Graphics::BuildStageStaticKey(info, static_key); }, back.input);
	Check(Precompile::StageKey(back.stage, back.shader_hash, back.user_data_count, back.code.size(), back.back_code.size(),
	                           static_key, back.push_data_cursor, back.specialization) == back.Key(),
	      "StageKey counts a back half");
	Check(Precompile::StageKey(back.stage, back.shader_hash, back.user_data_count, back.code.size(), 0, static_key,
	                           back.push_data_cursor, back.specialization) != back.Key(),
	      "StageKey of another back half is another key");
}

int main() {
	Config::Initialize(); // Origin() reads the options
	TestStageKey();
	const auto dir = std::filesystem::temp_directory_path() /
	                 ("kyty_shader_precompile_tests_" +
	                  std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	std::error_code error;
	std::filesystem::remove_all(dir, error);
	std::filesystem::create_directories(dir);
	TestJournal(dir);
	TestBufferTableMigration();
	TestHints(dir);
	TestProgramHints(dir);
	TestWarm(dir);
	TestPairing();
	TestPrediction();
	TestTableCursors();
	TestBeforeCreate();
	TestTrimOldest();
	TestStop();
	TestSuspectStage();
	TestSuspectLines(dir);
	std::filesystem::remove_all(dir, error);
	std::printf("ShaderPrecompileTests: ok\n");
	return 0;
}
