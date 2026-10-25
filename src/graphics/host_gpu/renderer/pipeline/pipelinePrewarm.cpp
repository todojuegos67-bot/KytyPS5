#include "graphics/host_gpu/renderer/pipeline/pipelinePrewarm.h"


#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <xxhash.h>

namespace Libs::Graphics::Prewarm {

namespace {

constexpr char     Magic[8]  = {'K', 'P', 'R', 'E', 'W', 'R', 'M', '1'};
constexpr uint32_t MaxWords  = 64u * 1024u * 1024u;
constexpr uint32_t MaxBinds  = 4096u;

enum class State : uint8_t { Queued, Building, Ready, Done };

struct Entry {
	State        state    = State::Queued;
	vk::Pipeline pipeline = nullptr;
	size_t       order    = 0;
};

struct Context {
	std::mutex                                              mutex;
	std::condition_variable                                 cv;
	std::unordered_map<uint64_t, std::vector<uint32_t>>     spirv_of_module;
	std::unordered_set<uint64_t>                            known; // keys in the file
	std::unordered_map<uint64_t, Entry>                     entries;
	std::vector<std::shared_ptr<Record>>                    queue;
	size_t                                                  next = 0;
	std::vector<std::thread>                                threads;
	std::FILE*                                              out = nullptr;
	std::filesystem::path                                   path;
	bool                                                    started = false;
	bool                                                    stop    = false;
	size_t                                                  built   = 0;
	size_t                                                  failed  = 0;
	uint64_t                                                served  = 0;
	uint64_t                                                missed  = 0;
	std::chrono::steady_clock::time_point                   start_time;
};

Context& Ctx() {
	static auto* ctx = new Context; // never destroyed: worker threads may outlive static destruction
	return *ctx;
}

uint64_t ModuleId(vk::ShaderModule module) {
	return reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(module));
}

uint64_t KeyOf(const Record& r) {
	XXH3_state_t* st = XXH3_createState();
	XXH3_64bits_reset(st);
	XXH3_64bits_update(st, r.spirv.data(), r.spirv.size() * sizeof(uint32_t));
	XXH3_64bits_update(st, r.bindings.data(), r.bindings.size() * sizeof(Binding));
	XXH3_64bits_update(st, &r.required_subgroup_size, sizeof(r.required_subgroup_size));
	XXH3_64bits_update(st, &r.flags, sizeof(r.flags));
	const uint64_t key = XXH3_64bits_digest(st);
	XXH3_freeState(st);
	return key == 0 ? 1 : key;
}

template <typename T>
bool ReadPod(std::FILE* f, T& v) {
	return std::fread(&v, sizeof(T), 1, f) == 1;
}

bool ReadRecord(std::FILE* f, Record& r) {
	uint32_t nb = 0;
	uint32_t nw = 0;
	if (!ReadPod(f, r.key) || !ReadPod(f, r.required_subgroup_size) || !ReadPod(f, r.flags) ||
	    !ReadPod(f, nb) || nb > MaxBinds) {
		return false;
	}
	r.bindings.resize(nb);
	if (nb != 0 && std::fread(r.bindings.data(), sizeof(Binding), nb, f) != nb) {
		return false;
	}
	if (!ReadPod(f, nw) || nw == 0 || nw > MaxWords) {
		return false;
	}
	r.spirv.resize(nw);
	return std::fread(r.spirv.data(), sizeof(uint32_t), nw, f) == nw;
}

void WriteRecord(std::FILE* f, const Record& r) {
	const auto nb = static_cast<uint32_t>(r.bindings.size());
	const auto nw = static_cast<uint32_t>(r.spirv.size());
	std::fwrite(&r.key, sizeof(r.key), 1, f);
	std::fwrite(&r.required_subgroup_size, sizeof(uint32_t), 1, f);
	std::fwrite(&r.flags, sizeof(uint32_t), 1, f);
	std::fwrite(&nb, sizeof(nb), 1, f);
	if (nb != 0) {
		std::fwrite(r.bindings.data(), sizeof(Binding), nb, f);
	}
	std::fwrite(&nw, sizeof(nw), 1, f);
	std::fwrite(r.spirv.data(), sizeof(uint32_t), nw, f);
	std::fflush(f);
}

void Report(Context& ctx, const char* when) {
	std::printf("Prewarm: %s: %zu of %zu recorded compute pipelines built (%zu failed), gameplay: %llu served, %llu missed\n",
	            when, ctx.built, ctx.queue.size(), ctx.failed, static_cast<unsigned long long>(ctx.served),
	            static_cast<unsigned long long>(ctx.missed));
	std::fflush(stdout);
}

void Worker(GraphicContext* graphics, vk::PipelineCache driver_cache) {
	auto& ctx = Ctx();
	for (;;) {
		std::shared_ptr<Record> rec;
		{
			std::lock_guard lock(ctx.mutex);
			if (ctx.stop) {
				return;
			}
			while (ctx.next < ctx.queue.size()) {
				auto& cand = ctx.queue[ctx.next++];
				auto  it   = ctx.entries.find(cand->key);
				if (it != ctx.entries.end() && it->second.state == State::Queued) {
					it->second.state = State::Building;
					rec              = cand;
					break;
				}
			}
			if (rec == nullptr) {
				return;
			}
		}
		vk::Pipeline pipeline = BuildComputePipeline(*graphics, *rec, driver_cache);
		rec->spirv.clear();
		rec->spirv.shrink_to_fit();
		{
			std::lock_guard lock(ctx.mutex);
			auto& e = ctx.entries[rec->key];
			if (pipeline != nullptr) {
				e.state    = State::Ready;
				e.pipeline = pipeline;
				ctx.built++;
			} else {
				e.state = State::Done;
				ctx.failed++;
			}
			const size_t total = ctx.queue.size();
			const size_t n     = ctx.built + ctx.failed;
			const size_t step  = std::max<size_t>(1, total / 20);
			if (n % step == 0 || n == total) {
				const auto s = std::chrono::duration_cast<std::chrono::seconds>(
				                   std::chrono::steady_clock::now() - ctx.start_time)
				                   .count();
				std::printf("Prewarm: %zu/%zu compute pipelines (%lld s)\n", n, total,
				            static_cast<long long>(s));
				std::fflush(stdout);
				if (n == total) {
					Report(ctx, "replay finished");
				}
			}
		}
		ctx.cv.notify_all();
	}
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_PREWARM");
		return v == nullptr || std::strcmp(v, "0") != 0;
	}();
	return enabled;
}

void NoteSpirv(vk::ShaderModule module, std::span<const uint32_t> spirv) {
	if (!Enabled() || module == nullptr || spirv.empty()) {
		return;
	}
	auto&           ctx = Ctx();
	std::lock_guard lock(ctx.mutex);
	ctx.spirv_of_module[ModuleId(module)].assign(spirv.begin(), spirv.end());
}

uint64_t Register(vk::ShaderModule module, std::span<const vk::DescriptorSetLayoutBinding> bindings,
                  uint32_t required_subgroup_size, uint32_t flags) {
	if (!Enabled() || module == nullptr) {
		return 0;
	}
	Record r;
	r.required_subgroup_size = required_subgroup_size;
	r.flags                  = flags;
	for (const auto& b: bindings) {
		if (b.pImmutableSamplers != nullptr) {
			return 0;
		}
		r.bindings.push_back({b.binding, static_cast<uint32_t>(b.descriptorType), b.descriptorCount,
		                      static_cast<uint32_t>(b.stageFlags)});
	}
	auto&           ctx = Ctx();
	std::lock_guard lock(ctx.mutex);
	auto            it = ctx.spirv_of_module.find(ModuleId(module));
	if (it == ctx.spirv_of_module.end()) {
		return 0;
	}
	r.spirv = std::move(it->second);
	ctx.spirv_of_module.erase(it);
	r.key = KeyOf(r);
	if (ctx.out != nullptr && ctx.known.insert(r.key).second) {
		WriteRecord(ctx.out, r);
	}
	return r.key;
}

void EnsureStarted(GraphicContext& graphics, const std::string& title_id, vk::PipelineCache driver_cache) {
	auto& ctx = Ctx();
	{
		std::lock_guard lock(ctx.mutex);
		if (ctx.started || !Enabled() || title_id.empty()) {
			ctx.started = true;
			return;
		}
		ctx.started    = true;
		ctx.start_time = std::chrono::steady_clock::now();
		std::error_code ec;
		std::filesystem::create_directories("_PipelineCache", ec);
		ctx.path = std::filesystem::path("_PipelineCache") / (title_id + ".prewarm");

		uint64_t valid_end = 0;
		if (std::FILE* f = std::fopen(ctx.path.string().c_str(), "rb"); f != nullptr) {
			char magic[8] = {};
			if (std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, Magic, 8) == 0) {
				valid_end = 8;
				for (;;) {
					auto rec = std::make_shared<Record>();
					if (!ReadRecord(f, *rec)) {
						break;
					}
					valid_end = static_cast<uint64_t>(std::ftell(f));
					if (KeyOf(*rec) != rec->key || !ctx.known.insert(rec->key).second) {
						continue;
					}
					ctx.entries[rec->key] = {State::Queued, nullptr, ctx.queue.size()};
					ctx.queue.push_back(std::move(rec));
				}
			}
			std::fclose(f);
		}
		if (valid_end == 0) {
			ctx.out = std::fopen(ctx.path.string().c_str(), "wb");
			if (ctx.out != nullptr) {
				std::fwrite(Magic, 1, 8, ctx.out);
				std::fflush(ctx.out);
			}
		} else {
			std::filesystem::resize_file(ctx.path, valid_end, ec); // drop a torn last record
			ctx.out = std::fopen(ctx.path.string().c_str(), "ab");
		}
		if (ctx.out == nullptr) {
			std::printf("Prewarm: cannot write %s; recording off\n", ctx.path.string().c_str());
		}
		unsigned threads = std::max(1u, std::thread::hardware_concurrency() > 2
		                                    ? std::thread::hardware_concurrency() - 2u
		                                    : 1u);
		if (const char* v = std::getenv("KYTY_PREWARM_THREADS"); v != nullptr && *v != '\0') {
			threads = std::clamp(static_cast<unsigned>(std::strtoul(v, nullptr, 10)), 1u, 64u);
		}
		threads = std::min<unsigned>(threads, static_cast<unsigned>(std::max<size_t>(1, ctx.queue.size())));
		std::printf("Prewarm: %s: %zu recorded compute pipelines, building them on %u threads\n",
		            ctx.path.string().c_str(), ctx.queue.size(), ctx.queue.empty() ? 0u : threads);
		std::fflush(stdout);
		if (!ctx.queue.empty()) {
			for (unsigned i = 0; i < threads; i++) {
				ctx.threads.emplace_back(Worker, &graphics, driver_cache);
			}
		}
	}
	static const bool wait = [] {
		const char* v = std::getenv("KYTY_PREWARM_WAIT");
		return v != nullptr && std::strcmp(v, "1") == 0;
	}();
	if (wait) {
		std::printf("Prewarm: waiting for the replay (KYTY_PREWARM_WAIT=1)\n");
		std::unique_lock lock(ctx.mutex);
		ctx.cv.wait(lock, [&] { return ctx.built + ctx.failed >= ctx.queue.size(); });
	}
}

vk::Pipeline Take(uint64_t key) {
	if (key == 0) {
		return nullptr;
	}
	auto&            ctx = Ctx();
	std::unique_lock lock(ctx.mutex);
	auto             it = ctx.entries.find(key);
	vk::Pipeline     result = nullptr;
	if (it != ctx.entries.end()) {
		if (it->second.state == State::Building) {
			ctx.cv.wait(lock, [&] { return it->second.state != State::Building; });
		}
		if (it->second.state == State::Ready) {
			result = it->second.pipeline;
		}
		it->second.state    = State::Done;
		it->second.pipeline = nullptr;
	}
	if (result != nullptr) {
		ctx.served++;
	} else {
		ctx.missed++;
	}
	if ((ctx.served + ctx.missed) % 200 == 0) {
		Report(ctx, "status");
	}
	return result;
}

void Shutdown() {
	auto& ctx = Ctx();
	{
		std::lock_guard lock(ctx.mutex);
		if (!ctx.started) {
			return;
		}
		ctx.stop = true;
		Report(ctx, "exit");
	}
	for (auto& t: ctx.threads) {
		t.join();
	}
	std::lock_guard lock(ctx.mutex);
	ctx.threads.clear();
	if (ctx.out != nullptr) {
		std::fclose(ctx.out);
		ctx.out = nullptr;
	}
}

} // namespace Libs::Graphics::Prewarm
