#include "graphics/host_gpu/renderer/pipeline/translationCache.h"

#include "graphics/shaderSessionStats.h"
#include "kytyGitVersion.h"

#include <xxhash.h>

#include <algorithm>
#include <atomic>
#include <type_traits>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace Libs::Graphics::TranslationCache {

namespace {

using ShaderRecompiler::IR::CompiledShaderInfo;
using ShaderRecompiler::IR::ShaderInfo;

constexpr char     FileMagic[8]  = {'K', 'Y', 'X', 'L', 'A', 'T', '0', '1'};
constexpr uint32_t FormatVersion = 2;

// What the cache is valid for: the translator sources (KYTY_TRANSLATOR_HASH, see
// generate_version.cmake) rather than the git revision, so builds that leave the translator alone
// keep their translations. KYTY_XLAT_BY_REVISION=1 restores the old per-revision behaviour.
std::string CacheIdentity() {
	const char* v = std::getenv("KYTY_XLAT_BY_REVISION");
	if (v != nullptr && v[0] == '1') return std::string("rev:") + KYTY_GIT_REVISION;
	return std::string("xlat:") + KYTY_TRANSLATOR_HASH;
}
constexpr uint32_t RecordMagic   = 0x43455258u; // "XREC"

// ---- serialization -------------------------------------------------------------------------

struct Writer {
	std::vector<uint8_t> out;
	void Bytes(const void* p, size_t n) {
		const auto* b = static_cast<const uint8_t*>(p);
		out.insert(out.end(), b, b + n);
	}
	template <typename T>
	void Pod(const T& v) {
		static_assert(std::is_trivially_copyable_v<T>);
		Bytes(&v, sizeof(v));
	}
	void U32s(const std::vector<uint32_t>& v) {
		Pod(static_cast<uint32_t>(v.size()));
		Bytes(v.data(), v.size() * sizeof(uint32_t));
	}
	void Str(const std::string& s) {
		Pod(static_cast<uint32_t>(s.size()));
		Bytes(s.data(), s.size());
	}
};

struct Reader {
	const uint8_t* p;
	const uint8_t* end;
	bool           ok = true;
	void Bytes(void* dst, size_t n) {
		if (!ok || static_cast<size_t>(end - p) < n) {
			ok = false;
			return;
		}
		std::memcpy(dst, p, n);
		p += n;
	}
	template <typename T>
	void Pod(T& v) {
		static_assert(std::is_trivially_copyable_v<T>);
		Bytes(&v, sizeof(v));
	}
	uint32_t Count(size_t element) {
		uint32_t n = 0;
		Pod(n);
		if (ok && static_cast<size_t>(end - p) / std::max<size_t>(element, 1) < n) ok = false;
		return ok ? n : 0;
	}
	void U32s(std::vector<uint32_t>& v) {
		const auto n = Count(sizeof(uint32_t));
		v.resize(n);
		Bytes(v.data(), n * sizeof(uint32_t));
	}
	void Str(std::string& s) {
		const auto n = Count(1);
		s.resize(n);
		Bytes(s.data(), n);
	}
};

// Every field, by name. A field forgotten here is caught by the round-trip comparison in
// Store (ShaderInfo and Bindings compare all members with a defaulted operator==).
template <typename IO, typename Info>
void Fields(IO& io, Info& c) {
	io.Pod(c.stage);
	io.Pod(c.shader_hash);
	io.Pod(c.wave_size);
	io.Pod(c.user_data_base);
	io.Pod(c.user_data_count);
	io.Pod(c.scratch_dwords);
	io.Pod(c.param_export_mask);
	io.Pod(c.has_address_writes);
	io.Pod(c.bindings.descriptor_counts);
	io.Pod(c.bindings.descriptor_mask);
	io.Pod(c.bindings.push_data_start_dword);
	io.Pod(c.bindings.dispatch_thread_dword);
	io.Pod(c.bindings.memory_offset_dword);
	auto& i = c.info;
	io.Pod(i.vertex_fetch_components);
	io.Pod(i.vertex_offset_sgpr);
	io.Pod(i.instance_offset_sgpr);
	io.Pod(i.has_bitwise_xor);
	io.Pod(i.uses_dma);
	io.Pod(i.uses_swizzle);
	io.Pod(i.uses_lds);
	io.Pod(i.uses_gds);
	io.Pod(i.uses_flattened_srt);
	io.Pod(i.uses_dispatch_threads);
	io.Pod(i.bvh);
	io.Pod(i.subgroup_ballot);
	io.Pod(i.subgroup_vote);
	io.Pod(i.subgroup_barrier);
	io.Pod(i.subgroup_shuffle);
	io.Pod(i.subgroup_arithmetic);
	io.Pod(i.subgroup_local_invocation_id);
	io.Pod(i.compute_derivatives);
	io.Pod(i.image_gather_extended);
	io.Pod(i.function_lds);
	io.Pod(i.function_scratch);
	io.Pod(i.pixel_valid_mask);
	io.Pod(i.buffer_int64_atomics);
	io.Pod(i.buffer_u8);
	io.Pod(i.buffer_u16);
	io.Pod(i.shared_int64_atomics);
	io.Pod(i.coherent_buffers);
	io.Pod(i.float64);
	io.U32s(i.user_data_registers);
}

void Write(Writer& w, const CompiledShaderInfo& c) {
	Fields(w, c);
	const auto& i = c.info;
	w.Pod(static_cast<uint32_t>(i.buffers.size()));
	for (const auto& b: i.buffers) {
		w.Pod(b.source); w.Pod(b.descriptor_index); w.Pod(b.first_use_pc); w.Pod(b.max_byte_extent);
		w.Pod(b.packed_stride); w.Pod(b.descriptor_format); w.Pod(b.descriptor_swizzle); w.Pod(b.image_alias);
		w.Pod(b.read); w.Pod(b.written); w.Pod(b.atomic); w.Pod(b.formatted); w.Pod(b.scalar); w.Pod(b.live);
		w.Pod(b.indirect_root); w.Pod(b.indirect_mapping_offset); w.Pod(b.indirect_search_iterations);
		w.Pod(b.runtime_stride_offset); w.U32s(b.indirect_resources);
	}
	w.Pod(static_cast<uint32_t>(i.images.size()));
	for (const auto& m: i.images) {
		w.Pod(m.source); w.Pod(m.descriptor_index); w.Pod(m.first_use_pc); w.Pod(m.resource_class);
		w.Pod(m.numeric_class); w.Pod(m.dimension); w.Pod(m.mip_mode); w.Pod(m.mip_count);
		w.Pod(m.conversion_format); w.Pod(m.shader_swizzle); w.Pod(m.read); w.Pod(m.written);
		w.Pod(m.atomic); w.Pod(m.atomic64); w.Pod(m.unfiltered_access_only); w.Pod(m.depth_compare);
		w.Pod(m.cube); w.Pod(m.r128); w.Pod(m.indirect_root); w.Pod(m.indirect_mapping_offset);
		w.Pod(m.indirect_search_iterations); w.U32s(m.indirect_resources);
	}
	w.Pod(static_cast<uint32_t>(i.samplers.size()));
	for (const auto& s: i.samplers) {
		w.Pod(s.source); w.Pod(s.first_use_pc); w.Pod(s.snapshot_index); w.Pod(s.indirect_root);
		w.Pod(s.indirect_mapping_offset); w.Pod(s.indirect_search_iterations); w.U32s(s.indirect_resources);
		w.Pod(s.force_point_filtering); w.Pod(s.depth_compare); w.Pod(s.integer_border); w.Pod(s.gather_lod);
	}
	w.Pod(static_cast<uint32_t>(i.sampled_pairs.size()));
	for (const auto& s: i.sampled_pairs) {
		w.Pod(s.image); w.Pod(s.sampler); w.Pod(s.first_use_pc);
	}
	w.Pod(static_cast<uint32_t>(i.inputs.size()));
	for (const auto& s: i.inputs) {
		w.Pod(s.kind); w.Pod(s.location); w.Pod(s.component_count); w.Str(s.debug_name); w.Pod(s.per_vertex);
	}
	w.Pod(static_cast<uint32_t>(i.outputs.size()));
	for (const auto& s: i.outputs) {
		w.Pod(s.kind); w.Pod(s.index); w.Pod(s.location); w.Str(s.debug_name);
	}
}

bool Read(Reader& r, CompiledShaderInfo& c) {
	Fields(r, c);
	auto& i = c.info;
	i.buffers.resize(r.Count(8));
	for (auto& b: i.buffers) {
		r.Pod(b.source); r.Pod(b.descriptor_index); r.Pod(b.first_use_pc); r.Pod(b.max_byte_extent);
		r.Pod(b.packed_stride); r.Pod(b.descriptor_format); r.Pod(b.descriptor_swizzle); r.Pod(b.image_alias);
		r.Pod(b.read); r.Pod(b.written); r.Pod(b.atomic); r.Pod(b.formatted); r.Pod(b.scalar); r.Pod(b.live);
		r.Pod(b.indirect_root); r.Pod(b.indirect_mapping_offset); r.Pod(b.indirect_search_iterations);
		r.Pod(b.runtime_stride_offset); r.U32s(b.indirect_resources);
	}
	i.images.resize(r.Count(8));
	for (auto& m: i.images) {
		r.Pod(m.source); r.Pod(m.descriptor_index); r.Pod(m.first_use_pc); r.Pod(m.resource_class);
		r.Pod(m.numeric_class); r.Pod(m.dimension); r.Pod(m.mip_mode); r.Pod(m.mip_count);
		r.Pod(m.conversion_format); r.Pod(m.shader_swizzle); r.Pod(m.read); r.Pod(m.written);
		r.Pod(m.atomic); r.Pod(m.atomic64); r.Pod(m.unfiltered_access_only); r.Pod(m.depth_compare);
		r.Pod(m.cube); r.Pod(m.r128); r.Pod(m.indirect_root); r.Pod(m.indirect_mapping_offset);
		r.Pod(m.indirect_search_iterations); r.U32s(m.indirect_resources);
	}
	i.samplers.resize(r.Count(8));
	for (auto& s: i.samplers) {
		r.Pod(s.source); r.Pod(s.first_use_pc); r.Pod(s.snapshot_index); r.Pod(s.indirect_root);
		r.Pod(s.indirect_mapping_offset); r.Pod(s.indirect_search_iterations); r.U32s(s.indirect_resources);
		r.Pod(s.force_point_filtering); r.Pod(s.depth_compare); r.Pod(s.integer_border); r.Pod(s.gather_lod);
	}
	i.sampled_pairs.resize(r.Count(8));
	for (auto& s: i.sampled_pairs) {
		r.Pod(s.image); r.Pod(s.sampler); r.Pod(s.first_use_pc);
	}
	i.inputs.resize(r.Count(8));
	for (auto& s: i.inputs) {
		r.Pod(s.kind); r.Pod(s.location); r.Pod(s.component_count); r.Str(s.debug_name); r.Pod(s.per_vertex);
	}
	i.outputs.resize(r.Count(8));
	for (auto& s: i.outputs) {
		r.Pod(s.kind); r.Pod(s.index); r.Pod(s.location); r.Str(s.debug_name);
	}
	return r.ok && r.p == r.end;
}

bool SameInfo(const CompiledShaderInfo& a, const CompiledShaderInfo& b) {
	return a.stage == b.stage && a.shader_hash == b.shader_hash && a.wave_size == b.wave_size &&
	       a.user_data_base == b.user_data_base && a.user_data_count == b.user_data_count &&
	       a.scratch_dwords == b.scratch_dwords && a.param_export_mask == b.param_export_mask &&
	       a.has_address_writes == b.has_address_writes && a.info == b.info && a.bindings == b.bindings;
}

// ---- file --------------------------------------------------------------------------------

struct KeyHash {
	size_t operator()(const Key& k) const noexcept { return static_cast<size_t>(k.a ^ (k.b * 0x9e3779b97f4a7c15ull)); }
};

struct Location {
	uint64_t offset = 0; // record start
	uint32_t spirv_words = 0;
	uint32_t info_bytes  = 0;
};

constexpr size_t RecordHeaderBytes = 4 + 16 + 4 + 4;

struct Cache {
	std::mutex                                   mutex;
	std::condition_variable                      cv;
	bool                                         loaded  = false;
	bool                                         usable  = false;
	std::filesystem::path                        path;
	std::ifstream                                in;
	std::unordered_map<Key, Location, KeyHash>   index;
	std::deque<std::vector<uint8_t>>             queue;   // serialized records awaiting append
	std::deque<std::pair<Key, Location>>         queued_locations;
	uint64_t                                     file_size = 0;
	bool                                         stop = false;
	std::thread                                  thread;
	std::atomic<uint64_t>                        hits {0}, misses {0}, stores {0}, rejects {0};

	~Cache() {
		{
			std::lock_guard lock(mutex);
			stop = true;
		}
		cv.notify_all();
		if (thread.joinable()) thread.join();
	}
};

Cache& Instance() {
	static Cache cache;
	return cache;
}

std::string Header() {
	std::string h(FileMagic, sizeof(FileMagic));
	const std::string revision = CacheIdentity();
	h.append(reinterpret_cast<const char*>(&FormatVersion), sizeof(FormatVersion));
	const auto len = static_cast<uint32_t>(revision.size());
	h.append(reinterpret_cast<const char*>(&len), sizeof(len));
	h += revision;
	return h;
}

// Background: load the index, then append queued records until stop.
void Run(Cache& c) {
	const auto header = Header();
	const auto begin  = std::chrono::steady_clock::now();
	std::error_code ec;
	uint64_t        valid = 0;
	size_t          records = 0;
	{
		std::ifstream f(c.path, std::ios::binary);
		std::string   got(header.size(), '\0');
		if (f && f.read(got.data(), static_cast<std::streamsize>(got.size())) && got == header) {
			valid = header.size();
			const auto size = std::filesystem::file_size(c.path, ec);
			for (;;) {
				uint8_t h[RecordHeaderBytes];
				if (!f.read(reinterpret_cast<char*>(h), sizeof(h))) break;
				uint32_t magic = 0; Key key; Location loc {.offset = valid};
				std::memcpy(&magic, h, 4);
				std::memcpy(&key.a, h + 4, 8);
				std::memcpy(&key.b, h + 12, 8);
				std::memcpy(&loc.spirv_words, h + 20, 4);
				std::memcpy(&loc.info_bytes, h + 24, 4);
				const uint64_t total = RecordHeaderBytes + uint64_t {loc.spirv_words} * 4 + loc.info_bytes + 8;
				if (magic != RecordMagic || ec || valid + total > size) break;
				f.seekg(static_cast<std::streamoff>(valid + total));
				if (!f) break;
				c.index[key] = loc;
				valid += total;
				++records;
			}
		}
	}
	bool fresh = valid == 0;
	if (!fresh && std::filesystem::file_size(c.path, ec) != valid) {
		std::filesystem::resize_file(c.path, valid, ec); // drop a torn tail record
		if (ec) { fresh = true; c.index.clear(); records = 0; }
	}
	if (fresh) {
		std::filesystem::create_directories(c.path.parent_path(), ec);
		std::ofstream out(c.path, std::ios::binary | std::ios::trunc);
		out.write(header.data(), static_cast<std::streamsize>(header.size()));
		valid = out ? header.size() : 0;
	}
	{
		std::lock_guard lock(c.mutex);
		c.file_size = valid;
		c.in.open(c.path, std::ios::binary);
		c.usable = valid != 0 && c.in.is_open();
		c.loaded = true;
	}
	c.cv.notify_all();
	ShaderSessionStats::g_xlat_loaded.store(records, std::memory_order_relaxed);
	std::printf("TranslationCache: %s, %zu translations loaded in %.0f ms%s\n",
	            c.path.string().c_str(), records,
	            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(),
	            fresh ? " (new file: missing or made by another build)" : "");
	std::fflush(stdout);
	if (!c.usable) return;

	std::ofstream out(c.path, std::ios::binary | std::ios::app);
	for (;;) {
		std::vector<uint8_t>     record;
		std::pair<Key, Location> where;
		{
			std::unique_lock lock(c.mutex);
			c.cv.wait(lock, [&] { return c.stop || !c.queue.empty(); });
			if (c.queue.empty()) return;
			record = std::move(c.queue.front());
			c.queue.pop_front();
			where = c.queued_locations.front();
			c.queued_locations.pop_front();
		}
		out.write(reinterpret_cast<const char*>(record.data()), static_cast<std::streamsize>(record.size()));
		out.flush();
		if (!out) return;
		std::lock_guard lock(c.mutex);
		where.second.offset = c.file_size;
		c.file_size += record.size();
		c.index[where.first] = where.second;
	}
}

} // namespace

KeyBuilder::KeyBuilder() {
	auto* s = XXH3_createState();
	XXH3_128bits_reset_withSeed(s, 0x4b595459584c4154ull);
	m_state = s;
	const std::string revision = CacheIdentity();
	Add(revision.data(), revision.size());
	// Recompiler switches read from the environment change its output.
	for (const char* name: {"KYTY_FATAL_UNSUPPORTED_SHADERS", "KYTY_ROUND_SEARCH", "KYTY_RUNTIME_STRIDE",
	                        "KYTY_STRICT_INDIRECT_IMAGES", "KYTY_DIRECT_RUNTIME_STRIDE"}) {
		const char* v = std::getenv(name);
		const std::string value = v != nullptr ? std::string("=") + v : std::string("-");
		Add(value.data(), value.size());
	}
}

KeyBuilder::~KeyBuilder() {
	XXH3_freeState(static_cast<XXH3_state_t*>(m_state));
}

void KeyBuilder::Add(const void* data, size_t bytes) {
	XXH3_128bits_update(static_cast<XXH3_state_t*>(m_state), data, bytes);
}

Key KeyBuilder::Finish() {
	const auto h = XXH3_128bits_digest(static_cast<XXH3_state_t*>(m_state));
	return {h.low64, h.high64};
}

bool Enabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TRANSLATION_CACHE");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

void Open(const std::filesystem::path& path) {
	if (!Enabled()) return;
	auto& c = Instance();
	std::lock_guard lock(c.mutex);
	if (c.thread.joinable()) return;
	c.path   = path;
	c.thread = std::thread([&c] { Run(c); });
}

std::optional<Entry> Find(const Key& key) {
	auto& c = Instance();
	std::unique_lock lock(c.mutex);
	if (!c.thread.joinable()) return std::nullopt;
	c.cv.wait(lock, [&] { return c.loaded; });
	if (!c.usable) return std::nullopt;
	const auto found = c.index.find(key);
	if (found == c.index.end()) {
		c.misses.fetch_add(1, std::memory_order_relaxed);
		static std::atomic<uint32_t> logged_miss {0};
		if (logged_miss.fetch_add(1) < 3) {
			std::printf("TranslationCache: lookup miss key=%016llx%016llx (index has %zu)\n",
			            static_cast<unsigned long long>(key.a), static_cast<unsigned long long>(key.b),
			            c.index.size());
			std::fflush(stdout);
		}
		return std::nullopt;
	}
	const auto loc = found->second;
	const uint64_t payload = uint64_t {loc.spirv_words} * 4 + loc.info_bytes;
	std::vector<uint8_t> bytes(RecordHeaderBytes + payload + 8);
	c.in.clear();
	c.in.seekg(static_cast<std::streamoff>(loc.offset));
	c.in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	const bool read_ok = static_cast<bool>(c.in);
	lock.unlock();
	uint64_t checksum = 0;
	std::memcpy(&checksum, bytes.data() + bytes.size() - 8, 8);
	Key stored;
	if (read_ok) {
		std::memcpy(&stored.a, bytes.data() + 4, 8);
		std::memcpy(&stored.b, bytes.data() + 12, 8);
	}
	if (!read_ok || stored != key || XXH3_64bits(bytes.data(), bytes.size() - 8) != checksum) {
		c.rejects.fetch_add(1, std::memory_order_relaxed);
		return std::nullopt;
	}
	Entry entry;
	entry.spirv.resize(loc.spirv_words);
	std::memcpy(entry.spirv.data(), bytes.data() + RecordHeaderBytes, uint64_t {loc.spirv_words} * 4);
	const auto* info = bytes.data() + RecordHeaderBytes + uint64_t {loc.spirv_words} * 4;
	Reader r {info, info + loc.info_bytes};
	if (entry.spirv.empty() || !Read(r, entry.info)) {
		c.rejects.fetch_add(1, std::memory_order_relaxed);
		return std::nullopt;
	}
	c.hits.fetch_add(1, std::memory_order_relaxed);
	ShaderSessionStats::g_xlat_hits.fetch_add(1, std::memory_order_relaxed);
	ShaderSessionStats::NoteDiskHit();
	Report();
	return entry;
}

void Store(const Key& key, std::span<const uint32_t> spirv, const CompiledShaderInfo& info) {
	auto& c = Instance();
	if (!Enabled() || spirv.empty()) return;
	Writer w;
	Write(w, info);
	{
		// Refuse to persist anything that does not read back identical.
		CompiledShaderInfo check;
		Reader r {w.out.data(), w.out.data() + w.out.size()};
		if (!Read(r, check) || !SameInfo(check, info)) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1) < 4) {
				std::printf("TranslationCache: reflection of shader %016llx does not round-trip; not cached\n",
				            static_cast<unsigned long long>(info.shader_hash));
				std::fflush(stdout);
			}
			c.rejects.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
	{
		static std::atomic<uint32_t> logged_store {0};
		if (logged_store.fetch_add(1) < 3) {
			std::printf("TranslationCache: store key=%016llx%016llx spirv=%zu words\n",
			            static_cast<unsigned long long>(key.a), static_cast<unsigned long long>(key.b),
			            spirv.size());
			std::fflush(stdout);
		}
	}
	std::vector<uint8_t> record;
	record.reserve(RecordHeaderBytes + spirv.size_bytes() + w.out.size() + 8);
	const auto push = [&](const void* p, size_t n) {
		const auto* b = static_cast<const uint8_t*>(p);
		record.insert(record.end(), b, b + n);
	};
	const auto words = static_cast<uint32_t>(spirv.size());
	const auto info_bytes = static_cast<uint32_t>(w.out.size());
	push(&RecordMagic, 4);
	push(&key.a, 8);
	push(&key.b, 8);
	push(&words, 4);
	push(&info_bytes, 4);
	push(spirv.data(), spirv.size_bytes());
	push(w.out.data(), w.out.size());
	const uint64_t checksum = XXH3_64bits(record.data(), record.size());
	push(&checksum, 8);
	{
		std::lock_guard lock(c.mutex);
		if (!c.thread.joinable() || (c.loaded && !c.usable) || c.index.contains(key)) return;
		c.queue.push_back(std::move(record));
		c.queued_locations.push_back({key, Location {.spirv_words = words, .info_bytes = info_bytes}});
	}
	c.stores.fetch_add(1, std::memory_order_relaxed);
	c.cv.notify_all();
	Report();
}

void Report() {
	static std::atomic<int64_t> next {0};
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch()).count();
	auto due = next.load(std::memory_order_relaxed);
	if (now < due || !next.compare_exchange_strong(due, now + 5000)) return;
	auto& c = Instance();
	const auto hits = c.hits.exchange(0), misses = c.misses.exchange(0), stores = c.stores.exchange(0),
	           rejects = c.rejects.exchange(0);
	if (hits + misses + stores + rejects == 0) return;
	std::printf("TranslationCache (5 s): %llu hits, %llu misses, %llu stored, %llu rejected\n",
	            static_cast<unsigned long long>(hits), static_cast<unsigned long long>(misses),
	            static_cast<unsigned long long>(stores), static_cast<unsigned long long>(rejects));
	std::fflush(stdout);
}

} // namespace Libs::Graphics::TranslationCache
