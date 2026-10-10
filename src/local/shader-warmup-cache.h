#pragma once

#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <bit>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <type_traits>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <xxhash.h>

// Store compiler inputs, never live resource snapshots. A warm
// permutation still has to match the freshly materialized specialization in Get.
// Files are scoped by the schema, title and device signature supplied by the owner.
// The owner keeps opaque driver binaries separately, scoped by executable identity.
namespace LocalShaderWarmup {
using namespace Libs::Graphics;
inline constexpr uint32_t NoShader = UINT32_MAX;

// References are indices in the same atomic cache file, never Vulkan handles.
struct PipelineRecord {
    uint32_t vertex = NoShader, pixel = NoShader, compute = NoShader;
    PipelineRenderingState rendering;
    PipelineVertexInputState vertex_input;
    PipelineStaticParameters state;
};

template<class A> bool VisitPipeline(A& a, PipelineRecord& r);

struct Record {
    ShaderType stage = ShaderType::Unknown;
    uint64_t hash = 0;
    uint32_t user_data_count = 0, push_cursor = 0;
    std::vector<uint32_t> code, back_code, static_key;
    ShaderVertexInputInfo vertex {};
    ShaderPixelInputInfo pixel {};
    ShaderComputeInputInfo compute {};
    ShaderRecompiler::IR::ResourceSpecialization specialization;
};

struct Writer {
    std::vector<uint32_t> words;
    template<class T> void operator()(T& value) { words.push_back(static_cast<uint32_t>(value)); }
    void operator()(float& value) { words.push_back(std::bit_cast<uint32_t>(value)); }
    void operator()(uint64_t& value) {
        words.push_back(uint32_t(value)); words.push_back(uint32_t(value >> 32u));
    }
    bool Good() const { return true; }
};

struct Reader {
    std::span<const uint32_t> words;
    size_t cursor = 0;
    bool good = true;
    template<class T> void operator()(T& value) {
        if (cursor == words.size()) { good = false; return; }
        const auto word = words[cursor++];
        if constexpr (std::is_same_v<T, bool>) {
            if (word > 1) { good = false; return; }
        }
        value = static_cast<T>(word);
    }
    void operator()(float& value) {
        uint32_t word = 0; (*this)(word); value = std::bit_cast<float>(word);
    }
    void operator()(uint64_t& value) {
        uint32_t lo = 0, hi = 0; (*this)(lo); (*this)(hi); value = lo | (uint64_t(hi) << 32u);
    }
    bool Good() const { return good; }
};

template<class A, class... T> void Fields(A& a, T&... values) { (a(values), ...); }
template<class A, class T> void Array(A& a, T& values) { for (auto& v : values) a(v); }
template<class A, class T, class F>
bool Vector(A& a, std::vector<T>& values, uint32_t bound, F visit) {
    uint32_t size = static_cast<uint32_t>(values.size()); a(size);
    if (!a.Good() || size > bound) return false;
    if constexpr (std::is_same_v<A, Reader>) {
        if (size > a.words.size() - a.cursor) return false;
        values.resize(size);
    }
    for (auto& value : values) { visit(value); if (!a.Good()) return false; }
    return true;
}
template<class A> bool Words(A& a, std::vector<uint32_t>& values, uint32_t bound) {
    return Vector(a, values, bound, [&](auto& v) { a(v); });
}

template<class A> void Workgroup(A& a, ShaderWorkgroupInputInfo& s) {
    Array(a, s.threads_num);
    Fields(a, s.lds_size_dwords, s.scratch_size_dwords, s.host_subgroup_size, s.wave_size);
}
template<class A> bool Visit(A& a, Record& r) {
    Fields(a, r.stage, r.hash, r.user_data_count, r.push_cursor);
    if (!Words(a, r.code, 256 * 1024) || !Words(a, r.back_code, 256 * 1024) ||
        !Words(a, r.static_key, 1024) || r.code.empty() || r.user_data_count > 108 ||
        r.push_cursor > 65536) return false;
    if (r.stage == ShaderType::Vertex || r.stage == ShaderType::Mesh) {
        auto& s = r.vertex;
        Fields(a, s.resources_num, s.fetch_attrib_reg, s.fetch_buffer_reg,
            s.scratch_size_dwords, s.pa_cl_vs_out_cntl, s.start_instance_sgpr, s.fetch_external,
            s.fetch_embedded, s.clip_space.enabled);
        Array(a, s.clip_space.scale); Array(a, s.clip_space.offset); Array(a, s.clip_space.half_extent);
        Workgroup(a, s.mesh);
        Fields(a, s.mesh.input_primitive, s.mesh.primitives_per_group, s.mesh.vertices_per_group,
            s.mesh.max_vertices, s.mesh.max_primitives, s.mesh.provoking_vertex);
        if (!a.Good() || s.resources_num < 0 || s.resources_num > s.RES_MAX ||
            s.start_instance_sgpr < -1 || s.start_instance_sgpr >= 8 + 32) return false;
        for (int i = 0; i < s.resources_num; ++i) {
            Array(a, s.resources[i].fields);
            auto& dst = s.resources_dst[i];
            Fields(a, dst.register_start, dst.registers_num, dst.attr_id, dst.fetch_index);
        }
        if ((r.stage == ShaderType::Mesh) != (s.mesh.threads_num[0] != 0)) return false;
    } else if (r.stage == ShaderType::Pixel) {
        auto& s = r.pixel;
        Fields(a, s.lod_stats_subgroup, s.input_num, s.ps_system_input_base,
            s.custom_interpolation_mask, s.ps_perspective_center_vgpr, s.scratch_size_dwords,
            s.ps_pos_x, s.ps_pos_y, s.ps_pos_z, s.ps_pos_w, s.ps_front_face, s.ps_ancillary,
            s.ps_no_perspective, s.ps_pixel_kill_enable, s.ps_depth_export_enable,
            s.ps_sample_mask_export_enable, s.ps_sample_shading, s.ps_early_z, s.ps_execute_on_noop);
        Array(a, s.interpolator_settings); Array(a, s.target_output_mode);
        for (auto& mapping : s.target_export_mapping) a(mapping.packed);
        if (s.input_num > 32) return false;
    } else if (r.stage == ShaderType::Compute) {
        auto& s = r.compute;
        Workgroup(a, s); Array(a, s.dispatch_threads_num); Array(a, s.group_id);
        Fields(a, s.dispatch_thread_dimensions, s.thread_ids_num, s.workgroup_register, s.tg_size_en);
    } else return false;
    if (!Vector(a, r.specialization.buffers, ShaderRecompiler::IR::ShaderInfo::MaxBuffers, [&](auto& b) {
        Fields(a, b.packed_stride, b.descriptor_format, b.descriptor_swizzle, b.byte_base_offset);
    })) return false;
    if (!Vector(a, r.specialization.images, ShaderRecompiler::IR::ShaderInfo::MaxImages, [&](auto& i) {
        Fields(a, i.numeric_class, i.dimension, i.mip_count, i.conversion_format, i.shader_swizzle,
            i.indirect_root, i.indirect_mapping_offset, i.indirect_search_iterations, i.cube, i.fmask);
    })) return false;
    return a.Good();
}

inline ShaderRecompiler::CompileOptions Options(Record& r, std::span<const uint32_t> user_data) {
    ShaderRecompiler::CompileOptions o;
    o.stage = r.stage; o.shader_hash = r.hash; o.user_data = user_data; o.back_code = r.back_code;
    o.enable_lod_stats = true; o.dump_ir = false; o.dump_label = "Shader warmup";
    if (r.stage == ShaderType::Vertex || r.stage == ShaderType::Mesh) {
        o.input_info.vertex = &r.vertex;
        o.user_data_base = r.stage == ShaderType::Vertex ? 8 : 0;
        o.scratch_dwords = r.stage == ShaderType::Vertex ? r.vertex.scratch_size_dwords : r.vertex.mesh.scratch_size_dwords;
        if (r.stage == ShaderType::Mesh) o.wave_size = r.vertex.mesh.wave_size;
    } else if (r.stage == ShaderType::Pixel) {
        o.input_info.pixel = &r.pixel; o.scratch_dwords = r.pixel.scratch_size_dwords;
    } else {
        o.input_info.compute = &r.compute; o.scratch_dwords = r.compute.scratch_size_dwords;
        o.wave_size = r.compute.wave_size;
    }
    return o;
}

inline bool ValidKey(Record& r) {
    std::vector<uint32_t> actual;
    if (r.stage == ShaderType::Vertex || r.stage == ShaderType::Mesh) BuildStageStaticKey(r.vertex, actual);
    else if (r.stage == ShaderType::Pixel) BuildStageStaticKey(r.pixel, actual);
    else if (r.stage == ShaderType::Compute) BuildStageStaticKey(r.compute, actual);
    else return false;
    return actual == r.static_key;
}

template<class A> bool VisitPipeline(A& a, PipelineRecord& r) {
    Fields(a, r.vertex, r.pixel, r.compute);
    if (r.compute != NoShader) return a.Good() && r.vertex == NoShader && r.pixel == NoShader;
    if (r.vertex == NoShader) return false;
    auto& t = r.rendering;
    Fields(a, t.color_count, t.depth_format, t.stencil_format); Array(a, t.color_formats);
    auto& v = r.vertex_input;
    Fields(a, v.binding_count, v.attribute_count);
    for (auto& b : v.bindings) Fields(a, b.stride, b.instance);
    for (auto& b : v.attributes) Fields(a, b.offset, b.binding);
    auto& s = r.state;
    Fields(a, s.negative_one_to_one, s.depth_clip_enable, s.topology, s.primitive_restart_enable,
        s.samples, s.sample_shading_enable, s.depth_bounds_test_enable, s.depth_min_bounds,
        s.depth_max_bounds, s.stencil_test_enable);
    for (auto* stencil : {&s.stencil_front, &s.stencil_back}) {
        Fields(a, stencil->failOp, stencil->passOp, stencil->depthFailOp, stencil->compareOp);
        if (uint32_t(stencil->failOp) > 7 || uint32_t(stencil->passOp) > 7 ||
            uint32_t(stencil->depthFailOp) > 7 || uint32_t(stencil->compareOp) > 7) return false;
    }
    Fields(a, s.cull_front, s.cull_back, s.face, s.provoking_vtx_last, s.polygon_mode);
    Array(a, s.color_mask); Array(a, s.color_srcblend); Array(a, s.color_comb_fcn);
    Array(a, s.color_destblend); Array(a, s.alpha_srcblend); Array(a, s.alpha_comb_fcn);
    Array(a, s.alpha_destblend); Array(a, s.separate_alpha_blend); Array(a, s.blend_enable);
    Array(a, s.blend_bypass);
    if (!a.Good() || t.color_count > RENDER_COLOR_ATTACHMENTS_MAX ||
        v.binding_count > v.bindings.size() || v.attribute_count > v.attributes.size() ||
        uint32_t(s.topology) > uint32_t(vk::PrimitiveTopology::ePatchList) ||
        uint32_t(s.polygon_mode) > uint32_t(vk::PolygonMode::ePoint) ||
        s.samples == 0 || s.samples > 64 || (s.samples & (s.samples - 1))) return false;
    for (uint32_t i = 0; i < v.attribute_count; ++i)
        if (v.attributes[i].binding >= v.binding_count) return false;
    for (uint32_t i = 0; i < t.color_count; ++i)
        if (t.color_formats[i] == vk::Format::eUndefined || s.color_mask[i] > 15 ||
            s.color_srcblend[i] > 20 || s.color_destblend[i] > 20 || s.color_comb_fcn[i] > 4 ||
            s.alpha_srcblend[i] > 20 || s.alpha_destblend[i] > 20 || s.alpha_comb_fcn[i] > 4) return false;
    return true;
}

class Cache {
public:
    static constexpr size_t MaxBytes = 256 * 1024 * 1024;
    static constexpr uint32_t MaxRecords = 65536, MaxPipelines = 262144;
    bool Enabled() const { return !path.empty(); }
    std::vector<std::vector<uint32_t>> records, pipelines;
    ~Cache() { StopWriter(); }

    bool Open(const std::filesystem::path& location, std::string signature) {
        StopWriter();
        records.clear(); pipelines.clear(); record_index.clear(); pipeline_index.clear();
        changed = false; total_bytes = 8;
        path = location; identity = "KytyShaderWarmup3:" + signature;
        if (!std::filesystem::exists(path)) return true;
        return Load(path, identity);
    }

    // The first line of a warm file: its schema, title and device signature.
    static std::string FileIdentity(const std::filesystem::path& location) {
        std::ifstream file(location, std::ios::binary);
        std::string signature;
        for (char c = 0; signature.size() < 1024 && file.get(c);) {
            signature += c;
            if (c == '\n') return signature;
        }
        return {};
    }

    // Takes over the inputs of another device signature's file (another driver version of
    // the same GPU, or another OS) while this one is still empty. They are compiler inputs,
    // validated again on load and against the live source on use; saved under this identity.
    bool Adopt(const std::filesystem::path& from) {
        if (!Enabled() || !records.empty() || !pipelines.empty()) return false;
        const auto signature = FileIdentity(from);
        if (!signature.starts_with("KytyShaderWarmup3:") || !Load(from, signature)) return false;
        changed = true;
        return true;
    }

    uint32_t Add(Record& record) {
        if (!Enabled()) return NoShader;
        Writer writer;
        if (!Visit(writer, record) || !ValidKey(record)) return NoShader;
        return AddWords(std::move(writer.words), false);
    }
    void AddPipeline(PipelineRecord& record) {
        if (!Enabled()) return;
        Writer writer;
        if (!VisitPipeline(writer, record) || !ValidPipeline(writer.words, records)) return;
        AddWords(std::move(writer.words), true);
    }

    void StartWriter() {
        if (!Enabled() || worker.joinable()) return;
        // Copy once at startup. The render thread subsequently enqueues only a
        // new immutable record; serialization/checksums/disk I/O run here.
        worker = std::jthread([this, saved = records, saved_pipelines = pipelines,
                              dirty = changed](std::stop_token stop) mutable {
            for (;;) {
                std::vector<Pending> updates;
                {
                    std::unique_lock lock(writer_mutex);
                    writer_cv.wait_for(lock, stop, std::chrono::seconds(5), [] { return false; });
                    updates.swap(pending);
                }
                for (auto& update : updates) {
                    (update.pipeline ? saved_pipelines : saved).push_back(std::move(update.words));
                    dirty = true;
                }
                if (dirty) {
                    writer_ok = SaveSnapshot(saved, saved_pipelines);
                    if (writer_ok) dirty = false;
                }
                if (stop.stop_requested()) break;
            }
        });
    }

    bool Save() {
        StopWriter();
        if (!Enabled() || !changed) return true;
        const bool ok = SaveSnapshot(records, pipelines);
        if (ok) changed = false;
        return ok;
    }

    // A seed file (tools/local/static-precompile): the same schema under its own identity line, with
    // records whose resource specialization is left to the compiler (see PipelineCache::WarmSeeds).
    static bool ReadSeeds(const std::filesystem::path& location, std::vector<std::vector<uint32_t>>& out_records,
                          std::vector<std::vector<uint32_t>>& out_pipelines) {
        const auto signature = FileIdentity(location);
        std::vector<uint32_t> contents;
        return signature.starts_with("KytyShaderSeeds2:") && ReadFile(location, signature, contents) &&
               Parse(contents, out_records, out_pipelines);
    }
    // A seed file, or a warmup file of any device signature (kyty_shader_precompile --make-hints reads recordings).
    static bool ReadInputs(const std::filesystem::path& location, std::vector<std::vector<uint32_t>>& out_records,
                           std::vector<std::vector<uint32_t>>& out_pipelines) {
        const auto signature = FileIdentity(location);
        std::vector<uint32_t> contents;
        return (signature.starts_with("KytyShaderSeeds2:") || signature.starts_with("KytyShaderWarmup3:")) &&
               ReadFile(location, signature, contents) && Parse(contents, out_records, out_pipelines);
    }

private:
    struct Pending { bool pipeline; std::vector<uint32_t> words; };
    using Index = std::unordered_multimap<uint64_t, uint32_t>;
    std::filesystem::path path;
    std::string identity;
    bool changed = false, writer_ok = true;
    size_t total_bytes = 8;
    Index record_index, pipeline_index;
    std::mutex writer_mutex;
    std::condition_variable_any writer_cv;
    std::vector<Pending> pending;
    std::jthread worker;

    void StopWriter() {
        if (!worker.joinable()) return;
        worker.request_stop(); worker.join();
        if (writer_ok) changed = false;
    }
    static uint64_t Hash(const std::vector<uint32_t>& words) {
        return XXH3_64bits(words.data(), words.size() * 4);
    }
    void Reindex() {
        for (bool pipeline : {false, true}) {
            const auto& entries = pipeline ? pipelines : records;
            auto& index = pipeline ? pipeline_index : record_index;
            for (uint32_t i = 0; i < entries.size(); ++i) {
                index.emplace(Hash(entries[i]), i);
                total_bytes += 4 + entries[i].size() * 4;
            }
        }
    }
    uint32_t AddWords(std::vector<uint32_t> words, bool pipeline) {
        auto& entries = pipeline ? pipelines : records;
        auto& index = pipeline ? pipeline_index : record_index;
        const auto hash = Hash(words);
        const auto [begin, end] = index.equal_range(hash);
        for (auto it = begin; it != end; ++it)
            if (entries[it->second] == words) return it->second;
        if (entries.size() >= (pipeline ? MaxPipelines : MaxRecords) ||
            total_bytes + words.size() * 4 + 4 > MaxBytes - identity.size() - 8) return NoShader;
        const auto id = uint32_t(entries.size());
        total_bytes += words.size() * 4 + 4;
        entries.push_back(std::move(words)); index.emplace(hash, id); changed = true;
        if (worker.joinable()) {
            std::lock_guard lock(writer_mutex);
            pending.push_back({pipeline, entries.back()});
        }
        return id;
    }
    // Nothing changes unless the whole file is valid.
    bool Load(const std::filesystem::path& location, const std::string& signature) {
        std::vector<uint32_t> contents;
        std::vector<std::vector<uint32_t>> loaded, loaded_pipelines;
        if (!ReadFile(location, signature, contents) || !Parse(contents, loaded, loaded_pipelines)) return false;
        records = std::move(loaded); pipelines = std::move(loaded_pipelines);
        Reindex(); return true;
    }
    // Records, then pipeline recipes valid against them, and nothing after.
    static bool Parse(const std::vector<uint32_t>& contents, std::vector<std::vector<uint32_t>>& out_records,
                      std::vector<std::vector<uint32_t>>& out_pipelines) {
        Reader reader {contents};
        if (!ReadRecords(reader, out_records)) return false;
        uint32_t count = 0; reader(count);
        if (!reader.Good() || count > MaxPipelines) return false;
        out_pipelines.resize(count);
        for (auto& words : out_pipelines)
            if (!Words(reader, words, 4096) || !ValidPipeline(words, out_records)) return false;
        return reader.Good() && reader.cursor == contents.size();
    }
    static bool ReadFile(const std::filesystem::path& location, const std::string& signature,
                         std::vector<uint32_t>& contents) {
        std::ifstream file(location, std::ios::binary | std::ios::ate);
        if (!file) return false;
        const auto size = file.tellg();
        if (size < std::streamoff(signature.size() + 8) || size > std::streamoff(MaxBytes)) return false;
        std::string saved(signature.size(), '\0'); uint64_t checksum = 0;
        const auto bytes = size_t(size) - saved.size() - 8;
        if (bytes % 4 != 0) return false;
        contents.resize(bytes / 4);
        file.seekg(0); file.read(saved.data(), saved.size()); file.read(reinterpret_cast<char*>(&checksum), 8);
        file.read(reinterpret_cast<char*>(contents.data()), bytes);
        return file && saved == signature && XXH3_64bits(contents.data(), bytes) == checksum;
    }
    static bool ReadRecords(Reader& reader, std::vector<std::vector<uint32_t>>& loaded) {
        uint32_t count = 0; reader(count);
        if (!reader.Good() || count > MaxRecords) return false;
        loaded.resize(count);
        for (auto& record : loaded) {
            if (!Words(reader, record, 1024 * 1024)) return false;
            Record decoded; Reader entry {record};
            if (!Visit(entry, decoded) || entry.cursor != record.size() || !ValidKey(decoded)) return false;
        }
        return reader.Good();
    }
    static bool ValidPipeline(const std::vector<uint32_t>& words,
                              const std::vector<std::vector<uint32_t>>& shaders) {
        PipelineRecord r; Reader reader {words};
        if (!VisitPipeline(reader, r) || reader.cursor != words.size()) return false;
        const auto stage = [&](uint32_t index) {
            return index < shaders.size() ? ShaderType(shaders[index][0]) : ShaderType::Unknown;
        };
        if (r.compute != NoShader) return stage(r.compute) == ShaderType::Compute;
        if (stage(r.vertex) != ShaderType::Vertex && stage(r.vertex) != ShaderType::Mesh) return false;
        if (r.pixel != NoShader && stage(r.pixel) != ShaderType::Pixel) return false;
        Record vertex; Reader input {shaders[r.vertex]};
        return Visit(input, vertex) && (vertex.stage == ShaderType::Mesh
            ? r.vertex_input.attribute_count == 0 && r.vertex_input.binding_count == 0
            : r.vertex_input.attribute_count == vertex.vertex.resources_num);
    }
    bool SaveSnapshot(std::vector<std::vector<uint32_t>>& saved,
                      std::vector<std::vector<uint32_t>>& saved_pipelines) {
        Writer writer;
        for (auto* entries : {&saved, &saved_pipelines}) {
            uint32_t count = uint32_t(entries->size()); writer(count);
            for (auto& record : *entries)
                if (!Words(writer, record, 1024 * 1024) ||
                    writer.words.size() * 4 > MaxBytes - identity.size() - 8) return false;
        }
        std::error_code ec;
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) return false;
        auto temporary = path; temporary += ".tmp";
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        const auto bytes = writer.words.size() * 4;
        const uint64_t checksum = XXH3_64bits(writer.words.data(), bytes);
        file.write(identity.data(), identity.size()); file.write(reinterpret_cast<const char*>(&checksum), 8);
        file.write(reinterpret_cast<const char*>(writer.words.data()), bytes); file.close();
        if (!file) return false;
        std::filesystem::rename(temporary, path, ec);
        return !ec;
    }
};
} // namespace LocalShaderWarmup
