#include "graphics/shader/recompiler/ir/passes/LinearSrt.h"
#ifdef KYTY_LOCAL_NATIVE_RESOURCES
#include "native-resource-state.h"
#endif
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/FunctionLdsLayout.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/ShaderInfoCollection.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/recompiler/ir/passes/WaterfallDescriptor.h"

#include <array>
#include <bit>
#include <cstring>
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderComputeInputInfo;
using Libs::Graphics::ShaderType;
namespace Decoder = Libs::Graphics::ShaderRecompiler::Decoder;

void Check(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool SameResourceSnapshot(const ResourceSnapshot &lhs,
                          const ResourceSnapshot &rhs) {
  return lhs.buffers == rhs.buffers && lhs.images == rhs.images &&
         lhs.samplers == rhs.samplers &&
         lhs.flattened_srt == rhs.flattened_srt &&
         lhs.user_data == rhs.user_data && lhs.uniform_fill == rhs.uniform_fill;
}

template <typename F>
void CheckFatal(F &&function, std::string_view expected, const char *message) {
  try {
    function();
  } catch (const std::runtime_error &error) {
    Check(std::string_view(error.what()).find(expected) !=
              std::string_view::npos,
          message);
    return;
  }
  Check(false, message);
}

struct Fixture {
  Program program;
  Block *block = nullptr;

  explicit Fixture(ShaderType stage = ShaderType::Compute) {
    program.stage = stage;
    program.user_data_count = 64;
    block = AddBlock();
  }

  Block *AddBlock() {
    auto storage = std::make_unique<Block>();
    auto *result = storage.get();
    program.block_storage.push_back(std::move(storage));
    program.blocks.push_back(result);
    program.block_info.push_back(
        {.id = static_cast<uint32_t>(program.block_info.size())});
    return result;
  }

  Value Emit(ValueOpcode opcode, std::initializer_list<Value> args = {},
             uint64_t flags = 0, Block *destination = nullptr) {
    if (NumArgsOf(opcode) != std::numeric_limits<size_t>::max() &&
        NumArgsOf(opcode) != args.size()) {
      throw std::runtime_error(std::string(ValueOpcodeName(opcode)) +
                               " argument count");
    }
    auto &inst = (destination != nullptr ? destination : block)
                     ->AppendNewInst(opcode, args, flags);
    return Value(&inst);
  }

  template <typename T>
  Value Emit(ValueOpcode opcode, std::initializer_list<Value> args, T flags,
             Block *destination = nullptr) {
    uint64_t bits = 0;
    std::memcpy(&bits, &flags, sizeof(flags));
    return Emit(opcode, args, bits, destination);
  }

  Value UserData(uint32_t index) {
    return Emit(ValueOpcode::GetUserData,
                {Value(static_cast<ScalarReg>(index))});
  }

  MemoryFlags AddMemory(MemoryInfo memory, uint32_t pc) {
    const auto index = static_cast<uint32_t>(program.memory_info.size());
    program.memory_info.push_back(memory);
    return {index, pc};
  }

  Value Buffer(std::array<Value, 4> dwords, uint32_t pc = 0) {
    return Emit(ValueOpcode::GetBufferResource,
                {dwords[0], dwords[1], dwords[2], dwords[3]},
                MemoryFlags{0, pc});
  }

  Value Address(Value low, Value high, uint32_t pc = 0) {
    return Emit(ValueOpcode::GetAddressResource, {low, high},
                MemoryFlags{0, pc});
  }

  Value Image(std::array<Value, 8> dwords, uint32_t pc = 0) {
    return Emit(ValueOpcode::GetImageResource,
                {dwords[0], dwords[1], dwords[2], dwords[3], dwords[4],
                 dwords[5], dwords[6], dwords[7]},
                MemoryFlags{0, pc});
  }

  Value Sampler(std::array<Value, 4> dwords, uint32_t pc = 0) {
    return Emit(ValueOpcode::GetSamplerResource,
                {dwords[0], dwords[1], dwords[2], dwords[3]},
                MemoryFlags{0, pc});
  }

  Value ImageAddress() {
    return Emit(ValueOpcode::MakeImageAddress,
                {Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
                 Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
                 Value(0u), Value(0u), Value(0u)});
  }

  void PlanAndTrack() {
    BuildSrtPlan(program);
    TrackResources(program);
  }
};

struct TestMemory {
  uint64_t base = 0x1000;
  std::array<uint32_t, 8> words{};
  uint32_t reads = 0;
  uint32_t fail_after = UINT32_MAX;
};

bool ReadTestMemory(void *userdata, uint64_t address, uint32_t *value) {
  auto *memory = static_cast<TestMemory *>(userdata);
  if (memory == nullptr || value == nullptr || address < memory->base ||
      address - memory->base >= memory->words.size() * sizeof(uint32_t) ||
      memory->reads >= memory->fail_after) {
    return false;
  }
  *value = memory->words[(address - memory->base) / sizeof(uint32_t)];
  memory->reads++;
  return true;
}

struct LinearTestMemory {
  uint64_t base = 0x1000;
  std::vector<uint32_t> words = std::vector<uint32_t>(0x2200 / 4);
  uint64_t fail_address = UINT64_MAX;
};

bool ReadLinearTestMemory(void *userdata, uint64_t address, uint32_t *value) {
  auto *memory = static_cast<LinearTestMemory *>(userdata);
  if (memory == nullptr || value == nullptr || address < memory->base ||
      address - memory->base >= memory->words.size() * sizeof(uint32_t) ||
      (address & 3u) != 0u || address == memory->fail_address) {
    return false;
  }
  *value = memory->words[(address - memory->base) / sizeof(uint32_t)];
  return true;
}

bool ReadLinearTestMemorySpan(void *userdata, uint64_t address, uint32_t *values,
                              uint32_t count, bool) {
  for (uint32_t i = 0; i < count; i++) {
    if (!ReadLinearTestMemory(userdata, address + uint64_t{i} * 4u, values + i)) {
      return false;
    }
  }
  return true;
}

uint64_t LinearTestMappingEnd(void *userdata, uint64_t address) {
  const auto *memory = static_cast<const LinearTestMemory *>(userdata);
  const auto end = memory->base + memory->words.size() * sizeof(uint32_t);
  return address >= memory->base && address < end ? end : 0;
}

std::unique_ptr<Fixture>
MakeIndirectImageFixture(bool malformed, uint32_t material_immediate = 0,
                         bool memory_backed_material = false,
                         bool immediate_member = false) {
  auto fixture = std::make_unique<Fixture>();
  std::array<Value, 4> material_words;
  std::array<Value, 4> heap_words;
  for (uint32_t dword = 0; dword < 4; dword++) {
    material_words[dword] = fixture->UserData(dword);
    heap_words[dword] = fixture->UserData(dword + 4u);
  }
  if (memory_backed_material) {
    const auto pointer_address =
        fixture->Address(fixture->UserData(9), fixture->UserData(10), 0x10b0);
    MemoryInfo pointer_word;
    pointer_word.kind = ResourceKind::ScalarAddress;
    const auto pointer =
        fixture->Emit(ValueOpcode::LoadAddressU32,
                      {pointer_address, Value(0u), Value(0u), Value(true)},
                      fixture->AddMemory(pointer_word, 0x10b0));
    const auto address = fixture->Address(pointer, Value(0u), 0x10c0);
    MemoryInfo descriptor_word;
    descriptor_word.kind = ResourceKind::ScalarAddress;
    material_words[0] =
        fixture->Emit(ValueOpcode::LoadAddressU32,
                      {address, Value(0u), Value(0u), Value(true)},
                      fixture->AddMemory(descriptor_word, 0x10c0));
  }
  const auto material = fixture->Buffer(material_words, 0x10d8);
  const auto heap = fixture->Buffer(heap_words, 0x10d8);
  if (memory_backed_material) {
    MemoryInfo shared_buffer;
    shared_buffer.kind = ResourceKind::Buffer;
    const auto load =
        fixture->Emit(ValueOpcode::LoadBufferU32,
                      {material, Value(0u), Value(0u), Value(0u), Value(true)},
                      fixture->AddMemory(shared_buffer, 0x10d8));
    fixture->Emit(ValueOpcode::ReferenceU32, {load});
  }
  const auto selector = fixture->Emit(ValueOpcode::ReadFirstLane,
                                      {fixture->UserData(8), Value(true)});
  const auto record =
      fixture->Emit(ValueOpcode::IMul32, {selector, Value(224u)});
  const auto member = fixture->Emit(ValueOpcode::IAdd32, {record, Value(4u)});
  fixture->Emit(ValueOpcode::ReferenceU32, {record});
  fixture->Emit(ValueOpcode::ReferenceU32, {member});
  MemoryInfo material_scalar;
  material_scalar.kind = ResourceKind::ScalarBuffer;
  material_scalar.offset = material_immediate;
  const auto key =
      fixture->Emit(ValueOpcode::ReadConstBuffer, {material, immediate_member ? record : member},
                    fixture->AddMemory(material_scalar, 0x10d8));
  const auto heap_offset =
      fixture->Emit(ValueOpcode::ShiftLeftLogical32, {key, Value(5u)});
  std::array<Value, 8> image_words;
  MemoryInfo heap_scalar;
  heap_scalar.kind = ResourceKind::ScalarBuffer;
  for (uint32_t dword = 0; dword < image_words.size(); dword++) {
    auto component = heap_scalar;
    component.offset = dword * sizeof(uint32_t);
    if (malformed && dword == image_words.size() - 1u) {
      component.offset += sizeof(uint32_t);
    }
    image_words[dword] =
        fixture->Emit(ValueOpcode::ReadConstBuffer, {heap, heap_offset},
                      fixture->AddMemory(component, 0x10d8));
  }
  const auto image = fixture->Image(image_words, 0x10f0);
  const auto sampler =
      fixture->Sampler({Value(0u), Value(0u), Value(0u), Value(0u)}, 0x10f0);
  MemoryInfo sample;
  sample.kind = ResourceKind::Image;
  sample.image_dimension = Decoder::ImageDimension::Dim2D;
  const auto sampled = fixture->Emit(ValueOpcode::ImageSampleRaw,
                                     {image, sampler, fixture->ImageAddress()},
                                     fixture->AddMemory(sample, 0x10f0));
  const auto sampled_x =
      fixture->Emit(ValueOpcode::CompositeExtractU32x4, {sampled, Value(0u)});
  fixture->Emit(ValueOpcode::ReferenceU32, {sampled_x});
  return fixture;
}

void TestInvariantIndirectImageMaterialization() {
  auto fixture = MakeIndirectImageFixture(false);
  fixture->PlanAndTrack();
  auto resource_plan = ExtractResourcePlan(fixture->program);
  EliminateDeadCode(fixture->program.blocks);
  ValidateProgram(fixture->program, true);

  Check(fixture->program.info.buffers.size() == 1 &&
            fixture->program.info.images.size() == 1 &&
            fixture->program.dynamic_reads.size() == 1,
        "indirect image key was not retained as a scalar-buffer read");
  const auto source = fixture->program.info.images[0].source;
  Check(source < fixture->program.descriptor_sources.size() &&
            fixture->program.descriptor_sources[source]
                .indirect_image.has_value(),
        "indirect image source was not retained for runtime proof");
  const auto image_handle =
      std::ranges::find_if(*fixture->block, [](const Inst &inst) {
        return inst.GetOpcode() == ValueOpcode::GetImageResource;
      });
  Check(image_handle != fixture->block->end() &&
            image_handle->Arg(0).ResolveInstruction() != nullptr &&
            image_handle->Arg(0).ResolveInstruction()->GetOpcode() ==
                ValueOpcode::ReadConstBuffer,
        "indirect image handle discarded the live material key");

  std::array<uint32_t, 9> user_data{0x1000u,    224u << 16u, 2u, 0u, 0x2000u,
                                    16u << 16u, 4u,          0u, 7u};
  LinearTestMemory memory;
  std::array<uint32_t, 8> image_descriptor{};
  image_descriptor[0] = 0x20u;
  image_descriptor[1] =
      static_cast<uint32_t>(
          Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float)
      << 20u;
  image_descriptor[2] = 3u | (3u << 14u);
  image_descriptor[3] =
      Libs::Graphics::DstSel(4, 5, 6, 7) |
      (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D)
       << 28u);
  for (uint32_t dword = 0; dword < image_descriptor.size(); dword++) {
    memory.words[(0x2000u - memory.base) / 4u + dword] =
        image_descriptor[dword];
    memory.words[(0x2020u - memory.base) / 4u + dword] =
        image_descriptor[dword];
  }
  memory.words[(0x2020u - memory.base) / 4u] ^= 1u;

  SrtRuntime runtime{.user_data = user_data,
                     .userdata = &memory,
                     .read_specialization_memory = ReadLinearTestMemory};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(resource_plan, runtime, snapshot, specialization) &&
            snapshot.images.size() == 1 &&
            std::equal(image_descriptor.begin(), image_descriptor.end(),
                       snapshot.images[0].dwords.begin()),
        "invariant indirect image table did not materialize");

  const auto prior_snapshot = snapshot;
  const auto prior_specialization = specialization;
  memory.fail_address = 0x1004u;
  Check(!MaterializeResources(resource_plan, runtime, snapshot,
                              specialization) &&
            SameResourceSnapshot(snapshot, prior_snapshot) &&
            specialization == prior_specialization,
        "rejected planning memory read mutated the snapshot");
  memory.fail_address = UINT64_MAX;

  memory.words[(0x1000u - memory.base + 36u) / 4u] = 1u;
  for (uint32_t dword = 0; dword < image_descriptor.size(); dword++) {
    memory.words[(0x2000u - memory.base) / 4u + dword] = 0u;
    memory.words[(0x2020u - memory.base) / 4u + dword] = 0u;
  }
  memory.words[(0x2000u - memory.base) / 4u + 1u] = image_descriptor[1];
  memory.words[(0x2000u - memory.base) / 4u + 3u] = image_descriptor[3];
  memory.words[(0x2020u - memory.base) / 4u + 1u] = image_descriptor[1];
  memory.words[(0x2020u - memory.base) / 4u + 3u] =
      image_descriptor[3] ^ (1u << 28u);
  ResourceSnapshot null_snapshot;
  ResourceSpecialization null_specialization;
  Check(MaterializeResources(resource_plan, runtime, null_snapshot,
                             null_specialization) &&
            std::ranges::all_of(null_snapshot.images[0].dwords,
                                [](uint32_t dword) { return dword == 0u; }),
        "stale typed null image descriptors were not canonicalized");

  for (uint32_t dword = 0; dword < image_descriptor.size(); dword++) {
    memory.words[(0x2000u - memory.base) / 4u + dword] =
        image_descriptor[dword];
    memory.words[(0x2020u - memory.base) / 4u + dword] =
        image_descriptor[dword];
  }
  memory.words[(0x2020u - memory.base) / 4u] ^= 1u;
  memory.words[(0x1000u - memory.base + 36u) / 4u] = 1u;
  ResourceSnapshot dynamic_snapshot;
  ResourceSpecialization dynamic_specialization;
  Check(MaterializeResources(resource_plan, runtime, dynamic_snapshot,
                             dynamic_specialization) &&
            dynamic_snapshot.images.size() == 2 &&
            dynamic_specialization.images.size() == 2,
        "dynamic indirect image table did not materialize");
  ApplyResourceSpecialization(fixture->program, dynamic_specialization);
  Check(fixture->program.info.images.size() == 2 &&
            fixture->program.info.images[0].indirect_root == 0 &&
            fixture->program.info.images[0].indirect_search_iterations != 0 &&
            fixture->program.info.images[0].indirect_resources.size() == 2 &&
            dynamic_snapshot.images.size() == 2,
        "dynamic indirect image table was not specialized transactionally");
  const auto &mapping = dynamic_specialization.images[0];
  const auto key_count = dynamic_snapshot.flattened_srt[mapping.indirect_mapping_offset];
  Check(mapping.indirect_search_iterations == std::bit_width(key_count) &&
            mapping.indirect_mapping_offset + 1u + key_count * 2u ==
                dynamic_snapshot.flattened_srt.size(),
        "indirect image mapping retained worst-case padding");

  for (uint32_t dword = 0; dword < image_descriptor.size(); dword++) {
    memory.words[(0x2000u - memory.base) / 4u + dword] =
        image_descriptor[dword];
    memory.words[(0x2020u - memory.base) / 4u + dword] =
        image_descriptor[dword];
  }
  memory.words[(0x2000u - memory.base) / 4u] += 0x100u;
  memory.words[(0x2020u - memory.base) / 4u] += 0x101u;
  ResourceSnapshot rebound_snapshot;
  ResourceSpecialization rebound_specialization;
  Check(MaterializeResources(resource_plan, runtime, rebound_snapshot,
                             rebound_specialization) &&
            rebound_specialization == dynamic_specialization,
        "stable indirect key mapping did not accept changed image addresses");
  memory.words[(0x2020u - memory.base) / 4u] =
      memory.words[(0x2000u - memory.base) / 4u];
  Check(MaterializeResources(resource_plan, runtime, rebound_snapshot,
                             rebound_specialization) &&
            rebound_specialization != dynamic_specialization,
        "collapsed indirect candidates did not select a new specialization");
  const auto collapsed_specialization = rebound_specialization;
  ResourceSnapshot capacity_snapshot;
  ResourceSpecialization capacity_specialization;
  for (const uint32_t records : {1u, 3u}) {
    user_data[2] = records;
    Check(MaterializeResources(resource_plan, runtime, capacity_snapshot,
                               capacity_specialization),
          "runtime indirect key mapping rejected a valid material-table size");
  }
  user_data[2] = 2u;
  memory.words[(0x2020u - memory.base) / 4u] =
      memory.words[(0x2000u - memory.base) / 4u] + 1u;
  memory.words[(0x2040u - memory.base) / 4u] =
      memory.words[(0x2000u - memory.base) / 4u] + 2u;
  for (uint32_t dword = 1; dword < image_descriptor.size(); dword++) {
    memory.words[(0x2040u - memory.base) / 4u + dword] =
        image_descriptor[dword];
  }
  memory.words[(0x1000u - memory.base + 68u) / 4u] = 2u;
  Check(MaterializeResources(resource_plan, runtime, rebound_snapshot,
                             rebound_specialization) &&
            rebound_specialization != collapsed_specialization,
        "larger indirect candidate topology reused the old specialization");

  auto memory_backed = MakeIndirectImageFixture(false, 0u, true);
  memory_backed->PlanAndTrack();
  auto memory_backed_plan = ExtractResourcePlan(memory_backed->program);
  EliminateDeadCode(memory_backed->program.blocks);
  std::array<uint32_t, 11> memory_backed_user_data{0x1000u, 224u << 16u, 2u, 0u,
                                                   0x2000u, 16u << 16u,  4u, 0u,
                                                   7u,      0x3100u,     0u};
  memory.words[(0x3100u - memory.base) / 4u] = 0x3000u;
  memory.words[(0x3000u - memory.base) / 4u] = 0x1000u;
  memory.fail_address = 0x3100u;
  SrtRuntime memory_backed_runtime{.user_data = memory_backed_user_data,
                                   .userdata = &memory,
                                   .read_specialization_memory =
                                       ReadLinearTestMemory};
  const auto memory_backed_prior_snapshot = snapshot;
  const auto memory_backed_prior_specialization = specialization;
  Check(!MaterializeResources(memory_backed_plan, memory_backed_runtime,
                              snapshot, specialization) &&
            SameResourceSnapshot(snapshot, memory_backed_prior_snapshot) &&
            specialization == memory_backed_prior_specialization,
        "rejected indirect table descriptor read mutated the snapshot");
  memory.fail_address = UINT64_MAX;

  auto malformed = MakeIndirectImageFixture(true);
  BuildSrtPlan(malformed->program);
  CheckFatal([&] { TrackResources(malformed->program); }, "not a valid runtime value",
             "malformed indirect image pattern was accepted");
  Check(!malformed->program.resource_tracking_complete &&
            malformed->program.info.images.empty() &&
            malformed->program.descriptor_sources.empty(),
        "malformed indirect image pattern was partially accepted");

  auto immediate_fixture = MakeIndirectImageFixture(false, 4u);
  immediate_fixture->PlanAndTrack();
  auto immediate_plan = ExtractResourcePlan(immediate_fixture->program);
  Check(immediate_fixture->program.resource_tracking_complete &&
            std::ranges::any_of(immediate_plan.descriptor_sources,
                                [](const DescriptorSource &source) {
                                  return source.indirect_image.has_value() &&
                                         source.indirect_image->selector_immediate == 4u;
                                }),
        "material key immediate was not carried into the indirect image source");
  LinearTestMemory immediate_memory;
  for (uint32_t dword = 0; dword < image_descriptor.size(); dword++) {
    immediate_memory.words[(0x2000u - immediate_memory.base) / 4u + dword] =
        image_descriptor[dword];
    immediate_memory.words[(0x2020u - immediate_memory.base) / 4u + dword] =
        image_descriptor[dword];
  }
  immediate_memory.words[(0x2020u - immediate_memory.base) / 4u] ^= 1u;
  SrtRuntime immediate_runtime{.user_data = user_data,
                               .userdata = &immediate_memory,
                               .read_specialization_memory = ReadLinearTestMemory};
  immediate_memory.words[(0x1000u - immediate_memory.base + 36u) / 4u] = 1u;
  ResourceSnapshot unshifted_snapshot;
  ResourceSpecialization unshifted_specialization;
  Check(MaterializeResources(immediate_plan, immediate_runtime, unshifted_snapshot,
                             unshifted_specialization) &&
            unshifted_snapshot.images.size() == 1,
        "material key immediate was applied to the wrapped dynamic offset");
  immediate_memory.words[(0x1000u - immediate_memory.base + 36u) / 4u] = 0u;
  immediate_memory.words[(0x1000u - immediate_memory.base + 40u) / 4u] = 1u;
  ResourceSnapshot shifted_snapshot;
  ResourceSpecialization shifted_specialization;
  Check(MaterializeResources(immediate_plan, immediate_runtime, shifted_snapshot,
                             shifted_specialization) &&
            shifted_snapshot.images.size() == 2 &&
            shifted_specialization.images.size() == 2,
        "material key immediate did not select the record field after the wrapped offset");
}

void TestComputeBufferFill() {
  struct Options {
    bool scalar = false;
    bool conditional = false;
    bool shifted = false;
    bool extra_store = false;
    bool clean = false;
    bool branch = false;
  };
  const auto Run = [](Options options) {
    Fixture fixture;
    fixture.program.block_info[0].terminator.kind =
        Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
    if (options.branch) {
      fixture.program.block_info[0].terminator.kind = Libs::Graphics::
          ShaderRecompiler::CFG::TerminatorKind::ConditionalBranch;
    }
    const auto buffer =
        fixture.Buffer({fixture.UserData(0), fixture.UserData(1),
                        fixture.UserData(2), fixture.UserData(3)});
    const auto local = fixture.Emit(
        ValueOpcode::GetBuiltin,
        {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)),
         Value(0u)});
    const auto group = fixture.Emit(
        ValueOpcode::GetBuiltin,
        {Value(static_cast<uint32_t>(StageInputKind::WorkgroupId)), Value(0u)});
    auto index =
        fixture.Emit(ValueOpcode::IAdd32,
                     {local, fixture.Emit(ValueOpcode::ShiftLeftLogical32,
                                          {group, Value(6u)})});
    if (options.shifted)
      index = fixture.Emit(ValueOpcode::IAdd32, {index, Value(1u)});
    Value value(0u);
    TestMemory memory;
    memory.words[0] = 0x40404040u;
    if (options.scalar) {
      const auto input =
          fixture.Buffer({Value(static_cast<uint32_t>(memory.base)),
                          Value(4u << 16), Value(1u), Value(0x14204u)});
      MemoryInfo load;
      load.kind = ResourceKind::ScalarBuffer;
      value = fixture.Emit(ValueOpcode::ReadConstBuffer, {input, Value(0u)},
                           fixture.AddMemory(load, 8));
    }
    MemoryInfo store;
    store.kind = ResourceKind::Buffer;
    store.formatted = true;
    store.idxen = true;
    const auto flags = fixture.AddMemory(store, 16);
    const auto predicate =
        options.conditional
            ? fixture.Emit(ValueOpcode::ULessThan32, {local, Value(32u)})
            : Value(true);
    const auto EmitStore = [&] {
      fixture.Emit(ValueOpcode::StoreBufferU32,
                   {buffer, index, Value(0u), Value(0u), value, predicate},
                   flags);
    };
    EmitStore();
    if (options.extra_store)
      EmitStore();
    fixture.PlanAndTrack();
    auto plan = ExtractResourcePlan(fixture.program);
    std::array<uint32_t, 4> userdata{0x200000u, 4u << 16, 0x4000u, 0x14204u};
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
    const auto Read = +[](void *data, uint64_t address, uint32_t *word) {
      auto &memory = *static_cast<TestMemory *>(data);
      if (address != memory.base)
        return false;
      ++memory.reads;
      *word = memory.words[0];
      return true;
    };
    Check(MaterializeResources(
              plan,
              {.user_data = userdata,
               .read_memory = Read,
               .userdata = &memory,
               .read_specialization_memory = options.clean ? Read : nullptr},
              snapshot, specialization),
          "fill fixture did not materialize");
    const bool expected = !options.conditional && !options.shifted &&
                          !options.extra_store && !options.branch &&
                          (!options.scalar || options.clean);
    Check((snapshot.uniform_fill.words != 0) == expected,
          "fill proof accepted an unsafe store or missed the real GTA3 clear");
    if (expected) {
      Check(snapshot.uniform_fill.words == 1 &&
                snapshot.uniform_fill.group_stride[0] == 64 &&
                snapshot.uniform_fill.value ==
                    (options.scalar ? 0x40404040u : 0u),
            "fill proof lost address coverage or the actual stored scalar");
    }
  };
  Run({});
  Run({.scalar = true, .clean = true});
  Run({.scalar = true});
  Run({.conditional = true, .clean = true});
  Run({.shifted = true, .clean = true});
  Run({.extra_store = true, .clean = true});
  Run({.clean = true, .branch = true});
}

void TestDenseBufferTracking() {
  Fixture fixture;
  std::array<Value, 8> userdata;
  for (uint32_t index = 0; index < userdata.size(); index++) {
    userdata[index] = fixture.UserData(index);
  }
  const auto first =
      fixture.Buffer({userdata[0], userdata[1], userdata[2], userdata[3]}, 4);
  const auto second =
      fixture.Buffer({userdata[4], userdata[5], userdata[6], userdata[7]}, 28);

  MemoryInfo load_info;
  load_info.kind = ResourceKind::Buffer;
  load_info.offset = 4;
  load_info.formatted = true;
  const auto load_flags = fixture.AddMemory(load_info, 4);
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {first, Value(0u), Value(0u), Value(0u), Value(true)},
               load_flags);

  auto store_info = load_info;
  store_info.offset = 12;
  const auto store_flags = fixture.AddMemory(store_info, 8);
  fixture.Emit(ValueOpcode::StoreBufferU32,
               {first, Value(0u), Value(0u), Value(0u), Value(7u), Value(true)},
               store_flags);

  auto atomic_info = load_info;
  atomic_info.offset = 0;
  const auto atomic_flags = fixture.AddMemory(atomic_info, 12);
  fixture.Emit(ValueOpcode::BufferAtomicIAdd32,
               {first, Value(0u), Value(0u), Value(1u), Value(0u), Value(true)},
               atomic_flags);

  const auto other_flags = fixture.AddMemory(load_info, 28);
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {second, Value(0u), Value(0u), Value(0u), Value(true)},
               other_flags);
  fixture.PlanAndTrack();

  Check(fixture.program.info.buffers.size() == 2,
        "typed buffer sources were not densely interned");
  Check(fixture.program.descriptor_sources.size() == 2,
        "descriptor source table did not match dense topology");
  const auto &resource = fixture.program.info.buffers[0];
  Check(resource.read && resource.written && resource.atomic &&
            resource.formatted && resource.max_byte_extent == 16 &&
            resource.first_use_pc == 4,
        "buffer access facts were not merged");
  Check(first.Instruction()->Flags<uint32_t>() == 0 &&
            second.Instruction()->Flags<uint32_t>() == 1,
        "typed handles were not assigned dense indices");
  Check(fixture.program.memory_info[load_flags.index].resource == 0 &&
            fixture.program.memory_info[store_flags.index].resource == 0 &&
            fixture.program.memory_info[other_flags.index].resource == 1,
        "typed memory metadata was not patched to dense indices");

  CheckFatal([&] { TrackResources(fixture.program); }, "already tracked",
             "resource tracking allowed a second mutation pass");
}

void TestScalarAndVectorBufferAlias() {
  Fixture fixture;
  const auto d0 = fixture.UserData(0);
  const auto d1 = fixture.UserData(1);
  const auto d2 = fixture.UserData(2);
  const auto d3 = fixture.UserData(3);
  const auto descriptor = fixture.Buffer({d0, d1, d2, d3}, 4);

  MemoryInfo scalar;
  scalar.kind = ResourceKind::ScalarBuffer;
  const auto scalar_flags = fixture.AddMemory(scalar, 4);
  fixture.Emit(ValueOpcode::ReadConstBuffer, {descriptor, fixture.UserData(4)},
               scalar_flags);
  MemoryInfo vector;
  vector.kind = ResourceKind::Buffer;
  const auto vector_flags = fixture.AddMemory(vector, 8);
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
               vector_flags);
  fixture.PlanAndTrack();

  Check(fixture.program.info.buffers.size() == 1 &&
            fixture.program.info.buffers[0].scalar,
        "typed scalar and vector uses of one descriptor were split");
  Check(fixture.program.memory_info[scalar_flags.index].resource == 0 &&
            fixture.program.memory_info[vector_flags.index].resource == 0,
        "scalar/vector alias did not share a dense index");
}

void TestRuntimeUnsignedMinDescriptor() {
  Fixture fixture;
  const auto word3 =
      fixture.Emit(ValueOpcode::UMin32, {fixture.UserData(0), Value(0x100u)});
  const auto descriptor =
      fixture.Buffer({Value(0u), Value(0u), Value(64u), word3}, 0x330);
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(memory, 0x330));
  fixture.PlanAndTrack();

  std::array<uint32_t, 1> user_data{0xffffffffu};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue value;
  const auto source = fixture.program.info.buffers[0].source;
  Check(EvaluateDescriptorSource(fixture.program, source, runtime, value) &&
            value.dwords[3] == 0x100u,
        "runtime descriptor unsigned minimum did not clamp its first operand");
  user_data[0] = 0x80u;
  Check(
      EvaluateDescriptorSource(fixture.program, source, runtime, value) &&
          value.dwords[3] == 0x80u,
      "runtime descriptor unsigned minimum did not preserve its first operand");
}

void TestRuntimeUnsignedGreaterEqual() {
  Fixture fixture;
  const auto predicate = fixture.Emit(ValueOpcode::UGreaterThanEqual32,
                                      {fixture.UserData(0), fixture.UserData(1)});
  BuildSrtPlan(fixture.program);
  Check(ValidateRuntimeValue(fixture.program, predicate),
        "uniform unsigned >= must be accepted by runtime descriptor evaluation");
  const std::array<std::array<uint32_t, 3>, 5> cases{{
      {0u, 0u, 1u}, {0u, 1u, 0u}, {1u, 0u, 1u},
      {0xffffffffu, 0x80000000u, 1u}, {0x7fffffffu, 0x80000000u, 0u}}};
  for (const auto& item : cases) {
    SrtRuntime runtime{.user_data = std::span(item.data(), 2)};
    uint32_t value = 42;
    Check(EvaluateUniformValues(fixture.program, std::span(&predicate, 1), runtime,
                                std::span(&value, 1)) && value == item[2],
          "runtime unsigned >= must preserve equality and unsigned high-bit ordering");
  }
}

void TestLargeRuntimeEvaluation() {
  Fixture fixture;
  const auto input = fixture.UserData(0);
  std::vector<Value> values;
  constexpr uint32_t count = 4096;
  for (uint32_t i = 0; i < count; ++i) {
    values.push_back(fixture.Emit(ValueOpcode::IAdd32, {input, Value(i)}));
    if (i % 16 == 0) {
      values.back() = fixture.Emit(ValueOpcode::Identity, {values.back()});
    }
  }
  for (uint32_t i = 0; i < count; ++i) {
    values.push_back(values[count - i - 1]);
  }
  BuildSrtPlan(fixture.program);
  std::vector<uint32_t> result(values.size());
  for (uint32_t input_value : {17u, 0xffffff00u}) {
    Check(EvaluateUniformValues(fixture.program, values,
                                {.user_data = std::span(&input_value, 1)}, result),
          "large runtime evaluation failed after cache growth");
    for (uint32_t i = 0; i < count; ++i) {
      Check(result[i] == input_value + i &&
                result[count + i] == input_value + count - i - 1,
            "runtime cache lost a value or retained data from a previous draw");
    }
  }
}

void TestExtractedRuntimeEvaluation() {
  Fixture fixture;
  const auto input = fixture.UserData(0);
  constexpr uint32_t count = 4096;
  for (uint32_t i = 0; i < count; ++i) {
    const auto value = fixture.Emit(ValueOpcode::IAdd32, {input, Value(i)});
    fixture.program.srt_reads.push_back({value, i});
  }
  fixture.program.srt_plan_complete = true;
  auto plan = ExtractResourcePlan(fixture.program);
  std::vector<uint32_t> result;
  for (uint32_t input_value : {17u, 0xffffff00u}) {
    Check(WalkSrt(plan, {.user_data = std::span(&input_value, 1)}, result),
          "extracted runtime evaluation failed");
    Check(result.size() == count, "extracted runtime evaluation lost outputs");
    for (uint32_t i = 0; i < count; ++i) {
      Check(result[i] == input_value + i, "extracted runtime cache returned stale data");
    }
  }
}

void TestImagesSamplersAndAliases() {
  Fixture fixture;
  std::array<Value, 8> image_words;
  for (uint32_t index = 0; index < image_words.size(); index++) {
    image_words[index] = fixture.UserData(index);
  }
  const auto image_address = fixture.ImageAddress();
  const std::array<Value, 4> sampler0{Value(0u), Value(1u), Value(2u),
                                      Value(0x1111u)};
  const std::array<Value, 4> sampler1{Value(0u), Value(1u), Value(2u),
                                      Value(0x2222u)};

  auto AddSample = [&](uint32_t pc, uint32_t sample_flags,
                       const auto &sampler_words) {
    const auto image = fixture.Image(image_words, pc);
    const auto sampler = fixture.Sampler(sampler_words, pc);
    MemoryInfo memory;
    memory.kind = ResourceKind::Image;
    memory.image_dimension = Decoder::ImageDimension::Dim2D;
    memory.image_sample_flags = sample_flags;
    fixture.Emit(ValueOpcode::ImageSampleRaw, {image, sampler, image_address},
                 fixture.AddMemory(memory, pc));
    return std::pair{image, sampler};
  };
  const auto normal = AddSample(4, 0, sampler0);
  const auto repeated = AddSample(8, 0, sampler1);
  const auto compare = AddSample(12, Decoder::ImageSampleFlagCompare, sampler0);

  const auto storage = fixture.Image(image_words, 16);
  MemoryInfo storage_memory;
  storage_memory.kind = ResourceKind::Image;
  storage_memory.image_dimension = Decoder::ImageDimension::Dim2D;
  fixture.Emit(ValueOpcode::ImageAtomicIAdd32,
               {storage, image_address, Value(1u), Value(true)},
               fixture.AddMemory(storage_memory, 16));

  const auto buffer = fixture.Buffer(
      {image_words[0], image_words[1], image_words[2], image_words[3]}, 20);
  MemoryInfo buffer_memory;
  buffer_memory.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {buffer, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(buffer_memory, 20));
  fixture.PlanAndTrack();

  Check(fixture.program.info.images.size() == 3 &&
            fixture.program.info.samplers.size() == 1 &&
            fixture.program.info.sampled_pairs.size() == 2,
        "typed image view classes or samplers were deduplicated incorrectly");
  Check(normal.first.Instruction()->Flags<uint32_t>() ==
                repeated.first.Instruction()->Flags<uint32_t>() &&
            compare.first.Instruction()->Flags<uint32_t>() !=
                normal.first.Instruction()->Flags<uint32_t>(),
        "image handles did not receive view-class indices");
  Check(normal.second.Instruction()->Flags<uint32_t>() == 0 &&
            repeated.second.Instruction()->Flags<uint32_t>() == 0,
        "unused sampler border colors prevented source interning");
  const auto sampler_source = fixture.program.info.samplers[0].source;
  Check(fixture.program.descriptor_sources[sampler_source].dwords[3].U32() == 0,
        "unused sampler border color was not canonicalized");
  Check(fixture.program.info.buffers[0].image_alias == 0,
        "buffer/image descriptor alias was not linked");
}

void TestSampleAdjustSamplerScratch() {
  Fixture fixture(ShaderType::Pixel);
  const auto active = fixture.Emit(
      ValueOpcode::IEqual32, {fixture.Emit(ValueOpcode::LaneId), Value(0u)});
  const auto lane =
      fixture.Emit(ValueOpcode::SelectU32, {active, Value(1u), Value(0u)});
  const auto low =
      fixture.Emit(ValueOpcode::BitwiseAnd32, {lane, Value(0xffu)});
  const auto high =
      fixture.Emit(ValueOpcode::BitwiseAnd32, {lane, Value(0xffu)});
  const auto quads = fixture.Emit(
      ValueOpcode::BitwiseOr32,
      {low, fixture.Emit(ValueOpcode::ShiftLeftLogical32, {high, Value(8u)})});
  const auto scratch =
      fixture.Emit(ValueOpcode::ShiftLeftLogical32, {quads, Value(12u)});
  const auto word3 =
      fixture.Emit(ValueOpcode::BitwiseOr32, {fixture.UserData(3), scratch});
  const auto image = fixture.Image({Value(0u), Value(0u), Value(0u), Value(0u),
                                    Value(0u), Value(0u), Value(0u), Value(0u)},
                                   0x1ec);
  const auto sampler = fixture.Sampler(
      {fixture.UserData(0), fixture.UserData(1), fixture.UserData(2), word3},
      0x1ec);
  MemoryInfo memory;
  memory.kind = ResourceKind::Image;
  memory.image_dimension = Decoder::ImageDimension::Dim2D;
  memory.image_sample_flags = Decoder::ImageSampleFlagAdjust;
  fixture.Emit(ValueOpcode::ImageSampleRaw,
               {image, sampler, fixture.ImageAddress()},
               fixture.AddMemory(memory, 0x1ec));
  fixture.PlanAndTrack();

  const auto source = fixture.program.info.samplers[0].source;
  const auto stored = fixture.program.descriptor_sources[source]
                          .dwords[3]
                          .Resolve()
                          .TryInstruction();
  Check(stored != nullptr && stored->GetOpcode() == ValueOpcode::GetUserData,
        "SampleAdjust reserved scratch remained in sampler identity");
  std::array<uint32_t, 4> user_data{4u, 1u, 2u, 0x80000abcu};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue descriptor;
  Check(EvaluateDescriptorSource(fixture.program, source, runtime, descriptor) &&
            descriptor.dwords[3] == 0x80000abcu,
        "SampleAdjust canonicalization lost sampler border fields");

  const auto CheckRejected = [](uint32_t flags, uint32_t shift,
                                const char *message) {
    Fixture rejected(ShaderType::Pixel);
    const auto condition = rejected.Emit(
        ValueOpcode::IEqual32, {rejected.Emit(ValueOpcode::LaneId), Value(0u)});
    const auto bit = rejected.Emit(ValueOpcode::SelectU32,
                                   {condition, Value(1u), Value(0u)});
    const auto dynamic =
        rejected.Emit(ValueOpcode::ShiftLeftLogical32, {bit, Value(shift)});
    const auto dynamic_word3 = rejected.Emit(ValueOpcode::BitwiseOr32,
                                             {rejected.UserData(3), dynamic});
    const auto rejected_image =
        rejected.Image({Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
                        Value(0u), Value(0u), Value(0u)},
                       0x200);
    const auto rejected_sampler =
        rejected.Sampler({rejected.UserData(0), rejected.UserData(1),
                          rejected.UserData(2), dynamic_word3},
                         0x200);
    MemoryInfo rejected_memory;
    rejected_memory.kind = ResourceKind::Image;
    rejected_memory.image_dimension = Decoder::ImageDimension::Dim2D;
    rejected_memory.image_sample_flags = flags;
    rejected.Emit(ValueOpcode::ImageSampleRaw,
                  {rejected_image, rejected_sampler, rejected.ImageAddress()},
                  rejected.AddMemory(rejected_memory, 0x200));
    BuildSrtPlan(rejected.program);
    CheckFatal([&] { TrackResources(rejected.program); },
               "not a valid runtime value", message);
  };
  CheckRejected(0u, 12u,
                "ordinary sampling accepted SampleAdjust reserved scratch");
  CheckRejected(Decoder::ImageSampleFlagAdjust, 30u,
                "SampleAdjust canonicalization discarded border-mode bits");
}

void TestFmaskLoadSpecialization() {
  namespace Prospero = Libs::Graphics::Prospero;
  Fixture fixture;
  std::array<Value, 8> words;
  for (uint32_t i = 0; i < words.size(); i++) {
    words[i] = fixture.UserData(i);
  }
  const auto fmask = fixture.Image(words, 4);
  const auto active = fixture.Emit(ValueOpcode::IEqual32,
                                    {fixture.UserData(8), Value(0u)});
  MemoryInfo load;
  load.kind = ResourceKind::Image;
  load.image_dimension = Decoder::ImageDimension::Dim2D;
  load.image_address_components = 2;
  load.dmask = 1;
  const auto mapping = fixture.Emit(
      ValueOpcode::ImageRead, {fmask, fixture.ImageAddress(), active},
      fixture.AddMemory(load, 4));
  const auto ordinary = fixture.Image(
      {Value(0x2000u),
       Value(static_cast<uint32_t>(Prospero::BufferFormat::k8UInt) << 20u),
       Value(3u | (3u << 14u)),
       Value(Libs::Graphics::DstSel(4, 5, 6, 7) |
             (static_cast<uint32_t>(Prospero::ImageType::kColor2D) << 28u)),
       Value(0u), Value(0u), Value(0u), Value(0u)}, 8);
  const auto ordinary_flags = fixture.AddMemory(load, 8);
  const auto color = fixture.Emit(
      ValueOpcode::ImageRead, {ordinary, fixture.ImageAddress(), Value(true)},
      ordinary_flags);
  const auto output = fixture.Buffer(
      {Value(0x3000u), Value(0u), Value(12u), Value(0u)}, 12);
  MemoryInfo store;
  store.kind = ResourceKind::Buffer;
  Value result;
  for (uint32_t i = 0; i < 2; i++) {
    const auto value = fixture.Emit(
        ValueOpcode::CompositeExtractU32x4,
        {i == 0 ? mapping : color, Value(0u)});
    fixture.Emit(ValueOpcode::StoreBufferU32,
                 {output, Value(0u), Value(i * 4u), Value(0u), value, Value(true)},
                 fixture.AddMemory(store, 12 + i * 4u));
    if (i == 0) result = value;
  }
  fixture.PlanAndTrack();
  const auto plan = ExtractResourcePlan(fixture.program);
  std::array<uint32_t, 9> user_data{
      0x303ac300u, 0xca100000u, 0x021bc3bfu, 0x91800004u,
      0u, 0x00700000u, 0u, 0u};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {.user_data = user_data}, snapshot,
                             specialization),
        "FMASK resources did not materialize");
  ApplyResourceSpecialization(fixture.program, specialization);
  RemoveIdentities(fixture.program.blocks);
  EliminateDeadCode(fixture.program.blocks);
  Check(fixture.program.info.images.size() == 1 && snapshot.images.size() == 1 &&
            snapshot.images[0].dwords[0] == 0x2000u &&
            ordinary.Instruction()->Flags<uint32_t>() == 0 &&
            fixture.program.memory_info[ordinary_flags.index].resource == 0,
        "FMASK removal did not preserve the remaining image and runtime descriptor");
  const auto *vector = result.Instruction()->Arg(0).Resolve().TryInstruction();
  Check(vector != nullptr &&
            vector->GetOpcode() == ValueOpcode::CompositeConstructU32x4,
        "FMASK load did not lower to a value vector");
  result = vector->Arg(0);
  uint32_t value = 0;
  Check(EvaluateUniformValues(plan, {&result, 1}, {.user_data = user_data}, {&value, 1}) &&
            value == 0x76543210u,
        "FMASK load did not return the native sample-to-fragment mapping");
  user_data[8] = 1;
  Check(EvaluateUniformValues(plan, {&result, 1}, {.user_data = user_data}, {&value, 1}) &&
            value == 0u,
        "inactive FMASK load did not preserve the execution mask");
  ShaderComputeInputInfo compute{};
  CollectShaderInfo(fixture.program, {.compute = &compute});
  AllocateBindings(fixture.program);
  const auto kind = DescriptorBindingForImage(fixture.program.info.images[0]);
  Check(kind.has_value() &&
            FindBinding(fixture.program.bindings, *kind)->resources ==
                std::vector<uint32_t>{0},
        "FMASK allocated an ordinary image descriptor");
  user_data[8] = 0;
  user_data[1] = static_cast<uint32_t>(Prospero::BufferFormat::k8UInt) << 20u;
  ResourceSpecialization rebound;
  Check(MaterializeResources(plan, {.user_data = user_data}, snapshot, rebound) &&
            rebound != specialization && snapshot.images.size() == 2,
        "rebinding FMASK as a texture reused the metadata specialization");
}

void TestDynamicStorageMipTracking() {
  Fixture fixture;
  std::array<Value, 8> image_words;
  for (uint32_t index = 0; index < image_words.size(); index++) {
    image_words[index] = fixture.UserData(index);
  }
  const auto data = fixture.Emit(ValueOpcode::CompositeConstructU32x4,
                                 {Value(1u), Value(2u), Value(3u), Value(4u)});
  const auto AddStore = [&](uint32_t pc, bool has_mip, Value lod) {
    const auto handle = fixture.Image(image_words, pc);
    const auto address = fixture.Emit(
        ValueOpcode::MakeImageAddress,
        {Value(0u), Value(0u), lod, Value(0u), Value(0u), Value(0u), Value(0u),
         Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u)});
    MemoryInfo memory;
    memory.kind = ResourceKind::Image;
    memory.image_dimension = Decoder::ImageDimension::Dim2D;
    memory.image_address_components = has_mip ? 3u : 2u;
    memory.image_has_mip = has_mip;
    const auto flags = fixture.AddMemory(memory, pc);
    fixture.Emit(ValueOpcode::ImageWrite, {handle, address, data, Value(true)},
                 flags);
    return std::pair{handle, flags.index};
  };

  const auto plain = AddStore(4, false, Value(0u));
  const auto mip1 = AddStore(8, true, Value(1u));
  const auto mip2 = AddStore(12, true, Value(2u));
  const auto dynamic = AddStore(16, true, fixture.UserData(8));
  fixture.PlanAndTrack();
  auto resource_plan = ExtractResourcePlan(fixture.program);

  const auto &images = fixture.program.info.images;
  Check(images.size() == 2 && images[0].mip_mode == ImageMipMode::None &&
            images[0].mip_count == 1 &&
            images[1].mip_mode == ImageMipMode::DynamicStorage &&
            images[1].mip_count == 1,
        "storage mip writes did not share one dynamic logical resource");
  Check(plain.first.Instruction()->Flags<uint32_t>() == 0 &&
            mip1.first.Instruction()->Flags<uint32_t>() == 1 &&
            mip2.first.Instruction()->Flags<uint32_t>() == 1 &&
            dynamic.first.Instruction()->Flags<uint32_t>() == 1 &&
            fixture.program.memory_info[plain.second].resource == 0 &&
            fixture.program.memory_info[mip1.second].resource == 1 &&
            fixture.program.memory_info[mip2.second].resource == 1 &&
            fixture.program.memory_info[dynamic.second].resource == 1,
        "dynamic storage mip handles and memory metadata were not patched");

  DescriptorValue descriptor{};
  descriptor.dwords[0] = 0x1000u;
  descriptor.dwords[1] =
      static_cast<uint32_t>(
          Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float)
      << 20u;
  descriptor.dwords[2] = 3u | (3u << 14u);
  descriptor.dwords[3] =
      Libs::Graphics::DstSel(4, 5, 6, 7) | (1u << 12u) | (3u << 16u) |
      (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D)
       << 28u);
  descriptor.dwords[5] = 3u << 4u;
  descriptor.dword_count = 8;
  std::array<uint32_t, 9> user_data{};
  std::copy(descriptor.dwords.begin(), descriptor.dwords.end(),
            user_data.begin());
  user_data[8] = 2u;
  SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(resource_plan, runtime, snapshot,
                             specialization),
        "dynamic storage resources did not materialize");
  ApplyResourceSpecialization(fixture.program, specialization);
  Check(fixture.program.info.images[1].mip_count == 3 &&
            snapshot.images.size() == fixture.program.info.images.size(),
        "base-1 through last-3 dynamic storage range was not specialized");
  ShaderComputeInputInfo compute{};
  CollectShaderInfo(fixture.program, {.compute = &compute});
  AllocateBindings(fixture.program);
  const auto storage_kind = DescriptorBindingForImage(images[0]);
  Check(storage_kind.has_value(), "storage image has no descriptor binding");
  const auto *storage_binding =
      FindBinding(fixture.program.bindings, *storage_kind);
  Check(storage_binding != nullptr &&
            storage_binding->resources == std::vector<uint32_t>({0, 1, 1, 1}),
        "dynamic storage mip descriptors were not expanded consecutively");

  Fixture null_fixture;
  const auto null_handle = null_fixture.Image(
      {Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
       Value(0u), Value(0u)});
  const auto null_address = null_fixture.Emit(
      ValueOpcode::MakeImageAddress,
      {Value(0u), Value(0u), null_fixture.UserData(0), Value(0u), Value(0u),
       Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
       Value(0u), Value(0u)});
  MemoryInfo null_memory;
  null_memory.kind = ResourceKind::Image;
  null_memory.image_dimension = Decoder::ImageDimension::Dim2D;
  null_memory.image_address_components = 3u;
  null_memory.image_has_mip = true;
  const auto null_data = null_fixture.Emit(
      ValueOpcode::CompositeConstructU32x4,
      {Value(1u), Value(2u), Value(3u), Value(4u)});
  null_fixture.Emit(ValueOpcode::ImageWrite,
                    {null_handle, null_address, null_data, Value(true)},
                    null_fixture.AddMemory(null_memory, 4));
  null_fixture.PlanAndTrack();
  auto null_plan = ExtractResourcePlan(null_fixture.program);
  ResourceSnapshot null_snapshot;
  ResourceSpecialization null_specialization;
  const std::array<uint32_t, 1> null_user_data{0u};
  Check(MaterializeResources(null_plan, {.user_data = null_user_data},
                             null_snapshot, null_specialization),
        "canonical null dynamic storage image did not materialize");
  ApplyResourceSpecialization(null_fixture.program, null_specialization);
  Check(null_fixture.program.info.images[0].mip_count == 1 &&
            null_snapshot.images.size() == 1,
        "canonical null dynamic storage image did not retain one descriptor");

  auto changed_user_data = user_data;
  changed_user_data[3] =
      (changed_user_data[3] & ~(0xfu << 16u)) | (2u << 16u);
  ResourceSnapshot changed_snapshot;
  ResourceSpecialization changed_specialization;
  Check(MaterializeResources(resource_plan, {.user_data = changed_user_data},
                             changed_snapshot, changed_specialization) &&
            changed_specialization != specialization,
        "a changed dynamic storage mip count reused the specialization key");
  changed_user_data[3] =
      (changed_user_data[3] & ~((0xfu << 12u) | (0xfu << 16u))) |
      (4u << 12u) | (3u << 16u);
  const auto valid_snapshot = changed_snapshot;
  const auto valid_specialization = changed_specialization;
  Check(!MaterializeResources(resource_plan, {.user_data = changed_user_data},
                              changed_snapshot, changed_specialization) &&
            SameResourceSnapshot(changed_snapshot, valid_snapshot) &&
            changed_specialization == valid_specialization,
        "an inverted dynamic storage mip range was accepted or mutated output");
}

void TestSrtFlatteningAndRuntimeMemoization() {
  Fixture fixture;
  const auto base =
      fixture.Address(fixture.UserData(0), fixture.UserData(1), 4);
  MemoryInfo scalar;
  scalar.kind = ResourceKind::ScalarAddress;
  scalar.offset = 4;
  const auto read0 = fixture.Emit(ValueOpcode::LoadAddressU32,
                                  {base, Value(0u), Value(0u), Value(true)},
                                  fixture.AddMemory(scalar, 4));
  const auto descriptor0 =
      fixture.Buffer({read0, Value(0u), Value(64u), Value(0u)}, 12);
  const auto descriptor1 =
      fixture.Buffer({read0, Value(0u), Value(64u), Value(0u)}, 16);
  MemoryInfo buffer;
  buffer.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {descriptor0, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(buffer, 12));
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {descriptor1, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(buffer, 16));
  fixture.PlanAndTrack();

  Check(fixture.program.srt_reads.size() == 1,
        "shared typed scalar read did not receive one flat SRT slot");
  Check(fixture.program.info.buffers.size() == 1 &&
            !fixture.program.info.uses_dma,
        "planning-only scalar reads leaked into resource topology");
  Check(fixture.program.memory_info[0].planning_only,
        "canonical runtime scalar read was not marked planning-only");

  std::array<uint32_t, 2> user_data{0x1000u, 0u};
  TestMemory memory;
  memory.words[1] = 0xdeadbeefu;
  SrtRuntime runtime{.user_data = user_data,
                     .read_memory = ReadTestMemory,
                     .userdata = &memory};
  std::vector<DescriptorValue> descriptors;
  std::vector<uint32_t> flat;
  std::vector<uint8_t> active_sources;
  const uint32_t request = fixture.program.info.buffers[0].source;
  Check(EvaluateRuntimeSources(fixture.program, std::span{&request, 1}, runtime,
                               descriptors, flat, {}, active_sources),
        "typed runtime source evaluation failed");
  Check(descriptors.size() == 1 && descriptors[0].dwords[0] == 0xdeadbeefu &&
            flat == std::vector<uint32_t>{0xdeadbeefu} && memory.reads == 1,
        "descriptor and flat SRT evaluation did not share one memoized read");

  memory.reads = 0;
  memory.fail_after = 0;
  descriptors = {{{1u}, 1u}};
  flat = {2u};
  active_sources = {3u};
  Check(!EvaluateRuntimeSources(fixture.program, std::span{&request, 1},
                                runtime, descriptors, flat, {}, active_sources) &&
            descriptors == std::vector<DescriptorValue>{{{1u}, 1u}} &&
            flat == std::vector<uint32_t>{2u} &&
            active_sources == std::vector<uint8_t>{3u},
        "runtime evaluation failure was not transactional");

  ShaderComputeInputInfo compute{};
  CollectShaderInfo(fixture.program, {.compute = &compute});
  AllocateBindings(fixture.program);
  Check(FindBinding(fixture.program.bindings,
                    DescriptorBindingKind::FlattenedSrt) != nullptr,
        "flattened typed SRT reads did not receive a binding");
}

void TestDynamicSrtReadRemainsExplicit() {
  Fixture fixture;
  const auto base =
      fixture.Address(fixture.UserData(0), fixture.UserData(1), 4);
  MemoryInfo scalar;
  scalar.kind = ResourceKind::ScalarAddress;
  const auto read =
      fixture.Emit(ValueOpcode::LoadAddressU32,
                   {base, fixture.UserData(2), Value(0u), Value(true)},
                   fixture.AddMemory(scalar, 4));
  const auto descriptor =
      fixture.Buffer({read, Value(0u), Value(64u), Value(0u)}, 8);
  MemoryInfo buffer;
  buffer.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(buffer, 8));
  fixture.PlanAndTrack();

  Check(fixture.program.srt_reads.empty() &&
            fixture.program.dynamic_reads.size() == 1 &&
            fixture.program.info.uses_dma,
        "dynamic scalar read was incorrectly flattened or lost");
  std::array<uint32_t, 3> user_data{0x1000u, 0u, 4u};
  TestMemory memory;
  memory.words[1] = 0xabcdef01u;
  SrtRuntime runtime{.user_data = user_data,
                     .read_memory = ReadTestMemory,
                     .userdata = &memory};
  DescriptorValue value;
  Check(EvaluateDescriptorSource(fixture.program,
                                 fixture.program.info.buffers[0].source, runtime, value) &&
            value.dwords[0] == 0xabcdef01u && memory.reads == 1,
        "dynamic typed scalar descriptor source was not evaluated");

  ShaderComputeInputInfo compute{};
  CollectShaderInfo(fixture.program, {.compute = &compute});
  AllocateBindings(fixture.program);
  Check(FindBinding(fixture.program.bindings,
                    DescriptorBindingKind::FlattenedSrt) == nullptr &&
            FindBinding(fixture.program.bindings,
                        DescriptorBindingKind::BdaPagetable) != nullptr &&
            FindBinding(fixture.program.bindings,
                        DescriptorBindingKind::FaultBuffer) != nullptr,
        "dynamic scalar read received the wrong resource bindings");
  Check(fixture.program.bindings.memory_offset_dword ==
                fixture.program.bindings.user_data_registers.size() &&
            fixture.program.bindings.memory_offset_count == 1u &&
            fixture.program.bindings.ShaderDataDwords() ==
                fixture.program.bindings.memory_offset_dword + 1u,
        "unified memory-offset layout is inconsistent");
}

void TestPhiValidation() {
  Fixture fixture;
  auto *left = fixture.block;
  auto *right = fixture.AddBlock();
  auto *merge = fixture.AddBlock();
  left->AddBranch(merge);
  right->AddBranch(merge);
  auto &phi = merge->AppendNewInst(ValueOpcode::Phi, {},
                                   static_cast<uint64_t>(Type::U32));
  phi.AddPhiOperand(left, Value(1u));
  phi.AddPhiOperand(right, Value(2u));
  const auto word3 =
      fixture.Emit(ValueOpcode::UMin32, {Value(&phi), Value(0x100u)}, 0, merge);
  const auto handle = fixture.Emit(ValueOpcode::GetBufferResource,
                                   {Value(0u), Value(0u), Value(0u), word3},
                                   MemoryFlags{0, 20}, merge);
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {handle, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(memory, 20), merge);

  BuildSrtPlan(fixture.program);
  CheckFatal([&] { TrackResources(fixture.program); }, "not a valid runtime value",
             "control-dependent descriptor phi was accepted");
  Check(!fixture.program.resource_tracking_complete &&
            fixture.program.info.buffers.empty() &&
            fixture.program.descriptor_sources.empty(),
        "control-dependent descriptor phi was not rejected transactionally");
}

Fixture MakeNullPathSamplerFixture(uint32_t null_dword) {
  Fixture fixture;
  auto *loaded = fixture.block;
  auto *nulled = fixture.AddBlock();
  auto *merge = fixture.AddBlock();
  loaded->AddBranch(merge);
  nulled->AddBranch(merge);

  std::array<Value, 4> sampler_words{};
  for (uint32_t index = 0; index < sampler_words.size(); index++) {
    const auto word = fixture.UserData(index);
    const bool flipped = index == null_dword;
    auto &phi = merge->AppendNewInst(ValueOpcode::Phi, {},
                                     static_cast<uint64_t>(Type::U32));
    phi.AddPhiOperand(loaded, flipped ? Value(0u) : word);
    phi.AddPhiOperand(nulled, flipped ? word : Value(0u));
    sampler_words[index] = Value(&phi);
  }
  std::array<Value, 8> image_words{};
  for (uint32_t index = 0; index < image_words.size(); index++) {
    image_words[index] = fixture.UserData(index + 8u);
  }
  const auto image = fixture.Emit(
      ValueOpcode::GetImageResource,
      {image_words[0], image_words[1], image_words[2], image_words[3],
       image_words[4], image_words[5], image_words[6], image_words[7]},
      MemoryFlags{0, 32}, merge);
  const auto sampler = fixture.Emit(
      ValueOpcode::GetSamplerResource,
      {sampler_words[0], sampler_words[1], sampler_words[2], sampler_words[3]},
      MemoryFlags{0, 32}, merge);
  MemoryInfo memory;
  memory.kind = ResourceKind::Image;
  memory.image_dimension = Decoder::ImageDimension::Dim2D;
  fixture.Emit(ValueOpcode::ImageSampleRaw,
               {image, sampler, fixture.ImageAddress()},
               fixture.AddMemory(memory, 32), merge);
  return fixture;
}

void TestNullDescriptorPathCollapse() {
  auto fixture = MakeNullPathSamplerFixture(UINT32_MAX);
  fixture.PlanAndTrack();
  Check(fixture.program.info.samplers.size() == 1,
        "sampler nulled on one branch was not tracked");

  const std::array<uint32_t, 16> user_data{0x11u, 0x22u, 0x33u, 0x44u};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue result;
  Check(EvaluateDescriptorSource(fixture.program,
                                 fixture.program.info.samplers[0].source,
                                 runtime, result),
        "sampler nulled on one branch did not evaluate");
  Check(result.dwords[0] == 0x11u && result.dwords[1] == 0x22u &&
            result.dwords[2] == 0x33u && result.dwords[3] == 0x44u,
        "sampler nulled on one branch did not collapse to the loaded value");
}

void TestMixedNullDescriptorPathsRejected() {
  auto fixture = MakeNullPathSamplerFixture(1u);
  BuildSrtPlan(fixture.program);
  CheckFatal([&] { TrackResources(fixture.program); },
             "not a valid runtime value",
             "descriptor nulled on mismatched branches was accepted");
  Check(!fixture.program.resource_tracking_complete &&
            fixture.program.info.samplers.empty(),
        "descriptor nulled on mismatched branches was partially accepted");
}

struct WaterfallFixture {
  Fixture fixture;
  Inst *phi = nullptr;
  Value entry;
  Value key;

  WaterfallFixture(uint32_t one = 1u, uint32_t lane_mask = 31u,
                   ValueOpcode clear = ValueOpcode::BitwiseXor32,
                   bool entry_in_loop = false, bool body = true,
                   bool ballot_from_key = true, bool invert_bit = false,
                   bool immediate_table = false) {
    auto *entry_block = fixture.block;
    auto *loop = fixture.AddBlock();
    entry_block->AddBranch(loop);
    loop->AddBranch(loop);

    key = fixture.UserData(4);
    const auto key_bits =
        fixture.Emit(ValueOpcode::BitwiseAnd32, {key, Value(31u)});
    const auto bit =
        fixture.Emit(ValueOpcode::ShiftLeftLogical32, {Value(1u), key_bits});
    const auto low = fixture.Emit(ValueOpcode::ReadLane, {bit, Value(31u)});
    const auto high = fixture.Emit(ValueOpcode::ReadLane, {bit, Value(63u)});
    const auto ballot = fixture.Emit(ValueOpcode::BitwiseOr32, {low, high});

    phi = &loop->AppendNewInst(ValueOpcode::Phi, {},
                               static_cast<uint64_t>(Type::U32));
    const auto mask = Value(phi);
    const auto lsb = fixture.Emit(ValueOpcode::FindILsb32, {mask}, 0, loop);
    const auto masked = fixture.Emit(ValueOpcode::BitwiseAnd32,
                                     {lsb, Value(lane_mask)}, 0, loop);
    const auto stepped = fixture.Emit(ValueOpcode::ShiftLeftLogical32,
                                      {Value(one), masked}, 0, loop);
    const auto clear_bit = invert_bit
        ? fixture.Emit(ValueOpcode::BitwiseNot32, {stepped}, 0, loop)
        : stepped;
    const auto cleared = fixture.Emit(clear, {clear_bit, mask}, 0, loop);
    if (body) {
      fixture.Emit(ValueOpcode::IEqual32, {lsb, key}, 0, loop);
      const auto scaled = fixture.Emit(ValueOpcode::ShiftLeftLogical32,
                                       {lsb, Value(5u)}, 0, loop);
      const auto based =
          immediate_table ? scaled : fixture.Emit(ValueOpcode::IAdd32, {scaled, Value(0x158u)}, 0, loop);
      const auto second =
          immediate_table ? based : fixture.Emit(ValueOpcode::IAdd32, {Value(16u), based}, 0, loop);
      std::array<Value, 8> dwords{};
      for (uint32_t index = 0; index < dwords.size(); index++) {
        const auto address = fixture.Address(fixture.UserData(0),
                                             fixture.UserData(1), 0x118);
        dwords[index] = fixture.Emit(
            ValueOpcode::LoadAddressU32,
            {address, index < 4u ? based : second, Value(0u), Value(true)},
            immediate_table ? fixture.AddMemory(MemoryInfo{.kind = ResourceKind::ScalarAddress, .offset = 0x158u + index * 4u}, 0x118)
                            : MemoryFlags{0, 0x118}, loop);
      }
      fixture.Emit(ValueOpcode::GetImageResource,
                   {dwords[0], dwords[1], dwords[2], dwords[3], dwords[4],
                    dwords[5], dwords[6], dwords[7]},
                   MemoryFlags{0, 0x118}, loop);
    }
    entry = entry_in_loop     ? cleared
            : ballot_from_key ? ballot
                              : fixture.UserData(7);
    phi->AddPhiOperand(entry_block, entry);
    phi->AddPhiOperand(loop, cleared);
  }
};

void TestDenseIndirectImageMaterialization() {
  Fixture fixture;
  const auto low = fixture.UserData(0);
  const auto high = fixture.UserData(1);
  const auto image_handle = fixture.Image({low, high, Value(0u), Value(0u),
                                           Value(0u), Value(0u), Value(0u),
                                           Value(0u)},
                                          0x118);
  MemoryInfo memory;
  memory.kind = ResourceKind::Image;
  memory.image_dimension = Decoder::ImageDimension::Dim2D;
  fixture.Emit(ValueOpcode::ImageSampleRaw,
               {image_handle, fixture.Sampler({Value(0u), Value(0u), Value(0u),
                                               Value(0u)},
                                              0x118),
                fixture.ImageAddress()},
               fixture.AddMemory(memory, 0x118));
  fixture.PlanAndTrack();
  Check(fixture.program.info.images.size() == 1,
        "dense indirect image fixture did not track one image");

  constexpr uint32_t kEntries = 4;
  constexpr uint32_t kTable = 0x100;
  auto &source = fixture.program
                     .descriptor_sources[fixture.program.info.images[0].source];
  source.dword_count = 2;
  source.dwords = {low, high};
  source.indirect_image = DescriptorSource::IndirectImage{
      .heap_source = fixture.program.info.images[0].source,
      .table_offset = kTable,
      .key_bound = kEntries};

  LinearTestMemory memory_image;
  std::array<uint32_t, 8> descriptor{};
  descriptor[0] = 0x40u;
  descriptor[1] =
      static_cast<uint32_t>(Libs::Graphics::Prospero::BufferFormat::k8_8_8_8UNorm)
      << 20u;
  descriptor[3] =
      Libs::Graphics::DstSel(4, 5, 6, 7) |
      (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D)
       << 28u);
  for (uint32_t entry = 0; entry < kEntries; entry++) {
    for (uint32_t dword = 0; dword < descriptor.size(); dword++) {
      const auto at = (kTable + entry * 32u) / 4u + dword;
      memory_image.words[at] = descriptor[dword];
    }
    memory_image.words[(kTable + entry * 32u) / 4u] += entry;
  }

  const std::array<uint32_t, 2> user_data{
      static_cast<uint32_t>(memory_image.base), 0u};
  SrtRuntime runtime{.user_data = user_data,
                     .userdata = &memory_image,
                     .read_specialization_memory = ReadLinearTestMemory};
  auto resource_plan = ExtractResourcePlan(fixture.program);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(resource_plan, runtime, snapshot, specialization),
        "dense indirect image table did not materialize");
  const auto mapping = specialization.images.empty()
                           ? UINT32_MAX
                           : specialization.images[0].indirect_mapping_offset;
  Check(specialization.images.size() == kEntries &&
            snapshot.images.size() == kEntries &&
            specialization.images[0].indirect_root == 0 &&
            mapping != UINT32_MAX &&
            mapping + 1u + kEntries * 2u <= snapshot.flattened_srt.size() &&
            snapshot.flattened_srt[mapping] == kEntries,
        "dense indirect image table did not enumerate its bound");
  const auto candidate_of = [&](const ResourceSnapshot &view, uint32_t key) {
    const auto offset = mapping + 1u + key * 2u;
    Check(view.flattened_srt[offset] == key,
          "dense indirect image keys are not the dense range");
    return view.images[view.flattened_srt[offset + 1u]];
  };
  for (uint32_t entry = 0; entry < kEntries; entry++) {
    Check(candidate_of(snapshot, entry).dwords[0] == 0x40u + entry,
          "dense indirect image read the wrong table entry");
  }

  memory_image.words[(kTable + 2u * 32u) / 4u + 4u] = (7u << 16u) | 3u;
  ResourceSnapshot stale_snapshot;
  ResourceSpecialization stale_specialization;
  Check(MaterializeResources(resource_plan, runtime, stale_snapshot,
                             stale_specialization),
        "dense indirect image table with a stale slot did not materialize");
  Check(std::ranges::all_of(candidate_of(stale_snapshot, 2u).dwords,
                            [](uint32_t dword) { return dword == 0u; }),
        "an out-of-range array base was kept as a usable descriptor");
  memory_image.words[(kTable + 2u * 32u) / 4u + 4u] = 0x84b1b500u;
  ResourceSnapshot foreign_snapshot;
  ResourceSpecialization foreign_specialization;
  Check(MaterializeResources(resource_plan, runtime, foreign_snapshot,
                             foreign_specialization),
        "dense indirect image table with a foreign slot did not materialize");
  Check(std::ranges::all_of(candidate_of(foreign_snapshot, 2u).dwords,
                            [](uint32_t dword) { return dword == 0u; }),
        "a slot with reserved texture bits set was kept as a usable descriptor");
  fixture.program.descriptor_sources[fixture.program.info.images[0].source]
      .indirect_image->key_bound = 0x10000u;
  auto unreadable_plan = ExtractResourcePlan(fixture.program);
  Check(!MaterializeResources(unreadable_plan, runtime, snapshot,
                              specialization),
        "dense indirect image accepted a table it could not read");
}

void TestLoopBoundedDenseIndirectImage() {
  Fixture fixture;
  auto *entry = fixture.block;
  auto *header = fixture.AddBlock();
  auto *body = fixture.AddBlock();
  auto *latch = fixture.AddBlock();
  auto *exit = fixture.AddBlock();
  entry->AddBranch(header);
  header->AddBranch(body);
  header->AddBranch(exit);
  body->AddBranch(latch);
  latch->AddBranch(header);
  const auto block_id = [&](Block *block) {
    return static_cast<uint32_t>(
        std::ranges::find(fixture.program.blocks, block) -
        fixture.program.blocks.begin());
  };
  const auto terminate = [&](Block *block, Block *taken, Block *not_taken,
                             Value condition) {
    auto &info = fixture.program.block_info[block_id(block)];
    using Kind = Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
    info.terminator.kind =
        not_taken != nullptr ? Kind::ConditionalBranch : Kind::Branch;
    info.terminator.true_block = block_id(taken);
    if (not_taken != nullptr) {
      info.terminator.false_block = block_id(not_taken);
    }
    info.condition = condition;
  };

  const auto low = fixture.UserData(0);
  const auto high = fixture.UserData(1);
  const auto count = fixture.UserData(5);
  const auto other = fixture.Emit(
      ValueOpcode::IEqual32,
      {fixture.Emit(ValueOpcode::BitwiseAnd32,
                    {fixture.Emit(ValueOpcode::LaneId), Value(1u)}),
       Value(0u)});
  auto *phi = &header->AppendNewInst(ValueOpcode::Phi, {},
                                     static_cast<uint64_t>(Type::U32));
  const auto key = Value(phi);
  const auto below =
      fixture.Emit(ValueOpcode::SLessThan32, {key, count}, 0, header);
  const auto gated = fixture.Emit(ValueOpcode::SelectU1,
                                  {below, other, Value(false)}, 0, header);
  const auto not_other =
      fixture.Emit(ValueOpcode::LogicalNot, {other}, 0, header);
  const auto either =
      fixture.Emit(ValueOpcode::LogicalOr, {gated, not_other}, 0, header);
  const auto active =
      fixture.Emit(ValueOpcode::LogicalAnd, {other, either}, 0, header);
  const auto any_active =
      fixture.Emit(ValueOpcode::AnyLane, {active}, 0, header);
  const auto leave =
      fixture.Emit(ValueOpcode::LogicalNot, {any_active}, 0, header);
  fixture.Emit(ValueOpcode::Reference, {leave}, 0, header);
  terminate(entry, header, nullptr, Value());
  terminate(header, exit, body, leave);
  terminate(body, latch, nullptr, Value());
  terminate(latch, header, nullptr, Value());

  const auto scaled =
      fixture.Emit(ValueOpcode::ShiftLeftLogical32, {key, Value(5u)}, 0, body);
  const auto based =
      fixture.Emit(ValueOpcode::IAdd32, {scaled, Value(0x6b0u)}, 0, body);
  const auto second =
      fixture.Emit(ValueOpcode::IAdd32, {Value(16u), based}, 0, body);
  const auto first_half = fixture.Address(low, high, 0x29c);
  const auto second_half = fixture.Address(low, high, 0x29c);
  std::array<Value, 8> dwords{};
  for (uint32_t index = 0; index < dwords.size(); index++) {
    MemoryInfo word;
    word.kind = ResourceKind::ScalarAddress;
    word.offset = (index % 4u) * sizeof(uint32_t);
    dwords[index] = fixture.Emit(
        ValueOpcode::LoadAddressU32,
        {index < 4u ? first_half : second_half, index < 4u ? based : second,
         Value(0u), Value(true)},
        fixture.AddMemory(word, 0x29c), body);
  }
  const auto image_handle =
      fixture.Emit(ValueOpcode::GetImageResource,
                   {dwords[0], dwords[1], dwords[2], dwords[3], dwords[4],
                    dwords[5], dwords[6], dwords[7]},
                   MemoryFlags{0, 0x29c}, body);
  MemoryInfo memory;
  memory.kind = ResourceKind::Image;
  memory.image_dimension = Decoder::ImageDimension::Dim2D;
  fixture.Emit(ValueOpcode::ImageSampleRaw,
               {image_handle,
                fixture.Sampler({Value(0u), Value(0u), Value(0u), Value(0u)},
                                0x29c),
                fixture.ImageAddress()},
               fixture.AddMemory(memory, 0x29c), body);
  const auto next = fixture.Emit(ValueOpcode::IAdd32, {key, Value(1u)}, 0, latch);
  phi->AddPhiOperand(entry, Value(0u));
  phi->AddPhiOperand(latch, next);
  fixture.PlanAndTrack();

  Check(fixture.program.info.images.size() == 1,
        "loop-bounded dense image was not tracked as one image");
  const auto &source = fixture.program
                           .descriptor_sources[fixture.program.info.images[0].source];
  Check(source.indirect_image.has_value() &&
            source.indirect_image->key_bound != 0u &&
            source.indirect_image->table_offset == 0x6b0u &&
            source.indirect_image->bound_source !=
                DescriptorSource::IndirectImage::NoBoundSource &&
            source.indirect_image->bound_signed,
        "loop counter key was not planned as a runtime-bounded dense table");

  LinearTestMemory memory_image;
  std::array<uint32_t, 8> descriptor{};
  descriptor[0] = 0x40u;
  descriptor[1] =
      static_cast<uint32_t>(Libs::Graphics::Prospero::BufferFormat::k8_8_8_8UNorm)
      << 20u;
  descriptor[3] =
      Libs::Graphics::DstSel(4, 5, 6, 7) |
      (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D)
       << 28u);
  for (uint32_t entry_index = 0; entry_index < 3u; entry_index++) {
    for (uint32_t dword = 0; dword < descriptor.size(); dword++) {
      const auto at = (0x6b0u + entry_index * 32u) / 4u + dword;
      memory_image.words[at] = descriptor[dword];
    }
    memory_image.words[(0x6b0u + entry_index * 32u) / 4u] += entry_index;
  }
  std::array<uint32_t, 8> user_data{static_cast<uint32_t>(memory_image.base),
                                    0u, 0u, 0u, 0u, 3u, 0u, 0u};
  SrtRuntime runtime{.user_data = user_data,
                     .userdata = &memory_image,
                     .read_specialization_memory = ReadLinearTestMemory};
  auto resource_plan = ExtractResourcePlan(fixture.program);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(resource_plan, runtime, snapshot, specialization),
        "loop-bounded dense table did not materialize");
  const auto mapping = specialization.images.empty()
                           ? UINT32_MAX
                           : specialization.images[0].indirect_mapping_offset;
  Check(specialization.images.size() == 3u && snapshot.images.size() == 3u &&
            mapping != UINT32_MAX &&
            mapping + 1u + 3u * 2u <= snapshot.flattened_srt.size() &&
            snapshot.flattened_srt[mapping] == 3u,
        "loop-bounded dense table did not enumerate the runtime count");
  for (uint32_t entry_index = 0; entry_index < 3u; entry_index++) {
    const auto offset = mapping + 1u + entry_index * 2u;
    Check(snapshot.flattened_srt[offset] == entry_index &&
              snapshot.images[snapshot.flattened_srt[offset + 1u]].dwords[0] ==
                  0x40u + entry_index,
          "loop-bounded dense table read the wrong entry");
  }
  user_data[5] = 0x80000000u;
  ResourceSnapshot zero_snapshot;
  ResourceSpecialization zero_specialization;
  Check(MaterializeResources(resource_plan, runtime, zero_snapshot,
                             zero_specialization),
        "zero-trip loop table did not materialize");
  Check(zero_specialization.images.size() == 1u &&
            zero_snapshot.images.size() == 1u &&
            std::ranges::all_of(zero_snapshot.images[0].dwords,
                                [](uint32_t dword) { return dword == 0u; }),
        "a zero-trip loop enumerated live descriptors");
  user_data[5] = 5000u;
  Check(!MaterializeResources(resource_plan, runtime, snapshot, specialization),
        "a loop bound past the enumeration cap was accepted");
}

void TestFindLsbDenseIndirectImage() {
  Fixture fixture;
  auto *entry = fixture.block;
  auto *header = fixture.AddBlock();
  auto *body = fixture.AddBlock();
  auto *latch = fixture.AddBlock();
  auto *exit = fixture.AddBlock();
  entry->AddBranch(header);
  header->AddBranch(body);
  header->AddBranch(exit);
  body->AddBranch(latch);
  latch->AddBranch(header);
  const auto block_id = [&](Block *block) {
    return static_cast<uint32_t>(
        std::ranges::find(fixture.program.blocks, block) -
        fixture.program.blocks.begin());
  };
  const auto terminate = [&](Block *block, Block *taken, Block *not_taken,
                             Value condition) {
    auto &info = fixture.program.block_info[block_id(block)];
    using Kind = Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
    info.terminator.kind =
        not_taken != nullptr ? Kind::ConditionalBranch : Kind::Branch;
    info.terminator.true_block = block_id(taken);
    if (not_taken != nullptr) {
      info.terminator.false_block = block_id(not_taken);
    }
    info.condition = condition;
  };

  const auto low = fixture.UserData(0);
  const auto high = fixture.UserData(1);
  const auto initial = fixture.UserData(5);
  auto *phi = &header->AppendNewInst(ValueOpcode::Phi, {},
                                     static_cast<uint64_t>(Type::U32));
  const auto mask = Value(phi);
  const auto nonzero =
      fixture.Emit(ValueOpcode::INotEqual32, {Value(0u), mask}, 0, header);
  const auto leave = fixture.Emit(ValueOpcode::LogicalNot, {nonzero}, 0, header);
  fixture.Emit(ValueOpcode::Reference, {leave}, 0, header);
  terminate(entry, header, nullptr, Value());
  terminate(header, exit, body, leave);
  terminate(body, latch, nullptr, Value());
  terminate(latch, header, nullptr, Value());

  const auto key = fixture.Emit(ValueOpcode::FindILsb32, {mask}, 0, body);
  const auto scaled =
      fixture.Emit(ValueOpcode::ShiftLeftLogical32, {key, Value(5u)}, 0, body);
  const auto table = fixture.Address(low, high, 0x29c);
  std::array<Value, 8> dwords{};
  for (uint32_t index = 0; index < dwords.size(); index++) {
    MemoryInfo word;
    word.kind = ResourceKind::ScalarAddress;
    word.offset = 0x6b0u + index * sizeof(uint32_t);
    dwords[index] = fixture.Emit(ValueOpcode::LoadAddressU32,
                                 {table, scaled, Value(0u), Value(true)},
                                 fixture.AddMemory(word, 0x29c), body);
  }
  const auto image_handle =
      fixture.Emit(ValueOpcode::GetImageResource,
                   {dwords[0], dwords[1], dwords[2], dwords[3], dwords[4],
                    dwords[5], dwords[6], dwords[7]},
                   MemoryFlags{0, 0x29c}, body);
  MemoryInfo memory;
  memory.kind = ResourceKind::Image;
  memory.image_dimension = Decoder::ImageDimension::Dim2D;
  fixture.Emit(ValueOpcode::ImageSampleRaw,
               {image_handle,
                fixture.Sampler({Value(0u), Value(0u), Value(0u), Value(0u)},
                                0x29c),
                fixture.ImageAddress()},
               fixture.AddMemory(memory, 0x29c), body);
  const auto bit = fixture.Emit(
      ValueOpcode::ShiftLeftLogical32,
      {Value(1u), fixture.Emit(ValueOpcode::BitwiseAnd32, {key, Value(31u)}, 0, latch)},
      0, latch);
  const auto next = fixture.Emit(ValueOpcode::BitwiseXor32, {mask, bit}, 0, latch);
  phi->AddPhiOperand(entry, initial);
  phi->AddPhiOperand(latch, next);
  fixture.PlanAndTrack();

  Check(fixture.program.info.images.size() == 1,
        "lsb-keyed dense image was not tracked as one image");
  const auto &source = fixture.program
                           .descriptor_sources[fixture.program.info.images[0].source];
  Check(source.indirect_image.has_value() &&
            source.indirect_image->key_bound == 32u &&
            source.indirect_image->table_offset == 0x6b0u &&
            source.indirect_image->bound_source ==
                DescriptorSource::IndirectImage::NoBoundSource,
        "find-lsb key was not planned as a 32-entry dense table");

  LinearTestMemory memory_image;
  std::array<uint32_t, 8> descriptor{};
  descriptor[0] = 0x40u;
  descriptor[1] =
      static_cast<uint32_t>(Libs::Graphics::Prospero::BufferFormat::k8_8_8_8UNorm)
      << 20u;
  descriptor[3] =
      Libs::Graphics::DstSel(4, 5, 6, 7) |
      (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D)
       << 28u);
  for (uint32_t entry_index = 0; entry_index < 32u; entry_index++) {
    for (uint32_t dword = 0; dword < descriptor.size(); dword++) {
      const auto at = (0x6b0u + entry_index * 32u) / 4u + dword;
      memory_image.words[at] = descriptor[dword];
    }
    memory_image.words[(0x6b0u + entry_index * 32u) / 4u] += entry_index;
  }
  std::array<uint32_t, 8> user_data{static_cast<uint32_t>(memory_image.base),
                                    0u, 0u, 0u, 0u, 5u, 0u, 0u};
  SrtRuntime runtime{.user_data = user_data,
                     .userdata = &memory_image,
                     .read_specialization_memory = ReadLinearTestMemory};
  auto resource_plan = ExtractResourcePlan(fixture.program);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(resource_plan, runtime, snapshot, specialization),
        "lsb-keyed dense table did not materialize");
  const auto mapping = specialization.images.empty()
                           ? UINT32_MAX
                           : specialization.images[0].indirect_mapping_offset;
  Check(specialization.images.size() == 32u && snapshot.images.size() == 32u &&
            mapping != UINT32_MAX &&
            mapping + 1u + 32u * 2u <= snapshot.flattened_srt.size() &&
            snapshot.flattened_srt[mapping] == 32u,
        "lsb-keyed dense table did not enumerate all 32 bit positions");
  for (uint32_t entry_index = 0; entry_index < 32u; entry_index++) {
    const auto offset = mapping + 1u + entry_index * 2u;
    Check(snapshot.flattened_srt[offset] == entry_index &&
              snapshot.images[snapshot.flattened_srt[offset + 1u]].dwords[0] ==
                  0x40u + entry_index,
          "lsb-keyed dense table read the wrong entry");
  }
}

void TestReadLaneProbeIndirectImage(bool first_lane = false, bool loop_mask = false,
                                    bool invalid_mask = false, bool spans = false) {
  Fixture fixture;
  const auto low = fixture.UserData(0);
  const auto high = fixture.UserData(1);
  const auto mask = fixture.UserData(4);
  const auto other =
      fixture.Emit(ValueOpcode::IEqual32, {fixture.UserData(6), Value(0u)});
  const auto nonzero = fixture.Emit(ValueOpcode::INotEqual32, {Value(0u), mask});
  const auto scan = fixture.Emit(ValueOpcode::FindILsb32, {mask});
  const auto item =
      fixture.Emit(ValueOpcode::SelectU32, {nonzero, scan, Value(32u)});
  const auto below =
      fixture.Emit(ValueOpcode::SGreaterThan32, {Value(32u), item});
  const auto enable = fixture.Emit(ValueOpcode::LogicalAnd, {other, below});
  const auto sixteen =
      fixture.Emit(ValueOpcode::ShiftLeftLogical32, {item, Value(4u)});
  const auto gated =
      fixture.Emit(ValueOpcode::SelectU32, {enable, sixteen, Value(0u)});
  const auto eight =
      fixture.Emit(ValueOpcode::ShiftLeftLogical32, {gated, Value(3u)});
  const auto nine = fixture.Emit(ValueOpcode::IAdd32, {eight, gated});
  const auto offset = fixture.Emit(ValueOpcode::IAdd32, {Value(0xc00u), nine});
  const auto records = fixture.Address(low, high, 0x1aec);
  MemoryInfo key_word;
  key_word.kind = ResourceKind::Flat;
  const auto loaded = fixture.Emit(ValueOpcode::LoadAddressU32,
                                   {records, offset, Value(0u), enable},
                                   fixture.AddMemory(key_word, 0x1aec));
  const auto per_lane =
      fixture.Emit(ValueOpcode::SelectU32, {enable, loaded, Value(0u)});
  const auto lane = fixture.Emit(
      ValueOpcode::BitwiseAnd32,
      {fixture.Emit(ValueOpcode::FindILsb32, {fixture.UserData(5)}), Value(63u)});
  Value active = invalid_mask ? other : enable;
  if (loop_mask) {
    auto* entry = fixture.block;
    auto* loop = fixture.AddBlock();
    fixture.block = loop;
    auto& phi = loop->AppendNewInst(ValueOpcode::Phi, {});
    active = Value(&phi);
    const auto carried = fixture.Emit(ValueOpcode::LogicalAnd, {active, other});
    phi.AddPhiOperand(entry, invalid_mask ? other : enable);
    phi.AddPhiOperand(loop, carried);
  }
  const auto key = fixture.Emit(first_lane ? ValueOpcode::ReadFirstLane : ValueOpcode::ReadLane,
                                {per_lane, first_lane ? active : lane});
  const auto scaled =
      fixture.Emit(ValueOpcode::ShiftLeftLogical32, {key, Value(5u)});
  const auto based = fixture.Emit(ValueOpcode::IAdd32, {scaled, Value(0x20e0u)});
  const auto second = fixture.Emit(ValueOpcode::IAdd32, {Value(16u), based});
  const auto first_half = fixture.Address(low, high, 0x1aec);
  const auto second_half = fixture.Address(low, high, 0x1aec);
  std::array<Value, 8> dwords{};
  for (uint32_t index = 0; index < dwords.size(); index++) {
    MemoryInfo word;
    word.kind = ResourceKind::ScalarAddress;
    word.offset = (index % 4u) * sizeof(uint32_t);
    dwords[index] = fixture.Emit(
        ValueOpcode::LoadAddressU32,
        {index < 4u ? first_half : second_half, index < 4u ? based : second,
         Value(0u), Value(true)},
        fixture.AddMemory(word, 0x1aec));
  }
  const auto image_handle =
      fixture.Emit(ValueOpcode::GetImageResource,
                   {dwords[0], dwords[1], dwords[2], dwords[3], dwords[4],
                    dwords[5], dwords[6], dwords[7]},
                   MemoryFlags{0, 0x1aec});
  MemoryInfo memory;
  memory.kind = ResourceKind::Image;
  memory.image_dimension = Decoder::ImageDimension::Dim2D;
  fixture.Emit(ValueOpcode::ImageSampleRaw,
               {image_handle,
                fixture.Sampler({Value(0u), Value(0u), Value(0u), Value(0u)},
                                0x1aec),
                fixture.ImageAddress()},
               fixture.AddMemory(memory, 0x1aec));
  if (invalid_mask) {
    CheckFatal([&] { fixture.PlanAndTrack(); },
               "first-lane mask does not retain the key load enable",
               "first-lane probe accepted an inactive load branch");
    return;
  }
  fixture.PlanAndTrack();

  Check(fixture.program.info.images.size() == 1,
        "readlane probe fixture did not track one image");
  const auto &source = fixture.program
                           .descriptor_sources[fixture.program.info.images[0].source];
  Check(source.indirect_image.has_value() &&
            source.indirect_image->item_bound == 32u &&
            source.indirect_image->selector_stride == 144u &&
            source.indirect_image->selector_offset == 0xc00u &&
            source.indirect_image->table_offset == 0x20e0u &&
            source.indirect_image->key_bound == 0u,
        "readlane waterfall was not planned as an address probe over its records");

  LinearTestMemory memory_image;
  for (uint32_t item = 0; item < 32u; item++) {
    memory_image.words[(0xc00u + item * 144u) / 4u] = item % 3u;
  }
  std::array<uint32_t, 8> descriptor{};
  descriptor[0] = 0x40u;
  descriptor[1] =
      static_cast<uint32_t>(Libs::Graphics::Prospero::BufferFormat::k8_8_8_8UNorm)
      << 20u;
  descriptor[3] =
      Libs::Graphics::DstSel(4, 5, 6, 7) |
      (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D)
       << 28u);
  for (uint32_t key_value = 0; key_value < 3u; key_value++) {
    for (uint32_t dword = 0; dword < descriptor.size(); dword++) {
      memory_image.words[(0x20e0u + key_value * 32u) / 4u + dword] =
          descriptor[dword];
    }
    memory_image.words[(0x20e0u + key_value * 32u) / 4u] += key_value;
  }
  std::array<uint32_t, 8> user_data{static_cast<uint32_t>(memory_image.base),
                                    0u, 0u, 0u, 0u, 0u, 0u, 0u};
  SrtRuntime runtime{.user_data = user_data,
                     .userdata = &memory_image,
                     .read_specialization_memory = ReadLinearTestMemory};
  auto resource_plan = ExtractResourcePlan(fixture.program);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(resource_plan, runtime, snapshot, specialization),
        "address probe table did not materialize");
  if (spans) {
    // The records' keys read a span at a time, and a stale record's key past the table's mapping (no read of its
    // entry): what word by word reads give.
    memory_image.words[(0xc00u + 31u * 144u) / 4u] = 0x12345u;
    Check(MaterializeResources(resource_plan, runtime, snapshot, specialization),
          "address probe table with a stale key did not materialize");
    auto span_runtime = runtime;
    span_runtime.try_read_memory_span = ReadLinearTestMemorySpan;
    span_runtime.mapping_end = LinearTestMappingEnd;
    ResourceSnapshot span_snapshot;
    ResourceSpecialization span_specialization;
    Check(MaterializeResources(resource_plan, span_runtime, span_snapshot,
                               span_specialization) &&
              span_snapshot.flattened_srt == snapshot.flattened_srt &&
              span_snapshot.images.size() == snapshot.images.size() &&
              span_specialization == specialization,
          "address probe span reads differ from word reads");
    for (size_t i = 0; i < snapshot.images.size(); i++) {
      Check(span_snapshot.images[i].dwords == snapshot.images[i].dwords,
            "address probe span reads found other candidates");
    }
    // A refused T# span (one word of key 1's T# unreadable) falls back to reads by key: the same candidates.
    memory_image.fail_address = memory_image.base + 0x20e0u + 32u + 4u;
    {
      ResourceSnapshot word_table, span_table;
      ResourceSpecialization word_table_specialization, span_table_specialization;
      const bool word_ok = MaterializeResources(resource_plan, runtime, word_table,
                                                word_table_specialization);
      const bool span_ok = MaterializeResources(resource_plan, span_runtime, span_table,
                                                span_table_specialization);
      Check(word_ok == span_ok &&
                (!word_ok || (word_table.flattened_srt == span_table.flattened_srt &&
                              word_table_specialization == span_table_specialization)),
            "address probe T# span reads differ where a T# is unreadable");
    }
    // A refused span (one record's key unreadable) falls back to word reads: the same outcome as word reads.
    memory_image.fail_address = memory_image.base + 0xc00u + 7u * 144u;
    ResourceSnapshot word_failed, span_failed;
    ResourceSpecialization word_failed_specialization, span_failed_specialization;
    const bool word_ok = MaterializeResources(resource_plan, runtime, word_failed,
                                              word_failed_specialization);
    const bool span_ok = MaterializeResources(resource_plan, span_runtime, span_failed,
                                              span_failed_specialization);
    Check(word_ok == span_ok &&
              (!word_ok || (word_failed.flattened_srt == span_failed.flattened_srt &&
                            word_failed_specialization == span_failed_specialization)),
          "address probe span reads skipped an unreadable record key");
    return;
  }
  const auto mapping = specialization.images.empty()
                           ? UINT32_MAX
                           : specialization.images[0].indirect_mapping_offset;
  Check(specialization.images.size() == 3u && snapshot.images.size() == 3u &&
            mapping != UINT32_MAX &&
            mapping + 1u + 3u * 2u <= snapshot.flattened_srt.size() &&
            snapshot.flattened_srt[mapping] == 3u,
        "address probe did not collect the records' distinct keys");
  for (uint32_t key_value = 0; key_value < 3u; key_value++) {
    const auto offset = mapping + 1u + key_value * 2u;
    Check(snapshot.flattened_srt[offset] == key_value &&
              snapshot.images[snapshot.flattened_srt[offset + 1u]].dwords[0] ==
                  0x40u + key_value,
          "address probe read the wrong table entry for a key");
  }
}


void TestWaterfallDescriptorMatch() {
  WaterfallFixture built;
  const auto matches = FindWaterfallDescriptors(built.fixture.program);
  Check(matches.size() == 1, "waterfall descriptor loop was not matched");
  Check(matches[0].mask_phi == built.phi &&
            matches[0].index->GetOpcode() == ValueOpcode::FindILsb32 &&
            matches[0].entry.Resolve() == built.entry.Resolve(),
        "waterfall loop was matched with the wrong mask operands");
  Check(matches[0].key.Resolve() == built.key.Resolve() &&
            matches[0].stride_shift == 5u && matches[0].table_offset == 0x158u &&
            matches[0].handle != nullptr &&
            matches[0].handle->GetOpcode() == ValueOpcode::GetImageResource &&
            !matches[0].heap.IsEmpty(),
        "waterfall descriptor table was extracted incorrectly");
}

void TestWaterfallImmediateTable() {
  WaterfallFixture built(1u, 31u, ValueOpcode::BitwiseAnd32, false, true, true, true, true);
  const auto matches = FindWaterfallDescriptors(built.fixture.program);
  Check(matches.size() == 1 && matches[0].table_offset == 0x158u,
        "AND-NOT waterfall with scalar-load immediate table was not matched");
  Check(RewriteWaterfallDescriptors(built.fixture.program) == 1,
        "immediate table waterfall was not rewritten");
  Check(matches[0].scaled->Arg(0).Resolve() == built.key.Resolve(),
        "immediate table still uses the scalar loop index");
  auto &fixture = built.fixture;
  built.key.Resolve().TryInstruction()->ReplaceUsesWith(
      fixture.Emit(ValueOpcode::LaneId));
  MemoryInfo image;
  image.kind = ResourceKind::Image;
  image.image_dimension = Decoder::ImageDimension::Dim2DArray;
  fixture.Emit(ValueOpcode::ImageSampleRaw,
               {Value(const_cast<Inst *>(matches[0].handle)),
                fixture.Sampler({Value(0u), Value(0u), Value(0u), Value(0u)}, 0xfc),
                fixture.ImageAddress()}, fixture.AddMemory(image, 0xfc));
  fixture.PlanAndTrack();
  Check(fixture.program.info.images.size() == 1,
        "immediate waterfall image was not tracked");
  const auto &source = fixture.program.descriptor_sources[
      fixture.program.info.images[0].source];
  Check(source.indirect_image && source.indirect_image->table_offset == 0x158u &&
            source.indirect_image->key_bound == 32u,
        "per-lane immediate waterfall did not produce the bounded texture table");
}

void TestWaterfallAndNotClear() {
  WaterfallFixture built(1u, 31u, ValueOpcode::BitwiseAnd32, false, true, true, true);
  Check(FindWaterfallDescriptors(built.fixture.program).size() == 1,
        "mask & ~lowest_bit waterfall was not matched");
  Check(RewriteWaterfallDescriptors(built.fixture.program) == 1,
        "AND-NOT waterfall was not rewritten");
  Check(built.phi->Arg(0).Resolve().U32() == 1u &&
            built.phi->Arg(1).Resolve().U32() == 0u,
        "AND-NOT waterfall does not terminate after one iteration");
  Check(FindWaterfallDescriptors(
            WaterfallFixture(1u, 31u, ValueOpcode::BitwiseAnd32).fixture.program)
            .empty(),
        "mask & lowest_bit incorrectly matched as clearing the bit");
  Check(FindWaterfallDescriptors(
            WaterfallFixture(2u, 31u, ValueOpcode::BitwiseAnd32, false, true, true, true)
                .fixture.program).empty(),
        "AND-NOT clearing the wrong bit was accepted");
}

void TestWaterfallNearMissesRejected() {
  Check(FindWaterfallDescriptors(WaterfallFixture(1u, 63u).fixture.program).empty(),
        "a 64-lane mask walk was matched as a waterfall");
  Check(FindWaterfallDescriptors(WaterfallFixture(2u, 31u).fixture.program).empty(),
        "a shifted-by-two mask walk was matched as a waterfall");
  Check(FindWaterfallDescriptors(
            WaterfallFixture(1u, 31u, ValueOpcode::BitwiseOr32).fixture.program)
            .empty(),
        "a mask loop that never clears its bit was matched as a waterfall");
  Check(FindWaterfallDescriptors(
            WaterfallFixture(1u, 31u, ValueOpcode::BitwiseXor32, true)
                .fixture.program)
            .empty(),
        "a mask loop with no outside entry was matched as a waterfall");
  Check(FindWaterfallDescriptors(
            WaterfallFixture(1u, 31u, ValueOpcode::BitwiseXor32, false, false)
                .fixture.program)
            .empty(),
        "a mask loop with no descriptor table was matched as a waterfall");
  Check(FindWaterfallDescriptors(
            WaterfallFixture(1u, 31u, ValueOpcode::BitwiseXor32, false, true, false)
                .fixture.program)
            .empty(),
        "a mask unrelated to the key was matched as a waterfall");
  {
    Fixture fixture;
    auto *entry_block = fixture.block;
    auto *loop = fixture.AddBlock();
    entry_block->AddBranch(loop);
    loop->AddBranch(loop);
    const auto attribute = fixture.UserData(6);
    const auto key = fixture.Emit(ValueOpcode::BitFieldUExtract,
                                  {attribute, Value(8u), Value(8u)});
    const auto ballot_field = fixture.Emit(ValueOpcode::BitFieldUExtract,
                                           {attribute, Value(8u), Value(8u)});
    const auto key_bits =
        fixture.Emit(ValueOpcode::BitwiseAnd32, {ballot_field, Value(31u)});
    const auto bit =
        fixture.Emit(ValueOpcode::ShiftLeftLogical32, {Value(1u), key_bits});
    const auto ballot = fixture.Emit(ValueOpcode::ReadLane, {bit, Value(31u)});
    auto *phi = &loop->AppendNewInst(ValueOpcode::Phi, {},
                                     static_cast<uint64_t>(Type::U32));
    const auto mask = Value(phi);
    const auto lsb = fixture.Emit(ValueOpcode::FindILsb32, {mask}, 0, loop);
    const auto masked =
        fixture.Emit(ValueOpcode::BitwiseAnd32, {lsb, Value(31u)}, 0, loop);
    const auto stepped =
        fixture.Emit(ValueOpcode::ShiftLeftLogical32, {Value(1u), masked}, 0, loop);
    const auto cleared =
        fixture.Emit(ValueOpcode::BitwiseXor32, {stepped, mask}, 0, loop);
    fixture.Emit(ValueOpcode::IEqual32, {lsb, key}, 0, loop);
    const auto scaled =
        fixture.Emit(ValueOpcode::ShiftLeftLogical32, {lsb, Value(5u)}, 0, loop);
    const auto based =
        fixture.Emit(ValueOpcode::IAdd32, {scaled, Value(0x158u)}, 0, loop);
    std::array<Value, 8> dwords{};
    for (uint32_t index = 0; index < dwords.size(); index++) {
      const auto address =
          fixture.Address(fixture.UserData(0), fixture.UserData(1), 0x660);
      dwords[index] =
          fixture.Emit(ValueOpcode::LoadAddressU32,
                       {address, based, Value(0u), Value(true)},
                       MemoryFlags{0, 0x660}, loop);
    }
    fixture.Emit(ValueOpcode::GetImageResource,
                 {dwords[0], dwords[1], dwords[2], dwords[3], dwords[4],
                  dwords[5], dwords[6], dwords[7]},
                 MemoryFlags{0, 0x660}, loop);
    phi->AddPhiOperand(entry_block, ballot);
    phi->AddPhiOperand(loop, cleared);
    Check(FindWaterfallDescriptors(fixture.program).size() == 1,
          "a waterfall keying off a middle byte field was not matched");
  }
}

void TestWaterfallRewriteDescalarizes() {
  WaterfallFixture built;
  Check(RewriteWaterfallDescriptors(built.fixture.program) == 1,
        "waterfall loop was not rewritten");
  const auto matches = FindWaterfallDescriptors(built.fixture.program);
  Check(matches.empty(),
        "rewritten waterfall still matches, so it would be rewritten twice");
  bool bounded = false;
  for (auto &inst : *built.fixture.program.blocks[1]) {
    bounded = bounded || (inst.GetOpcode() == ValueOpcode::ULessThan32 &&
                          inst.Arg(0).Resolve() == built.key.Resolve() &&
                          inst.Arg(1).Resolve().IsImmediate() &&
                          inst.Arg(1).Resolve().U32() == 32u);
  }
  Check(bounded, "rewritten waterfall does not bound the key to the mask width");
  const auto entry = built.phi->Arg(0).Resolve();
  const auto latch = built.phi->Arg(1).Resolve();
  Check(entry.IsImmediate() && entry.U32() == 1u && latch.IsImmediate() &&
            latch.U32() == 0u,
        "rewritten waterfall mask does not run exactly one iteration");
  Check(!built.entry.Resolve().TryInstruction()->HasUses(),
        "rewritten waterfall still keeps the ballot alive");
}

void TestLoopCycleEnteredThroughRuntimeValue() {
  Fixture fixture;
  auto *entry = fixture.block;
  auto *loop = fixture.AddBlock();
  const auto initial = fixture.UserData(0);
  entry->AddBranch(loop);
  loop->AddBranch(loop);
  auto &phi = loop->AppendNewInst(ValueOpcode::Phi, {},
                                  static_cast<uint64_t>(Type::U32));
  const auto carried = fixture.Emit(ValueOpcode::BitwiseAnd32,
                                    {Value(&phi), Value(0xffffffffu)}, 0, loop);
  phi.AddPhiOperand(entry, initial);
  phi.AddPhiOperand(loop, carried);
  fixture.Emit(ValueOpcode::GetBufferResource,
               {carried, Value(0u), Value(0u), Value(0u)}, MemoryFlags{0, 12},
               loop);

  BuildSrtPlan(fixture.program);
}

void TestInvariantLoopPhi() {
  Fixture fixture;
  auto *entry = fixture.block;
  auto *loop = fixture.AddBlock();
  entry->AddBranch(loop);
  loop->AddBranch(loop);
  const auto invariant = fixture.UserData(0);
  auto &phi = loop->AppendNewInst(ValueOpcode::Phi, {},
                                  static_cast<uint64_t>(Type::U32));
  phi.AddPhiOperand(entry, invariant);
  phi.AddPhiOperand(loop, Value(&phi));
  const auto handle = fixture.Emit(
      ValueOpcode::GetBufferResource,
      {Value(&phi), Value(0u), Value(0u), Value(0u)}, MemoryFlags{0, 4}, loop);
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {handle, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(memory, 4), loop);
  fixture.PlanAndTrack();

  std::array<uint32_t, 1> user_data{0x12345678u};
  SrtRuntime runtime{.user_data = user_data};
  DescriptorValue descriptor;
  Check(EvaluateDescriptorSource(fixture.program,
                                 fixture.program.info.buffers[0].source, runtime, descriptor) &&
            descriptor.dwords[0] == user_data[0],
        "loop-invariant descriptor phi was not evaluated through typed SSA");
}

void TestDmaAddressMaterialization() {
  Fixture fixture;
  const auto based =
      fixture.Address(fixture.UserData(0), fixture.UserData(1), 4);
  MemoryInfo global;
  global.kind = ResourceKind::Global;
  global.offset = static_cast<uint32_t>(-8);
  fixture.Emit(ValueOpcode::LoadAddressU32,
               {based, Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(global, 4));

  const auto undef = fixture.Emit(ValueOpcode::UndefU32);
  const auto unbased = fixture.Address(undef, undef, 8);
  MemoryInfo flat;
  flat.kind = ResourceKind::Flat;
  flat.address_is_full = true;
  fixture.Emit(ValueOpcode::StoreAddressU32,
               {unbased, Value(0u), Value(0u), Value(9u), Value(true)},
               fixture.AddMemory(flat, 8));
  fixture.PlanAndTrack();
  auto resource_plan = ExtractResourcePlan(fixture.program);

  Check(fixture.program.info.uses_dma,
        "typed address operations did not enable DMA");
  std::array<uint32_t, 2> user_data{0x2008u, 0u};
  SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(resource_plan, runtime, snapshot,
                             specialization),
        "DMA shader resources did not materialize");
  ApplyResourceSpecialization(fixture.program, specialization);
}

void TestDynamicFlatAddressesUseDma() {
  Fixture fixture;
  const auto low_root = fixture.UserData(0);
  const auto high_root = fixture.UserData(1);
  const auto active =
      fixture.Emit(ValueOpcode::INotEqual32, {fixture.UserData(2), Value(0u)});
  const auto inactive_low = fixture.Emit(ValueOpcode::UndefU32);
  const auto inactive_high = fixture.Emit(ValueOpcode::UndefU32);
  const auto low =
      fixture.Emit(ValueOpcode::SelectU32, {active, low_root, inactive_low});
  const auto high =
      fixture.Emit(ValueOpcode::SelectU32, {active, high_root, inactive_high});
  const auto address = fixture.Address(low, high, 0xa4);
  MemoryInfo flat;
  flat.kind = ResourceKind::Flat;
  flat.address_is_full = true;
  fixture.Emit(ValueOpcode::LoadAddressU8, {address, low, high, active},
               fixture.AddMemory(flat, 0xa4));
  fixture.PlanAndTrack();
  auto resource_plan = ExtractResourcePlan(fixture.program);

  Check(fixture.program.info.uses_dma,
        "exec-masked FLAT address did not enable DMA");
  std::array<uint32_t, 3> user_data{0x23456780u, 1u, 1u};
  SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(resource_plan, runtime, snapshot,
                             specialization),
        "exec-masked FLAT shader resources did not materialize");

  Fixture mismatch;
  const auto mismatch_active = mismatch.Emit(ValueOpcode::INotEqual32,
                                             {mismatch.UserData(2), Value(0u)});
  const auto other_active =
      mismatch.Emit(ValueOpcode::LogicalNot, {mismatch_active});
  const auto mismatch_low = mismatch.Emit(
      ValueOpcode::SelectU32, {mismatch_active, mismatch.UserData(0),
                               mismatch.Emit(ValueOpcode::UndefU32)});
  const auto mismatch_high = mismatch.Emit(
      ValueOpcode::SelectU32, {mismatch_active, mismatch.UserData(1),
                               mismatch.Emit(ValueOpcode::UndefU32)});
  const auto mismatch_address =
      mismatch.Address(mismatch_low, mismatch_high, 0xa4);
  mismatch.Emit(ValueOpcode::LoadAddressU8,
                {mismatch_address, mismatch_low, mismatch_high, other_active},
                mismatch.AddMemory(flat, 0xa4));
  mismatch.PlanAndTrack();
  Check(mismatch.program.info.uses_dma,
        "dynamic FLAT address did not enable DMA");
}

void TestBufferSwizzleSpecialization() {
  Fixture fixture;
  const auto handle = fixture.Buffer({fixture.UserData(0), fixture.UserData(1),
                                      fixture.UserData(2), fixture.UserData(3)},
                                     4);
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  memory.formatted = true;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {handle, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(memory, 4));
  fixture.PlanAndTrack();
  auto resource_plan = ExtractResourcePlan(fixture.program);

  constexpr auto swizzle = Libs::Graphics::DstSel(4, 5, 0, 1);
  std::array<uint32_t, 4> user_data{
      0, 16u << 16u, 1,
      swizzle |
          (static_cast<uint32_t>(
               Libs::Graphics::Prospero::BufferFormat::k32_32Float)
           << 12u) |
          (1u << 24u)};
  SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(resource_plan, runtime, snapshot,
                             specialization),
        "buffer resources did not materialize");
  ApplyResourceSpecialization(fixture.program, specialization);
  Check(fixture.program.info.buffers[0].descriptor_swizzle == swizzle &&
            specialization.buffers[0].descriptor_swizzle == swizzle,
        "buffer destination selectors were not specialized");

  user_data[3] ^= 1u << 9u;
  ResourceSnapshot changed_snapshot;
  ResourceSpecialization changed_specialization;
  Check(MaterializeResources(resource_plan, runtime, changed_snapshot,
                             changed_specialization) &&
            changed_specialization != specialization,
        "buffer swizzle change did not select a new specialization key");

  user_data[3] ^= 1u << 9u;
  user_data[0] = 0x4000;
  Check(MaterializeResources(resource_plan, runtime, changed_snapshot, changed_specialization) &&
            changed_specialization == specialization,
        "aligned buffer relocation changed the specialization");
  user_data[0] = 0x4001;
  Check(MaterializeResources(resource_plan, runtime, changed_snapshot, changed_specialization) &&
            changed_specialization != specialization &&
            changed_specialization.buffers[0].byte_base_offset,
        "byte-aligned buffer did not enable its low-bit address path");
  const auto byte_specialization = changed_specialization;
  user_data[0] = 0x8003;
  Check(MaterializeResources(resource_plan, runtime, changed_snapshot, changed_specialization) &&
            changed_specialization == byte_specialization,
        "different byte offsets unnecessarily created distinct shader variants");
}

enum class ConditionalBufferUse { Optional, Shared, Loop, Writable };

ResourcePlan ConditionalBufferPlan(ConditionalBufferUse use) {
  namespace CFG = Libs::Graphics::ShaderRecompiler::CFG;
  Fixture fixture;
  auto *entry = fixture.block;
  auto *optional = fixture.AddBlock();
  auto *done = fixture.AddBlock();
  auto *condition_block = entry;
  uint32_t condition_index = 0;
  fixture.program.block_info[0].id = 11;
  fixture.program.block_info[1].id = 27;
  fixture.program.block_info[2].id = 42;
  if (use == ConditionalBufferUse::Loop) {
    condition_block = fixture.AddBlock();
    condition_index = 3;
    fixture.program.block_info[3].id = 55;
    fixture.program.block_info[0].terminator = {
        .kind = CFG::TerminatorKind::Branch, .true_block = 55};
    entry->AddBranch(condition_block);
  }
  condition_block->AddBranch(optional);
  condition_block->AddBranch(done);
  optional->AddBranch(use == ConditionalBufferUse::Loop ? condition_block : done);
  fixture.program.block_info[condition_index].terminator = {
      .kind = CFG::TerminatorKind::ConditionalBranch,
      .true_block = 27, .false_block = 42};
  fixture.program.block_info[1].terminator = {
      .kind = CFG::TerminatorKind::Branch,
      .true_block = use == ConditionalBufferUse::Loop ? 55u : 42u};

  const auto control = fixture.Buffer(
      {fixture.UserData(0), fixture.UserData(1), fixture.UserData(2),
       fixture.UserData(3)}, 4);
  MemoryInfo scalar;
  scalar.kind = ResourceKind::ScalarBuffer;
  auto flag = fixture.Emit(ValueOpcode::ReadConstBuffer,
                           {control, Value(0u)}, fixture.AddMemory(scalar, 4));
  if (use == ConditionalBufferUse::Loop) {
    auto &phi = condition_block->AppendNewInst(ValueOpcode::Phi, {},
                                               static_cast<uint64_t>(Type::U32));
    phi.AddPhiOperand(entry, flag);
    phi.AddPhiOperand(optional, Value(1u));
    flag = Value(&phi);
  }
  fixture.program.block_info[condition_index].condition =
      fixture.Emit(ValueOpcode::INotEqual32, {flag, Value(0u)}, 0, condition_block);

  const auto payload = fixture.Buffer(
      {fixture.UserData(4), fixture.UserData(5), fixture.UserData(6),
       fixture.UserData(7)}, 8);
  MemoryInfo vector;
  vector.kind = ResourceKind::Buffer;
  const auto load = [&](Block *block) {
    fixture.Emit(ValueOpcode::LoadBufferU32,
                 {payload, Value(0u), Value(0u), Value(0u), Value(true)},
                 fixture.AddMemory(vector, 8), block);
  };
  load(optional);
  if (use == ConditionalBufferUse::Shared) {
    load(done);
  }
  if (use == ConditionalBufferUse::Writable) {
    fixture.Emit(ValueOpcode::StoreBufferU32,
                 {control, Value(0u), Value(0u), Value(0u), Value(1u),
                  Value(true)}, fixture.AddMemory(vector, 12));
  }
  fixture.PlanAndTrack();
  return ExtractResourcePlan(fixture.program);
}

void TestConditionalBufferMaterialization() {
  auto plan = ConditionalBufferPlan(ConditionalBufferUse::Optional);
  // GTA III leaves packet words in s[12:15] when its scalar control word is zero.
  std::array<uint32_t, 8> user_data{
      0x1000, 16u << 16u, 1, 0x4dfac,
      0xc0107600, 0x8c, 0x97730000, 0x100020};
  TestMemory memory;
  SrtRuntime runtime{.user_data = user_data, .userdata = &memory,
                     .read_specialization_memory = ReadTestMemory};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.buffers.size() == 2 &&
            snapshot.buffers[1].dword_count == 4 &&
            snapshot.buffers[1].dwords == std::array<uint32_t, 8>{},
        "untaken scalar branch materialized stale buffer words");
  Check(snapshot.user_data == std::vector<uint32_t>(user_data.begin(), user_data.end()),
        "resource reachability changed native shader user data");

  runtime.user_data = std::span(user_data).first(4);
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "untaken branch evaluated its unavailable descriptor");
  const auto prior = snapshot;
  memory.words[0] = 1;
  Check(!MaterializeResources(plan, runtime, snapshot, specialization) &&
            SameResourceSnapshot(snapshot, prior),
        "taken branch accepted an unavailable descriptor or changed the snapshot");

  runtime.user_data = user_data;
  const auto CheckActive = [&] {
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.buffers.size() == 2 &&
              std::equal(user_data.begin() + 4, user_data.end(),
                         snapshot.buffers[1].dwords.begin()),
          "potentially executed buffer descriptor was discarded");
  };
  CheckActive();
  memory.words[0] = 0;
  memory.fail_after = memory.reads;
  CheckActive();
  runtime.read_specialization_memory = nullptr;
  CheckActive();
}

void TestConservativeBufferReachability() {
  std::array<uint32_t, 8> user_data{
      0x1000, 16u << 16u, 1, 0x4dfac,
      0x2000, 16u << 16u, 1, 0x4dfac};
  TestMemory memory;
  const SrtRuntime runtime{.user_data = user_data, .userdata = &memory,
                           .read_specialization_memory = ReadTestMemory};
  for (const auto use : {ConditionalBufferUse::Shared, ConditionalBufferUse::Loop,
                         ConditionalBufferUse::Writable}) {
    auto plan = ConditionalBufferPlan(use);
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.buffers.size() == 2 &&
              std::equal(user_data.begin() + 4, user_data.end(),
                         snapshot.buffers[1].dwords.begin()),
          "shared, loop-dependent, or writable-alias resource was pruned");
  }
}

void TestConditionalIndirectImageMaterialization() {
  namespace CFG = Libs::Graphics::ShaderRecompiler::CFG;
  auto fixture = MakeIndirectImageFixture(false);
  auto *body = fixture->block;
  auto *entry = fixture->AddBlock();
  auto *done = fixture->AddBlock();
  entry->AddBranch(body);
  entry->AddBranch(done);
  body->AddBranch(done);
  const auto flag = fixture->Emit(ValueOpcode::GetUserData,
                                  {Value(static_cast<ScalarReg>(8))}, 0, entry);
  fixture->program.block_info[1].condition = fixture->Emit(
      ValueOpcode::INotEqual32, {flag, Value(0u)}, 0, entry);
  fixture->program.block_info[1].terminator = {
      .kind = CFG::TerminatorKind::ConditionalBranch,
      .true_block = 0, .false_block = 2};
  fixture->program.block_info[0].terminator = {
      .kind = CFG::TerminatorKind::Branch, .true_block = 2};
  std::swap(fixture->program.blocks[0], fixture->program.blocks[1]);
  std::swap(fixture->program.block_info[0], fixture->program.block_info[1]);
  fixture->PlanAndTrack();
  auto plan = ExtractResourcePlan(fixture->program);
  std::array<uint32_t, 9> user_data{
      0x1000, 224u << 16u, 2, 0, 0x2000, 16u << 16u, 4, 0, 0};
  uint32_t reads = 0;
  const SrtRuntime runtime{
      .user_data = user_data, .userdata = &reads,
      .read_specialization_memory = [](void *data, uint64_t, uint32_t *) {
        ++*static_cast<uint32_t *>(data);
        return false;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            reads == 0 && snapshot.images.size() == 1 &&
            snapshot.images[0].dwords == std::array<uint32_t, 8>{},
        "untaken indirect image branch probed its descriptor table");
  user_data[8] = 1;
  Check(!MaterializeResources(plan, runtime, snapshot, specialization) && reads != 0,
        "taken indirect image branch did not require its descriptor table");
}

void TestShaderInfoAndBindingLayout() {
  Fixture fixture;
  const auto handle = fixture.Buffer(
      {fixture.UserData(3), fixture.UserData(4), Value(64u), Value(0u)}, 4);
  MemoryInfo buffer;
  buffer.kind = ResourceKind::Buffer;
  fixture.Emit(ValueOpcode::LoadBufferU32,
               {handle, Value(0u), Value(0u), Value(0u), Value(true)},
               fixture.AddMemory(buffer, 4));
  fixture.Emit(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::GlobalInvocationId)),
       Value(2u)});
  fixture.Emit(ValueOpcode::BitwiseXor32, {Value(1u), Value(2u)});
  MemoryInfo gds;
  gds.kind = ResourceKind::Gds;
  fixture.Emit(ValueOpcode::WriteSharedU32, {Value(0u), Value(1u), Value(true)},
               fixture.AddMemory(gds, 8));
  fixture.PlanAndTrack();

  ShaderComputeInputInfo compute{};
  compute.dispatch_thread_dimensions = true;
  CollectShaderInfo(fixture.program, {.compute = &compute});
  Check(fixture.program.info.has_bitwise_xor &&
            !fixture.program.info.inputs.empty() &&
            fixture.program.info.inputs[0].kind ==
                StageInputKind::GlobalInvocationId,
        "typed shader values were not reflected in shader info");

  AllocateBindings(fixture.program);
  Check(FindBinding(fixture.program.bindings, DescriptorBindingKind::Buffers) !=
                nullptr &&
            FindBinding(fixture.program.bindings, DescriptorBindingKind::Gds) !=
                nullptr &&
            FindBinding(fixture.program.bindings,
                        DescriptorBindingKind::ShaderData) == nullptr &&
	        fixture.program.bindings.UsesPushData(),
        "typed resources were not assigned native bindings");
  Check(NativeBinding(ShaderType::Compute, DescriptorBindingKind::Buffers) ==
                static_cast<uint32_t>(DescriptorBindingKind::Buffers) &&
            NativeBinding(ShaderType::Vertex, DescriptorBindingKind::Buffers) ==
                static_cast<uint32_t>(DescriptorBindingKind::Buffers) &&
            NativeBinding(ShaderType::Pixel, DescriptorBindingKind::Buffers) ==
                static_cast<uint32_t>(DescriptorBindingKind::Count) +
                    static_cast<uint32_t>(DescriptorBindingKind::Buffers),
        "fixed stage binding ranges are inconsistent");
  Check(fixture.program.bindings.user_data_registers ==
            std::vector<uint32_t>({3u, 4u}),
        "binding layout did not collect live typed user-data values");
}

void TestLodStatsBindingLayout() {
  for (const auto stage : {ShaderType::Compute, ShaderType::Pixel}) {
    for (const bool enabled : {false, true}) {
      for (const uint32_t count : {1u, 64u}) {
        Fixture f;
        f.program.stage = stage;
        f.program.shader_info_complete = true;
        ImageResource image;
        image.resource_class = ImageResourceClass::Sampled;
        image.numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float;
        image.dimension = Decoder::ImageDimension::Dim2D;
        f.program.info.images.resize(count, image);
        AllocateBindings(f.program, 0, enabled);
        const bool active = enabled && stage == ShaderType::Pixel;
        Check((FindBinding(f.program.bindings, DescriptorBindingKind::LodStats) != nullptr) == active,
              "LOD instrumentation leaked into disabled or compute layout");
        Check(f.program.bindings.ShaderDataDwords() == (active ? count : 0),
              "LOD metadata allocation does not cover every image");
        if (active) {
          Check(f.program.bindings.UsesPushData() == (count <= PushData::DwordCount),
                "LOD metadata overflow did not switch to shader-data storage");
          Check((FindBinding(f.program.bindings, DescriptorBindingKind::ShaderData) != nullptr) ==
                    (count > PushData::DwordCount),
                "large LOD metadata has no storage descriptor");
        }
      }
    }
  }
}

void TestImageBindingAbi() {
  using NumericClass = Libs::Graphics::Prospero::TextureNumericClass;

  Check(ImageBindingCount == 43u &&
            static_cast<uint32_t>(DescriptorBindingKind::Buffers) == 0u &&
            static_cast<uint32_t>(DescriptorBindingKind::Samplers) == 44u &&
            static_cast<uint32_t>(DescriptorBindingKind::Gds) == 45u &&
            static_cast<uint32_t>(DescriptorBindingKind::BdaPagetable) == 46u &&
            static_cast<uint32_t>(DescriptorBindingKind::FaultBuffer) == 47u &&
            static_cast<uint32_t>(DescriptorBindingKind::FlattenedSrt) == 48u &&
            static_cast<uint32_t>(DescriptorBindingKind::ShaderData) == 49u &&
            static_cast<uint32_t>(DescriptorBindingKind::LodStats) == 50u &&
            static_cast<uint32_t>(DescriptorBindingKind::Count) == 51u,
        "native descriptor binding anchors changed");

  const std::array sampled_dimensions{
      Decoder::ImageDimension::Dim1D,
      Decoder::ImageDimension::Dim1DArray,
      Decoder::ImageDimension::Dim2D,
      Decoder::ImageDimension::Dim2DArray,
      Decoder::ImageDimension::Dim2DMsaa,
      Decoder::ImageDimension::Dim2DMsaaArray,
      Decoder::ImageDimension::Dim3D,
  };
  const std::array storage_dimensions{
      Decoder::ImageDimension::Dim1D, Decoder::ImageDimension::Dim1DArray,
      Decoder::ImageDimension::Dim2D, Decoder::ImageDimension::Dim2DArray,
      Decoder::ImageDimension::Dim3D,
  };
  const std::array sampled_classes{NumericClass::Float, NumericClass::Uint,
                                   NumericClass::Sint};
  const std::array storage_classes{NumericClass::Float, NumericClass::Uint};
  uint32_t index = 0;
  const auto CheckBinding =
      [&](ImageResourceClass resource_class, NumericClass numeric_class,
          Decoder::ImageDimension dimension, bool atomic, bool comparison = false) {
        ImageResource image;
        image.resource_class = resource_class;
        image.numeric_class = numeric_class;
        image.dimension = dimension;
        image.atomic = atomic;
        image.depth_compare = comparison;
        const auto kind = DescriptorBindingForImage(image);
        Check(kind.has_value() &&
                  static_cast<uint32_t>(*kind) == FirstImageBinding + index &&
                  ImageBindingIndex(*kind) == index &&
                  ImageBindingResourceClass(*kind) == resource_class &&
                  NativeBinding(ShaderType::Compute, *kind) ==
                      FirstImageBinding + index &&
                  NativeBinding(ShaderType::Pixel, *kind) ==
                      static_cast<uint32_t>(DescriptorBindingKind::Count) +
                          FirstImageBinding + index,
              "generated image descriptor binding changed ABI");
        index++;
      };
  for (const auto numeric_class : sampled_classes) {
    for (const auto dimension : sampled_dimensions) {
      CheckBinding(ImageResourceClass::Sampled, numeric_class, dimension,
                   false);
    }
  }
  for (const auto dimension : sampled_dimensions) {
    CheckBinding(ImageResourceClass::Sampled, NumericClass::Float, dimension,
                 false, true);
  }
  for (const auto numeric_class : storage_classes) {
    for (const auto dimension : storage_dimensions) {
      CheckBinding(ImageResourceClass::Storage, numeric_class, dimension,
                   false);
    }
  }
  for (const auto dimension : storage_dimensions) {
    CheckBinding(ImageResourceClass::Storage, NumericClass::Uint, dimension,
                 true);
  }
  Check(index == ImageBindingCount, "image descriptor ABI case count changed");

  const auto Invalid = [](ImageResource image) {
    return !DescriptorBindingForImage(image).has_value();
  };
  ImageResource image;
  Check(Invalid(image), "untyped image received a descriptor binding");
  image.resource_class = ImageResourceClass::Sampled;
  image.numeric_class = NumericClass::Float;
  image.dimension = Decoder::ImageDimension::Unknown;
  Check(Invalid(image),
        "unknown sampled dimension received a descriptor binding");
  image.dimension = Decoder::ImageDimension::Dim2D;
  image.numeric_class = NumericClass::Unsupported;
  Check(Invalid(image),
        "unsupported sampled class received a descriptor binding");
  image.numeric_class = NumericClass::Uint;
  image.depth_compare = true;
  Check(Invalid(image), "integer comparison image received a descriptor binding");
  image.depth_compare = false;
  image.numeric_class = static_cast<NumericClass>(UINT32_MAX);
  Check(Invalid(image), "invalid sampled class received a descriptor binding");
  image.numeric_class = NumericClass::Float;
  image.dimension = static_cast<Decoder::ImageDimension>(UINT32_MAX);
  Check(Invalid(image),
        "invalid sampled dimension received a descriptor binding");
  image.dimension = Decoder::ImageDimension::Dim2D;
  image.atomic = true;
  Check(Invalid(image), "atomic sampled image received a descriptor binding");
  image.resource_class = ImageResourceClass::Storage;
  image.atomic = false;
  image.numeric_class = NumericClass::Sint;
  Check(Invalid(image), "signed storage image received a descriptor binding");
  image.numeric_class = NumericClass::Float;
  image.dimension = Decoder::ImageDimension::Dim2DMsaa;
  Check(Invalid(image),
        "multisampled storage image received a descriptor binding");
  image.dimension = Decoder::ImageDimension::Dim2D;
  image.atomic = true;
  Check(Invalid(image), "float atomic image received a descriptor binding");
}

void TestGraphicsPushConstantLayout() {
  const auto AddUserData = [](Fixture &fixture, uint32_t count) {
    for (uint32_t index = 0; index < count; index++) {
      fixture.Emit(ValueOpcode::ReferenceU32, {fixture.UserData(index)});
    }
    fixture.program.shader_info_complete = true;
  };
  uint32_t cursor = 0;
  Fixture pixel(ShaderType::Pixel);
  AddUserData(pixel, 4);
  AllocateBindings(pixel.program, cursor);
  Check(
      pixel.program.bindings.UsesPushData() &&
          pixel.program.bindings.push_data_start_dword == 0 &&
          FindBinding(pixel.program.bindings,
                      DescriptorBindingKind::ShaderData) == nullptr,
      "pixel shader did not start the shared push-data block");
  pixel.program.bindings.AdvancePushData(cursor);

  Fixture vertex(ShaderType::Vertex);
  AddUserData(vertex, 9);
  AllocateBindings(vertex.program, cursor);
  Check(vertex.program.bindings.UsesPushData() &&
            vertex.program.bindings.push_data_start_dword == 4,
        "vertex shader did not follow pixel data in the shared push-data block");
  vertex.program.bindings.AdvancePushData(cursor);
  Check(cursor == 13, "graphics push-data cursor advanced incorrectly");

  Fixture edge(ShaderType::Pixel);
  AddUserData(edge, NativePushConstantSize / sizeof(uint32_t));
  AllocateBindings(edge.program);
  Check(edge.program.bindings.UsesPushData() &&
            FindBinding(edge.program.bindings,
                        DescriptorBindingKind::ShaderData) == nullptr,
        "the full shared push-data block did not fit");

  Fixture spill(ShaderType::Pixel);
  AddUserData(spill, 20);
  AllocateBindings(spill.program, cursor);
  Check(
      !spill.program.bindings.UsesPushData() &&
          spill.program.bindings.push_data_start_dword == PushData::NoStart &&
          FindBinding(spill.program.bindings,
                      DescriptorBindingKind::ShaderData) != nullptr,
      "a stage that exceeded the remaining shared push data did not spill to storage");
  const auto spill_layout = spill.program.bindings;
  spill.program.bindings.AdvancePushData(cursor);
  Check(cursor == 13, "a spilled stage consumed shared push-data space");

  Fixture repeated_spill(ShaderType::Pixel);
  AddUserData(repeated_spill, 20);
  AllocateBindings(repeated_spill.program, 20);
  Check(repeated_spill.program.bindings == spill_layout,
        "storage fallback retained an irrelevant attempted push-data position");
}

void TestResourceLimitIsTransactional() {
  Fixture fixture;
  MemoryInfo memory;
  memory.kind = ResourceKind::Buffer;
  for (uint32_t index = 0; index <= ShaderInfo::MaxBuffers; index++) {
    const auto handle = fixture.Buffer(
        {Value(index), Value(index + 1u), Value(index + 2u), Value(index + 3u)},
        index * 4u);
    fixture.Emit(ValueOpcode::LoadBufferU32,
                 {handle, Value(0u), Value(0u), Value(0u), Value(true)},
                 fixture.AddMemory(memory, index * 4u));
  }
  BuildSrtPlan(fixture.program);
  CheckFatal([&] { TrackResources(fixture.program); },
             "buffer resource limit exceeded",
             "resource-limit failure was not reported");
  Check(!fixture.program.resource_tracking_complete &&
            fixture.program.info.buffers.empty() &&
            fixture.program.descriptor_sources.empty(),
        "resource-limit failure partially mutated typed resource state");
}

void TestMalformedMemoryKindsRejected() {
  {
    Fixture fixture;
    const auto address = fixture.Address(Value(0u), Value(0u), 4);
    MemoryInfo memory;
    memory.kind = ResourceKind::Buffer;
    fixture.Emit(ValueOpcode::StoreAddressU32,
                 {address, Value(0u), Value(0u), Value(1u), Value(true)},
                 fixture.AddMemory(memory, 4));
    BuildSrtPlan(fixture.program);
    CheckFatal(
        [&] { TrackResources(fixture.program); },
        "address operation has invalid resource kind",
        "resource tracking accepted an address opcode with buffer metadata");
  }
  {
    Fixture fixture;
    const auto image =
        fixture.Image({Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
                       Value(0u), Value(0u), Value(0u)},
                      8);
    MemoryInfo memory;
    memory.kind = ResourceKind::Flat;
    fixture.Emit(ValueOpcode::ImageRead,
                 {image, fixture.ImageAddress(), Value(true)},
                 fixture.AddMemory(memory, 8));
    BuildSrtPlan(fixture.program);
    CheckFatal(
        [&] { TrackResources(fixture.program); },
        "image operation has invalid resource kind",
        "resource tracking accepted an image opcode with address metadata");
  }
}

void TestFunctionLdsLayout() {
  Fixture fixture(ShaderType::Pixel);
  const auto lane_address = [&]() {
    return fixture.Emit(ValueOpcode::ShiftLeftLogical32,
                        {fixture.Emit(ValueOpcode::LaneId), Value(2u)});
  };
  const auto add_access = [&](uint32_t offset, bool write) {
    const auto flags = fixture.AddMemory(
        MemoryInfo{.kind = ResourceKind::Lds, .offset = offset}, 0);
    return write ? fixture.Emit(ValueOpcode::WriteSharedU32,
                               {lane_address(), Value(offset), Value(true)}, flags)
                 : fixture.Emit(ValueOpcode::LoadSharedU32,
                                {lane_address(), Value(true)}, flags);
  };
  const auto high_write = add_access(1280, true);
  const auto low_write = add_access(0, true);
  const auto high_read = add_access(1280, false);
  auto layout = PlanFunctionLdsLayout(fixture.program);
  Check(layout.dwords == 2 && layout.slots.size() == 3,
        "private lane-relative LDS must use only the distinct scalar slots");
  Check(layout.slots.at(high_write.TryInstruction()) == layout.slots.at(high_read.TryInstruction()) &&
            layout.slots.at(high_write.TryInstruction()) != layout.slots.at(low_write.TryInstruction()),
        "LDS slot compression must preserve aliases and separate distinct offsets");
  fixture.program.stage = ShaderType::Compute;
  Check(PlanFunctionLdsLayout(fixture.program).slots.empty(),
        "workgroup-shared compute LDS must never be made private");
  fixture.program.stage = ShaderType::Pixel;
  fixture.program.memory_info[0].offset = 1281;
  Check(PlanFunctionLdsLayout(fixture.program).slots.empty(),
        "unaligned access must reject the whole private LDS layout");
  fixture.program.memory_info[0].offset = 1280;
  auto address = high_read.TryInstruction()->Arg(0).TryInstruction();
  address->SetArg(1, Value(3u));
  Check(PlanFunctionLdsLayout(fixture.program).slots.empty(),
        "different lane scale must reject all slots, not partially compact LDS");
  address->SetArg(1, Value(2u));
  const auto extra = fixture.AddMemory(MemoryInfo{.kind = ResourceKind::Lds}, 0);
  fixture.Emit(ValueOpcode::LoadSharedU16, {lane_address(), Value(true)}, extra);
  Check(PlanFunctionLdsLayout(fixture.program).slots.empty(),
        "mixed subword accesses must retain the original LDS representation");
}

} // namespace

void TestNativePredicateBitfields() {
#if defined(KYTY_LOCAL_NATIVE_RESOURCES) && defined(__x86_64__)
  struct Memory {
    uint32_t bits;
    unsigned fail;
    std::vector<std::pair<uint64_t, bool>> calls;
    bool Read(uint64_t address, uint32_t* value, bool clean) {
      calls.emplace_back(address, clean);
      if (calls.size() == fail) return false;
      if (address == 0x1000) *value = 0x2000;
      else if (address == 0x1004) *value = 0;
      else if (address == 0x200c) *value = bits;
      else if (address == 0x3000 || address == 0x3004) *value = uint32_t(address) ^ 0x83f051a7u;
      else throw std::runtime_error("native predicate followed an invalid pointer");
      return true;
    }
  };
  const auto raw = +[](void* p, uint64_t a, uint32_t* v) { return static_cast<Memory*>(p)->Read(a, v, false); };
  const auto clean = +[](void* p, uint64_t a, uint32_t* v) { return static_cast<Memory*>(p)->Read(a, v, true); };
  const std::array<std::pair<uint32_t, uint32_t>, 9> ranges = {{{10, 1}, {0, 32}, {31, 1}, {0, 0},
                            {32, 0}, {16, 16}, {31, 2}, {33, 0}, {0, 33}}};
  for (const auto [offset, width] : ranges) {
    Fixture fixture;
    auto& plan = fixture.program;
    const auto load = [&](Value low, Value high, uint32_t offset) {
      const MemoryInfo info{.kind = ResourceKind::ScalarAddress, .offset = offset};
      return fixture.Emit(ValueOpcode::LoadAddressU32,
          {fixture.Address(low, high), Value(0u), Value(0u), Value(true)}, fixture.AddMemory(info, 0));
    };
    const auto table = fixture.Emit(ValueOpcode::GetSrtResource);
    const auto flat = [&](uint32_t index) { return fixture.Emit(ValueOpcode::ReadConst, {table, Value(index)}); };
    plan.srt_reads = {{load(Value(0x1000u), Value(0u), 0), 0},
                      {load(Value(0x1000u), Value(0u), 4), 1},
                      {load(flat(0), flat(1), 12), 2}};
    const auto bit = fixture.Emit(ValueOpcode::BitFieldUExtract, {flat(2), Value(offset), Value(width)});
    const auto condition = fixture.Emit(ValueOpcode::LogicalNot,
        {fixture.Emit(ValueOpcode::IEqual32, {bit, Value(1u)})});
    plan.descriptor_sources.resize(2);
    for (uint32_t i = 0; i < 2; ++i) {
      plan.descriptor_sources[i].dword_count = 1;
      plan.descriptor_sources[i].dwords[0] = load(Value(0x3000u + i * 4), Value(0u), 0);
    }
    plan.materialization_sources = {0, 1};
    plan.control_flow = {{condition, {1, 2}, {}}, {{}, {}, {0}}, {{}, {}, {1}}};
    plan.srt_plan_complete = true;
    BuildLinearSrtPlan(plan);
    Check(bool(plan.linear_srt), "bitfield predicate lost the conservative leaf fallback");
    const bool valid = offset <= 32 && width <= 32 - offset;
    Check(bool(plan.linear_srt->single_condition) == valid, "bitfield predicate range eligibility differs");
    const auto compiled = plan.linear_srt;
    for (const uint32_t bits : {0u, 1u, 0x400u, 0x80000000u, 0xdeadbeefu, UINT32_MAX})
      for (unsigned failure = 0; failure < 10; ++failure) for (bool can_clean : {false, true}) {
        Memory old_memory{bits, failure}, new_memory{bits, failure};
        SrtRuntime runtime{.read_memory = raw, .userdata = &old_memory,
                           .read_specialization_memory = can_clean ? clean : nullptr};
        std::vector<DescriptorValue> expected, actual;
        std::vector<uint32_t> expected_flat, actual_flat;
        std::vector<uint8_t> expected_active, actual_active;
        plan.linear_srt.reset();
        const bool old_ok = EvaluateRuntimeSources(plan, plan.materialization_sources, runtime,
            expected, expected_flat, {}, expected_active);
        plan.linear_srt = compiled;
        runtime.userdata = &new_memory;
        const bool new_ok = EvaluateRuntimeSources(plan, plan.materialization_sources, runtime,
            actual, actual_flat, {}, actual_active);
        Check(old_ok == new_ok && old_memory.calls == new_memory.calls,
              "bitfield predicate changed pointer reads, order, or failure behavior");
        Check(!old_ok || (expected == actual && expected_flat == actual_flat && expected_active == actual_active),
              "bitfield predicate changed the selected resource transaction");
      }
  }
#endif
}

void TestControlledLinearSrt() {
#if defined(__x86_64__) || defined(_M_X64)
  struct Memory {
    int first=0,second=0,fail=0;
    std::vector<std::pair<uint64_t,bool>> calls;
  };
  const auto raw=+[](void* ptr,uint64_t address,uint32_t* word) {
    auto& memory=*static_cast<Memory*>(ptr);memory.calls.emplace_back(address,false);
    if(int(memory.calls.size())==memory.fail)return false;
    *word=uint32_t(address)^0x73846521u;return true;
  };
  const auto clean=+[](void* ptr,uint64_t address,uint32_t* word) {
    auto& memory=*static_cast<Memory*>(ptr);memory.calls.emplace_back(address,true);
    const auto flag=address==0x8000u ? memory.first : memory.second;
    if(flag<0 || int(memory.calls.size())==memory.fail)return false;
    *word=uint32_t(flag);return true;
  };
  for(const bool loop:{false,true}) for(uint32_t shape=0;shape<4;++shape) {
    Fixture fixture;
    auto& plan=fixture.program;
    const auto read=[&](uint32_t address) {
      MemoryInfo info;info.kind=ResourceKind::ScalarAddress;
      return fixture.Emit(ValueOpcode::LoadAddressU32,
        {fixture.Address(Value(address),Value(0u)),Value(0u),Value(0u),Value(true)},fixture.AddMemory(info,0));
    };
    const auto a=fixture.Emit(ValueOpcode::INotEqual32,{read(0x8000),Value(0u)});
    const auto b=fixture.Emit(ValueOpcode::INotEqual32,{read(0x8004),Value(0u)});
    plan.descriptor_sources.resize(4);
    for(uint32_t i=0;i<4;++i) {
      auto& source=plan.descriptor_sources[i];source.dword_count=1;
      source.dwords[0]=read(0x1000u*(i+1));
    }
    // Duplicate descriptor sources share a read; inactive descriptors are zero.
    // Flat values still execute in their original order, even if a related
    // descriptor is inactive. Back edges must terminate by the visited mask.
    plan.materialization_sources={3,1,0,2,1};
    plan.srt_reads={{plan.descriptor_sources[0].dwords[0],0},
                    {plan.descriptor_sources[3].dwords[0],1}};
    plan.control_flow={{a,{1,2},{0}},{b,{3,2},{1}},{{},{},{2}},
                       {{},{loop?0u:2u},{3}}};
    if(shape==1 || shape==3) plan.control_flow[1].condition={};
    if(shape==2) { plan.control_flow[0].condition={}; plan.control_flow[0].successors={2}; }
    if(shape==3) {
      auto body=std::move(plan.control_flow);
      plan.control_flow.resize(70);
      for(uint32_t i=0;i<70;++i) plan.control_flow[i].successors={i+1};
      for(auto& block:body) {
        for(auto& next:block.successors) next+=70;
        plan.control_flow.push_back(std::move(block));
      }
    }
    plan.srt_plan_complete=true;
    BuildLinearSrtPlan(plan);
    Check(plan.linear_srt && !plan.linear_srt->control_variants.empty() &&
      plan.linear_srt->control_variants.size()<=9,"conditional source combinations did not compile");
    const auto compiled=plan.linear_srt;
#ifdef KYTY_LOCAL_NATIVE_RESOURCES
    Check(bool(compiled->single_condition)==(shape==1 || shape==3),
      "native predicate did not respect unique reachable condition eligibility");
#endif
    for(const int first:{0,1,-1})for(const int second:{0,1,-1})
      for(const int fail:{0,1,2,3,4,5,6,7,8})for(const bool clean_reader:{false,true}) {
        Memory original{first,second,fail},native=original;
        SrtRuntime runtime{.read_memory=raw,.userdata=&original,
                           .read_specialization_memory=clean_reader?clean:nullptr};
        std::vector<DescriptorValue> expected(1),actual=expected;
        expected[0].dwords[0]=actual[0].dwords[0]=0xdeadbeefu;
        std::vector<uint32_t> expected_flat{0x1234},actual_flat=expected_flat;
        std::vector<uint8_t> expected_active{7},actual_active=expected_active;
        plan.linear_srt.reset();
        const bool reference=EvaluateRuntimeSources(plan,plan.materialization_sources,runtime,
          expected,expected_flat,{},expected_active);
        runtime.userdata=&native;plan.linear_srt=compiled;
        const bool result=EvaluateRuntimeSources(plan,plan.materialization_sources,runtime,
          actual,actual_flat,{},actual_active);
        Check(reference==result && expected==actual && expected_flat==actual_flat &&
          expected_active==actual_active && original.calls==native.calls,
          "controlled graph changed predicates, inactive reads, memoization, failure output or read order");
      }
    plan.clean_flat_slots={1,0};BuildLinearSrtPlan(plan);
    Check(!plan.linear_srt,"conditional graph with shared clean-flat memo was accepted");
    plan.clean_flat_slots.clear();plan.control_flow.push_back({a,{0,1},{}});
    plan.control_flow.push_back({a,{0,1},{}});
    BuildLinearSrtPlan(plan);
    Check(!plan.linear_srt,"unbounded conditional variant enumeration was accepted");
  }
  auto optional=ConditionalBufferPlan(ConditionalBufferUse::Optional);
  Check(optional.linear_srt && !optional.linear_srt->control_variants.empty(),
    "production-extracted conditional resource plan did not compile");
#endif
}


void TestLinearSrtDifferential() {
#if defined(__x86_64__) || defined(_M_X64)
  Fixture fixture;
  const auto a=fixture.Emit(ValueOpcode::CompositeConstructU64,{fixture.UserData(0),fixture.UserData(1)});
  const auto b=fixture.Emit(ValueOpcode::CompositeConstructU64,{fixture.UserData(2),fixture.UserData(3)});
  const ValueOpcode operations[] {
    ValueOpcode::IAdd32,ValueOpcode::IAdd64,ValueOpcode::ISub32,ValueOpcode::ISub64,
    ValueOpcode::IMul32,ValueOpcode::IMul64,ValueOpcode::BitwiseAnd32,ValueOpcode::BitwiseAnd64,
    ValueOpcode::BitwiseOr32,ValueOpcode::BitwiseXor32,ValueOpcode::ShiftLeftLogical32,
    ValueOpcode::ShiftLeftLogical64,ValueOpcode::ShiftRightLogical32,ValueOpcode::ShiftRightLogical64,
    ValueOpcode::ShiftRightArithmetic32,ValueOpcode::ShiftRightArithmetic64,ValueOpcode::CompositeConstructU64,
    ValueOpcode::IEqual32,ValueOpcode::INotEqual32,ValueOpcode::ULessThan32,ValueOpcode::UGreaterThan32,
    ValueOpcode::UGreaterThanEqual32,ValueOpcode::UMin32,ValueOpcode::LogicalAnd,ValueOpcode::LogicalOr,ValueOpcode::LogicalXor};
  auto output=[&](Value value){fixture.program.srt_reads.push_back({value,uint32_t(fixture.program.srt_reads.size())});};
  for(const auto op:operations) {
    const auto v=fixture.Emit(op,{a,b});output(v);
    output(fixture.Emit(ValueOpcode::CompositeExtractU64,{v,Value(1u)}));
  }
  output(fixture.Emit(ValueOpcode::BitwiseNot32,{a}));
  output(fixture.Emit(ValueOpcode::LogicalNot,{a}));
  output(fixture.Emit(ValueOpcode::SelectU32,{fixture.UserData(0),a,b}));
  fixture.program.srt_plan_complete=true;
  auto plan=ExtractResourcePlan(fixture.program);
  Check(plan.linear_srt!=nullptr,"whole-graph arithmetic fixture did not compile");
  const auto compiled=plan.linear_srt;
  uint32_t seed=0x743512fe;
  for(uint32_t trial=0;trial<1024;++trial) {
    std::array<uint32_t,4> data;
    for(auto& word:data){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;word=seed;}
    if(trial<128)data[2]=trial;
    if(trial==128)data.fill(0);
    if(trial==129)data.fill(UINT32_MAX);
    std::vector<uint32_t> expected{0xdeadbeef},actual=expected;
    const auto words=trial%23==0?std::span<const uint32_t>{}:std::span<const uint32_t>{data};
    plan.linear_srt.reset();const bool old_ok=WalkSrt(plan,{.user_data=words},expected);
    plan.linear_srt=compiled;const bool new_ok=WalkSrt(plan,{.user_data=words},actual);
    Check(old_ok==new_ok && expected==actual,"linear native arithmetic/failure differs from evaluator");
  }
#endif
}

void TestCompiledSrtSpanBounds() {
#if defined(__x86_64__) || defined(_M_X64)
	struct Memory {
		std::vector<uint64_t> events;
		uint64_t              groups = 0;
		static uint32_t Word(uint64_t a) { return uint32_t(a) ^ uint32_t(a >> 32) ^ 0x93814762u; }
	};
	const auto read = +[](void* p, uint64_t a, uint32_t* v) {
		static_cast<Memory*>(p)->events.push_back(a);
		*v = Memory::Word(a);
		return true;
	};
	const auto span = +[](void* p, uint64_t a, uint32_t* v, uint32_t n, bool clean) {
		++static_cast<Memory*>(p)->groups;
		Check(!clean && n >= 2 && n <= 16, "invalid boundary probe");
		for (uint32_t i = 0; i < n; ++i) {
			static_cast<Memory*>(p)->events.push_back(a + i * 4);
			v[i] = Memory::Word(a + i * 4);
		}
		return true;
	};
	uint64_t used = 0;
	for (bool buffer: {false, true})
		for (int32_t immediate: {INT32_MIN, -65, -4, 0, 3, INT32_MAX - 64})
			for (uint32_t offset: {0u, 3u, 7u, 0x80000000u, UINT32_MAX - 3u}) {
				Fixture    f;
				const auto lo = f.UserData(0), hi = f.UserData(1), records = f.UserData(2);
				const auto handle =
				    buffer ? f.Buffer({lo, hi, records, Value(0u)}) : f.Address(lo, hi);
				for (uint32_t i = 0; i < 16; ++i) {
					MemoryInfo info;
					info.kind   = buffer ? ResourceKind::ScalarBuffer : ResourceKind::ScalarAddress;
					info.offset = static_cast<uint32_t>(int64_t {immediate} + i * 4u);
					info.planning_only = true;
					const auto flags   = f.AddMemory(info, 0);
					const auto value =
					    buffer
					        ? f.Emit(ValueOpcode::ReadConstBuffer, {handle, Value(offset)}, flags)
					        : f.Emit(ValueOpcode::LoadAddressU32,
					                 {handle, Value(offset), Value(0u), Value(true)}, flags);
					f.program.srt_reads.push_back({value, i});
				}
				f.program.srt_plan_complete = true;
				auto plan                   = ExtractResourcePlan(f.program);
				Check(bool(plan.linear_srt), "boundary graph failed to compile");
				for (uint64_t address: {0ull, 3ull, 0x1003ull, 0x0000ffffffffffb0ull,
				                        0x0000fffffffffffcull, 0x1234000000000003ull})
					for (uint32_t count: {0u, 4u, 63u, 64u, UINT32_MAX})
						for (uint32_t stride: {0u, 1u, 0x3fffu}) {
							const uint32_t data[] = {
							    uint32_t(address), uint32_t(address >> 32) | (stride << 16), count};
							Memory     expected, actual;
							SrtRuntime runtime {
							    .user_data = data, .read_memory = read, .userdata = &expected};
							std::vector<uint32_t> ev {0xdeadbeefu}, av = ev;
							auto                  compiled = std::move(plan.linear_srt);
							const bool            ok       = WalkSrt(plan, runtime, ev);
							plan.linear_srt                = std::move(compiled);
							runtime.userdata               = &actual;
							runtime.try_read_memory_span   = span;
							const bool got                 = WalkSrt(plan, runtime, av);
							used += actual.groups;
							Check(ok == got && ev == av && expected.events == actual.events,
							      "compiled span changed signed offset, masked base, extent "
							      "or partial failure semantics");
						}
			}
	// A completed group must still be counted if a later user-data access fails.
	Fixture    fail;
	const auto address = fail.Address(fail.UserData(0), Value(0u));
	for (uint32_t i = 0; i < 2; ++i) {
		MemoryInfo info;
		info.kind          = ResourceKind::ScalarAddress;
		info.offset        = i * 4;
		info.planning_only = true;
		const auto value =
		    fail.Emit(ValueOpcode::LoadAddressU32, {address, Value(0u), Value(0u), Value(true)},
		              fail.AddMemory(info, 0));
		fail.program.srt_reads.push_back({value, i});
	}
	fail.program.srt_reads.push_back({fail.UserData(1), 2});
	fail.program.srt_plan_complete = true;
	auto                  plan     = ExtractResourcePlan(fail.program);
	const uint32_t        data[]   = {0x1000};
	Memory                memory;
	std::vector<uint32_t> output {0xdeadbeef};
	Check(!WalkSrt(plan,
	               {.user_data            = data,
	                .read_memory          = read,
	                .userdata             = &memory,
	                .try_read_memory_span = span},
	               output) &&
	          output == std::vector<uint32_t> {0xdeadbeef} &&
	          memory.events == std::vector<uint64_t> {0x1000, 0x1004} && memory.groups == 1,
	      "failed transaction lost completed span counters or published output");
	Check(used > 0, "compiled boundary tests did not use generated groups");
#endif
}

void TestLinearSrtSpans() {
#if defined(__x86_64__) || defined(_M_X64)
	struct Memory {
		std::vector<std::pair<uint64_t, bool>> calls;
		int                                    fail   = -1;
		uint32_t                               probes = 0, spans = 0;
		bool                                   accept = true;
		static uint32_t                        Word(uint64_t a, bool clean) {
            return uint32_t(a ^ (a >> 32)) ^ (clean ? 0x8761u : 0x9123u);
		}
		bool Read(uint64_t a, uint32_t* v, bool clean) {
			calls.emplace_back(a, clean);
			if (int(calls.size()) == fail) return false;
			*v = Word(a, clean);
			return true;
		}
	};
	const auto raw = +[](void* p, uint64_t a, uint32_t* v) {
		return static_cast<Memory*>(p)->Read(a, v, false);
	};
	const auto clean =
	    +[](void* p, uint64_t a, uint32_t* v) { return static_cast<Memory*>(p)->Read(a, v, true); };
	const auto span = +[](void* p, uint64_t a, uint32_t* v, uint32_t count, bool clean) {
		auto& m = *static_cast<Memory*>(p);
		++m.probes;
		if (!m.accept || m.fail != -1) return false;
		Check(count >= 2 && count <= 16, "invalid compiled span width");
		++m.spans;
		for (uint32_t i = 0; i < count; ++i) {
			m.calls.emplace_back(a + i * 4, clean);
			v[i] = Memory::Word(a + i * 4, clean);
		}
		return true;
	};
	for (bool buffer: {false, true})
		for (bool mixed: {false, true})
			for (uint32_t step: {4u, 8u}) {
				Fixture    f;
				const auto low = f.UserData(0), high = f.UserData(1), records = f.UserData(2);
				const auto handle =
				    buffer ? f.Buffer({low, high, records, Value(0u)}) : f.Address(low, high);
				for (uint32_t i = 0; i < 32; ++i) {
					MemoryInfo info;
					info.kind   = buffer ? ResourceKind::ScalarBuffer : ResourceKind::ScalarAddress;
					info.offset = i * step;
					info.planning_only = true;
					const auto flags   = f.AddMemory(info, 0);
					const auto read =
					    buffer ? f.Emit(ValueOpcode::ReadConstBuffer, {handle, Value(0u)}, flags)
					           : f.Emit(ValueOpcode::LoadAddressU32,
					                    {handle, Value(0u), Value(0u), Value(true)}, flags);
					f.program.srt_reads.push_back({read, i});
				}
				f.program.srt_plan_complete = true;
				auto plan                   = ExtractResourcePlan(f.program);
				if (mixed) {
					plan.clean_flat_slots.resize(32);
					for (uint32_t i = 0; i < 16; ++i)
						plan.clean_flat_slots[i] = 1;
					BuildLinearSrtPlan(plan);
				}
				Check(bool(plan.linear_srt), "span fixture did not compile");
				const auto compiled = plan.linear_srt;
				for (uint32_t trial = 0; trial < 9; ++trial) {
					// Disable/decline spans, fail an intermediate scalar read, and fail
					// later bounds after earlier reads: all retain the evaluator's exact
					// sequence.
					const uint32_t data[] = {trial == 8 ? 0xfffffffcu : 0x1003u,
					                         trial == 8 ? 0xffffu : 0u, trial == 7 ? 20u : 256u};
					Memory reference, actual;
					reference.fail = actual.fail = trial == 5 ? 3 : trial == 6 ? 17 : -1;
					actual.accept                = trial != 4;
					SrtRuntime                   r {.user_data                  = data,
					                                .read_memory                = raw,
					                                .userdata                   = &reference,
					                                .read_specialization_memory = clean};
					std::vector<DescriptorValue> er, ar;
					std::vector<uint32_t>        ev {0xdeadbeef}, av = ev;
					std::vector<uint8_t>         ea {99}, aa         = ea;
					plan.linear_srt.reset();
					const bool ok =
					    EvaluateRuntimeSources(plan, {}, r, er, ev, plan.clean_flat_slots, ea);
					r.userdata             = &actual;
					r.try_read_memory_span = (trial == 1 || trial == 2) ? nullptr : span;
					if (trial == 3) r.user_data = {};
					plan.linear_srt = compiled;
					const bool got =
					    EvaluateRuntimeSources(plan, {}, r, ar, av, plan.clean_flat_slots, aa);
					if (trial == 3)
						Check(!got && av == std::vector<uint32_t> {0xdeadbeef} &&
						          actual.calls.empty() && !actual.probes,
						      "span read crossed failing user data access");
					else
						Check(ok == got && ev == av && ea == aa && reference.calls == actual.calls,
						      "span execution changed data, read order, bounds or "
						      "transaction failure");
					if (trial == 0 && step == 4)
						Check(actual.spans == 2 && actual.probes == 2,
						      "contiguous scalar groups were not executed");
					if (trial == 1 || trial == 2 || step == 8)
						Check(!actual.spans, "disabled or noncontiguous span was read");
					if (trial >= 4 && trial <= 6)
						Check(!actual.spans, "declined probe performed a read");
				}
			}
	Fixture    p;
	MemoryInfo info;
	info.kind          = ResourceKind::ScalarAddress;
	info.planning_only = true;
	auto load          = [&](Value handle, uint32_t offset) {
        info.offset = offset;
        return p.Emit(ValueOpcode::LoadAddressU32, {handle, Value(0u), Value(0u), Value(true)},
		                       p.AddMemory(info, 0));
	};
	const auto parent           = load(p.Address(p.UserData(0), Value(0u)), 0);
	const auto child            = p.Address(parent, Value(0u));
	p.program.srt_reads         = {{parent, 0}, {load(child, 0), 1}, {load(child, 4), 2}};
	p.program.srt_plan_complete = true;
	auto plan                   = ExtractResourcePlan(p.program);
	Check(bool(plan.linear_srt), "pointer span fixture did not compile");
	struct Pointers {
		uint32_t              base = 0x2000, probes = 0;
		std::vector<uint64_t> calls;
		const ResourcePlan*   plan = nullptr;
	};
	const auto pointer_read = +[](void* v, uint64_t a, uint32_t* word) {
		auto& m = *static_cast<Pointers*>(v);
		m.calls.push_back(a);
		if (a == 0x1000) {
			*word = m.base;
			return true;
		}
		if (a == m.base || a == m.base + 4) {
			*word = uint32_t(a);
			return true;
		}
		return false;
	};
	const auto pointer_span = +[](void* v, uint64_t a, uint32_t* words, uint32_t n, bool clean) {
		auto& m = *static_cast<Pointers*>(v);
		++m.probes;
		Check(!clean && a == m.base && n == 2,
		      "span crossed a parent dependency or retained old child address");
		// A span callback can reenter the same generated function with different
		// inputs; nested scratch and outputs must not overwrite this transaction.
		Pointers inner;
		inner.base            = m.base + 0x1000;
		const auto inner_read = +[](void* v, uint64_t a, uint32_t* word) {
			auto& m = *static_cast<Pointers*>(v);
			if (a == 0x1000) {
				*word = m.base;
				return true;
			}
			if (a == m.base || a == m.base + 4) {
				*word = uint32_t(a);
				return true;
			}
			return false;
		};
		const uint32_t        inputs[] = {0x1000};
		std::vector<uint32_t> result;
		const auto inner_span = +[](void* v, uint64_t a, uint32_t* words, uint32_t n, bool clean) {
			auto& m = *static_cast<Pointers*>(v);
			Check(!clean && a == m.base && n == 2, "nested span address mismatch");
			for (uint32_t i = 0; i < n; ++i)
				words[i] = uint32_t(a + i * 4);
			return true;
		};
		Check(WalkSrt(*m.plan,
		              {.user_data            = inputs,
		               .read_memory          = inner_read,
		               .userdata             = &inner,
		               .try_read_memory_span = inner_span},
		              result) &&
		          result == std::vector<uint32_t> {inner.base, inner.base, inner.base + 4},
		      "nested span evaluation failed");
		for (uint32_t i = 0; i < n; ++i) {
			m.calls.push_back(a + i * 4);
			words[i] = uint32_t(a + i * 4);
		}
		return true;
	};
	Pointers memory;
	memory.plan           = &plan;
	const uint32_t data[] = {0x1000};
	for (uint32_t base: {0x2000u, 0x4000u}) {
		memory.base = base;
		memory.calls.clear();
		std::vector<uint32_t> values;
		Check(WalkSrt(plan,
		              {.user_data            = data,
		               .read_memory          = pointer_read,
		               .userdata             = &memory,
		               .try_read_memory_span = pointer_span},
		              values) &&
		          values == std::vector<uint32_t> {base, base, base + 4} &&
		          memory.calls == std::vector<uint64_t> {0x1000, base, base + 4},
		      "pointer span read stale child data");
	}
	Check(memory.probes == 2, "dependent child groups were not executed");
#endif
}

void TestLinearSrtReads() {
#if defined(__x86_64__) || defined(_M_X64)
  struct Memory {std::vector<std::pair<uint64_t,bool>> calls;int fail=-1;uint32_t salt=0;};
  const auto raw=+[](void* p,uint64_t a,uint32_t* v){
    auto& m=*static_cast<Memory*>(p);m.calls.emplace_back(a,false);
    if(int(m.calls.size())==m.fail)return false;*v=uint32_t(a^(a>>32))^m.salt^0x6821u;return true;
  };
  const auto clean=+[](void* p,uint64_t a,uint32_t* v){
    auto& m=*static_cast<Memory*>(p);m.calls.emplace_back(a,true);
    if(int(m.calls.size())==m.fail)return false;*v=uint32_t(a^(a>>32))^m.salt^0x1847u;return true;
  };
  for(bool buffer:{false,true})for(uint32_t immediate:{0u,4u,0x7ffffffcu,0xfffffffcu}) {
    Fixture f;
    MemoryInfo info;info.kind=buffer?ResourceKind::ScalarBuffer:ResourceKind::ScalarAddress;info.offset=immediate;info.planning_only=true;
    const auto flags=f.AddMemory(info,0);
    const auto handle=buffer?f.Buffer({f.UserData(0),f.UserData(1),f.UserData(3),f.UserData(4)}):f.Address(f.UserData(0),f.UserData(1));
    const auto value=buffer?f.Emit(ValueOpcode::ReadConstBuffer,{handle,f.UserData(2)},flags):
      f.Emit(ValueOpcode::LoadAddressU32,{handle,f.UserData(2),Value(0u),Value(true)},flags);
    const auto alias=f.Emit(ValueOpcode::ReadConst,{Value(0u),Value(0u)});
    f.program.srt_reads={{value,0},{value,1},{alias,2}};f.program.srt_plan_complete=true;
    auto plan=ExtractResourcePlan(f.program);
    // Distinct raw/clean evaluator caches must be preserved when the same
    // raw node is reached through a clean alias and as a raw output.
    plan.clean_flat_slots={1,0,0};BuildLinearSrtPlan(plan);
    Check(plan.linear_srt!=nullptr,"whole-graph scalar fixture did not compile");
    const auto compiled=plan.linear_srt;
    uint32_t seed=0x37bf5182;
    for(uint32_t trial=0;trial<256;++trial) {
      std::array<uint32_t,5> data;
      for(auto& word:data){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;word=seed;}
      if(trial<64){data[0]=0x1000;data[1]&=0xffff;data[2]=trial;data[3]=trial/2;}
      if(trial==64){data[0]=0;data[1]=0;data[2]=0;}
      if(trial==65){data[0]=0xfffffffc;data[1]=0xffff;data[2]=UINT32_MAX;}
      Memory reference,native;reference.fail=native.fail=trial%7==0?1:trial%7==1?2:-1;reference.salt=native.salt=trial;
      std::vector<DescriptorValue> ev,av;std::vector<uint32_t> expected{1234},actual=expected;std::vector<uint8_t> ea{9},aa=ea;
      SrtRuntime r{.user_data=data,.read_memory=raw,.userdata=&reference,.read_specialization_memory=clean};
      plan.linear_srt.reset();
      const bool old_ok=EvaluateRuntimeSources(plan,{},r,ev,expected,plan.clean_flat_slots,ea);
      r.userdata=&native;plan.linear_srt=compiled;
      const bool new_ok=EvaluateRuntimeSources(plan,{},r,av,actual,plan.clean_flat_slots,aa);
      Check(old_ok==new_ok && expected==actual && ea==aa && reference.calls==native.calls,
            "linear scalar read changed reader, order, memoization, address, or failure");
      Check(native.calls.size()<=2,"shared read was repeated inside a graph");
    }
  }
  // Reader callbacks may reenter evaluation. Each invocation needs distinct
  // scratch storage even when both calls execute the same generated function.
  Fixture reentrant;
  MemoryInfo read_info;read_info.kind=ResourceKind::ScalarAddress;read_info.planning_only=true;
  const auto read_flags=reentrant.AddMemory(read_info,0);
  const auto address=reentrant.Address(reentrant.UserData(0),Value(0u));
  const auto read=reentrant.Emit(ValueOpcode::LoadAddressU32,{address,Value(0u),Value(0u),Value(true)},read_flags);
  const auto sum=reentrant.Emit(ValueOpcode::IAdd32,{reentrant.UserData(1),read});
  reentrant.program.srt_reads={{sum,0},{read,1}};reentrant.program.srt_plan_complete=true;
  auto reentrant_plan=ExtractResourcePlan(reentrant.program);
  Check(reentrant_plan.linear_srt!=nullptr,"reentrant graph was not compiled");
  struct Nested {const ResourcePlan* plan;bool nested=false,ok=false;uint32_t calls=0;};
  Nested nested{&reentrant_plan};
  const auto recursive=+[](void* data,uint64_t address,uint32_t* output) {
    auto& state=*static_cast<Nested*>(data);++state.calls;
    if(state.nested){*output=17;return true;}
    state.nested=true;
    const uint32_t words[]={0x2000,99};std::vector<uint32_t> result;
    const auto inner=+[](void* data,uint64_t address,uint32_t* output){
      auto& state=*static_cast<Nested*>(data);++state.calls;*output=17;return address==0x2000 && state.nested;
    };
    state.ok=WalkSrt(*state.plan,{.user_data=words,.read_memory=inner,.userdata=data},result) &&
             result==std::vector<uint32_t>{116,17};
    state.nested=false;*output=31;return state.ok && address==0x1000;
  };
  const uint32_t inputs[]={0x1000,7};std::vector<uint32_t> reentrant_result;
  Check(WalkSrt(reentrant_plan,{.user_data=inputs,.read_memory=recursive,.userdata=&nested},reentrant_result) &&
        nested.ok && nested.calls==2 && reentrant_result==std::vector<uint32_t>{38,31},
        "reader reentry corrupted an outer graph or repeated a shared load");
  // Reject unsupported graphs before evaluating anything; never partially run
  // a generated prefix and then retry the original reader sequence.
  Fixture unsupported;unsupported.program.srt_plan_complete=true;
  unsupported.program.srt_reads={{unsupported.Emit(ValueOpcode::UndefU32,{}),0}};
  auto no_plan=ExtractResourcePlan(unsupported.program);
  Check(!no_plan.linear_srt,"unsupported graph acquired executable code");
  std::vector<uint32_t> unchanged{0xbeef};
  Check(!WalkSrt(no_plan,{},unchanged) && unchanged==std::vector<uint32_t>{0xbeef},"fallback changed failed output");
#endif
}


int main(int argc, char** argv) {
#ifdef KYTY_LOCAL_NATIVE_RESOURCES
  // The same cases through the compiled SRT paths.
  if (const char* mode = std::getenv("KYTY_SRT_NATIVE")) {
    kyty_local_srt_native_mode.store(std::strtoul(mode, nullptr, 10));
  }
  if (const char* mode = std::getenv("KYTY_SRT_PREDICATES")) {
    kyty_local_srt_predicate_mode.store(std::strtoul(mode, nullptr, 10));
  }
#endif
  if (argc == 2 && std::strcmp(argv[1], "--benchmark-srt") == 0) {
    Fixture fixture;
    const auto input = fixture.UserData(0);
    constexpr uint32_t count = 1024, repeats = 5000;
    for (uint32_t i = 0; i < count; ++i) {
      auto value = fixture.Emit(ValueOpcode::IAdd32, {input, Value(i)});
      value = fixture.Emit(ValueOpcode::BitwiseAnd32, {value, Value(0xffffu)});
      fixture.program.srt_reads.push_back({value, i});
    }
    fixture.program.srt_plan_complete = true;
    auto plan = ExtractResourcePlan(fixture.program);
    std::vector<uint32_t> result;
    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t iteration = 0; iteration < repeats; ++iteration) {
      Check(WalkSrt(plan, {.user_data = std::span(&iteration, 1)}, result), "benchmark evaluation failed");
      checksum += result.back();
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "SRT nodes=" << plan.value_storage.size() << " batches=" << repeats
              << " seconds=" << seconds << " checksum=" << checksum << '\n';
    return 0;
  }

  try {
    // KYTY_TEST_FILTER: only the tests whose name contains it.
    const char *filter = std::getenv("KYTY_TEST_FILTER");
    const auto Run = [filter](const char *name, auto test) {
      if (filter != nullptr && std::strstr(name, filter) == nullptr) return;
      try {
        test();
      } catch (const std::exception &exception) {
        throw std::runtime_error(std::string(name) + ": " + exception.what());
      }
    };
    Run("native predicate bitfields", TestNativePredicateBitfields);
    Run("controlled linear SRT", TestControlledLinearSrt);
    Run("linear SRT arithmetic", TestLinearSrtDifferential);
    Run("linear SRT reads", TestLinearSrtReads);
	Run("linear SRT grouped reads", TestLinearSrtSpans);
	Run("compiled SRT span bounds", TestCompiledSrtSpanBounds);
	Run("private LDS scalar slots", TestFunctionLdsLayout);
	Run("dense buffers", TestDenseBufferTracking);
    Run("compute buffer fill", TestComputeBufferFill);
    Run("scalar/vector alias", TestScalarAndVectorBufferAlias);
    Run("runtime unsigned greater equal", TestRuntimeUnsignedGreaterEqual);
    Run("runtime unsigned min", TestRuntimeUnsignedMinDescriptor);
    Run("images and samplers", TestImagesSamplersAndAliases);
    Run("SampleAdjust sampler scratch", TestSampleAdjustSamplerScratch);
    Run("FMASK load specialization", TestFmaskLoadSpecialization);
    Run("dynamic storage mips", TestDynamicStorageMipTracking);
    Run("invariant indirect images", TestInvariantIndirectImageMaterialization);
    Run("SRT runtime", TestSrtFlatteningAndRuntimeMemoization);
    Run("dynamic SRT", TestDynamicSrtReadRemainsExplicit);
    Run("phi validation", TestPhiValidation);
    Run("dense indirect images", TestDenseIndirectImageMaterialization);
    Run("loop-bounded dense images", TestLoopBoundedDenseIndirectImage);
    Run("readlane probe images", [] { TestReadLaneProbeIndirectImage(false); });
    Run("readfirstlane probe images", [] { TestReadLaneProbeIndirectImage(true); });
    Run("readfirstlane probe images (span reads)",
        [] { TestReadLaneProbeIndirectImage(true, false, false, true); });
    Run("readfirstlane loop mask", [] { TestReadLaneProbeIndirectImage(true, true); });
    Run("readfirstlane invalid mask", [] { TestReadLaneProbeIndirectImage(true, false, true); });
    Run("readfirstlane invalid loop mask", [] { TestReadLaneProbeIndirectImage(true, true, true); });
    Run("lsb-keyed dense images", TestFindLsbDenseIndirectImage);
    Run("waterfall descriptor match", TestWaterfallDescriptorMatch);
    Run("waterfall AND-NOT clear", TestWaterfallAndNotClear);
    Run("waterfall immediate table", TestWaterfallImmediateTable);
    Run("waterfall near misses", TestWaterfallNearMissesRejected);
    Run("waterfall rewrite", TestWaterfallRewriteDescalarizes);
    Run("null descriptor path", TestNullDescriptorPathCollapse);
    Run("mixed null descriptor paths", TestMixedNullDescriptorPathsRejected);
    Run("runtime-rooted loop", TestLoopCycleEnteredThroughRuntimeValue);
    Run("large runtime evaluation", TestLargeRuntimeEvaluation);
    Run("extracted runtime evaluation", TestExtractedRuntimeEvaluation);
    Run("invariant loop phi", TestInvariantLoopPhi);
    Run("DMA address materialization", TestDmaAddressMaterialization);
    Run("dynamic FLAT address", TestDynamicFlatAddressesUseDma);
    Run("buffer swizzle specialization", TestBufferSwizzleSpecialization);
    Run("conditional buffer materialization", TestConditionalBufferMaterialization);
    Run("conservative buffer reachability", TestConservativeBufferReachability);
    Run("conditional indirect image", TestConditionalIndirectImageMaterialization);
    Run("shader info and bindings", TestShaderInfoAndBindingLayout);
    Run("image binding ABI", TestImageBindingAbi);
    Run("LOD feedback binding layout", TestLodStatsBindingLayout);
    Run("graphics push constants", TestGraphicsPushConstantLayout);
    Run("resource limit", TestResourceLimitIsTransactional);
    Run("malformed memory kinds", TestMalformedMemoryKindsRejected);
  } catch (const std::exception &exception) {
    std::cerr << "resource tracking test failed: " << exception.what() << '\n';
    return 1;
  }
  std::cout << "resource tracking tests passed\n";
  return 0;
}

// The full emulator supplies these assertion hooks through common. This focused
// target links only fmt; keep assertion failures observable without widening
// its focused build manifest.
namespace Common {
int DbgExitHandler(const char *, int, std::string_view text) {
  throw std::runtime_error(std::string(text));
}

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view text) {
  throw std::runtime_error(std::string(text));
}

int DbgExitIfHandler(const char *expression, const char *file, int line) {
  throw std::runtime_error(std::string("typed IR assertion: ") + expression +
                           " at " + file + ':' + std::to_string(line));
}

int DbgNotImplementedHandler(const char *expression, const char *file,
                             int line) {
  throw std::runtime_error(std::string("typed IR not implemented: ") +
                           expression + " at " + file + ':' +
                           std::to_string(line));
}

void DbgExit(int) { throw std::runtime_error("typed IR assertion failed"); }
} // namespace Common

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.cpp"
