#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ImageForms.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "SrtWalkerTestMemory.h"

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ResourceMaterializationTests: failed: %s\n", text);
    std::abort();
  }
}

// Every extracted plan below is the one the game's precompile loads from the stage store: exactly its bytes load, the
// loaded plan saves back to them, and a cut or padded record does not load.
Libs::Graphics::ShaderRecompiler::IR::ResourcePlan
Extract(const Libs::Graphics::ShaderRecompiler::IR::Program &program) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const auto bytes = SaveResourcePlan(ExtractResourcePlan(program));
  auto extended = bytes;
  extended.push_back(0u);
  ResourcePlan loaded, truncated, padded;
  Check(!bytes.empty() && LoadResourcePlan(bytes, loaded) && SaveResourcePlan(loaded) == bytes &&
            !LoadResourcePlan(std::span(bytes).first(bytes.size() - 1u), truncated) &&
            !LoadResourcePlan(extended, padded),
        "resource plan did not round-trip through its exact stage store bytes");
  // A Phi predecessor past the blocks (a corrupt file) does not load: the walker indexes its reach bytes by it.
  if (!loaded.control_phi_predecessors.empty()) {
    auto &blocks = loaded.control_phi_predecessors.begin()->second;
    const auto kept = blocks;
    blocks[0] = static_cast<uint32_t>(loaded.control_flow.size());
    const auto corrupt_bytes = SaveResourcePlan(loaded);
    blocks.push_back(0u);
    const auto extra_bytes = SaveResourcePlan(loaded);
    blocks = kept;
    ResourcePlan corrupt, extra;
    Check(!corrupt_bytes.empty() && !LoadResourcePlan(corrupt_bytes, corrupt) && !LoadResourcePlan(extra_bytes, extra),
          "a plan with a Phi predecessor past its blocks, or one too many, loaded");
  }
  return loaded;
}

// The x86 walk and the interpreter's refresh agree on everything both know, and the walker knows
// nothing the interpreter does not.
void CheckWalkAgrees(const Libs::Graphics::ShaderRecompiler::IR::ResourcePlan &plan,
                     const Libs::Graphics::ShaderRecompiler::IR::SrtRuntime &runtime, const char *text,
                     uint32_t reach_smaller = 0) {
  const auto comparison = Libs::Graphics::ShaderRecompiler::IR::CompareWalk(plan, runtime);
  const bool agrees = comparison.walked && comparison.refreshed && comparison.mismatches == 0 &&
                      comparison.walker_only == 0 && comparison.reach_smaller == reach_smaller;
  if (!agrees) {
    std::fprintf(stderr,
                 "ResourceMaterializationTests: walked %d refreshed %d, mismatches %u, walker only %u, "
                 "reach smaller %u: %s\n",
                 comparison.walked, comparison.refreshed, comparison.mismatches, comparison.walker_only,
                 comparison.reach_smaller, comparison.first.c_str());
  }
  Check(agrees, text);
}

// Serves the walker's loads from a test runtime's readers (the strict one first). A load neither
// reads fails the walk, as an unmapped load outside the null page does in the emulator.
bool ReadThroughRuntime(void *userdata, uint64_t address, std::span<uint32_t> words) {
  const auto &runtime = *static_cast<const Libs::Graphics::ShaderRecompiler::IR::SrtRuntime *>(userdata);
  for (const auto reader : {runtime.read_specialization_memory, runtime.read_memory}) {
    if (reader != nullptr && reader(runtime.userdata, address, words)) return true;
  }
  Libs::Graphics::ShaderRecompiler::IR::SrtWalkerFaultedLoad(UINT64_MAX);
  return false;
}

void CheckWalkAgreesWith(const Libs::Graphics::ShaderRecompiler::IR::ResourcePlan &plan,
                         const Libs::Graphics::ShaderRecompiler::IR::SrtRuntime &runtime, const char *text,
                         bool walks = true, uint32_t reach_smaller = 0) {
  FakeGuestMemory memory(ReadThroughRuntime, const_cast<void *>(static_cast<const void *>(&runtime)));
  auto direct = runtime;
  direct.direct_reads = true;
  // On the GPU thread both readers give the guest's words.
  if (direct.read_specialization_memory == nullptr) direct.read_specialization_memory = direct.read_memory;
  const auto comparison = Libs::Graphics::ShaderRecompiler::IR::CompareWalk(plan, direct);
  const bool agrees = comparison.walked == walks && comparison.mismatches == 0 &&
                      comparison.walker_only == 0 && comparison.reach_smaller == reach_smaller;
  if (!agrees) {
    std::fprintf(stderr,
                 "ResourceMaterializationTests: walked %d refreshed %d, mismatches %u, walker only %u, "
                 "reach smaller %u: %s\n",
                 comparison.walked, comparison.refreshed, comparison.mismatches, comparison.walker_only,
                 comparison.reach_smaller, comparison.first.c_str());
  }
  Check(agrees, text);
}

bool RejectSpecializationRead(void *userdata, uint64_t, std::span<uint32_t>) {
  ++*static_cast<uint32_t *>(userdata);
  return false;
}

bool SameSnapshot(const Libs::Graphics::ShaderRecompiler::IR::ResourceSnapshot &a,
                  const Libs::Graphics::ShaderRecompiler::IR::ResourceSnapshot &b) {
  return a.buffers == b.buffers && a.images == b.images && a.samplers == b.samplers &&
         a.flattened_srt == b.flattened_srt && a.image_heap.base == b.image_heap.base &&
         a.image_heap.size == b.image_heap.size && a.sampler_heap.base == b.sampler_heap.base &&
         a.sampler_heap.size == b.sampler_heap.size && a.user_data == b.user_data &&
         a.specialization_reads == b.specialization_reads && a.uniform_fill == b.uniform_fill &&
         a.buffer_writes == b.buffer_writes && a.buffer_write_ranges == b.buffer_write_ranges;
}

Libs::Graphics::ShaderRecompiler::IR::Block &
AddValueBlock(Libs::Graphics::ShaderRecompiler::IR::Program &program) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto block = std::make_unique<Block>();
  auto *result = block.get();
  program.blocks.push_back(result);
  result->id = static_cast<uint32_t>(program.blocks.size() - 1u);
  program.block_storage.push_back(std::move(block));
  return *result;
}

Libs::Graphics::ShaderRecompiler::IR::Program SrtProgram(uint64_t address) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  const auto low = Value(static_cast<uint32_t>(address));
  const auto high = Value(static_cast<uint32_t>(address >> 32u));
  auto &handle =
      value_block.AppendNewInst(ValueOpcode::GetAddressResource, {low, high});
  auto &raw = value_block.AppendNewInst(
      ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(0u), Value(0u), Value(true)});
  raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40});
  program.srt_reads.push_back({Value(&raw), 0});

  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  auto &flat = value_block.AppendNewInst(ValueOpcode::ReadConst,
                                         {Value(&srt), Value(0u)});
  DescriptorSource source;
  source.dwords[0] = Value(&flat);
  source.dwords[1] = Value(0u);
  source.dword_count = 2;
  program.descriptor_sources.push_back(source);
  return program;
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UnbasedFlatPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);
  return Extract(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UserDataBufferPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  auto &user_data = value_block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  DescriptorSource source;
  source.dwords[0] = Value(&user_data);
  source.dwords[1] = Value(0u);
  source.dwords[2] = Value(0u);
  source.dwords[3] = Value(0u);
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  return Extract(program);
}

Libs::Graphics::ShaderRecompiler::IR::Program MixedSamplerProgram() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);

  const auto AddSource = [&program](uint32_t dword_count) {
    DescriptorSource source;
    source.dword_count = dword_count;
    for (uint32_t i = 0; i < dword_count; i++) {
      source.dwords[i] = Value(0u);
    }
    program.descriptor_sources.push_back(source);
    return static_cast<uint32_t>(program.descriptor_sources.size() - 1u);
  };

  const auto image0 = AddSource(8);
  const auto image1 = AddSource(8);
  const auto sampler0 = AddSource(4);
  const auto sampler1 = AddSource(4);
  program.descriptor_sources[image1].dwords[0] = Value(1u);
  program.descriptor_sources[image1].dwords[1] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::BufferFormat::k11_11_10UInt) << 20u);
  program.descriptor_sources[image1].dwords[3] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::ImageType::kColor2D) << 28u);
  for (uint32_t index = 0; index < 2; ++index) {
    auto &value = block.AppendNewInst(
        ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(index))});
    program.descriptor_sources[sampler0 + index].dwords[0] = Value(&value);
  }
  program.info.images.push_back(
      {.source = image0,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.images.push_back(
      {.source = image1,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.samplers.push_back({.source = sampler0});
  program.info.samplers.push_back({.source = sampler1});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 0});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 1});
  program.info.sampled_pairs.push_back({.image = 1, .sampler = 1});
  return program;
}

void TestMappedSrtUsesDirectReaderByDefault() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const uint32_t dword = 0x12345678;
  auto plan = Extract(SrtProgram(reinterpret_cast<uint64_t>(&dword)));
  uint32_t specialization_reads = 0;
  const SrtRuntime runtime{.userdata = &specialization_reads,
                           .read_specialization_memory =
                               RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mapped SRT stage materialization failed");
  Check(specialization_reads == 0,
        "ordinary SRT read used the specialization reader");
  Check(snapshot.flattened_srt.size() == 1 &&
            snapshot.flattened_srt[0] == dword,
        "cache rematerialization did not use the direct reader by default");
}

void TestNativeNestedSrtRefresh() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  std::array<uint32_t, 3> first{0u, 0x12345678u, 0xabcdef01u};
  std::array<uint32_t, 3> second{0u, 0x87654321u, 0x10fedcbau};
  const auto pointer_words = [](const void *pointer) {
    const auto address = reinterpret_cast<uint64_t>(pointer);
    return std::array<uint32_t, 2>{static_cast<uint32_t>(address) | 3u,
                                  static_cast<uint32_t>(address >> 32u) | 0xabcd0000u};
  };
  auto table = pointer_words(first.data());
  const auto user_data = pointer_words(table.data());
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 2;
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto low = emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))});
  const auto high = emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))});
  const auto root = emit(ValueOpcode::GetAddressResource, {low, high});
  const auto read = [&](Value address, uint32_t offset) {
    const auto value = emit(ValueOpcode::LoadAddressU32,
                            {address, Value(offset), Value(0u), Value(true)});
    value.Instruction()->SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back(
        {value, static_cast<uint32_t>(program.srt_reads.size())});
    return value;
  };
  const auto pointer_low = read(root, 0u);
  const auto pointer_high = read(root, 4u);
  auto *leaf = program.srt_reads[0].value.Instruction();
  leaf->Arg(0).Instruction()->SetArg(0, pointer_low);
  leaf->Arg(0).Instruction()->SetArg(1, pointer_high);
  leaf->SetArg(1, Value(7u));
  read(leaf->Arg(0), 11u);
  auto plan = Extract(program);
  Check(plan.srt_code != nullptr, "nested user-data tree did not compile an x86 walker");
  // The descriptor uses the first leaf word and the pointer words address the leaves; only the shader reads the last.
  Check(plan.srt_read_host == std::vector<uint8_t>{1u, 1u, 1u, 0u},
        "pointer words not told apart from shader-only words");
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const SrtRuntime runtime{.user_data = user_data};
  {
    SrtWalker walker(plan, runtime);
    Check(walker.RefreshFlatBuffer(snapshot.flattened_srt),
          "native SRT refresh failed");
    const auto loaded = first[1];
    first[1] ^= 0xffffffffu;
    uint32_t value = 0;
    Check(walker.Evaluate(plan.srt_reads[0].value, value) && value == loaded,
          "descriptor evaluation reloaded a value already copied by the native walker");
    first[1] = loaded;
  }
  auto direct = runtime;
  direct.direct_reads = true;
  for (const auto *data : {&first, &second}) {
    table = pointer_words(data->data());
    for (const auto &walk : {runtime, direct}) {
      Check(MaterializeResources(plan, walk, snapshot, specialization) &&
                plan.srt_walked == walk.direct_reads &&
                snapshot.flattened_srt ==
                    std::vector<uint32_t>{(*data)[1], table[0], table[1], (*data)[2]},
            "native SRT refresh lost pointer masking, alignment, or current values");
    }
    CheckWalkAgrees(plan, direct, "the walker and the interpreter refreshed a nested SRT differently");
  }
  {
    // A logged materialization that walked repeats while the walk gives the same again.
    SrtReadLog log;
    table = pointer_words(first.data());
    Check(MaterializeResources(plan, log.Recording(direct), snapshot, specialization) && plan.srt_walked,
          "a logged materialization did not walk");
    log.RecordWalk(plan, snapshot.flattened_srt);
    Check(log.Repeats(direct, plan), "an unchanged walk did not repeat");
    first[2] ^= 1u;
    Check(!log.Repeats(direct, plan), "a walk over a changed word repeated");
    first[2] ^= 1u;
    Check(log.Repeats(direct, plan), "the restored words did not repeat");
    table = pointer_words(second.data());
    Check(!log.Repeats(direct, plan), "a walk through another table repeated");
    Check(!log.Repeats(runtime, plan), "a walk repeated for a runtime without direct reads");
    // The changed walk the check made is the materialization's.
    const std::vector<uint32_t> *walked = nullptr;
    Check(!log.Repeats(direct, plan, &walked) && walked != nullptr &&
              *walked == std::vector<uint32_t>{second[1], table[0], table[1], second[2]},
          "a changed walk was not handed on");
    ResourceSnapshot handed;
    ResourceSpecialization handed_specialization;
    Check(MaterializeResources(plan, direct, handed, handed_specialization, walked) && plan.srt_walked &&
              handed.flattened_srt == *walked,
          "a handed walk did not materialize");
    Check(MaterializeResources(plan, direct, snapshot, specialization) &&
              snapshot.flattened_srt == handed.flattened_srt && specialization == handed_specialization,
          "a handed walk materialized differently from a fresh walk");
    Check(!log.Repeats(runtime, plan, &walked) && walked == nullptr,
          "a check without a walk handed one on");
  }
  {
    // Loads through a null pointer fault and read zero (bdf48dcf3).
    FakeGuestMemory unmapped;
    table = {3u, 0xabcd0000u};
    Check(MaterializeResources(plan, direct, snapshot, specialization) && plan.srt_walked &&
              snapshot.flattened_srt == std::vector<uint32_t>{0u, 3u, 0xabcd0000u, 0u},
          "native SRT walk did not read a null table as zero");
  }
  Check(!MaterializeResources(plan, {.user_data = std::span(user_data).first(1)},
                              snapshot, specialization),
        "native SRT walk accepted truncated user data");
  const std::array<uint32_t, 2> fake_user_data{0x1003u, 0xabcd0000u};
  uint32_t reads = 0;
  const SrtRuntime callback_runtime{
      .user_data = fake_user_data,
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++*static_cast<uint32_t *>(data);
        switch (address) {
          case 0x1000u: words[0] = 0x2003u; break;
          case 0x1004u: words[0] = 0xabcd0000u; break;
          case 0x2004u: words[0] = 0x1234u; break;
          case 0x2008u: words[0] = 0x5678u; break;
          default: return false;
        }
        return true;
      },
      .userdata = &reads};
  Check(MaterializeResources(plan, callback_runtime, snapshot, specialization) && reads == 4 &&
            snapshot.flattened_srt ==
                std::vector<uint32_t>{0x1234u, 0x2003u, 0xabcd0000u, 0x5678u},
        "native-eligible SRT bypassed its callback or repeated nested pointer reads");
  leaf->SetArg(0, root);
  leaf->SetArg(1, Value(0u));
  program.srt_reads.resize(1);
  for (const int32_t offset : {-1, 7}) {
    program.memory_info[0].offset = static_cast<uint32_t>(offset);
    auto bounded = Extract(program);
    auto base = pointer_words(second.data() + 1);
    for (const bool walk : {false, true}) {
      Check(bounded.srt_code != nullptr &&
                MaterializeResources(bounded, {.user_data = base, .direct_reads = walk}, snapshot,
                                     specialization) &&
                snapshot.flattened_srt[0] == second[offset < 0 ? 0 : 2],
            "native SRT applied a signed immediate offset incorrectly");
    }
    base = offset < 0 ? std::array<uint32_t, 2>{0u, 0u}
                      : std::array<uint32_t, 2>{0xfffffffcu, 0xffffu};
    for (const bool walk : {false, true}) {
      Check(!MaterializeResources(bounded, {.user_data = base, .direct_reads = walk}, snapshot,
                                  specialization),
            "native SRT accepted address underflow or 48-bit overflow");
    }
  }
  // The x86 walker (upstream d7856753e) also takes pointer words that are not adjacent, and agrees
  // with the interpreter on them.
  table = pointer_words(first.data());
  const std::array<uint32_t, 3> spread{user_data[0], 0u, user_data[1]};
  high.Instruction()->SetArg(0, Value(ScalarReg(6)));
  program.user_data_count = 3;
  {
    auto spread_plan = Extract(program);
    Check(spread_plan.srt_code != nullptr, "the walker did not compile nonadjacent user-data pointer words");
    CheckWalkAgrees(spread_plan, {.user_data = spread, .direct_reads = true},
                    "nonadjacent user-data pointer words walked differently");
  }
  high.Instruction()->SetArg(0, Value(ScalarReg(5)));
  program.user_data_count = 2;
  program.memory_info[0].offset = 0;
  program.srt_reads.push_back({pointer_low, 1});
  program.srt_reads.push_back({pointer_high, 2});
  leaf->SetArg(0, emit(ValueOpcode::GetAddressResource, {pointer_low, pointer_high}));
  pointer_high.Instruction()->SetArg(1, Value(8u));
  // (Not walked: the high word lies past the two-word table, so the pointer would be garbage.)
  Check(Extract(program).srt_code != nullptr,
        "the walker did not compile nonadjacent nested pointer words");
}

void TestIntegerRuntimeValueFollowsSrtReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = SrtProgram(0x10000);
  const auto root = plan.descriptor_sources.front().dwords[0];
  Check(ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT read was rejected");

  Block values;
  auto &comparison = values.AppendNewInst(ValueOpcode::FPOrdLessThanEqual32,
                                          {Value::F32(1.f), Value::F32(0.f)});
  auto &selection = values.AppendNewInst(
      ValueOpcode::SelectU32, {Value(&comparison), Value(1u), Value(0u)});
  plan.srt_reads[0].value = Value(&selection);
  Check(ValidateRuntimeValue(plan, root),
        "ordinary SRT validation rejected a floating-point dependency");
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT validation missed a hidden floating-point dependency");

  auto &first =
      values.AppendNewInst(ValueOpcode::ReadFirstLane, {root, Value(true)});
  Check(!ValidateRuntimeValue(plan, Value(&first), RuntimeValueType::Integer),
        "read-first-lane lost integer-only SRT validation");

  auto &active = values.AppendNewInst(ValueOpcode::ReadFirstLane,
                                      {Value(&selection), Value(&comparison)});
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point execution mask was accepted as integer-only");

  auto &lane = values.AppendNewInst(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)),
       Value(0u)});
  auto &mask =
      values.AppendNewInst(ValueOpcode::INotEqual32, {Value(&lane), Value(0u)});
  selection.SetArg(0, Value(&mask));
  active.SetArg(1, Value(&mask));
  Check(ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "nonuniform integer execution mask was rejected");
  auto &float_value =
      values.AppendNewInst(ValueOpcode::BitCastU32F32, {Value::F32(1.f)});
  selection.SetArg(2, Value(&float_value));
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point inactive arm was accepted as integer-only");

  plan.srt_reads[0].value = Value(&first);
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "cyclic SRT read-first-lane dependency was accepted");
}

void TestSrtAliasesRetainReadPolicy() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = SrtProgram(0x1000u);
  auto &block = *program.blocks[0];
  const auto append_read = [&](uint32_t address) {
    auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                       {Value(address), Value(0u)});
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(0u), Value(0u), Value(true)});
    // Different reads deliberately share MemoryInfo but have different guest PCs.
    read.SetFlags(MemoryFlags{.index = 0, .pc = address});
    const auto slot = static_cast<uint32_t>(program.srt_reads.size());
    program.srt_reads.push_back({Value(&read), slot});
    auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
    return Value(&block.AppendNewInst(ValueOpcode::ReadConst,
                                      {Value(&srt), Value(slot)}));
  };
  const auto ordinary = append_read(0x2000u);
  const auto pointer = append_read(0x3000u);
  program.srt_reads[0].value.Instruction()->Arg(0).Instruction()->SetArg(0, pointer);
  auto &mask = block.AppendNewInst(ValueOpcode::INotEqual32, {ordinary, Value(0u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane,
                                    {Value(9u), Value(&mask)});
  program.descriptor_sources.push_back({.dwords = {Value(&first), Value(0u)},
                                       .dword_count = 2});
  DescriptorSource indirect;
  indirect.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1};
  program.descriptor_sources.push_back(indirect);
  auto plan = Extract(program);
  for (const auto &inst : plan.value_storage) {
    Check(inst.GetOpcode() != ValueOpcode::ReadConst &&
              inst.GetOpcode() != ValueOpcode::GetSrtResource,
          "retained plan still contains flattened-read aliases");
  }
  Check(plan.descriptor_sources[0].dwords[0] == plan.srt_reads[0].value &&
            plan.srt_reads[0].value.Instruction()->Flags<SrtReadFlags>().clean == 1u &&
            plan.srt_reads[1].value.Instruction()->Flags<SrtReadFlags>().clean == 0u &&
            plan.srt_reads[2].value.Instruction()->Flags<SrtReadFlags>().clean == 1u,
        "alias normalization lost read identity or retained discarded EXEC provenance");
  // Upstream's walker (d7856753e) compiles plans with captured strict reads too; it runs on the GPU thread only.
  Check(plan.srt_code != nullptr, "the walker did not compile a plan with captured strict reads");
  struct Reads { uint32_t strict = 0; uint32_t ordinary = 0; bool dirty = false; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        if (address == 0x2000u) words[0] = 0x2222u;
        else if (address == 0x3000u) words[0] = 0x8000u;
        else if (address == 0x8000u) words[0] = 0xeeeeu;
        else return false;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address,
                                        std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (reads.dirty) return false;
        if (address == 0x3000u) words[0] = 0x1000u;
        else if (address == 0x1000u) words[0] = 0x1111u;
        else return false;
        return true;
      }};
  std::vector<uint32_t> flat;
  DescriptorValue descriptor;
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, &clean);
    Check(walker.RefreshFlatBuffer(flat) &&
              flat == std::vector<uint32_t>{0x1111u, 0x2222u, 0x1000u} &&
              walker.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == 0x1111u &&
              reads.strict == 2 && reads.ordinary == 1,
          "normalized aliases changed nested strict reads or repeated a shared read");
  }
  Check(!SrtWalker(plan, runtime).RefreshFlatBuffer(flat),
        "strict flat read accepted a missing clean evaluator");
  auto no_reader = runtime;
  no_reader.read_specialization_memory = nullptr;
  {
    SrtWalker clean(plan, CleanRuntime(no_reader));
    Check(!SrtWalker(plan, no_reader, &clean).RefreshFlatBuffer(flat),
          "strict flat read accepted a missing strict reader");
  }
  Check(SrtWalker(plan, runtime).EvaluateDescriptor(0, descriptor) &&
            descriptor.dwords[0] == 0xeeeeu && reads.strict == 2 && reads.ordinary == 3,
        "direct descriptor evaluation without a clean evaluator changed read domains");
  reads.dirty = true;
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    Check(!SrtWalker(plan, runtime, &clean).RefreshFlatBuffer(flat) &&
              reads.strict == 3 && reads.ordinary == 3,
          "dirty strict pointer fell back to an ordinary read");
  }
}

void TestUniformVectorDescriptorRead() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  auto &handle = block.AppendNewInst(ValueOpcode::GetBufferResource,
      {Value(0x1000u), Value(4u << 16u), Value(1u), Value(0x16204u)});
  program.memory_info.push_back({.kind = ResourceKind::Buffer});
  auto &count = block.AppendNewInst(ValueOpcode::LoadBufferU32,
      {Value(&handle), Value(0u), Value(0u), Value(0u), Value(true)});
  count.SetFlags(MemoryFlags{.index = 0});
  // The captured indirect kernel shares one read across sibling scalar lane reads,
  // enclosed by a different EXEC mask. Its resource plan must share that read too.
  auto &lane = block.AppendNewInst(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
  auto &active = block.AppendNewInst(ValueOpcode::ULessThan32, {Value(&lane), Value(32u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(true)});
  auto &next = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&count), Value(1u)});
  auto &second = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&next), Value(true)});
  auto &sum = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&first), Value(&second)});
  auto &small = block.AppendNewInst(ValueOpcode::ULessThanEqual32, {Value(&count), Value(72u)});
  auto &selected = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&small), Value(&sum), Value(&count)});
  auto &masked = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&active), Value(&selected), Value(0u)});
  auto &records = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&masked), Value(&active)});
  DescriptorSource source;
  source.dword_count = 4;
  source.dwords = {Value(0x2000u), Value(4u << 16u), Value(&records), Value(0x16204u)};
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  Check(ValidateRuntimeValue(program, Value(&count)), "uniform DWORD count was rejected");
  struct Reads { uint32_t value = 72; uint32_t strict = 0; uint32_t ordinary = 0; bool clean = true; } reads;
  const SrtRuntime runtime{
      .read_memory = [](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        words[0] = 999;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = [](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (!reads.clean || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const bool written : {false, true}) {
    program.info.buffers[0].written = written;
    auto plan = Extract(program);
    Check(plan.control_flow.empty() && plan.requires_specialization_memory &&
              plan.capture_specialization_reads,
          "vector descriptor read depended on incidental control-flow capture");
    reads = {};
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.buffers[0].dwords[2] == 145 && reads.strict == 1 && reads.ordinary == 0 &&
              snapshot.specialization_reads ==
                  std::vector<std::pair<uint64_t, uint64_t>>{{0x1000u, 4u}},
          "vector descriptor input was not read and captured exactly once");
    reads.clean = false;
    reads.strict = 0;
    Check(!MaterializeResources(plan, runtime, snapshot, specialization) &&
              reads.strict == 1 && reads.ordinary == 0,
          "dirty vector descriptor input fell back to an ordinary memory read");
  }
  count.SetArg(4, Value(false));
  auto &inactive = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(false)});
  program.descriptor_sources.push_back({.dwords = {Value(&inactive)}, .dword_count = 1});
  reads.strict = 0;
  uint32_t result = 99;
  {
    const auto plan = Extract(program);
    Check(SrtWalker(plan, runtime).Evaluate(plan.descriptor_sources.back().dwords[0], result) &&
              result == 0 && reads.strict == 0 && reads.ordinary == 0,
          "literal false EXEC read vector memory");
  }
  count.SetArg(4, Value(true));
  handle.SetArg(3, Value(0x204u));
  program.descriptor_sources.back().dwords[0] = Value(&count);
  {
    const auto plan = Extract(program);
    Check(SrtWalker(plan, runtime).Evaluate(plan.descriptor_sources.back().dwords[0], result) &&
              result == 0 && reads.strict == 0 && reads.ordinary == 0,
          "invalid vector buffer format read memory");
  }
  handle.SetArg(3, Value(0x16204u));
  count.SetArg(1, Value(&lane));
  Check(!ValidateRuntimeValue(program, Value(&count)),
        "varying vector address was treated as a uniform descriptor read");
}

void TestExactReciprocalDescriptorArithmetic() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  auto &block = AddValueBlock(program);
  for (const float divisor : {64.f, 128.f, 256.f, 512.f,
                              std::numeric_limits<float>::min(),
                              std::bit_cast<float>(253u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(SrtWalker(program, {}).Evaluate(Value(&reciprocal), result) &&
              result == std::bit_cast<uint32_t>(1.f / divisor),
          "power-of-two reciprocal was not exact");
  }
  for (const float divisor : {0.f, 3.f, -64.f, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::denorm_min(),
                              std::bit_cast<float>(254u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(!SrtWalker(program, {}).Evaluate(Value(&reciprocal), result),
          "unsupported reciprocal rounding or exceptional input was accepted");
  }
}

void TestUnbasedFlatCacheHitMaterializes() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UnbasedFlatPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {}, snapshot, specialization),
        "unbased FLAT stage materialization failed");
  Check(snapshot.buffers.empty() && snapshot.images.empty(),
        "unbased FLAT plan produced unexpected descriptors");
}

void TestWrittenDescriptorPredicateReads(bool memory_condition) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.user_data_count = 1;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &offset = block.AppendNewInst(ValueOpcode::GetUserData,
                                     {Value(static_cast<ScalarReg>(0))});
  auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(&offset), Value(0u), Value(false)});
  read.SetFlags(MemoryFlags{.index = 0});
  DescriptorSource source;
  source.dwords = {Value(&read), Value(0u), Value(4u), Value(0u)};
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  // Only memory-derived host decisions need clean, disjoint descriptor reads.
  auto &condition = block.AppendNewInst(ValueOpcode::IEqual32,
      {memory_condition ? Value(&read) : Value(&offset),
       Value(memory_condition ? 0x8000u : 4u)});
  auto &store_block = AddValueBlock(program);
  AddValueBlock(program);
  program.blocks[0]->condition = Value(&condition);
  program.blocks[0]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::ConditionalBranch;
  program.blocks[0]->terminator.true_block = program.blocks[1];
  program.blocks[0]->terminator.false_block = program.blocks[2];
  program.blocks[1]->id = 1;
  program.blocks[1]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.blocks[2]->id = 2;
  program.blocks[2]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  auto &output = store_block.AppendNewInst(ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]});
  store_block.AppendNewInst(ValueOpcode::StoreBufferU32,
      {Value(&output), Value(0u), Value(0u), Value(0u), Value(1u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  auto plan = Extract(program);
  Check(plan.capture_specialization_reads == memory_condition && !plan.control_flow.empty(),
        "conditional writable descriptor used the wrong memory dependency policy");
  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; bool clean = false; } reads;
  const std::array<uint32_t, 1> user_data{4u};
  const SrtRuntime runtime{
      .user_data = user_data,
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        words[0] = 0x8000u;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (!reads.clean || address != 0x1004u) return false;
        words[0] = 0x8000u;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const bool clean : {false, true}) {
    reads = {.clean = clean};
    // A failed predicate explores both edges; its dirty read also rejects the writable descriptor.
    const bool expected = !memory_condition || clean;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) == expected &&
              reads.ordinary == (memory_condition ? 0u : 1u) &&
              reads.strict == (memory_condition ? 1u : 0u),
          "writable descriptor changed reader policy or bypassed a dirty memory predicate");
    if (!expected) continue;
    if (clean) {
      CheckWalkAgreesWith(plan, runtime, "a writable descriptor's predicate walked differently");
      reads = {.clean = clean};
      Check(MaterializeResources(plan, runtime, snapshot, specialization), "rematerializing failed");
    }
    Check(snapshot.buffers[0].dwords[0] == 0x8000u &&
              snapshot.specialization_reads == (memory_condition
                  ? std::vector<std::pair<uint64_t, uint64_t>>{{0x1004u, 4u}}
                  : std::vector<std::pair<uint64_t, uint64_t>>{}),
          "writable descriptor lost its address or captured an immutable predicate");
  }
}

void TestFailedMaterializationRejectsStage() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UserDataBufferPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, {}, snapshot, specialization),
        "missing runtime user data did not reject the cached stage");
}

void TestSharedControlPredicates() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  constexpr uint32_t depth = 8;
  for (const auto operation : {ValueOpcode::LogicalAnd, ValueOpcode::LogicalOr}) {
    Program program;
    program.user_data_count = 1;
    program.srt_plan_complete = program.resource_tracking_complete = true;
    auto &block = AddValueBlock(program);
    const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
      return Value(&block.AppendNewInst(opcode, args));
    };
    const auto user = emit(ValueOpcode::GetUserData, {Value(ScalarReg(0))});
    const auto uniform = emit(ValueOpcode::INotEqual32, {user, Value(0u)});
    const auto lane = emit(ValueOpcode::GetBuiltin,
        {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
    auto varying = emit(ValueOpcode::INotEqual32, {lane, Value(0u)});
    // Both failed and partially uniform subgraphs share their operands exponentially.
    for (uint32_t i = 0; i < depth; ++i)
      varying = emit(ValueOpcode::LogicalOr, {varying, varying});
    auto condition = emit(operation, {varying, uniform});
    for (uint32_t i = 0; i < depth; ++i)
      condition = emit(ValueOpcode::LogicalOr, {condition, condition});
    condition = emit(ValueOpcode::LogicalNot, {condition});
    condition = emit(ValueOpcode::ConditionRef, {condition});
    for (uint32_t i = 0; i < 3; ++i) AddValueBlock(program);
    for (uint32_t i = 0; i < 2; ++i) {
      program.blocks[i]->condition = condition;
      program.blocks[i]->terminator.kind = TerminatorKind::ConditionalBranch;
      program.blocks[i]->terminator.true_block = program.blocks[i == 0 ? 1 : 3];
      program.blocks[i]->terminator.false_block = program.blocks[2];
    }
    auto plan = Extract(program);
    Check(plan.value_storage.size() <= 2u * depth + 12u &&
              plan.control_flow.size() == 4 &&
              plan.control_flow[0].condition == plan.control_flow[1].condition &&
              !plan.capture_specialization_reads,
          "shared predicates expanded into trees or retained varying dependencies");
    for (const uint32_t flag : {0u, 1u}) {
      const std::array<uint32_t, 1> user_data{flag};
      SrtWalker walker(plan, {.user_data = user_data});
      uint32_t result = 99;
      const bool conjunction = operation == ValueOpcode::LogicalAnd;
      const bool known = conjunction ? flag == 0 : flag != 0;
      Check(walker.Evaluate(plan.control_flow[0].condition, result) == known &&
                (!known || result == uint32_t{conjunction}),
            "shared predicate changed its uniform factor or cached runtime values");
      std::vector<uint32_t> flat;
      Check(walker.RefreshFlatBuffer(flat) &&
                plan.visited_blocks == (known
                    ? (conjunction ? std::vector<uint8_t>{1, 1, 0, 1}
                                   : std::vector<uint8_t>{1, 0, 1, 0})
                    : std::vector<uint8_t>{1, 1, 1, 1}),
            "shared predicate skipped an unknown branch or followed a rejected branch");
      CheckWalkAgrees(plan, {.user_data = user_data, .direct_reads = true},
                      "the walker and the interpreter took different shared-predicate branches");
    }
  }
}

void TestNestedExecPredicatesReuseReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  constexpr uint32_t depth = 12;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 3; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &use = *program.blocks[1];
  auto &exit = *program.blocks[2];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&entry.AppendNewInst(opcode, args));
  };
  const auto lane = emit(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  const auto address = emit(ValueOpcode::GetAddressResource, {Value(0x1000u), Value(0u)});
  const auto count = emit(ValueOpcode::LoadAddressU32,
      {address, Value(0u), Value(0u), Value(true)});
  count.Instruction()->SetFlags(MemoryFlags{.index = 0});
  auto condition = emit(ValueOpcode::LogicalNot,
      {emit(ValueOpcode::ULessThanEqual32, {count, lane})});
  // A compact shared DAG must not become a tree of fresh EXEC evaluations.
  for (uint32_t i = 0; i < depth; ++i) {
    const auto mask = emit(ValueOpcode::INotEqual32, {lane, Value(i)});
    condition = emit(ValueOpcode::LogicalAnd,
        {mask, emit(ValueOpcode::LogicalOr, {condition, condition})});
  }
  entry.condition = condition;
  entry.terminator.kind = TerminatorKind::ConditionalBranch;
  entry.terminator.true_block = &use;
  entry.terminator.false_block = &exit;
  entry.AddBranch(&use);
  entry.AddBranch(&exit);
  use.terminator.kind = TerminatorKind::Branch;
  use.terminator.true_block = &exit;
  use.AddBranch(&exit);
  DescriptorSource source;
  source.dwords = {Value(0x8000u), Value(1u << 16u), Value(16u), Value(3u << 28u)};
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .read = true});
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  const auto buffer = Value(&use.AppendNewInst(ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]}));
  use.AppendNewInst(ValueOpcode::LoadBufferU32,
      {buffer, Value(0u), Value(0u), Value(0u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  auto plan = Extract(program);
  struct Reads { uint32_t value = 0; bool clean = true; uint32_t strict = 0, ordinary = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t>) {
        ++static_cast<Reads *>(data)->ordinary;
        return false;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        Check(reads.strict <= depth + 1,
              "shared EXEC predicates repeated scalar reads beyond their distinct masks");
        if (!reads.clean || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const auto state : {Reads{64u, true}, Reads{0u, true}, Reads{0u, false}, Reads{64u, true}}) {
    reads = state;
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "nested EXEC predicates rejected a valid resource snapshot");
    const bool inactive = reads.clean && reads.value == 0;
    Check(snapshot.buffers[0].dwords[0] == (inactive ? 0u : 0x8000u),
          "nested EXEC predicates lost an unknown branch or retained stale runtime facts");
    Check(reads.strict > 0 && reads.ordinary == 0 &&
              snapshot.specialization_reads.size() <= depth + 1,
          "shared EXEC predicates repeated scalar reads beyond their distinct masks");
  }
}

void TestFiniteImageRefreshReusesScalarReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
  for (uint32_t index = 0; index < 3; ++index) {
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(index * 4u), Value(0u), Value(true)});
    read.SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back({Value(&read), index});
    auto &flat = block.AppendNewInst(ValueOpcode::ReadConst,
                                     {Value(&srt), Value(index)});
    DescriptorSource source;
    source.dword_count = 8;
    source.dwords.fill(Value(0u));
    source.dwords[0] = Value(&flat);
    source.dwords[1] = Value(static_cast<uint32_t>(
        Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float) << 20u);
    source.dwords[3] = Value(Libs::Graphics::DstSel(4, 5, 6, 7) |
        (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u));
    program.descriptor_sources.push_back(source);
  }
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1, 2, 1};
  program.descriptor_sources.push_back(root);
  program.info.images.push_back({
      .source = 3,
      .resource_class = ImageResourceClass::Sampled,
      .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
      .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  auto plan = Extract(program);
  struct Reads {
    std::array<uint32_t, 3> words{0x100u, 0x200u, 0x200u};
    std::array<uint32_t, 3> counts{};
    uint32_t ordinary = 0;
  } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t>) {
        ++static_cast<Reads *>(data)->ordinary;
        return false;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address,
                                        std::span<uint32_t> words) {
        if (words.size() != 1 || address < 0x1000u || address >= 0x100cu ||
            (address & 3u) != 0) return false;
        auto &reads = *static_cast<Reads *>(data);
        const auto index = (address - 0x1000u) / 4u;
        ++reads.counts[index];
        words[0] = reads.words[index];
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const auto capacities = [&] {
    return std::array{snapshot.images.capacity(), snapshot.flattened_srt.capacity(),
                      snapshot.specialization_reads.capacity(), specialization.images.capacity()};
  };
  const auto check_mapping = [&](std::array<uint32_t, 4> ordinals) {
    const auto offset = specialization.images[0].indirect_mapping_offset;
    Check(snapshot.images.size() == 2 && snapshot.flattened_srt[offset] == 4,
          "finite image candidates were not deduplicated");
    for (uint32_t key = 0; key < ordinals.size(); ++key) {
      Check(snapshot.flattened_srt[offset + 1u + key * 2u] == key &&
                snapshot.flattened_srt[offset + 2u + key * 2u] == ordinals[key],
            "finite image selector mapping is stale or incorrect");
    }
  };
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image materialization failed");
  Check(reads.ordinary == 0 && reads.counts == std::array<uint32_t, 3>{1, 1, 1} &&
            snapshot.specialization_reads.size() == 3,
        "finite image candidates repeated scalar reads or bypassed clean provenance");
  check_mapping({0, 1, 1, 1});
  const auto warm_capacities = capacities();
  reads.words = {0x300u, 0x300u, 0x400u};
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image refresh failed after descriptor changes");
  Check(reads.ordinary == 0 && reads.counts == std::array<uint32_t, 3>{2, 2, 2} &&
            snapshot.specialization_reads.size() == 3 &&
            snapshot.images[0].dwords[0] == 0x300u && snapshot.images[1].dwords[0] == 0x400u,
        "finite image refresh retained old scalar values or repeated reads");
  check_mapping({0, 0, 1, 0});
  Check(capacities() == warm_capacities,
        "finite image refresh grew reusable resource storage after warmup");
  CheckWalkAgreesWith(plan, runtime, "a finite image refresh walked differently");
}

void TestEmptyTraversalSkipsUnmappedDescriptor(bool independent_mask, bool counted_loop = false) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 6; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &loop = *program.blocks[1];
  auto &body = *program.blocks[2];
  auto &merge = *program.blocks[3];
  auto &use = *program.blocks[4];
  auto &exit = *program.blocks[5];
  const auto emit = [](Block &block, ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto branch = [](Block &block, Block &target, Value condition = {}, Block *other = nullptr) {
    block.terminator.kind = other ? TerminatorKind::ConditionalBranch : TerminatorKind::Branch;
    block.terminator.true_block = &target;
    block.terminator.false_block = other;
    block.condition = condition;
    block.AddBranch(&target);
    if (other) block.AddBranch(other);
  };
  Value lane, active, initial;
  if (!counted_loop) {
    lane = emit(entry, ValueOpcode::GetBuiltin,
        {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
    active = emit(entry, ValueOpcode::INotEqual32, {lane, Value(0u)});
    const auto initial_mask = independent_mask
        ? emit(entry, ValueOpcode::IEqual32, {lane, Value(0u)}) : active;
    initial = emit(entry, ValueOpcode::SelectU32,
        {initial_mask, Value(UINT32_MAX), Value(0u)});
  }
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  const auto address = emit(entry, ValueOpcode::GetAddressResource, {Value(0x1000u), Value(0u)});
  const auto count = emit(entry, ValueOpcode::LoadAddressU32,
      {address, Value(0u), Value(0u), Value(true)});
  count.Instruction()->SetFlags(MemoryFlags{.index = 0});
  if (counted_loop) {
    branch(entry, loop);
  } else {
    const auto rejected = emit(entry, ValueOpcode::ULessThanEqual32, {count, lane});
    const auto accepted = emit(entry, ValueOpcode::LogicalNot, {rejected});
    branch(entry, loop, emit(entry, ValueOpcode::LogicalAnd, {active, accepted}), &merge);
  }

  // A nonempty traversal contains a varying loop-carried value. Its cycle must
  // remain unknown; only the proven empty traversal can select the bypass arm.
  auto &carried = loop.AppendNewInst(ValueOpcode::Phi);
  carried.SetFlags(Type::U32);
  const auto next = emit(body, ValueOpcode::IAdd32, {Value(&carried), Value(1u)});
  carried.AddPhiOperand(&entry, counted_loop ? Value(0u) : lane);
  carried.AddPhiOperand(&body, next);
  const auto condition = counted_loop
      ? emit(loop, ValueOpcode::ULessThan32, {Value(&carried), count})
      : emit(loop, ValueOpcode::INotEqual32, {Value(&carried), Value(UINT32_MAX)});
  branch(loop, counted_loop ? use : body, condition, &merge);
  loop.terminator.loop_header = true;
  loop.terminator.continue_block = &body;
  loop.terminator.merge_block = &merge;
  branch(body, loop);
  if (counted_loop) {
    branch(merge, exit);
    branch(use, body);
  } else {
    auto &selected = merge.AppendNewInst(ValueOpcode::Phi);
    selected.SetFlags(Type::U32);
    selected.AddPhiOperand(&entry, initial);
    selected.AddPhiOperand(&loop, Value(&carried));
    const auto valid = emit(merge, ValueOpcode::INotEqual32, {Value(&selected), Value(UINT32_MAX)});
    branch(merge, use, emit(merge, ValueOpcode::LogicalAnd, {active, valid}), &exit);
    branch(use, exit);
  }
  DescriptorSource source;
  const auto descriptor_address = emit(use, ValueOpcode::GetAddressResource,
      {Value(0x2000u), Value(0u)});
  for (uint32_t i = 0; i < 4; ++i) {
    const auto index = static_cast<uint32_t>(program.memory_info.size());
    program.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
    source.dwords[i] = emit(use, ValueOpcode::LoadAddressU32,
        {descriptor_address, Value(i * 4u), Value(0u), Value(true)});
    source.dwords[i].Instruction()->SetFlags(MemoryFlags{.index = index});
  }
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .read = true});
  const auto buffer_index = static_cast<uint32_t>(program.memory_info.size());
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  const auto buffer = emit(use, ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]});
  emit(use, ValueOpcode::LoadBufferU32, {buffer, Value(0u), Value(0u), Value(0u), Value(true)})
      .Instruction()->SetFlags(MemoryFlags{.index = buffer_index});
  auto plan = Extract(program);
  // The same plan interpreted every time: a replayed refresh must give what it gives.
  auto interpreted = Extract(program);
  interpreted.srt_trace = false;
  struct Reads { uint32_t count = 0; bool clean = true; uint32_t descriptors = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        constexpr std::array<uint32_t, 4> descriptor{0x8000u, 1u << 16u, 16u,
                                                   (3u << 28u) | 0x4dfacu};
        if (address < 0x2000u || address >= 0x2010u || address % 4u || words.size() != 1)
          return false;
        ++static_cast<Reads *>(data)->descriptors;
        words[0] = descriptor[(address - 0x2000u) / 4u];
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        const auto &reads = *static_cast<Reads *>(data);
        if (!reads.clean || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.count;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  // Each state twice: the second refresh of a state replays the first, a changed state gives up.
  for (const auto state : {Reads{0u, true}, Reads{0u, true}, Reads{4u, true}, Reads{4u, true},
                           Reads{0u, false}, Reads{0u, false}, Reads{0u, true}}) {
    reads = state;
    if (!counted_loop) {
      uint32_t condition = 99;
      Check(!SrtWalker(plan, runtime).Evaluate(plan.control_flow[3].condition, condition),
            "direct merge evaluation used missing or previous runtime reachability");
    }
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "partial traversal proof rejected a valid resource snapshot");
    const bool inactive = !independent_mask && reads.clean && reads.count == 0u;
    Check(snapshot.buffers[0].dwords[0] == (inactive ? 0u : 0x8000u),
          "empty traversal kept its descriptor or an uncertain traversal lost it");
    Check(specialization.buffers[0].zero_stride_oob == inactive,
          "inactive traversal did not expose dead vector reads to specialization");
    Check(reads.descriptors == (inactive ? 0u : 4u),
          "inactive traversal read its descriptor or an uncertain traversal skipped it");
    if (inactive) {
      Check(!snapshot.specialization_reads.empty(), "empty traversal lost its clean scalar read");
      for (const auto &read : snapshot.specialization_reads)
        Check(read == std::pair<uint64_t, uint64_t>{0x1000u, 4u},
              "empty traversal proof captured an unrelated memory range");
    }
    const auto traced_reads = reads.descriptors;
    reads = state;
    ResourceSnapshot expected;
    ResourceSpecialization expected_specialization;
    Check(MaterializeResources(interpreted, runtime, expected, expected_specialization) &&
              SameSnapshot(snapshot, expected) && specialization == expected_specialization &&
              reads.descriptors == traced_reads && plan.active_sources == interpreted.active_sources &&
              plan.visited_blocks == interpreted.visited_blocks,
          "a replayed resource refresh differed from its interpretation");
    // A dirty count read fails the walk: the interpreter refreshes instead.
    CheckWalkAgreesWith(interpreted, runtime, "an empty traversal walked differently", state.clean);
  }
  Check(plan.trace_replays >= 3u && interpreted.trace_replays == 0u,
        "repeated resource refreshes were not replayed");
}

void TestCyclicControlPredicatesRemainUnknown() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 4; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &left = *program.blocks[1];
  auto &right = *program.blocks[2];
  auto &exit = *program.blocks[3];
  const auto lane = Value(&entry.AppendNewInst(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)}));
  const auto u = Value(&entry.AppendNewInst(ValueOpcode::INotEqual32, {lane, Value(0u)}));
  const auto v = Value(&entry.AppendNewInst(ValueOpcode::INotEqual32, {lane, Value(1u)}));
  auto &p = left.AppendNewInst(ValueOpcode::Phi);
  auto &q = right.AppendNewInst(ValueOpcode::Phi);
  p.SetFlags(Type::U1);
  q.SetFlags(Type::U1);
  const auto a = Value(&left.AppendNewInst(ValueOpcode::LogicalAnd, {u, Value(&p)}));
  const auto b = Value(&right.AppendNewInst(ValueOpcode::LogicalAnd, {v, Value(&q)}));
  p.AddPhiOperand(&entry, Value(true));
  p.AddPhiOperand(&right, b);
  q.AddPhiOperand(&entry, Value(true));
  q.AddPhiOperand(&left, a);
  const auto branch = [](Block &block, Value condition, Block &yes, Block &no) {
    block.condition = condition;
    block.terminator.kind = TerminatorKind::ConditionalBranch;
    block.terminator.true_block = &yes;
    block.terminator.false_block = &no;
    block.AddBranch(&yes);
    block.AddBranch(&no);
  };
  // Two loop entries have no single header whose Phi can be discarded up front.
  branch(entry, u, left, right);
  branch(left, a, right, exit);
  branch(right, b, left, exit);
  auto plan = Extract(program);
  SrtWalker walker(plan, {});
  uint32_t value = 99;
  Check(!walker.Evaluate(plan.control_flow[1].condition, value) &&
            !walker.Evaluate(plan.control_flow[2].condition, value),
        "cyclic varying control acquired an unsupported uniform value");
  std::vector<uint32_t> flat;
  Check(walker.RefreshFlatBuffer(flat) && plan.visited_blocks == std::vector<uint8_t>({1, 1, 1, 1}),
        "cyclic control lost a possible branch during reachability");
  CheckWalkAgrees(plan, {.direct_reads = true}, "cyclic varying control walked differently");
}

void TestEmptyScalarBufferSpecialization() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct Fixture {
    Program program;
    std::vector<std::pair<Inst *, Type>> uses;
    Inst *nonnull = nullptr;
  };
  const auto make_fixture = [] {
    Fixture fixture;
    auto &program = fixture.program;
    program.stage = Libs::Graphics::ShaderType::Compute;
    program.srt_plan_complete = program.resource_tracking_complete = true;
    program.user_data_count = 4u;
    auto &block = AddValueBlock(program);
    DescriptorSource source;
    source.dword_count = 4u;
    for (uint32_t word = 0; word < 4u; ++word)
      source.dwords[word] = Value(&block.AppendNewInst(ValueOpcode::GetUserData,
          {Value(static_cast<ScalarReg>(word))}));
    program.descriptor_sources.push_back(source);
    program.info.buffers.push_back({.source = 0u, .read = true, .scalar = true});
    const auto table = Value(&block.AppendNewInst(ValueOpcode::GetBufferResource,
        {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]}));
    std::array<Value, 4> words;
    for (uint32_t word = 0; word < 4u; ++word) {
      program.memory_info.push_back({.kind = ResourceKind::ScalarBuffer, .resource = 0u,
                                     .offset = word * 4u});
      auto &read = block.AppendNewInst(ValueOpcode::ReadConstBuffer, {table, Value(0u)});
      read.SetFlags(MemoryFlags{.index = word});
      words[word] = Value(&read);
    }
    const auto handle = Value(&block.AppendNewInst(ValueOpcode::GetBufferResource,
        {words[0], words[1], words[2], words[3]}));
    for (const auto opcode : {ValueOpcode::ReadConstBuffer, ValueOpcode::LoadBufferU8,
                              ValueOpcode::LoadBufferU16, ValueOpcode::LoadBufferU32,
                              ValueOpcode::LoadBufferU32x2, ValueOpcode::LoadBufferU32x3,
                              ValueOpcode::LoadBufferU32x4}) {
      const auto index = static_cast<uint32_t>(program.memory_info.size());
      program.memory_info.push_back({.kind = ResourceKind::IndirectBuffer, .resource = UINT32_MAX,
          .data_dwords = BufferComponentCount(opcode),
          .data_bits = opcode == ValueOpcode::LoadBufferU8 ? 8u : opcode == ValueOpcode::LoadBufferU16 ? 16u : 32u,
          .formatted = opcode == ValueOpcode::LoadBufferU32});
      auto &read = opcode == ValueOpcode::ReadConstBuffer
          ? block.AppendNewInst(opcode, {handle, Value(0u)})
          : block.AppendNewInst(opcode, {handle, Value(0u), Value(0u), Value(0u), Value(true)});
      read.SetFlags(MemoryFlags{.index = index});
      fixture.uses.emplace_back(&block.AppendNewInst(ValueOpcode::Identity, {Value(&read)}), read.GetType());
    }
    const auto nonnull = Value(&block.AppendNewInst(ValueOpcode::GetBufferResource,
        {words[0], words[1], words[2], Value(1u)}));
    auto &read = block.AppendNewInst(ValueOpcode::ReadConstBuffer, {nonnull, Value(0u)});
    read.SetFlags(MemoryFlags{.index = 4u});
    fixture.nonnull = &block.AppendNewInst(ValueOpcode::Identity, {Value(&read)});
    return fixture;
  };
  auto fixture = make_fixture();
  auto plan = Extract(fixture.program);
  std::array<uint32_t, 4> descriptor{0x1000u, 8u << 16u, 0u, 3u << 28u};
  uint32_t reads = 0u;
  const SrtRuntime runtime{.user_data = descriptor, .read_memory = RejectSpecializationRead,
                           .userdata = &reads, .read_specialization_memory = RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization, empty, populated;
  for (const uint32_t records : {0u, 1u, 0x20000000u, 0u}) {
    descriptor[2] = records;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) && reads == 0u &&
              specialization.buffers[0].scalar_empty == (records == 0u),
          "scalar emptiness read table contents or truncated its 64-bit byte size");
    if (records == 0u) {
      if (empty.buffers.empty()) empty = specialization;
      else Check(specialization == empty, "returning to an empty scalar table changed its specialization");
    }
    else if (records == 1u) populated = specialization;
    else Check(specialization == populated, "nonempty scalar size created a shader permutation");
    auto current = make_fixture();
    ApplyResourceSpecialization(current.program, specialization);
    for (const auto &[use, type] : current.uses) {
      const auto value = use->Arg(0).Resolve();
      Check(value.GetType() == type, "empty scalar-table folding changed the load result type");
      if (records != 0u) {
        Check(value.TryInstruction() != nullptr && BufferAccessOf(value.Instruction()->GetOpcode()) == BufferAccess::Read,
              "nonempty scalar table lost a runtime buffer load");
      } else if (value.IsImmediate()) {
        Check(value == Value(uint8_t{0}) || value == Value(uint16_t{0}) || value == Value(0u),
              "empty scalar table did not produce a typed zero");
      } else {
        for (size_t component = 0; component < value.Instruction()->NumArgs(); ++component)
          Check(value.Instruction()->Arg(component) == Value(0u),
                "empty scalar table retained a nonzero vector component");
      }
    }
    Check(current.nonnull->Arg(0).Resolve().TryInstruction() != nullptr,
          "a partly nonzero indirect descriptor was treated as unbound");
  }
  Check(empty != populated, "empty-to-populated scalar transition reused the folded specialization");
  plan.info.buffers[0].scalar = false;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            !specialization.buffers[0].scalar_empty,
        "vector-only resource acquired scalar-empty specialization");
  empty.buffers[0].indirect_root = 0u;
  empty.buffers.push_back(empty.buffers[0]);
  empty.buffers[1].scalar_empty = false;
  ApplyResourceSpecialization(fixture.program, empty);
  for (const auto &[use, type] : fixture.uses)
    Check(use->Arg(0).GetType() == type && use->Arg(0).Resolve().TryInstruction() != nullptr &&
              BufferAccessOf(use->Arg(0).ResolveInstruction()->GetOpcode()) == BufferAccess::Read,
          "empty root folded a scalar read with nonempty candidates");
}

void TestMixedSamplerVariantsShareRuntimeDescriptor() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = MixedSamplerProgram();
  auto plan = Extract(program);
  std::array<uint32_t, 2> user_data{0x11111111u, 0x22222222u};
  const SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mixed sampler materialization failed");
  ApplyResourceSpecialization(program, specialization);
  const auto &samplers = program.info.samplers;
  Check(snapshot.samplers.size() == 2 && samplers.size() == 3 &&
            samplers[0].snapshot_index == 0 && samplers[1].snapshot_index == 1 &&
            samplers[2].snapshot_index == 1 &&
            samplers[0].source == plan.info.samplers[0].source &&
            samplers[1].source == plan.info.samplers[1].source &&
            samplers[2].source == samplers[1].source &&
            !samplers[1].force_point_filtering && samplers[2].force_point_filtering &&
            program.info.sampled_pairs[2].sampler == 2,
        "native sampler variants lost their source identity or binding order");
  const auto capacity = snapshot.samplers.capacity();
  user_data[1] = 0x33333333u;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.samplers.size() == 2 && snapshot.samplers.capacity() == capacity &&
            snapshot.samplers[samplers[0].snapshot_index].dwords[0] == user_data[0] &&
            snapshot.samplers[samplers[1].snapshot_index].dwords[0] == user_data[1] &&
            snapshot.samplers[samplers[2].snapshot_index].dwords[0] == user_data[1],
        "sampler variants retained stale or duplicated descriptors after refresh");
}

void TestSignedStorageSpecialization() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using F = Libs::Graphics::Prospero::BufferFormat;
  using N = Libs::Graphics::Prospero::TextureNumericClass;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  program.user_data_count = 1u;
  auto &block = AddValueBlock(program);
  DescriptorSource source;
  source.dword_count = 8u;
  source.dwords.fill(Value(0u));
  source.dwords[0] = Value(0x100u);
  source.dwords[1] = Value(&block.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(0))}));
  constexpr auto swizzle = Libs::Graphics::DstSel(5, 4, 1, 0);
  source.dwords[3] = Value(swizzle |
      (uint32_t(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u));
  program.descriptor_sources.push_back(source);
  program.info.images.push_back({.source = 0u, .resource_class = ImageResourceClass::Storage,
      .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D, .written = true});
  auto plan = Extract(program);
  std::array<uint32_t, 1> data{};
  const SrtRuntime runtime{.user_data = data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization, signed16;
  for (const auto format : {F::k16SInt, F::k32SInt, F::k16SInt}) {
    data[0] = uint32_t(format) << 20u;
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "write-only signed storage image failed materialization");
    const auto conversion = format == F::k16SInt ? format : F::kInvalid;
    const auto view = format == F::k16SInt ? F::k16UInt : F::k32UInt;
    Check(specialization.images[0].numeric_class == N::Uint &&
              specialization.images[0].conversion_format == conversion &&
              specialization.images[0].shader_swizzle == swizzle &&
              snapshot.images[0].dwords[1] == data[0] &&
              RawSignedStorageFormat(plan.info.images[0], format) == view,
          "signed storage specialization lost guest format, conversion, swizzle or raw view width");
    if (format == F::k16SInt) {
      if (signed16.images.empty()) signed16 = specialization;
      else Check(specialization == signed16, "signed storage format refresh retained a stale specialization");
    } else {
      Check(specialization != signed16, "signed 16/32-bit stores shared incompatible conversion state");
    }
    ApplyResourceSpecialization(program, specialization);
    const auto binding = DescriptorBindingForImage(program.info.images[0]);
    Check(binding && ImageBindingResourceClass(*binding) == ImageResourceClass::Storage,
          "signed storage conversion did not reuse the unsigned storage binding");
  }
  for (const auto format : {F::k8SInt, F::k16SInt, F::k32SInt}) {
    data[0] = uint32_t(format) << 20u;
    auto &image = plan.info.images[0];
    image.read = true;
    Check(RawSignedStorageFormat(image, format) == F::kInvalid &&
              !MaterializeResources(plan, runtime, snapshot, specialization),
          "signed raw storage view accepted a non-atomic read/write image");
    image.atomic = true;
    Check((RawSignedStorageFormat(image, format) != F::kInvalid) == (format == F::k32SInt) &&
              MaterializeResources(plan, runtime, snapshot, specialization) == (format == F::k32SInt),
          "signed storage conversion changed atomic format eligibility");
    image.atomic64 = true;
    Check(RawSignedStorageFormat(image, format) == F::kInvalid &&
              !MaterializeResources(plan, runtime, snapshot, specialization),
          "signed 16/32-bit image acquired a 64-bit atomic raw view");
    image.resource_class = ImageResourceClass::Sampled;
    image.written = image.atomic = image.atomic64 = false;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              specialization.images[0].numeric_class == N::Sint &&
              specialization.images[0].conversion_format == F::kInvalid &&
              Libs::Graphics::Prospero::NumBytesPerElement(format) ==
                  (format == F::k8SInt ? 1u : format == F::k16SInt ? 2u : 4u) &&
              RawSignedStorageFormat(image, format) == F::kInvalid,
          "storage conversion changed ordinary signed image reads");
    image.resource_class = ImageResourceClass::Storage;
    image.written = true;
    image.read = false;
  }
}

// A written storage image of a 16-bit packed UNorm format is an R16_UINT view the shader packs (PackedStorageFormat):
// unsigned, converted from the guest format, with its swizzle. A sampled read of it, and any other format, is not.
void TestPackedStorageSpecialization() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using F = Libs::Graphics::Prospero::BufferFormat;
  using N = Libs::Graphics::Prospero::TextureNumericClass;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  program.user_data_count = 1u;
  auto &block = AddValueBlock(program);
  DescriptorSource source;
  source.dword_count = 8u;
  source.dwords.fill(Value(0u));
  source.dwords[0] = Value(0x100u);
  source.dwords[1] = Value(&block.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(0))}));
  constexpr auto swizzle = Libs::Graphics::DstSel(6, 5, 4, 7);
  source.dwords[3] = Value(swizzle |
      (uint32_t(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u));
  program.descriptor_sources.push_back(source);
  program.info.images.push_back({.source = 0u, .resource_class = ImageResourceClass::Storage,
      .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D, .written = true});
  auto plan = Extract(program);
  std::array<uint32_t, 1> data{};
  const SrtRuntime runtime{.user_data = data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const auto format : {F::k5_6_5UNorm, F::k5_5_5_1UNorm, F::k4_4_4_4UNorm, F::k8_8_8_8UNorm}) {
    const bool packed = format != F::k8_8_8_8UNorm;
    data[0] = uint32_t(format) << 20u;
    auto &image = plan.info.images[0];
    image.resource_class = ImageResourceClass::Storage;
    image.written = true;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              specialization.images[0].numeric_class == (packed ? N::Uint : N::Float) &&
              specialization.images[0].conversion_format == (packed ? format : F::kInvalid) &&
              specialization.images[0].shader_swizzle == swizzle &&
              PackedStorageFormat(image, format) == (packed ? F::k16UInt : F::kInvalid),
          "16-bit packed UNorm storage did not take an unsigned R16 view with the guest format's conversion");
    image.resource_class = ImageResourceClass::Sampled;
    image.written = false;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              specialization.images[0].numeric_class == N::Float &&
              specialization.images[0].conversion_format == F::kInvalid &&
              PackedStorageFormat(image, format) == F::kInvalid,
          "packed storage conversion changed sampled reads of a 16-bit packed UNorm texture");
  }
}

// One descriptor serves a store, an atomic that a user-data condition guards, and an image no
// block lists. Unreached, the atomic reads a zero descriptor instead of the RGBA8 format the
// others use; reached, it rejects that format.
// An image memory entry whose instructions dead-code elimination removed keeps the translator's T# register
// number, which may lie past the image list (PPSA03671 pixel shaders 5b68b75638ea51a3, c8eb03340a65ab88 and
// dffe92699dfd73c2: s44 -> 11 with 6 images). Specialization remaps only the entries an instruction uses.
// A shader with form slots (ImageForms.h): a dispatch whose slots each hold one of the forms gets the forms module's
// specialization, the root of each slot its first form and one candidate a further form, its descriptor on its form's
// binding and null ones on the others, and its forms in the form dword. One outside them keeps its own specialization.
void TestImageForms() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::Prospero::TextureNumericClass;
  using Dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension;
  ResourcePlan plan;
  plan.shader_hash = ImageFormsShader;
  for (uint32_t image = 0; image < 16u; ++image)
    plan.info.images.push_back({.resource_class = ImageResourceClass::Sampled, .dimension = Dimension::Dim2DArray,
                                .read = true});
  plan.memory_info.push_back({.kind = ResourceKind::Image, .resource = 1u,
                              .image_sample_flags = Libs::Graphics::ShaderRecompiler::Decoder::ImageSampleFlagLevelZero});
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (uint32_t image = 0; image < 16u; ++image) {
    DescriptorValue value{.dword_count = 8u};
    value.dwords[0] = image + 1u;
    snapshot.images.push_back(value);
    specialization.images.push_back({.numeric_class = TextureNumericClass::Float, .dimension = Dimension::Dim2DArray});
  }
  // Slot 0 (image 1) holds form 3, slot 1 (image 2) form 5, the others form 0.
  specialization.images[1].numeric_class = TextureNumericClass::Uint;
  specialization.images[1].dimension     = Dimension::Dim3D;
  specialization.images[2].cube          = true;
  const auto original_snapshot       = snapshot;
  const auto original_specialization = specialization;

  auto forms = specialization;
  auto forms_snapshot = snapshot;
  Check(ApplyImageForms(plan, forms_snapshot, forms), "image forms did not apply to a dispatch of its forms");
  const auto child = [](uint32_t slot, uint32_t form) { return 16u + slot * 6u + form - 1u; };
  Check(forms.images.size() == 58u && forms_snapshot.images.size() == 58u &&
            forms_snapshot.image_forms == (3u | 5u << 3u),
        "image forms made the wrong number of candidates or form dword");
  bool roots = true;
  for (uint32_t slot = 0; slot < ImageFormSlots; ++slot) {
    const auto  index = ImageFormFirstSlot + slot;
    const auto &root  = forms.images[index];
    roots = roots && root.numeric_class == TextureNumericClass::Float && root.dimension == Dimension::Dim2DArray &&
            !root.cube && root.indirect_root == index && root.indirect_mapping_offset == ImageFormMapping &&
            root.indirect_search_iterations == slot;
    for (uint32_t form = 1; form < ImageForms.size(); ++form) {
      const auto &candidate = forms.images[child(slot, form)];
      roots = roots && candidate.indirect_root == index && candidate.indirect_mapping_offset == ImageFormMapping &&
              candidate.indirect_search_iterations == slot &&
              candidate.numeric_class == ImageForms[form].numeric_class &&
              candidate.dimension == ImageForms[form].dimension && candidate.cube == ImageForms[form].cube &&
              candidate.mip_count == 1u;
    }
  }
  Check(roots, "image forms made a root or candidate of the wrong form");
  Check(forms_snapshot.images[1].dwords[0] == 0u && forms_snapshot.images[child(0, 3)].dwords[0] == 2u &&
            forms_snapshot.images[2].dwords[0] == 0u && forms_snapshot.images[child(1, 5)].dwords[0] == 3u &&
            forms_snapshot.images[3].dwords[0] == 4u && forms_snapshot.images[child(2, 1)].dwords[0] == 0u &&
            forms_snapshot.images[child(0, 1)].dwords[0] == 0u,
        "image forms put a descriptor on another form's binding");
  Check(forms.images[0] == original_specialization.images[0] && forms.images[8] == original_specialization.images[8] &&
            forms_snapshot.images[0] == original_snapshot.images[0] &&
            forms_snapshot.images[15] == original_snapshot.images[15],
        "image forms changed an image that is no form slot");

  // A form outside the title's (Uint 1D), a written image, an image read with derivatives and another shader each keep
  // the specialization and snapshot as they were.
  const auto unchanged = [&](ResourcePlan &other, ResourceSpecialization changed, const char *text) {
    auto other_snapshot = original_snapshot;
    const auto before   = changed;
    Check(!ApplyImageForms(other, other_snapshot, changed) && changed == before &&
              other_snapshot.images == original_snapshot.images && other_snapshot.image_forms == 0u,
          text);
  };
  auto uint_1d = original_specialization;
  uint_1d.images[4].numeric_class = TextureNumericClass::Uint;
  uint_1d.images[4].dimension     = Dimension::Dim1D;
  unchanged(plan, uint_1d, "image forms applied to a form outside the title's");
  plan.info.images[5].written = true;
  unchanged(plan, original_specialization, "image forms applied to a written image");
  plan.info.images[5].written = false;
  plan.memory_info.push_back({.kind = ResourceKind::Image, .resource = 6u,
                              .image_sample_flags = Libs::Graphics::ShaderRecompiler::Decoder::ImageSampleFlagDerivative});
  unchanged(plan, original_specialization, "image forms applied to an image read with derivatives");
  plan.memory_info.pop_back();
  plan.shader_hash = ImageFormsShader + 1u;
  unchanged(plan, original_specialization, "image forms applied to another shader");
}

void TestDeadImageMemoryKeepsTranslatorIndex() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  using Libs::Graphics::Prospero::BufferFormat;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  program.user_data_count = 1u;
  auto &entry = AddValueBlock(program);
  DescriptorSource source;
  source.dword_count = 8u;
  source.dwords.fill(Value(0u));
  source.dwords[0] = Value(0x100u);
  source.dwords[1] = Value(&entry.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(0))}));
  source.dwords[3] = Value(uint32_t(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u);
  program.descriptor_sources.push_back(source);
  for (uint32_t image = 0; image < 2u; ++image)
    program.info.images.push_back({.source = 0u, .resource_class = ImageResourceClass::Storage,
        .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D, .written = true});
  program.memory_info.push_back({.kind = ResourceKind::Image, .resource = 1u});
  program.memory_info.push_back({.kind = ResourceKind::Image, .resource = 11u});
  entry.AppendNewInst(ValueOpcode::ImageWrite, {Value(0u), Value(0u), Value(0u), Value(true)})
      .SetFlags(MemoryFlags{.index = 0});
  entry.terminator.kind = TerminatorKind::Return;
  auto plan = Extract(program);
  std::array<uint32_t, 1> data{uint32_t(BufferFormat::k8_8_8_8UNorm) << 20u};
  const SrtRuntime runtime{.user_data = data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) && specialization.images.size() == 2u,
        "image fixture materialization failed");
  ApplyResourceSpecialization(program, specialization);
  Check(program.memory_info[0].resource == 1u && program.memory_info[1].resource == 11u,
        "specialization remapped a dead image memory entry or lost a live one");
}

void TestUnreachedImageIgnoresSharedDescriptor(bool atomic64) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  using Libs::Graphics::Prospero::BufferFormat;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  program.user_data_count = 2u;
  auto &entry = AddValueBlock(program);
  auto &guarded = AddValueBlock(program);
  auto &done = AddValueBlock(program);
  const auto user = [&](uint32_t index) {
    return Value(&entry.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(index))}));
  };
  DescriptorSource source;
  source.dword_count = 8u;
  source.dwords.fill(Value(0u));
  source.dwords[0] = Value(0x100u);
  source.dwords[1] = user(0);
  source.dwords[3] = Value(uint32_t(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u);
  program.descriptor_sources.push_back(source);
  const auto dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D;
  for (const bool atomic : {false, true, false})
    program.info.images.push_back({.source = 0u, .resource_class = ImageResourceClass::Storage,
        .dimension = dimension, .written = true, .atomic = atomic, .atomic64 = atomic && atomic64});
  for (uint32_t image = 0; image < 2u; ++image)
    program.memory_info.push_back({.kind = ResourceKind::Image, .resource = image});
  entry.AppendNewInst(ValueOpcode::ImageWrite, {Value(0u), Value(0u), Value(0u), Value(true)})
      .SetFlags(MemoryFlags{.index = 0});
  guarded.AppendNewInst(atomic64 ? ValueOpcode::ImageAtomicUMax64 : ValueOpcode::ImageAtomicUMax32,
                        {Value(0u), Value(0u), Value(0u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  entry.condition = Value(&entry.AppendNewInst(ValueOpcode::INotEqual32, {user(1), Value(0u)}));
  entry.terminator.kind = TerminatorKind::ConditionalBranch;
  entry.terminator.true_block = &guarded;
  entry.terminator.false_block = &done;
  guarded.terminator.kind = TerminatorKind::Branch;
  guarded.terminator.true_block = &done;
  done.terminator.kind = TerminatorKind::Return;
  auto plan = Extract(program);
  std::array<uint32_t, 2> data{uint32_t(BufferFormat::k8_8_8_8UNorm) << 20u, 0u};
  const SrtRuntime runtime{.user_data = data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.images[0].dwords[1] == data[0] && snapshot.images[2].dwords[1] == data[0] &&
            snapshot.images[1] == DescriptorValue{.dword_count = 8u},
        "unreached atomic image kept the shared descriptor or an unlisted image lost it");
  data[1] = 1u;
  Check(!MaterializeResources(plan, runtime, snapshot, specialization),
        "reached atomic image accepted a format its width rejects");
}

void TestIndirectImageSamplerClasses(bool mixed) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::Prospero::BufferFormat;
  using Libs::Graphics::Prospero::TextureNumericClass;
  auto program = MixedSamplerProgram();
  auto &first = program.descriptor_sources[0];
  first.dwords[0] = Value(0x100u);
  first.dwords[1] = Value(static_cast<uint32_t>(BufferFormat::k8UNorm) << 20u);
  first.dwords[3] = program.descriptor_sources[1].dwords[3];
  program.descriptor_sources[1].dwords[1] = Value(static_cast<uint32_t>(
      mixed ? BufferFormat::k8UInt : BufferFormat::k8UNorm) << 20u);
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1, 0};
  program.info.images.resize(1);
  program.info.images[0].source = static_cast<uint32_t>(program.descriptor_sources.size());
  program.descriptor_sources.push_back(root);
  program.info.samplers.resize(1);
  program.info.sampled_pairs.resize(1);
  auto plan = Extract(program);
  std::array<uint32_t, 1> user_data{0x11111111u};
  uint32_t reads = 0;
  const SrtRuntime runtime{.user_data = user_data, .userdata = &reads,
                           .read_specialization_memory = RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) && reads == 0 &&
            snapshot.images.size() == 2 && snapshot.images[0].dwords[0] == 0x100u &&
            snapshot.images[1].dwords[0] == 1u &&
            specialization.images[0].numeric_class == TextureNumericClass::Float &&
            specialization.images[1].numeric_class ==
                (mixed ? TextureNumericClass::Uint : TextureNumericClass::Float),
        "indirect image materialization rejected or changed a typed candidate");
  const auto offset = specialization.images[0].indirect_mapping_offset;
  Check(snapshot.flattened_srt[offset] == 3u &&
            snapshot.flattened_srt[offset + 2u] == 0u &&
            snapshot.flattened_srt[offset + 4u] == 1u &&
            snapshot.flattened_srt[offset + 6u] == 0u,
        "mixed image classes changed selector ordinals or duplicated candidates");
  ApplyResourceSpecialization(program, specialization);
  Check(program.info.images[0].indirect_resources == std::vector<uint32_t>({0, 1}) &&
            snapshot.samplers.size() == 1 && program.info.samplers.size() == (mixed ? 2u : 1u),
        "indirect images lost candidates or created unnecessary sampler variants");
  user_data[0] = 0x33333333u;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mixed image sampler refresh failed");
  for (uint32_t index = 0; index < program.info.samplers.size(); ++index) {
    const auto &sampler = program.info.samplers[index];
    Check(sampler.snapshot_index == 0u && sampler.source == plan.info.samplers[0].source &&
              sampler.force_point_filtering == (index != 0u) &&
              sampler.integer_border == (index != 0u) &&
              snapshot.samplers[sampler.snapshot_index].dwords[0] == user_data[0],
          "typed sampler variants lost filtering, border class, or shared live descriptor");
  }
}

// Two states of one image heap that a written storage image indexes: other descriptors, in another order and
// number, of the same two types. Grouped, both give one specialization (one module), and every table word still
// selects the descriptor its ordinal selected: dense words and key/ordinal pairs.
void TestGroupIndirectStorageImages() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using N = Libs::Graphics::Prospero::TextureNumericClass;
  using D = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension;
  ShaderInfo base;
  base.images.push_back({.resource_class = ImageResourceClass::Storage, .dimension = D::Dim2D, .written = true});
  const auto make = [](uint32_t candidates, uint32_t seed, bool pairs, ResourceSnapshot &snapshot,
                       ResourceSpecialization &specialization, std::vector<DescriptorValue> &selects) {
    snapshot = {};
    specialization = {};
    selects.clear();
    specialization.images.push_back({.numeric_class = N::Float, .dimension = D::Dim2D, .indirect_root = 0u,
                                     .indirect_mapping_offset = 3u, .indirect_search_iterations = pairs ? 7u : 0u});
    snapshot.images.push_back({.dword_count = 8u}); // the root, null
    for (uint32_t candidate = 0; candidate < candidates; ++candidate) {
      auto type = specialization.images[0];
      type.indirect_mapping_offset = type.indirect_search_iterations = 0u;
      type.shader_swizzle = (candidate * 7u + seed) % 3u == 0u ? 0x688u : 0xfacu;
      specialization.images.push_back(type);
      DescriptorValue value{.dword_count = 8u};
      value.dwords[0] = seed * 1000u + candidate + 1u;
      snapshot.images.push_back(value);
    }
    snapshot.flattened_srt = {9u, 9u, 9u, candidates * 2u};
    for (uint32_t word = 0; word < candidates * 2u; ++word) {
      const auto ordinal = (word * 13u + seed) % (candidates + 1u);
      if (pairs) snapshot.flattened_srt.push_back(word * 4u);
      snapshot.flattened_srt.push_back(ordinal);
      selects.push_back(snapshot.images[ordinal]);
    }
  };
  for (const bool pairs : {false, true}) {
    std::array<ResourceSnapshot, 2> snapshots;
    std::array<ResourceSpecialization, 2> specializations;
    std::array<std::vector<DescriptorValue>, 2> selects;
    for (uint32_t state = 0; state < 2u; ++state) {
      make(state == 0u ? 40u : 37u, state == 0u ? 1u : 2u, pairs, snapshots[state], specializations[state],
           selects[state]);
      GroupIndirectImages(base, snapshots[state], specializations[state]);
      const auto &snapshot = snapshots[state];
      const auto &images = specializations[state].images;
      Check(images.size() == 3u && snapshot.images.size() == 3u && snapshot.image_groups.size() == 3u &&
                images[1].indirect_root == 0u && images[2].indirect_root == 0u &&
                images[1].indirect_capacity == 64u && images[2].indirect_capacity == 64u &&
                images[0].indirect_capacity == 0u && snapshot.image_groups[0].empty() &&
                snapshot.image_groups[1].size() + snapshot.image_groups[2].size() == (state == 0u ? 40u : 37u),
            "grouped storage candidates lost a candidate or a group's capacity");
      const auto offset = images[0].indirect_mapping_offset;
      for (uint32_t word = 0; word < snapshot.flattened_srt[offset]; ++word) {
        const auto mapped = snapshot.flattened_srt[offset + 1u + (pairs ? word * 2u + 1u : word)];
        const auto group = mapped >> IndirectImageGroupShift;
        const auto rank = mapped & ((1u << IndirectImageGroupShift) - 1u);
        const auto &selected = group == 0u ? snapshot.images[0] : snapshot.image_groups[group].at(rank);
        Check(group < 3u && (group != 0u || rank == 0u) && selected == selects[state][word] &&
                  (!pairs || snapshot.flattened_srt[offset + 1u + word * 2u] == word * 4u),
              "a grouped table word selects another descriptor than its ordinal did");
      }
    }
    Check(specializations[0] == specializations[1],
          "two states of one heap with the same candidate types gave two specializations");
  }
  // Too few candidates, or a sampled root: left as they were.
  for (const bool sampled : {false, true}) {
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
    std::vector<DescriptorValue> selects;
    make(sampled ? 40u : IndirectImageGroupMin - 2u, 1u, false, snapshot, specialization, selects);
    auto kept = base;
    if (sampled) kept.images[0].resource_class = ImageResourceClass::Sampled;
    const auto before = specialization;
    const auto mapping = snapshot.flattened_srt;
    GroupIndirectImages(kept, snapshot, specialization);
    Check(specialization == before && snapshot.flattened_srt == mapping && snapshot.image_groups.empty(),
          "a short or sampled indirect image table was grouped");
  }
}

// A written storage image picked from a whole image table by key: a plan reuses the table's candidates and groups
// while the table's words and geometry are unchanged, and every call gives what a plan without that reuse gives.
void TestImageTableReuse() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::Prospero::BufferFormat;
  using Libs::Graphics::Prospero::ImageType;
  using D = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension;
  constexpr uint32_t table = 0x10000u, entries = 48u;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);
  // The table's V#: raw bytes, so its size is its record count.
  std::array<uint32_t, 4> table_words{table, 0u, entries * 32u, 0x5204u};
  DescriptorSource table_source;
  table_source.dword_count = 4;
  for (uint32_t word = 0; word < 4u; ++word) table_source.dwords[word] = Value(table_words[word]);
  program.descriptor_sources.push_back(table_source);
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor = DescriptorSource::IndirectDescriptor {};
  auto &indirect = *root.indirect_descriptor;
  indirect.table_source = 0u;
  indirect.table_stride = 32u;
  program.descriptor_sources.push_back(root);
  program.info.images.push_back({.source = 1u, .resource_class = ImageResourceClass::Storage,
                                 .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
                                 .dimension = D::Dim2D, .written = true});
  struct Memory {
    std::vector<uint32_t> words;
    uint32_t reads = 0;
  } memory;
  memory.words.resize(entries * 8u);
  const auto entry = [&](uint32_t index, uint32_t address, bool second_type) {
    const std::array<uint32_t, 8> image{address, uint32_t(BufferFormat::k32Float) << 20u, 0u,
        (second_type ? Libs::Graphics::DstSel(4, 0, 0, 1) : Libs::Graphics::DstSel(4, 5, 6, 7)) |
            (uint32_t(ImageType::kColor2D) << 28u), 0u, 0u, 0u, 0u};
    std::copy(image.begin(), image.end(), memory.words.begin() + index * 8u);
  };
  for (uint32_t index = 0; index < entries; ++index) {
    if (index % 5u == 4u) continue; // empty slots stay null
    entry(index, 0x100u + index, index % 3u == 0u);
  }
  const SrtRuntime runtime{
      .userdata = &memory,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &memory = *static_cast<Memory *>(data);
        ++memory.reads;
        if (address < table || (address - table) / 4u + words.size() > memory.words.size()) return false;
        std::copy_n(memory.words.begin() + (address - table) / 4u, words.size(), words.begin());
        return true;
      }};
  const auto plan = Extract(program);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  // What a plan that has never seen the table makes of it.
  const auto check_fresh = [&](const char *text) {
    const auto fresh_plan = Extract(program);
    ResourceSnapshot fresh;
    ResourceSpecialization fresh_specialization;
    Check(MaterializeResources(fresh_plan, runtime, fresh, fresh_specialization), "fresh image table failed");
    Check(SameSnapshot(snapshot, fresh) && snapshot.image_groups == fresh.image_groups &&
              specialization == fresh_specialization,
          text);
  };
  Check(MaterializeResources(plan, runtime, snapshot, specialization), "image table materialization failed");
  Check(snapshot.image_groups.size() == 3u && specialization.images.size() == 3u &&
            specialization.images[1].indirect_capacity == 64u && specialization.images[2].indirect_capacity == 64u &&
            snapshot.image_groups[1].size() + snapshot.image_groups[2].size() == entries - entries / 5u,
        "the image table's storage candidates were not grouped");
  check_fresh("the first image table materialization differs from a fresh plan's");
  const auto reads = memory.reads;
  const auto first_snapshot = snapshot;
  const auto first_specialization = specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization), "repeated image table failed");
  Check(SameSnapshot(snapshot, first_snapshot) && snapshot.image_groups == first_snapshot.image_groups &&
            specialization == first_specialization && memory.reads > reads,
        "a repeated image table changed its result or was not read again");
  // One entry changes: the table is read and grouped afresh.
  entry(7u, 0x900u, false);
  Check(MaterializeResources(plan, runtime, snapshot, specialization), "changed image table failed");
  Check(snapshot.image_groups != first_snapshot.image_groups, "a changed image table kept its old candidates");
  check_fresh("a changed image table differs from a fresh plan's");
  Check(MaterializeResources(plan, runtime, snapshot, specialization), "repeated changed image table failed");
  check_fresh("a repeated changed image table differs from a fresh plan's");
  // The same words under another geometry (fewer records): not the same table.
  table_words[2] = (entries - 8u) * 32u;
  program.descriptor_sources[0].dwords[2] = Value(table_words[2]);
  const auto shorter = Extract(program);
  for (uint32_t pass = 0; pass < 2u; ++pass) {
    Check(MaterializeResources(shorter, runtime, snapshot, specialization), "shorter image table failed");
    check_fresh("a shorter image table differs from a fresh plan's");
  }
  // A plan whose table becomes a short one (too few candidates to group) keeps up as well.
  for (uint32_t index = 8u; index < entries; ++index) std::fill_n(memory.words.begin() + index * 8u, 8u, 0u);
  for (uint32_t pass = 0; pass < 2u; ++pass) {
    Check(MaterializeResources(shorter, runtime, snapshot, specialization), "short image table failed");
    Check(snapshot.image_groups.empty() || std::ranges::all_of(snapshot.image_groups, [](const auto &group) {
            return group.empty();
          }),
          "a short image table was grouped");
    check_fresh("a short image table differs from a fresh plan's");
  }
}

// A written storage image picked from a table whose entries are k8_8_8_8UNorm and k5_5_5_1UNorm textures (as in
// Marvel's Wolverine's fight): the table materializes, and its candidates are grouped by type, the 5_5_5_1 ones as an
// unsigned R16 store with the guest format's conversion (PackedStorageFormat), the others as they were.
void TestPackedStorageImageTable() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::Prospero::BufferFormat;
  using Libs::Graphics::Prospero::ImageType;
  using N = Libs::Graphics::Prospero::TextureNumericClass;
  using D = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension;
  constexpr uint32_t table = 0x10000u, entries = 48u;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);
  const std::array<uint32_t, 4> table_words{table, 0u, entries * 32u, 0x5204u};
  DescriptorSource table_source;
  table_source.dword_count = 4;
  for (uint32_t word = 0; word < 4u; ++word) table_source.dwords[word] = Value(table_words[word]);
  program.descriptor_sources.push_back(table_source);
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor = DescriptorSource::IndirectDescriptor {};
  auto &indirect = *root.indirect_descriptor;
  indirect.table_source = 0u;
  indirect.table_stride = 32u;
  program.descriptor_sources.push_back(root);
  program.info.images.push_back({.source = 1u, .resource_class = ImageResourceClass::Storage,
                                 .numeric_class = N::Float, .dimension = D::Dim2D, .written = true});
  std::vector<uint32_t> words(entries * 8u);
  for (uint32_t index = 0; index < entries; ++index) {
    const auto format = index % 4u == 1u ? BufferFormat::k5_5_5_1UNorm : BufferFormat::k8_8_8_8UNorm;
    const std::array<uint32_t, 8> image{0x100u + index, uint32_t(format) << 20u, 0u,
        Libs::Graphics::DstSel(4, 5, 6, 7) | (uint32_t(ImageType::kColor2D) << 28u), 0u, 0u, 0u, 0u};
    std::copy(image.begin(), image.end(), words.begin() + index * 8u);
  }
  const SrtRuntime runtime{
      .userdata = &words,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> out) {
        const auto &words = *static_cast<std::vector<uint32_t> *>(data);
        if (address < table || (address - table) / 4u + out.size() > words.size()) return false;
        std::copy_n(words.begin() + (address - table) / 4u, out.size(), out.begin());
        return true;
      }};
  const auto plan = Extract(program);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "a storage image table of k8_8_8_8UNorm and k5_5_5_1UNorm candidates failed materialization");
  uint32_t packed = 0, plain = 0;
  for (uint32_t index = 1; index < specialization.images.size(); ++index) {
    const auto &image = specialization.images[index];
    const auto count = static_cast<uint32_t>(index < snapshot.image_groups.size() ? snapshot.image_groups[index].size() : 0u);
    if (image.conversion_format == BufferFormat::k5_5_5_1UNorm && image.numeric_class == N::Uint) packed += count;
    if (image.conversion_format == BufferFormat::kInvalid && image.numeric_class == N::Float) plain += count;
  }
  Check(packed == entries / 4u && plain == entries - entries / 4u,
        "a mixed storage image table did not group its 5_5_5_1 candidates as converted R16 stores");
}

// A descriptor dword that is an SRT read, a user-data word or an immediate is fetched after the refresh from its flat
// slot, the user data or the immediate (ResourcePlan::descriptor_fetch): the same words as evaluating it.
void TestDescriptorFetch() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Fetch = ResourcePlan::DescriptorFetch;
  std::array<uint32_t, 2> table{0x11223344u, 0x55667788u};
  auto program = SrtProgram(reinterpret_cast<uint64_t>(table.data()));
  program.user_data_count = 2;
  auto &block = *program.blocks[0];
  auto &user = block.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(1))});
  auto &source = program.descriptor_sources[0];
  source.dwords[2] = Value(&user);
  source.dwords[3] = Value(9u);
  source.dword_count = 4;
  program.info.buffers.push_back({.source = 0});
  auto plan = Extract(program);
  Check(plan.descriptor_fetch.size() == 1, "no fetches for the descriptor source");
  const auto &fetch = plan.descriptor_fetch[0];
  Check(fetch[0].kind == Fetch::Flat && fetch[0].index == 0 && fetch[1].kind == Fetch::Immediate &&
            fetch[1].index == 0 && fetch[2].kind == Fetch::UserData && fetch[2].index == 1 &&
            fetch[3].kind == Fetch::Immediate && fetch[3].index == 9,
        "descriptor dwords were not fixed to their flat slot, user data word and immediates");
  auto walked = Extract(program);
  walked.descriptor_fetch.clear();
  const std::array<uint32_t, 2> user_data{0u, 0xcafef00du};
  const SrtRuntime runtime{.user_data = user_data};
  for (int pass = 0; pass < 3; ++pass) { // a recorded refresh, then replays
    ResourceSnapshot fetched_snapshot, walked_snapshot;
    ResourceSpecialization fetched_specialization, walked_specialization;
    Check(MaterializeResources(plan, runtime, fetched_snapshot, fetched_specialization) &&
              MaterializeResources(walked, runtime, walked_snapshot, walked_specialization),
          "materialization with or without fetches failed");
    Check(SameSnapshot(fetched_snapshot, walked_snapshot) && fetched_specialization == walked_specialization,
          "fetched descriptor differs from the evaluated one");
    Check(fetched_snapshot.buffers[0].dwords[0] == table[0] && fetched_snapshot.buffers[0].dwords[2] == user_data[1] &&
              fetched_snapshot.buffers[0].dwords[3] == 9u,
          "fetched descriptor lost the read, the user data word or the immediate");
    table[0] += 0x10u;
  }
  {
    // Before its refresh a walker evaluates; after it, the flat buffer stands for the read as the memo does.
    SrtWalker walker(plan, runtime);
    DescriptorValue value;
    Check(walker.EvaluateDescriptor(0, value) && value.dwords[0] == table[0], "unrefreshed walker did not evaluate");
  }
  {
    std::vector<uint32_t> flat;
    SrtWalker fresh(plan, runtime);
    Check(fresh.RefreshFlatBuffer(flat) && flat[0] == table[0], "refresh failed");
    const auto loaded = table[0];
    table[0] ^= 0xffffffffu;
    DescriptorValue value;
    Check(fresh.EvaluateDescriptor(0, value) && value.dwords[0] == loaded && value.dwords[2] == user_data[1],
          "fetch after the refresh read the guest again");
    table[0] = loaded;
  }
  // User data past the runtime's words: the walker's failure, not a fetch.
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, {.user_data = std::span(user_data).first(1)}, snapshot, specialization),
        "a fetch read user data past the runtime's words");
}

} // namespace

namespace Common {

void DbgExitHandler(const char *, int, std::string_view) { std::abort(); }

void DbgExitIfHandler(const char *, const char *, int) { std::abort(); }

} // namespace Common

// Upstream's walker tests (d7856753e, bdf48dcf3), adapted: lean keeps its interpreter, so each also
// checks what the interpreter gives for the same plan.

// header: p = Phi(entry: 0, body: 1); table[1] == p enters body, which returns to header. The walker
// grows reachability from the entry and restarts when body reaches the evaluated Phi; lean's interpreter
// starts from every block reached and shrinks. With table[1] == 1 the walker leaves body unreached (the
// GPU never runs it: p is 0 there), the interpreter keeps it.
void TestWalkerBackEdgeReachesPhiPredecessor() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.user_data_count = 3;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 4; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &header = *program.blocks[1];
  auto &body = *program.blocks[2];
  auto &exit = *program.blocks[3];
  const auto branch = [](Block &block, Block &target, Value condition = {}, Block *other = nullptr) {
    block.terminator.kind = other ? TerminatorKind::ConditionalBranch : TerminatorKind::Branch;
    block.terminator.true_block = &target;
    block.terminator.false_block = other;
    block.condition = condition;
    block.AddBranch(&target);
    if (other) block.AddBranch(other);
  };
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  const auto table = Value(&entry.AppendNewInst(ValueOpcode::GetAddressResource,
      {Value(&entry.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(0))})),
       Value(&entry.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(1))}))}));
  const auto load = [&](Block &block, uint32_t offset) -> Inst & {
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32, {table, Value(offset), Value(0u), Value(true)});
    read.SetFlags(MemoryFlags{.index = 0});
    return read;
  };
  auto &p = header.AppendNewInst(ValueOpcode::Phi);
  p.SetFlags(Type::U32);
  p.AddPhiOperand(&entry, Value(0u));
  p.AddPhiOperand(&body, Value(1u));
  branch(entry, header);
  branch(header, body, Value(&header.AppendNewInst(ValueOpcode::IEqual32, {Value(&load(header, 4u)), Value(&p)})),
         &exit);
  branch(body, header);
  exit.terminator.kind = TerminatorKind::Return;
  program.srt_reads.push_back({Value(&load(exit, 0u)), 0});
  exit.AppendNewInst(ValueOpcode::ReadConst,
                     {Value(&exit.AppendNewInst(ValueOpcode::GetSrtResource)), Value(0u)});
  auto plan = Extract(program);
  Check(plan.control_flow.size() == 4 && plan.control_phi_predecessors.size() == 1 && plan.srt_code != nullptr,
        "back-edge plan lost its control flow or Phi");
  plan.capture_specialization_reads = true;
  const std::array<uint32_t, 2> user_data{0x1000u, 0u};
  for (uint32_t selected : {0u, 1u}) {
    FakeGuestMemory fake(+[](void *userdata, uint64_t address, std::span<uint32_t> words) {
          words[0] = address == 0x1000u ? 0x5678u : *static_cast<uint32_t *>(userdata);
          return address == 0x1000u || address == 0x1004u;
        }, &selected);
    std::vector<uint32_t> flat;
    std::vector<std::pair<uint64_t, uint64_t>> loads;
    Check(WalkSrt(plan, {.user_data = user_data, .direct_reads = true}, flat, &loads) &&
              flat == std::vector<uint32_t>{0x5678u},
          "back edge to an evaluated Phi left its exit unreachable");
    Check(loads == std::vector<std::pair<uint64_t, uint64_t>>{{0x1004u, 4u}, {0x1000u, 4u}},
          "Phi restart kept the loads of the abandoned pass");
    std::vector<uint8_t> reached;
    for (uint32_t block = 0; block < 4; ++block) reached.push_back(WalkedReached(plan, block));
    Check(reached == (selected == 0u ? std::vector<uint8_t>{1, 1, 1, 1} : std::vector<uint8_t>{1, 1, 0, 1}),
          "Phi restart followed a rejected edge or missed a reachable one");
  }
  for (uint32_t selected : {0u, 1u}) {
    struct Memory { uint32_t selected; } memory{selected};
    const SrtRuntime runtime{
        .user_data = user_data,
        .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
          words[0] = address == 0x1000u ? 0x5678u : static_cast<Memory *>(data)->selected;
          return address == 0x1000u || address == 0x1004u;
        },
        .userdata = &memory,
        .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
          words[0] = address == 0x1000u ? 0x5678u : static_cast<Memory *>(data)->selected;
          return address == 0x1000u || address == 0x1004u;
        }};
    CheckWalkAgreesWith(plan, runtime, "the walker's reach on a back edge was not the interpreter's or a subset of it",
                        true, selected == 1u ? 1u : 0u);
  }
}

// Reads sharing one MemoryInfo walk once each; a pointer read through an unreadable table reads zero
// (in the emulator a load outside the null page region fails the walk instead).
void TestWalkerAliasesShareReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = SrtProgram(0x1000u);
  auto &block = *program.blocks[0];
  const auto append_read = [&](uint32_t address) {
    auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                       {Value(address), Value(0u)});
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(0u), Value(0u), Value(true)});
    read.SetFlags(MemoryFlags{.index = 0, .pc = address});
    const auto slot = static_cast<uint32_t>(program.srt_reads.size());
    program.srt_reads.push_back({Value(&read), slot});
    auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
    return Value(&block.AppendNewInst(ValueOpcode::ReadConst,
                                      {Value(&srt), Value(slot)}));
  };
  const auto ordinary = append_read(0x2000u);
  const auto pointer = append_read(0x3000u);
  program.srt_reads[0].value.Instruction()->Arg(0).Instruction()->SetArg(0, pointer);
  auto &mask = block.AppendNewInst(ValueOpcode::INotEqual32, {ordinary, Value(0u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane,
                                    {Value(9u), Value(&mask)});
  program.descriptor_sources.push_back({.dwords = {Value(&first), Value(0u)},
                                       .dword_count = 2});
  DescriptorSource indirect;
  indirect.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1};
  program.descriptor_sources.push_back(indirect);
  auto plan = Extract(program);
  struct Reads { uint32_t count = 0; bool dirty = false; } reads;
  FakeGuestMemory fake(+[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.count;
        if (address == 0x1000u) words[0] = 0x1111u;
        else if (address == 0x2000u) words[0] = 0x2222u;
        else if (address == 0x3000u && !reads.dirty) words[0] = 0x1000u;
        else return false;
        return true;
      }, &reads);
  std::vector<uint32_t> flat;
  Check(WalkSrt(plan, {.direct_reads = true}, flat, nullptr) &&
            flat == std::vector<uint32_t>{0x1111u, 0x2222u, 0x1000u} && reads.count == 3,
        "normalized aliases changed nested reads or repeated a shared read");
  reads = {.dirty = true};
  Check(WalkSrt(plan, {.direct_reads = true}, flat, nullptr) &&
            flat == std::vector<uint32_t>{0u, 0x2222u, 0u} && reads.count == 3,
        "unreadable pointer did not read as zero");
  reads = {};
  const SrtRuntime runtime{.read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
                             auto &reads = *static_cast<Reads *>(data);
                             if (address == 0x1000u) words[0] = 0x1111u;
                             else if (address == 0x2000u) words[0] = 0x2222u;
                             else if (address == 0x3000u && !reads.dirty) words[0] = 0x1000u;
                             else return false;
                             return true;
                           },
                           .userdata = &reads};
  CheckWalkAgreesWith(plan, runtime, "aliased reads walked differently");
}

void TestWalkerExactReciprocalArithmetic() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  auto &block = AddValueBlock(program);
  for (const float divisor : {64.f, 128.f, 256.f, 512.f,
                              std::numeric_limits<float>::min(),
                              std::bit_cast<float>(253u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(EvaluateSrtValue(program, {}, Value(&reciprocal), result) &&
              result == std::bit_cast<uint32_t>(1.f / divisor),
          "power-of-two reciprocal was not exact");
  }
  for (const float divisor : {0.f, 3.f, -64.f, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::denorm_min(),
                              std::bit_cast<float>(254u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(!EvaluateSrtValue(program, {}, Value(&reciprocal), result),
          "unsupported reciprocal rounding or exceptional input was accepted");
  }
  // Every pure op the walker emits gives the interpreter's bits, on edge inputs.
  const std::array<uint32_t, 10> inputs{0u, 1u, 31u, 32u, 33u, 0x7fffffffu, 0x80000000u, 0xffffffffu,
                                        0x4f800000u /* 2^32 */, 0x7fc00000u /* NaN */};
  for (const auto op : {ValueOpcode::IAdd32, ValueOpcode::ISub32, ValueOpcode::IMul32, ValueOpcode::UMulHi,
                        ValueOpcode::UMin32, ValueOpcode::UMax32, ValueOpcode::BitwiseAnd32, ValueOpcode::BitwiseOr32,
                        ValueOpcode::BitwiseXor32, ValueOpcode::ShiftLeftLogical32, ValueOpcode::ShiftRightLogical32,
                        ValueOpcode::ShiftRightArithmetic32, ValueOpcode::IEqual32, ValueOpcode::INotEqual32,
                        ValueOpcode::UGreaterThan32, ValueOpcode::SGreaterThanEqual32, ValueOpcode::FPMul32,
                        ValueOpcode::FPOrdLessThanEqual32, ValueOpcode::FPOrdGreaterThanEqual32}) {
    for (const auto a : inputs) {
      for (const auto b : inputs) {
        auto &inst = block.AppendNewInst(op, {Value(a), Value(b)});
        uint32_t walked = 0, interpreted = 0;
        const bool known = EvaluateSrtValue(program, {}, Value(&inst), walked);
        SrtWalker walker(program, {});
        Check(known == walker.Evaluate(Value(&inst), interpreted) && (!known || walked == interpreted),
              "a binary op gave other bits in the walker than in the interpreter");
      }
    }
  }
  for (const auto op : {ValueOpcode::ConvertF32U32, ValueOpcode::ConvertU32F32, ValueOpcode::ConvertS32F32,
                        ValueOpcode::FPTrunc32, ValueOpcode::FPIsNan32, ValueOpcode::BitwiseNot32,
                        ValueOpcode::LogicalNot}) {
    for (const auto a : inputs) {
      for (const auto value : {Value(a), Value::F32(std::bit_cast<float>(a)), Value(a != 0u)}) {
        auto &inst = block.AppendNewInst(op, {value});
        uint32_t walked = 0, interpreted = 0;
        const bool known = EvaluateSrtValue(program, {}, Value(&inst), walked);
        SrtWalker walker(program, {});
        Check(known == walker.Evaluate(Value(&inst), interpreted) && (!known || walked == interpreted),
              "a unary op gave other bits in the walker than in the interpreter");
      }
    }
  }
  for (const auto op : {ValueOpcode::BitFieldUExtract, ValueOpcode::BitFieldSExtract}) {
    for (const uint32_t offset : {0u, 1u, 16u, 31u, 32u, 33u}) {
      for (const uint32_t width : {0u, 1u, 15u, 16u, 31u, 32u}) {
        auto &inst = block.AppendNewInst(op, {Value(0x8badf00du), Value(offset), Value(width)});
        uint32_t walked = 0, interpreted = 0;
        const bool known = EvaluateSrtValue(program, {}, Value(&inst), walked);
        SrtWalker walker(program, {});
        Check(known == walker.Evaluate(Value(&inst), interpreted) && (!known || walked == interpreted),
              "a bit field extract gave other bits in the walker than in the interpreter");
      }
    }
  }
}

int main() {
  TestMappedSrtUsesDirectReaderByDefault();
  TestNativeNestedSrtRefresh();
  TestIntegerRuntimeValueFollowsSrtReads();
  TestSrtAliasesRetainReadPolicy();
  TestUniformVectorDescriptorRead();
  TestExactReciprocalDescriptorArithmetic();
  TestUnbasedFlatCacheHitMaterializes();
  for (const bool memory_condition : {false, true})
    TestWrittenDescriptorPredicateReads(memory_condition);
  TestFailedMaterializationRejectsStage();
  TestSharedControlPredicates();
  TestNestedExecPredicatesReuseReads();
  TestFiniteImageRefreshReusesScalarReads();
  for (const bool independent_mask : {false, true})
    TestEmptyTraversalSkipsUnmappedDescriptor(independent_mask);
  TestEmptyTraversalSkipsUnmappedDescriptor(false, true);
  TestCyclicControlPredicatesRemainUnknown();
  TestEmptyScalarBufferSpecialization();
  TestMixedSamplerVariantsShareRuntimeDescriptor();
  TestSignedStorageSpecialization();
  TestPackedStorageSpecialization();
  TestPackedStorageImageTable();
  TestGroupIndirectStorageImages();
  TestImageTableReuse();
  TestDescriptorFetch();
  for (const bool atomic64 : {false, true}) TestUnreachedImageIgnoresSharedDescriptor(atomic64);
  TestDeadImageMemoryKeepsTranslatorIndex();
  TestImageForms();
  for (const bool mixed : {false, true}) TestIndirectImageSamplerClasses(mixed);
  TestWalkerBackEdgeReachesPhiPredecessor();
  TestWalkerAliasesShareReads();
  TestWalkerExactReciprocalArithmetic();
  std::puts("ResourceMaterializationTests: all cases passed");
  return 0;
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
