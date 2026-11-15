#include "loader/wolverineRenderStep.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/virtualMemory.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <fmt/format.h>
#include <string>
#include <thread>

namespace Loader {

using namespace WolverineRenderStep;

namespace {

std::atomic<uint64_t> g_base {0}; // The eboot's base once it is known to be 1.05.

uint32_t Guest32(uint64_t base, uint64_t va) {
	return *reinterpret_cast<const volatile uint32_t*>(base + va);
}

// The game fills its step table at boot, after this hook. One log line, once the game's step is the pinned one (or
// after two minutes without), shows what it filled and which step it uses.
void ReportOnce(uint64_t base, uint32_t index, bool pro) {
	std::thread([base, index, pro] {
		uint32_t active = 0;
		for (int second = 0; second < 120; second++) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			active = Guest32(base, ACTIVE_VA);
			if (active == index) {
				break;
			}
		}
		const uint32_t count = Guest32(base, COUNT_VA);
		std::string    table;
		for (uint32_t i = 0; i < count && i < 16; i++) {
			table += " " + std::to_string(Guest32(base, TABLE_VA + 20 * i)) + "x" +
			         std::to_string(Guest32(base, TABLE_VA + 20 * i + 4));
		}
		const bool ok = active == index && index < count &&
		                Guest32(base, TABLE_VA + 20 * index) == StepAt(pro, index).width &&
		                Guest32(base, TABLE_VA + 20 * index + 4) == StepAt(pro, index).height;
		Log::WriteToConsoleAndLog(fmt::format("Wolverine render resolution: {} the game's step is {} ({} steps:{})\n",
		                                      ok ? "pinned," : "NOT PINNED,", active, count, table));
	}).detach();
}

} // namespace

namespace WolverineRenderStep {

bool CanReadActiveStep() {
	return g_base.load(std::memory_order_acquire) != 0;
}

std::optional<ActiveStep> ReadActiveStep() {
	const uint64_t base = g_base.load(std::memory_order_acquire);
	if (base == 0) {
		return std::nullopt;
	}
	const uint32_t active = Guest32(base, ACTIVE_VA);
	const uint64_t entry  = TABLE_VA + 20ull * std::min(active, 15u);
	return MakeActiveStep(active, Guest32(base, COUNT_VA), {Guest32(base, entry), Guest32(base, entry + 4)});
}

} // namespace WolverineRenderStep

void InstallWolverineRenderStep(Program* main_program) {
	const bool pro  = Config::Ps5ProEnabled();
	const bool dlss = pro && !Config::GetDlssFolder().empty() && Config::GetRenderWidth() == 0;
	const uint32_t width  = dlss ? DLSS_PRO_STEP.width : Config::GetRenderWidth();
	const uint32_t height = dlss ? DLSS_PRO_STEP.height : Config::GetRenderHeight();
	const bool     pin    = width != 0 && height != 0;
	const int      index  = FindStep(width, height, pro);
	std::string    title_id;
	const bool     wolverine = main_program != nullptr && SystemContentParamSfoGetString("TITLE_ID", &title_id) &&
	                       title_id == "PPSA03671";
	if (!wolverine || (pin && index < 0)) {
		if (pin) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Wolverine render resolution: {}x{} applies to Marvel's Wolverine 1.05 only; not pinned\n", width, height));
		}
		return;
	}
	const auto base = main_program->base_vaddr;
	auto*      code = reinterpret_cast<uint8_t*>(base + PATCH_VA);
	if (std::memcmp(code, ORIGINAL.data(), ORIGINAL.size()) != 0 ||
	    std::memcmp(reinterpret_cast<const void*>(base + STORE_VA), STORE.data(), STORE.size()) != 0) {
		if (pin) {
			Log::WriteToConsoleAndLog("Wolverine render resolution: the eboot is not 1.05's; not pinned\n");
		}
		return;
	}
	g_base.store(base, std::memory_order_release);
	if (!pin) {
		return;
	}
	const auto patch = BuildPatch(static_cast<uint32_t>(index));
	std::memcpy(code, patch.data(), patch.size());
	EXIT_IF(!Common::VirtualMemory::FlushInstructionCache(base + PATCH_VA, patch.size()));
	Log::WriteToConsoleAndLog(fmt::format("Wolverine render resolution: {}x{} (step {} of the game's {} table{})\n", width,
	                                      height, index, pro ? "Pro" : "base", dlss ? ", for DLSS" : ""));
	ReportOnce(base, static_cast<uint32_t>(index), pro);
}

} // namespace Loader
