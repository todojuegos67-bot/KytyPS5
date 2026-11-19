#include "graphics/host_gpu/renderer/emitStats.h"

#include "common/perfStats.h"

#include <fmt/format.h>

namespace Libs::Graphics::EmitStats {

namespace {
std::array<uint64_t, static_cast<size_t>(Diff::Count)> g_diffs {};
uint64_t                                               g_diffs_logged = 0;

constexpr std::array<const char*, static_cast<size_t>(Diff::Count)> DiffNames {
    "firma", "plantilla", "omitir set", "empujar set"};
} // namespace

void NoteDiff(Diff kind, const std::string& what) {
	g_diffs[static_cast<size_t>(kind)]++;
	if (g_diffs_logged < 16) {
		g_diffs_logged++;
		Log::WriteUnmuted(fmt::format("Emision: diferencia del verificador (KYTY_EMIT_FAST=2), {}: {}\n",
		                              DiffNames[static_cast<size_t>(kind)], what));
	}
}

void Report(double cycles_per_us, double seconds, uint64_t draws) {
	using Detail::g_count;
	using Detail::g_cycles;
	using Detail::g_timed;
	if (draws == 0) {
		return;
	}
	const double d = static_cast<double>(draws);
	// Per draw: how many, and the microseconds they take (their timed average times their count).
	const auto n = [&](Item item) { return static_cast<double>(g_count[item]) / d; };
	const auto us = [&](Item item) {
		if (g_timed[item] == 0 || cycles_per_us <= 0.0) {
			return 0.0;
		}
		return static_cast<double>(g_cycles[item]) / static_cast<double>(g_timed[item]) /
		       cycles_per_us * static_cast<double>(g_count[item]) / d;
	};
	const auto call = [&](const char* name, Item item) {
		return fmt::format("{} {:.2f} ({:.3f} us)", name, n(item), us(item));
	};
	std::string line = fmt::format(
	    "Emision draw ({:.1f} s, KYTY_EMIT_FAST={}): llamadas por draw (us por draw): {}, {}, {}, {}, "
	    "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
	    seconds, Mode(), call("push constants", CallPushConstants),
	    call("push descriptors", CallPushDescriptors), call("push con plantilla", CallPushTemplate),
	    call("update+bind set", CallUpdateBindSet), call("set con plantilla", CallUpdateTemplateSet),
	    call("bind sets", CallBindSets), call("bind pipeline", CallBindPipeline),
	    call("bind vertex", CallBindVertex), call("bind index", CallBindIndex),
	    call("estado dinamico", CallState), call("barreras", CallBarrier),
	    call("begin rendering", CallBeginRendering), call("end rendering", CallEndRendering),
	    call("draw", CallDraw), call("copias", CallCopy), call("consultas", CallQuery));
	line += fmt::format(
	    " | omitidas por draw: pipeline {:.2f}, estado {:.2f}, vertex {:.2f}, index {:.2f}, set "
	    "{:.2f}, constantes {:.2f}",
	    n(SkipPipeline), n(SkipState), n(SkipVertex), n(SkipIndex), n(SkipSet), n(SkipConstants));
	line += fmt::format(
	    " | pasos (n/draw, us/draw): esperas del lock {}, transiciones {} (con barrera {:.2f}), "
	    "escrituras {}, set y constantes {}, set del pool {} (vkAllocateDescriptorSets {:.3f}), "
	    "plantillas creadas {}, estado dinamico {}, vistas de targets {}, anotaciones {}, "
	    "consultas {}",
	    fmt::format("{:.3f} ({:.3f})", n(StepLockWait), us(StepLockWait)),
	    fmt::format("{:.2f} ({:.3f})", n(StepTransit), us(StepTransit)), n(CountTransitBarriers),
	    fmt::format("{:.2f} ({:.3f})", n(StepWrites), us(StepWrites)),
	    fmt::format("{:.2f} ({:.3f})", n(StepSet), us(StepSet)),
	    fmt::format("{:.2f} ({:.3f})", n(StepHeapCommit), us(StepHeapCommit)),
	    n(CountPoolAllocations), g_count[StepTemplate],
	    fmt::format("{:.2f} ({:.3f})", n(StepDynamic), us(StepDynamic)),
	    fmt::format("{:.2f} ({:.3f})", n(StepTargetViews), us(StepTargetViews)),
	    fmt::format("{:.2f} ({:.3f})", n(StepAnnotate), us(StepAnnotate)),
	    fmt::format("{:.2f} ({:.3f})", n(StepQueries), us(StepQueries)));
	line += fmt::format(
	    " | por draw: escrituras {:.2f}, infos {:.2f}, comandos encolados {:.2f}, bytes copiados "
	    "{:.0f}, bytes de plantilla {:.0f} | renderings terminados por draw: en commit {:.3f}, en "
	    "targets {:.3f}, en emision {:.3f}, antes {:.3f}",
	    n(CountWrites), n(CountInfos), n(CountRecords), n(CountStashBytes), n(CountTemplateBytes),
	    n(EndInCommit), n(EndInRendering), n(EndInEmit), n(EndInPrepare));
	if (Mode() == 2) {
		line += " | verificador, diferencias:";
		for (size_t i = 0; i < g_diffs.size(); i++) {
			line += fmt::format(" {} {}", DiffNames[i], g_diffs[i]);
		}
	}
	Log::WriteUnmuted(line + "\n");
	g_count.fill(0);
	g_cycles.fill(0);
	g_timed.fill(0);
}

} // namespace Libs::Graphics::EmitStats
