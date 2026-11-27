#pragma once
// (bda-repite) Con KYTY_TECHO=1 (solo mide; apagado = una lectura de un static por llamada): cuantas preparaciones BDA
// (EvaluateBdaReadPlan + PrepareBdaReadRanges: RenderExecutor::TablePrepareBda y PrepareBdaBindings) repiten EXACTAMENTE
// las entradas de otra del mismo fotograma o del anterior, y si su salida (los rangos) es la misma, para saber si una
// cache por entradas seria correcta y cuanto quitaria. Linea "[bda-repite]" cada 5 s (desde LiveControl::Flip, techo).
// Clave (por etapa con plan): la direccion del plan y su contenido entero (por span: tipo e indice/inmediato de las
// palabras low/high, begin, end, scalar_base; complete; numero de spans) + el valor de cada palabra que la evaluacion lee
// (user_data[i] o flattened_srt[i], o una marca si el indice queda fuera: la evaluacion falla) + el tamano de user_data y
// flattened_srt; mas que etapas evaluaron (1a/2a). Nada mas entra en EvaluateBdaReadPlan (TRACKER_ADDRESS_SIZE es fijo).
// Salida comparada: bounded y los rangos combinados entregados a PrepareBdaReadRanges.
#include "graphics/shader/recompiler/ir/BdaReadPlan.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "local/techo.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace BdaRepite {

struct Key {
	std::vector<uint64_t> words;
	uint64_t              hash = 0;
	bool                  operator==(const Key& o) const { return hash == o.hash && words == o.words; }
};
struct KeyHash {
	size_t operator()(const Key& k) const { return static_cast<size_t>(k.hash); }
};
struct Out {
	bool                                       bounded = false;
	std::vector<::Libs::Graphics::GuestRange> ranges;
	bool operator==(const Out& o) const { return bounded == o.bounded && ranges == o.ranges; }
};

inline void Push(Key& k, uint64_t v) {
	k.words.push_back(v);
	k.hash = (k.hash ^ v) * 0x100000001b3ULL + (k.hash >> 29);
}

// Una etapa: el plan y las palabras que la evaluacion lee de la instantanea.
inline void AddStage(Key& k, uint32_t stage, const ::Libs::Graphics::ShaderRecompiler::IR::BdaReadPlan& plan,
                     std::span<const uint32_t> user_data, std::span<const uint32_t> flat) {
	using W = ::Libs::Graphics::ShaderRecompiler::IR::BdaReadWord;
	Push(k, 0xB0A0000000000000ULL | stage);
	Push(k, reinterpret_cast<uintptr_t>(&plan));
	Push(k, (uint64_t {plan.complete} << 32u) | plan.spans.size());
	Push(k, (uint64_t(user_data.size()) << 32u) | flat.size());
	const auto word = [&](const W& w) {
		Push(k, (uint64_t(static_cast<uint8_t>(w.kind)) << 32u) | w.value);
		if (w.kind == W::Kind::Immediate) return;
		const auto values = w.kind == W::Kind::Flat ? flat : user_data;
		Push(k, w.value < values.size() ? values[w.value] : 0xFFFFFFFF00000000ULL);
	};
	for (const auto& s: plan.spans) {
		word(s.low);
		word(s.high);
		Push(k, static_cast<uint64_t>(s.begin));
		Push(k, static_cast<uint64_t>(s.end));
		Push(k, s.scalar_base ? 1 : 0);
	}
}

struct State {
	std::mutex                             m;
	std::unordered_map<Key, Out, KeyHash> cur, prev;
	uint64_t calls = 0, same_frame = 0, same_frame_ok = 0, prev_frame = 0, prev_frame_ok = 0, any = 0, any_ok = 0;
	uint64_t eval_ns = 0, total_ns = 0, frames = 0;
	int64_t  last_print = 0;
};
inline State& S() {
	static State s;
	return s;
}

// Alrededor de una preparacion: Begin antes, Eval despues de evaluar (rangos combinados), End al final.
struct Scope {
	bool                                  on;
	std::chrono::steady_clock::time_point t0, t1;
	Key                                   key;
	explicit Scope(): on(Techo::On()) {
		if (on) t0 = std::chrono::steady_clock::now();
	}
	void Eval(bool bounded, std::span<const ::Libs::Graphics::GuestRange> ranges) {
		if (!on) return;
		t1 = std::chrono::steady_clock::now();
		Out out;
		out.bounded = bounded;
		out.ranges.assign(ranges.begin(), ranges.end());
		auto& s = S();
		std::lock_guard lock(s.m);
		++s.calls;
		s.eval_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
		const auto c = s.cur.find(key);
		const auto p = s.prev.find(key);
		if (c != s.cur.end()) {
			++s.same_frame;
			if (c->second == out) ++s.same_frame_ok;
		}
		if (p != s.prev.end()) {
			++s.prev_frame;
			if (p->second == out) ++s.prev_frame_ok;
		}
		if (c != s.cur.end() || p != s.prev.end()) {
			++s.any;
			if ((c != s.cur.end() ? c->second : p->second) == out) ++s.any_ok;
		}
		if (c == s.cur.end()) s.cur.emplace(std::move(key), std::move(out));
	}
	~Scope() {
		if (!on) return;
		auto& s = S();
		std::lock_guard lock(s.m);
		s.total_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;
};

// Desde el Flip (hilo de render), con KYTY_TECHO=1.
inline void Flip(int64_t flip_ns) {
	if (!Techo::On()) return;
	auto& s = S();
	std::lock_guard lock(s.m);
	++s.frames;
	s.prev.swap(s.cur);
	s.cur.clear();
	if (s.last_print == 0) s.last_print = flip_ns;
	if (flip_ns - s.last_print < 5'000'000'000ll) return;
	const double f = static_cast<double>(s.frames), n = s.calls ? static_cast<double>(s.calls) : 1.0;
	std::printf("[bda-repite] %.0f llam/fot; mismas entradas: mismo fot %.0f/fot (%.1f%%, salida igual %llu de %llu), "
	            "fot anterior %.0f/fot (%.1f%%, igual %llu de %llu), cualquiera %.1f%% (igual %llu de %llu); "
	            "%.2f us/llam (evaluar %.2f us), %.2f ms/fot\n",
	            s.calls / f, s.same_frame / f, 100.0 * s.same_frame / n, (unsigned long long)s.same_frame_ok,
	            (unsigned long long)s.same_frame, s.prev_frame / f, 100.0 * s.prev_frame / n,
	            (unsigned long long)s.prev_frame_ok, (unsigned long long)s.prev_frame, 100.0 * s.any / n,
	            (unsigned long long)s.any_ok, (unsigned long long)s.any, s.total_ns / n / 1e3, s.eval_ns / n / 1e3,
	            s.total_ns / f / 1e6);
	std::fflush(stdout);
	s.calls = s.same_frame = s.same_frame_ok = s.prev_frame = s.prev_frame_ok = s.any = s.any_ok = 0;
	s.eval_ns = s.total_ns = s.frames = 0;
	s.last_print = flip_ns;
}

} // namespace BdaRepite
