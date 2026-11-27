#pragma once
// (rando-regiones) Reescritor generico de las pruebas de region de los scripts de IA (*_battle.lua / *_logic.lua).
//
// Ghidra (eboot): las funciones Lua de IA que miran regiones del MSB son IsInsideTargetRegion (registrada en 0x104a01,
// FUN_000f6490) e IsInsideObserveRegion (0x1056ef); las dos acaban en FUN_00201f90, que busca la region por su id de
// entidad en el MSB cargado (FUN_0020d6d0) y devuelve 0 (falso) si no existe. AddObserveRegion / DeleteObserveRegion
// solo dan de alta / baja la region a vigilar (interrupcion INTERUPT_Inside_ObserveArea): si la region no existe nunca
// salta, sin mas efecto. Fuera de su mapa (u otro sitio del mismo mapa) un enemigo cuyo script pregunta "esta el jugador
// en mi region" recibe siempre falso: no ataca, no persigue o vuelve a su sitio sin parar.
//
// Reglas (por forma de llamada, no por texto de cada archivo):
//   IsInsideTargetRegion(TARGET_ENE_0, R) / IsInsideObserveRegion(n) con n dado de alta para TARGET_ENE_0
//       -> (ai:GetDist(TARGET_ENE_0) <= 20)     el jugador "esta en la zona" si esta a 20 m o menos: ataca y persigue
//   IsInsideTargetRegion(TARGET_SELF, R) / observe de TARGET_SELF
//       -> (ai:GetDist(POINT_INITIAL) <= 15)    "estoy en mi zona" = a 15 m o menos de donde aparecio
//       -> false si R es una zona prohibida (la que corta el movimiento con INTERUPT_Inside_ObserveArea + ClearSubGoal)
//   local XXX_HOME_DIST / BACKHOME_DIST = N (N < 15) -> 15: vuelve a su sitio solo si se aleja mas de 15 m
//   Cualquier otra forma (otro objetivo, slot de observe con varias regiones, otra funcion ai:*Region*) se deja tal cual
//   y se cuenta como desconocida. Las lineas comentadas (--) no se tocan.
// Cada sustitucion es una expresion entre parentesis en lugar de una llamada completa: la sintaxis Lua no cambia.
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace Rando::Lua {

struct RegionStats {
	int target   = 0; // IsInsideTargetRegion cambiadas
	int observe  = 0; // IsInsideObserveRegion cambiadas
	int home     = 0; // distancias de vuelta a casa subidas
	int unknown  = 0; // formas no reconocidas (se dejan)
	std::string unknown_text;
	int  Total() const { return target + observe + home; }
};

namespace Detail2 {

inline bool InLineComment(const std::string& t, size_t p) {
	size_t s = t.rfind('\n', p);
	s        = (s == std::string::npos) ? 0 : s + 1;
	return t.find("--", s) < p;
}
inline std::string Trim(const std::string& s) {
	size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
	return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}
struct Call {
	size_t                   begin = 0, end = 0; // [begin, end) = "ai:Name( ... )"
	std::string              name;
	std::vector<std::string> args;
};
// Todas las llamadas "ai:<algo>Region<algo>(...)" fuera de comentarios de linea.
inline std::vector<Call> FindCalls(const std::string& t) {
	std::vector<Call> out;
	for (size_t p = t.find("ai:"); p != std::string::npos; p = t.find("ai:", p + 3)) {
		if (p > 0 && (std::isalnum(static_cast<unsigned char>(t[p - 1])) || t[p - 1] == '_')) continue;
		size_t q = p + 3;
		while (q < t.size() && (std::isalnum(static_cast<unsigned char>(t[q])) || t[q] == '_')) q++;
		std::string name = t.substr(p + 3, q - p - 3);
		if (name.find("Region") == std::string::npos || InLineComment(t, p)) continue;
		size_t o = q;
		while (o < t.size() && (t[o] == ' ' || t[o] == '\t')) o++;
		if (o >= t.size() || t[o] != '(') continue;
		int         depth = 0;
		size_t      e     = o;
		std::string cur;
		Call        c;
		bool        ok = false;
		for (; e < t.size(); e++) {
			char ch = t[e];
			if (ch == '\n') break;
			if (ch == '(') {
				if (depth++ == 0) continue;
			} else if (ch == ')') {
				if (--depth == 0) {
					ok = true;
					break;
				}
			} else if (ch == ',' && depth == 1) {
				c.args.push_back(Trim(cur));
				cur.clear();
				continue;
			}
			cur.push_back(ch);
		}
		if (!ok) continue;
		if (!Trim(cur).empty() || !c.args.empty()) c.args.push_back(Trim(cur));
		c.begin = p;
		c.end   = e + 1;
		c.name  = name;
		out.push_back(std::move(c));
	}
	return out;
}

} // namespace Detail2

// Reescribe |t|. Devuelve true si cambio algo.
inline bool RewriteRegionCalls(std::string* t, RegionStats* st) {
	using namespace Detail2;
	auto calls = FindCalls(*t);
	// Altas de observe: slot -> (objetivo, region); slot con dos altas distintas = ambiguo.
	std::map<std::string, std::pair<std::string, std::string>> slot;
	std::set<std::string>                                      ambiguous;
	std::set<std::string>                                      prohibited; // regiones SELF que cortan el movimiento
	for (const auto& c: calls)
		if (c.name == "AddObserveRegion" && c.args.size() == 3) {
			auto v  = std::make_pair(c.args[1], c.args[2]);
			auto it = slot.find(c.args[0]);
			if (it != slot.end() && it->second != v) ambiguous.insert(c.args[0]);
			slot[c.args[0]] = v;
		}
	for (const auto& c: calls)
		if (c.name == "IsInsideObserveRegion" && c.args.size() == 1 && slot.count(c.args[0])) {
			const size_t a = c.begin > 400 ? c.begin - 400 : 0;
			const bool   intr = t->substr(a, c.begin - a).find("INTERUPT_Inside_ObserveArea") != std::string::npos;
			const bool   clr  = t->substr(c.end, 200).find("ClearSubGoal") != std::string::npos;
			if (intr && clr && slot[c.args[0]].first == "TARGET_SELF") prohibited.insert(slot[c.args[0]].second);
		}
	auto expr = [&](const std::string& tgt, const std::string& reg) -> std::string {
		if (tgt == "TARGET_ENE_0") return "(ai:GetDist(TARGET_ENE_0) <= 20)";
		if (tgt == "TARGET_SELF") return prohibited.count(reg) ? "(false)" : "(ai:GetDist(POINT_INITIAL) <= 15)";
		return {};
	};
	std::string out;
	size_t      last = 0;
	for (const auto& c: calls) {
		std::string rep;
		if (c.name == "IsInsideTargetRegion" && c.args.size() == 2) {
			rep = expr(c.args[0], c.args[1]);
			if (!rep.empty()) st->target++;
		} else if (c.name == "IsInsideObserveRegion" && c.args.size() == 1 && slot.count(c.args[0]) && !ambiguous.count(c.args[0])) {
			rep = expr(slot[c.args[0]].first, slot[c.args[0]].second);
			if (!rep.empty()) st->observe++;
		} else if (c.name == "AddObserveRegion" || c.name == "DeleteObserveRegion") {
			continue; // inofensivas: sin region en el MSB nunca saltan
		}
		if (rep.empty()) {
			st->unknown++;
			if (st->unknown_text.size() < 200) st->unknown_text += (st->unknown_text.empty() ? "" : " | ") + t->substr(c.begin, c.end - c.begin);
			continue;
		}
		out.append(*t, last, c.begin - last);
		out += rep;
		last = c.end;
	}
	out.append(*t, last, std::string::npos);
	// local <...>HOME_DIST = N  (N < 15) -> 15
	for (size_t p = out.find("HOME_DIST"); p != std::string::npos; p = out.find("HOME_DIST", p + 9)) {
		size_t ls = out.rfind('\n', p);
		ls        = ls == std::string::npos ? 0 : ls + 1;
		if (Trim(out.substr(ls, p - ls)).rfind("local", 0) != 0 || InLineComment(out, p)) continue;
		size_t q = p + 9;
		while (q < out.size() && (out[q] == ' ' || out[q] == '\t')) q++;
		if (q >= out.size() || out[q] != '=') continue;
		q++;
		while (q < out.size() && (out[q] == ' ' || out[q] == '\t')) q++;
		size_t n = q;
		while (n < out.size() && (std::isdigit(static_cast<unsigned char>(out[n])) || out[n] == '.')) n++;
		if (n == q) continue;
		if (std::atof(out.substr(q, n - q).c_str()) >= 15.0) continue;
		out.replace(q, n - q, "15");
		st->home++;
	}
	if (st->Total() == 0) return false;
	t->swap(out);
	return true;
}

// Scripts de IA que el randomizer cambia de sitio dentro de su propio mundo: enemigos normales c2xxx / c3xxx
// (scripts 2xxxxx / 3xxxxx). Jefes (5xxxxx), NPC (8xxxxx) y el resto siguen en su sitio y conservan sus regiones.
inline bool MovedInOwnWorld(const std::string& lower_name) {
	size_t s = lower_name.find_last_of("\\/");
	s        = s == std::string::npos ? 0 : s + 1;
	return s < lower_name.size() && (lower_name[s] == '2' || lower_name[s] == '3');
}

} // namespace Rando::Lua
