#include "graphics/host_gpu/renderer/replay.h"

#include "common/perfStats.h"
#include "graphics/guest_gpu/pm4.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <unordered_map>
#include <vector>
#include <xxhash.h>

// gt7-replay-1119: KYTY_REPLAY (replay.h). The chunks' shapes and the draws' places, the
// counters and the line. What is kept at a place lives with the code that resolves it
// (descriptors.cpp: a stage's images and samplers; colorRenderTarget.cpp: a color target).
namespace Libs::Graphics::Replay {

int Mode() {
	static const int mode = [] {
		const char* value   = std::getenv("KYTY_REPLAY");
		const int   m       = value != nullptr ? std::atoi(value) : 0;
		const int   clamped = m < 0 || m > 2 ? 0 : m;
		Log::WriteUnmuted(fmt::format(
		    "Replay: {} (KYTY_REPLAY=1 reutiliza la traduccion de cada draw por su lugar en el "
		    "trozo, =2 la verifica contra el camino normal)\n",
		    clamped == 0 ? "apagado" : clamped == 1 ? "encendido" : "verificador"));
		return clamped;
	}();
	return mode;
}

namespace {

constexpr uint64_t Golden = 0x9E3779B97F4A7C15ull;

// Bounded maps of what was seen (cleared whole when full: only the counts depend on them).
constexpr size_t SeenLimit = 1u << 17u;

struct Counters {
	uint64_t chunks         = 0; // chained buffers started
	uint64_t shape_known    = 0; // structural shape seen before
	uint64_t values_known   = 0; // shape with every register value (addresses normalized) seen before
	uint64_t key_known      = 0; // shape chained with the chunk before it seen before
	uint64_t chunk_dwords   = 0;
	int64_t  shape_ticks    = 0; // time hashing the shapes
	uint64_t draws          = 0; // draws with a place
	uint64_t draws_whole    = 0; // every part of the draw from the replay (=2: would be)
	uint64_t draws_none     = 0; // draws with no part the replay keeps
	int64_t  ticks_whole    = 0;
	int64_t  ticks_other    = 0;
	std::array<uint64_t, static_cast<size_t>(Part::Count)> hits {};
	std::array<uint64_t, static_cast<size_t>(Part::Count)> asked {};
	std::array<uint64_t, static_cast<size_t>(Part::Count)> stored {};
	std::array<uint64_t, static_cast<size_t>(Part::Count)> dropped {};
	std::array<uint64_t, static_cast<size_t>(Part::Count)> verified {};
	std::array<uint64_t, static_cast<size_t>(Part::Count)> differences {};
	std::array<std::array<uint64_t, static_cast<size_t>(Miss::Count)>,
	           static_cast<size_t>(Part::Count)>
	    misses {};
};
Counters g_counters;

thread_local StreamState* t_stream = nullptr;
thread_local uint64_t     t_draw_key = 0;
thread_local uint32_t     t_parts    = 0;
thread_local uint32_t     t_missed   = 0;

int64_t Now() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// GPU thread: the shapes seen (structural, with values) and the chained keys seen.
struct Seen {
	std::unordered_map<uint64_t, uint32_t> shapes;
	std::unordered_map<uint64_t, uint32_t> values;
	std::unordered_map<uint64_t, uint32_t> keys;
};
Seen& GetSeen() {
	static Seen seen;
	return seen;
}
bool Insert(std::unordered_map<uint64_t, uint32_t>& map, uint64_t key) {
	if (map.size() >= SeenLimit) {
		map.clear();
	}
	return !map.try_emplace(key, 0u).second;
}

// A value pair of a SET_SH_REG run that is a 64-bit guest address (lo 4-byte aligned, hi the
// bits 32..39 of a 40-bit address): GT7's user data hold descriptor tables this way
// (FUN_000e62f0 stores FUN_0000acf0's pointer into two SGPRs), at a new address every frame.
// Shader program registers (PGM_LO = address >> 8, PGM_HI = address >> 40, which is 0) stay.
bool AddressPair(uint32_t lo, uint32_t hi) {
	return hi != 0 && hi <= 0xffu && (lo & 3u) == 0;
}

// The two hashes of a chunk: its structure (opcodes, lengths, register offsets: the place's
// key) and its values (registers set with their values, addresses normalized: statistics).
void ShapeHashes(const uint32_t* commands, uint32_t dwords, uint64_t& structure, uint64_t& values) {
	thread_local std::vector<uint32_t> shape;
	thread_local std::vector<uint32_t> valued;
	shape.clear();
	valued.clear();
	uint32_t at = 0;
	while (at < dwords) {
		const uint32_t header = commands[at];
		if (header == 0x80000000u || (header >> 30u) == 2u) {
			at++;
			continue;
		}
		if ((header >> 30u) == 0u) {
			const uint32_t len = ((header >> 16u) & 0x3fffu) + 2u;
			shape.push_back(header);
			at += len;
			continue;
		}
		const uint32_t len = KYTY_PM4_LEN(header);
		if (len < 2u || len > dwords - at) {
			shape.push_back(0xdeadbeefu); // a broken tail: hashed as such
			break;
		}
		const uint32_t  opcode = (header >> 8u) & 0xffu;
		const uint32_t* body   = commands + at + 1;
		const uint32_t  count  = len - 1u;
		shape.push_back(header);
		valued.push_back(header);
		switch (opcode) {
			case Pm4::IT_SET_CONTEXT_REG:
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX:
				shape.push_back(body[0]);
				valued.insert(valued.end(), body, body + count);
				break;
			case Pm4::IT_SET_SH_REG: {
				shape.push_back(body[0]);
				valued.push_back(body[0]);
				for (uint32_t i = 1; i < count; i++) {
					if (i + 1u < count && AddressPair(body[i], body[i + 1u])) {
						valued.push_back(0xa0a0a0a0u);
						valued.push_back(0xa0a0a0a0u);
						i++;
						continue;
					}
					valued.push_back(body[i]);
				}
				break;
			}
			default: break; // draws, indirect lists, chaining, events: their addresses change
		}
		at += len;
	}
	structure = XXH3_64bits(shape.data(), shape.size() * sizeof(uint32_t));
	values    = XXH3_64bits(valued.data(), valued.size() * sizeof(uint32_t));
}

} // namespace

void ChunkBegin(StreamState& stream, const uint32_t* commands, uint32_t dwords) {
	if (Mode() == 0) {
		return;
	}
	const int64_t begin     = Now();
	uint64_t      structure = 0;
	uint64_t      values    = 0;
	ShapeHashes(commands, dwords, structure, values);
	// The state a chunk inherits is that of the chunks before it: the key chains them.
	const uint64_t key =
	    XXH3_64bits_withSeed(&structure, sizeof(structure), stream.chain ^ Golden) | 1u;
	stream.chain       = key;
	stream.ordinal     = 0;
	stream.last_packet = nullptr;
	auto& seen         = GetSeen();
	auto& c            = g_counters;
	c.chunks++;
	c.chunk_dwords += dwords;
	c.shape_known += Insert(seen.shapes, structure) ? 1u : 0u;
	c.values_known += Insert(seen.values, values) ? 1u : 0u;
	c.key_known += Insert(seen.keys, key) ? 1u : 0u;
	c.shape_ticks += Now() - begin;
}

void NotePacket(StreamState& stream, const uint32_t* packet) {
	if (stream.last_packet != packet) {
		stream.last_packet = packet;
		stream.ordinal++;
	}
}

void StreamEnd(StreamState& stream) {
	stream = {};
}

StreamScope::StreamScope(StreamState* stream): m_previous(t_stream) {
	t_stream = stream;
}
StreamScope::~StreamScope() {
	t_stream = m_previous;
}

DrawScope::DrawScope()
    : m_previous_key(t_draw_key), m_previous_parts(t_parts), m_previous_missed(t_missed),
      m_begin(0), m_active(false) {
	t_draw_key = 0;
	t_parts    = 0;
	t_missed   = 0;
	if (Mode() == 0 || t_stream == nullptr) {
		return;
	}
	const std::array<uint64_t, 2> place {t_stream->chain, t_stream->ordinal};
	t_draw_key = XXH3_64bits(place.data(), sizeof(place)) | 1u;
	m_active   = true;
	m_begin    = Now();
}

DrawScope::~DrawScope() {
	if (m_active) {
		const int64_t ticks = Now() - m_begin;
		auto&         c     = g_counters;
		c.draws++;
		if (t_parts == 0) {
			c.draws_none++;
			c.ticks_other += ticks;
		} else if (t_missed == 0) {
			c.draws_whole++;
			c.ticks_whole += ticks;
		} else {
			c.ticks_other += ticks;
		}
	}
	t_draw_key = m_previous_key;
	t_parts    = m_previous_parts;
	t_missed   = m_previous_missed;
}

uint64_t DrawKey() {
	return t_draw_key;
}

uint64_t PartKey(uint64_t draw_key, uint32_t kind, uint32_t index) {
	return (draw_key ^ ((static_cast<uint64_t>(kind) << 8u | index) + 1u) * Golden) | 1u;
}

void NoteHit(Part part) {
	g_counters.asked[static_cast<size_t>(part)]++;
	g_counters.hits[static_cast<size_t>(part)]++;
	t_parts++;
}

void NoteMiss(Part part, Miss why) {
	g_counters.asked[static_cast<size_t>(part)]++;
	g_counters.misses[static_cast<size_t>(part)][static_cast<size_t>(why)]++;
	t_parts++;
	t_missed++;
}

void NoteVerified(Part part, const char* what, uint64_t shader_hash, uint32_t index) {
	g_counters.verified[static_cast<size_t>(part)]++;
	if (what == nullptr) {
		return;
	}
	g_counters.differences[static_cast<size_t>(part)]++;
	static uint32_t logged = 0;
	if (logged++ < 16) {
		static constexpr std::array names {"imagen", "samplers", "target"};
		Log::WriteUnmuted(fmt::format(
		    "Replay verificador: diferencia en {} ({}) shader {:016x} indice {}\n",
		    names[static_cast<size_t>(part)], what, shader_hash, index));
	}
}

void NoteStored(Part part) {
	g_counters.stored[static_cast<size_t>(part)]++;
}

void NoteDropped(Part part) {
	g_counters.dropped[static_cast<size_t>(part)]++;
}

void Report(double seconds) {
	if (Mode() == 0) {
		return;
	}
	auto&      c       = g_counters;
	const auto percent = [](uint64_t n, uint64_t of) {
		return of != 0 ? 100.0 * static_cast<double>(n) / static_cast<double>(of) : 0.0;
	};
	const auto us_each = [](int64_t ticks, uint64_t n) {
		return n != 0 ? static_cast<double>(ticks) / 1000.0 / static_cast<double>(n) : 0.0;
	};
	static constexpr std::array part_names {"imagenes", "samplers", "targets"};
	static constexpr std::array miss_names {"no apta",          "sin entrada",
	                                        "otro programa",    "otras palabras/registros",
	                                        "no guardable",     "memo de textura cambiado",
	                                        "imagenes creadas o borradas", "imagen retirada"};
	const bool  verifier = Mode() == 2;
	std::string line     = fmt::format(
        "Replay ({:.1f} s, KYTY_REPLAY={}): trozos {} ({:.0f} dwords cada uno, firma {:.2f} us "
            "cada uno): forma conocida {:.1f}%, valores iguales {:.1f}%, forma y entrada "
            "conocidas {:.1f}% | draws {} con lugar, {} {} ({:.1f}%), {} sin partes; us/draw {} "
            "{:.2f} frente al resto {:.2f}",
        seconds, Mode(), c.chunks,
        c.chunks != 0 ? static_cast<double>(c.chunk_dwords) / static_cast<double>(c.chunks) : 0.0,
        us_each(c.shape_ticks, c.chunks), percent(c.shape_known, c.chunks),
        percent(c.values_known, c.chunks), percent(c.key_known, c.chunks), c.draws,
        verifier ? "reutilizables enteros" : "reutilizados enteros", c.draws_whole,
        percent(c.draws_whole, c.draws), c.draws_none,
        verifier ? "(camino normal) de los reutilizables" : "reutilizados",
        us_each(c.ticks_whole, c.draws_whole), us_each(c.ticks_other, c.draws - c.draws_whole));
	for (size_t p = 0; p < part_names.size(); p++) {
		line += fmt::format(" | {} {}/{} ({:.1f}%), guardadas {}", part_names[p], c.hits[p],
		                    c.asked[p], percent(c.hits[p], c.asked[p]), c.stored[p]);
		if (c.dropped[p] != 0) {
			line += fmt::format(", vaciadas {}", c.dropped[p]);
		}
		std::string misses;
		for (size_t m = 0; m < miss_names.size(); m++) {
			if (c.misses[p][m] != 0) {
				misses += fmt::format("{}{} {}", misses.empty() ? "" : ", ", miss_names[m],
				                      c.misses[p][m]);
			}
		}
		if (!misses.empty()) {
			line += "; fallos: " + misses;
		}
		if (verifier) {
			line += fmt::format("; comparadas {}, diferencias {}", c.verified[p], c.differences[p]);
		}
	}
	Log::WriteUnmuted(line + "\n");
	c = {};
}

} // namespace Libs::Graphics::Replay
