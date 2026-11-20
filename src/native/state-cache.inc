// (g1) KYTY_G1_PROBE / KYTY_G1_STATE (native/g1-probe.h): the GPU side of the *_REG_INDIRECT register blocks.
// Included once, by graphics/guest_gpu/command_processor/pm4Handlers.cpp, after Pm4Regfile (it uses its proven
// descriptors: Raw = the handler is a plain store into the context / shader / user-config object, Ign = it changes
// nothing; proven at the first use by Pm4Regfile::Build against the command processor's bytes) and inside
// namespace Libs::Graphics.
//
// Each CpOpIndirect*Regs handler holds one G1State::Block:
//   Block g1(space, cp, address, pairs);     before SyncCommandMemory (the probe's "sync" stretch starts)
//   if (g1.Synced(pairs_ptr)) return ...;    after it: the probe classifies the pairs; STATE (ON window) applies the
//                                            block here and the handler returns; STATE=2 applies it on the registers
//                                            and puts them back, then the normal loop runs and the destructor compares
//   (destructor)                             the probe's times: sync, apply, total
//
// STATE: a block's decisions are kept by its content (XXH3 of its pairs + the pairs compared word by word: identical
// bytes give identical decisions, wherever the block lies in the command buffer - the address memo of otros2 found
// 12 hits for 1,435 misses a frame because the binder's block moves every frame). Applying a kept block: a proven raw
// register is stored only when the stored value differs (the true delta), a proven-ignored or fake register is not
// called, every other register's handler is called with its value, in the block's order, as the normal loop does.
namespace G1State {

using IndirectFunc = void (*)(CommandProcessor&, uint32_t, uint32_t);

[[nodiscard]] inline IndirectFunc FuncOf(uint32_t space, uint32_t at) {
	return space == 0 ? g_hw_ctx_indirect_func[at & (Pm4::CX_NUM - 1)]
	       : space == 1 ? g_hw_sh_indirect_func[at]
	                    : g_hw_uc_indirect_func[at];
}

// What the normal loop does with one pair: 0 skip (sentinel, NOP, fake, out of range, no handler), 1 raw store,
// 2 ignored (proven), 3 call. `report`: the normal loop reports this pair (no handler).
enum PairKind : uint8_t { PSkip = 0, PFake = 1, PRaw = 2, PIgn = 3, PCall = 4 };
struct Pair {
	PairKind              kind;
	uint32_t              at;
	const Pm4Regfile::Desc* desc;
	bool                  report;
};
[[nodiscard]] inline Pair Decide(uint32_t space, uint32_t raw_offset) {
	const uint32_t off = NormalizeRegisterOffset(raw_offset);
	const Pm4Regfile::Desc* d = nullptr;
	if (space == 0) {
		if (HwCtxTrySetFakeRegister(off, 0)) return {PFake, off, nullptr, false}; // (a no-op for a fake register)
		if (raw_offset == 0xffffffffu || off >= Pm4::CX_NUM) return {PSkip, off, nullptr, false};
		if (g_hw_ctx_indirect_func[off & (Pm4::CX_NUM - 1)] == nullptr) return {PSkip, off, nullptr, true};
		d = &Pm4Regfile::g_cx[off & (Pm4::CX_NUM - 1)];
	} else if (space == 1) {
		if (off == Pm4::SH_NOP || raw_offset == 0xffffffffu) return {PSkip, off, nullptr, false};
		if (off >= Pm4::SH_NUM || g_hw_sh_indirect_func[off] == nullptr) return {PSkip, off, nullptr, true};
		d = &Pm4Regfile::g_sh[off];
	} else {
		if (off == Pm4::UC_NOP) return {PSkip, off, nullptr, false};
		if (off >= Pm4::UC_NUM || g_hw_uc_indirect_func[off] == nullptr) return {PSkip, off, nullptr, true};
		d = &Pm4Regfile::g_uc[off];
	}
	if (d->kind == Pm4Regfile::Raw) return {PRaw, off, d, false};
	if (d->kind == Pm4Regfile::Ign) return {PIgn, off, d, false};
	return {PCall, off, d, false};
}

// ---- probe state (GPU thread only) ----
struct ProbeState {
	std::vector<uint64_t> shadow[3];  // (generation << 32) | value written by an indirect block
	uint64_t              generation = 1;
	uint64_t              observer_generation = 0;
	std::vector<uint64_t> seen;       // content hash -> (hash, frame) pairs, open addressing
	uint64_t              last_hash[3] {};
};
inline ProbeState& Probe() {
	static ProbeState p = [] {
		ProbeState s;
		s.shadow[0].assign(Pm4::CX_NUM, 0);
		s.shadow[1].assign(Pm4::SH_NUM, 0);
		s.shadow[2].assign(Pm4::UC_NUM, 0);
		s.seen.assign(2u * 65536u, 0);
		return s;
	}();
	return p;
}

// ---- STATE cache (GPU thread only) ----
struct Op {
	uint8_t  kind; // 0 raw store, 1 call
	uint8_t  object;
	uint16_t pad;
	uint32_t at;
	uint32_t value;
};
struct Entry {
	uint64_t              hash  = 0;
	uint32_t              space = ~0u;
	uint32_t              pairs = 0;
	uint32_t              not_called = 0; // pairs neither stored nor called (ignored, fake, skipped)
	std::vector<uint32_t> words;
	std::vector<Op>       ops;
};
constexpr uint32_t CacheEntries = 4096;
inline std::vector<Entry>& Cache() {
	static std::vector<Entry> c(CacheEntries);
	return c;
}

inline void EnsureDescriptors(CommandProcessor& cp) {
	if (!Pm4Regfile::g_ready) Pm4Regfile::Build(cp, kyty_local_pm4_regfile.load(std::memory_order_relaxed));
}

class Block {
public:
	Block(uint32_t space, CommandProcessor& cp, uint32_t pairs): m_cp(cp), m_space(space), m_pairs(pairs) {
		if (!G1::On()) return;
		if (!G1::t_gpu) {
			if (G1::ProbeMode() != 0) G1::g_other_blocks.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		m_on = true;
		m_t0 = __rdtsc();
	}
	Block(const Block&)            = delete;
	Block& operator=(const Block&) = delete;

	// After SyncCommandMemory returned `applies` (false: the handler returns without applying). True: the block was
	// applied here (STATE ON).
	[[nodiscard]] bool Synced(const uint32_t* pairs, bool applies) {
		if (!m_on) return false;
		m_t1 = __rdtsc();
		if (!applies) return false;
		if (G1::ProbeMode() != 0) {
			EnsureDescriptors(m_cp);
			Classify(pairs);
			m_probe = __rdtsc() - m_t1;
		}
		if (!G1::StateActive() || kyty_local_pm4_regfile.load(std::memory_order_relaxed) == 2 || Spec::Current() != nullptr) return false;
		EnsureDescriptors(m_cp);
		if (G1::StateMode() == 2) {
			// Both from the same registers: this path's result kept aside, the registers put back for the normal loop.
			for (int o = 0; o < 3; ++o) {
				Before(o).assign(Object(o), Object(o) + Size(o));
			}
			Apply(pairs);
			for (int o = 0; o < 3; ++o) {
				After(o).assign(Object(o), Object(o) + Size(o));
				std::memcpy(Object(o), Before(o).data(), Size(o));
			}
			m_verify = true;
			return false;
		}
		Apply(pairs);
		return true;
	}

	~Block() {
		if (!m_on) return;
		const uint64_t end = __rdtsc();
		auto&          b   = G1::g_blocks;
		if (m_t1 == 0) m_t1 = end;
		b.blocks[m_space] += 1;
		b.pairs[m_space] += m_pairs;
		b.cyc_sync[m_space] += m_t1 - m_t0;
		b.cyc_apply[m_space] += end - m_t1 - m_probe;
		b.cyc_total[m_space] += end - m_t0 - m_probe;
		b.probe_cycles += m_probe;
		if (m_verify) {
			++b.st_verified;
			for (int o = 0; o < 3; ++o) {
				if (std::memcmp(Object(o), After(o).data(), Size(o)) == 0) continue;
				if (b.st_discrepancias++ < 8)
					std::printf("[g1] DISCREPANCIA STATE espacio %u pares %u objeto %d (%s)\n", m_space, m_pairs, o,
					            o == 0 ? "contexto" : o == 1 ? "shader" : "user config");
			}
		}
	}

private:
	[[nodiscard]] uint8_t* Object(int o) const {
		return o == 0 ? reinterpret_cast<uint8_t*>(&m_cp.GetCtx())
		       : o == 1 ? reinterpret_cast<uint8_t*>(&m_cp.GetShCtx())
		                : reinterpret_cast<uint8_t*>(&m_cp.GetUcfg());
	}
	[[nodiscard]] static size_t Size(int o) {
		return o == 0 ? sizeof(HW::Context) : o == 1 ? sizeof(HW::Shader) : sizeof(HW::UserConfig);
	}
	static std::vector<uint8_t>& Before(int o) {
		static std::vector<uint8_t> v[3];
		return v[o];
	}
	static std::vector<uint8_t>& After(int o) {
		static std::vector<uint8_t> v[3];
		return v[o];
	}

	void Classify(const uint32_t* pairs) {
		auto& p = Probe();
		auto& b = G1::g_blocks;
		if (DrawStateObserver::g_shadow.generation != p.observer_generation) { // (the registers were reset: nothing known)
			p.observer_generation = DrawStateObserver::g_shadow.generation;
			++p.generation;
		}
		const uint64_t gen   = p.generation << 32u;
		auto&          shadow = p.shadow[m_space];
		for (uint32_t i = 0; i < m_pairs; ++i) {
			const uint32_t value = pairs[2 * i + 1];
			const Pair     d     = Decide(m_space, pairs[2 * i]);
			switch (d.kind) {
				case PSkip: ++b.skipped[m_space]; break;
				case PFake: ++b.fake[m_space]; break;
				case PIgn: ++b.ign[m_space]; break;
				case PRaw:
					++b.raw[m_space];
					if (std::memcmp(Object(d.desc->object) + d.desc->byte_off, &value, 4) != 0) ++b.raw_diff[m_space];
					break;
				case PCall:
					++b.call[m_space];
					if (shadow[d.at] != (gen | value)) ++b.call_diff[m_space];
					break;
			}
			if (d.kind == PRaw || d.kind == PCall) shadow[d.at] = gen | value;
		}
		const uint64_t h     = XXH3_64bits(pairs, size_t {m_pairs} * 8u) ^ (uint64_t {m_space} << 62u) ^ m_pairs;
		const uint64_t frame = G1::g_frame.load(std::memory_order_relaxed);
		const size_t   slot  = static_cast<size_t>(h & 65535u) * 2u;
		if (p.seen[slot] == h && p.seen[slot + 1] + 1u >= frame) {
			++b.repeat_blocks[m_space];
			b.repeat_pairs[m_space] += m_pairs;
		}
		p.seen[slot]     = h;
		p.seen[slot + 1] = frame;
		if (p.last_hash[m_space] == h) ++b.same_as_previous[m_space];
		p.last_hash[m_space] = h;
	}

	void Apply(const uint32_t* pairs) {
		auto&          b     = G1::g_blocks;
		const size_t   bytes = size_t {m_pairs} * 8u;
		const uint64_t h     = XXH3_64bits(pairs, bytes) ^ (uint64_t {m_space} << 62u) ^ m_pairs;
		Entry&         e     = Cache()[h % CacheEntries];
		if (e.hash == h && e.space == m_space && e.pairs == m_pairs && std::memcmp(e.words.data(), pairs, bytes) == 0) {
			++b.st_hit[m_space];
		} else {
			++b.st_miss[m_space];
			e.hash       = h;
			e.space      = m_space;
			e.pairs      = m_pairs;
			e.not_called = 0;
			e.words.assign(pairs, pairs + 2u * m_pairs);
			e.ops.clear();
			for (uint32_t i = 0; i < m_pairs; ++i) {
				const uint32_t value = pairs[2 * i + 1];
				const Pair     d     = Decide(m_space, pairs[2 * i]);
				if (d.report) // (as the normal loop: reported, skipped)
					ReportIndirectRegister(m_space == 0 ? "cx" : m_space == 1 ? "sh" : "uc", reinterpret_cast<uint64_t>(pairs), m_pairs, i,
					                       pairs[2 * i], value);
				if (d.kind == PRaw) e.ops.push_back({0, d.desc->object, 0, d.desc->byte_off, value});
				else if (d.kind == PCall) e.ops.push_back({1, 0, 0, d.at, value});
				else ++e.not_called;
			}
		}
		uint8_t* const base[3] = {Object(0), Object(1), Object(2)};
		uint64_t       stores = 0, kept = 0, calls = 0;
		for (const Op& op: e.ops) {
			if (op.kind == 0) {
				uint8_t* const at = base[op.object] + op.at;
				if (std::memcmp(at, &op.value, 4) != 0) {
					std::memcpy(at, &op.value, 4);
					++stores;
				} else {
					++kept;
				}
			} else {
				FuncOf(m_space, op.at)(m_cp, op.at, op.value);
				++calls;
			}
		}
		b.st_stores[m_space] += stores;
		b.st_kept[m_space] += kept;
		b.st_calls[m_space] += calls;
		b.st_not_called[m_space] += e.not_called;
	}

	CommandProcessor& m_cp;
	uint32_t          m_space;
	uint32_t          m_pairs;
	bool              m_on     = false;
	bool              m_verify = false;
	uint64_t          m_t0 = 0, m_t1 = 0, m_probe = 0;
};

} // namespace G1State
