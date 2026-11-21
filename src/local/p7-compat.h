#pragma once
// (p7) KYTY_P7: fixes for the flicker AMD (RDNA4: RX 9070) and Intel Arc users see and NVIDIA users do not
// (native/grafico/INFORME.md section 6b, native/parpadeo7-amd-intel/INFORME.md). Not KYTY_PARPADEO7: that name is the
// XPR prefetch of local/parpadeo7.h, which stays as it was.
//
// KYTY_P7 = auto (default, also when unset) / 0 (every fix off: the build before this one) / 1 (every fix on, any GPU).
// KYTY_P7_C1 .. KYTY_P7_C7 = 0 / 1 decide one cause each, over KYTY_P7 (C1 also 2 and 3, C6 also 2). In auto a fix is
// on where its cause can happen: not on NVIDIA GPUs, except the true correctness fix C3, which only acts where it
// changes something.
//   C1  compute chains (p7b: precise). 1 = a barrier inside a chain of dispatches only where a dispatch touches what an
//       earlier dispatch of the same chain wrote, or writes what one read (RAW, WAW, WAR on the buffer ranges of their
//       V#s, the indirect arguments as a read; a program with flat/global reads reads anything): compute -> compute |
//       draw-indirect, once, which orders the whole chain so far (ChainStep). 2 = no chains at all
//       (KYTY_COMPUTE_CHAIN=0, "gpu_seguro": ~25-35% less fps measured on NVIDIA; only by hand). 3 = the p7 blanket
//       version: the indirect arguments barrier on every chained indirect dispatch (measured on an RTX 5080: ~500 a
//       frame, with C2 +4.1 +- 1.4 ms a frame).
//       The hazards are counted in any mode but 0 (NVIDIA too: they are API hazards whatever the vendor), fixed or not.
//   C2  barrier dedupe, precise (p7b). The dedupes stay on (the p7 version turned them off: measured with C1 above);
//       an image transition is dropped only where it is proven redundant: same layout and access, the destination
//       stages covered by the last barrier's, and - for the repeat of a write inside one transit group - no command
//       that writes recorded on this thread since that image's last transition, and a whole-image state (not a
//       speculation's copy). What the old rule would have dropped and this one keeps is counted in any mode.
//   C3  compute subgroup size: where the GPU cannot require 64 lanes but its subgroups can be smaller than 32 (Intel:
//       8/16/32), every translated compute program requires 32 (what its wave64 emulation is written for). A
//       correctness fix (on in auto); nothing changes on a GPU whose compute subgroups are always 32 (NVIDIA) or that
//       requires them already (AMD).
//   C4  pixel shader subgroup size: AMD (32-64 lanes): 64 required (what the translation of a wave64 pixel shader
//       assumes: lane 63 exists, ballots are 64 bits; the issue #24 mosaics); Intel (8-32): 32 required. Only where the
//       GPU lets a pipeline require a fragment subgroup size and its sizes vary; elsewhere a warning in the log.
//   C5  labels (EOP/RELEASE_MEM data writes) written when the GPU has run the work before them, not when they are
//       translated. Off in auto: it adds the GPU's latency to every wait of the game on a label (cost unmeasured).
//   C6  guest readbacks on the graphics queue, not the transfer queue (KYTY_READBACK_QUEUE=0, live). The buffers are
//       already shared by both families (VK_SHARING_MODE_CONCURRENT: no ownership transfer is missing); what this
//       removes is the copy engine's ordering only against the graphics timeline. p7b: off in auto everywhere (no
//       readback hazard found in the code; measured with C7 +3.3 +- 1.3 ms a frame on an RTX 5080). 2 = the queue's own
//       check (KYTY_READBACK_QUEUE=3): every copy compared with a graphics-queue copy, READBACK_QUEUE_MISMATCH lines.
//   C7  new images (fresh or from the pool) with attachment or storage use cleared to 0 before their first use that is
//       not a transfer write: no read of undefined contents (AMD's DCC metadata).
// KYTY_P7_ALTERNA=1: the live fixes (C1, C2, C5, C6, C7) in windows ON OFF OFF ON ..., in the same process; the "[p7]"
// line gives the frame and submission times of each state and the cost per ABBA block with its error (p7-report.h).
// KYTY_P7_VENTANA_MS = the window (default 5000), KYTY_P7_DESDE_S = windows counted from this second on (default 0).
// C3 and C4 are fixed when pipelines are made.
//
// Diagnostics (always, one relaxed load when nothing to count): "[p7] GPU ..." lines at start (vendor, driver,
// subgroup properties, the decision per cause and why); "[p7]" every 5 s while a fix is on or alternating (else every
// 60 s); with KYTY_VULKAN_SYNCVAL=1 a "[p7-syncval]" summary every 30 s and at exit: hazards per message id, per
// probable cause and per command pair.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern "C" {
extern volatile std::atomic<uint32_t> kyty_local_compute_chain_mode;
extern volatile std::atomic<uint32_t> kyty_local_image_barrier_dedupe;
extern volatile std::atomic<uint32_t> kyty_local_global_barrier_dedupe;
extern volatile std::atomic<uint32_t> kyty_local_readback_queue_mode;
}

namespace P7 {

enum Vendor : int { VendorUnknown = 0, VendorNvidia = 1, VendorAmd = 2, VendorIntel = 3, VendorOther = 4 };

inline const char* VendorName(int vendor) noexcept {
	switch (vendor) {
		case VendorNvidia: return "NVIDIA";
		case VendorAmd: return "AMD";
		case VendorIntel: return "Intel";
		case VendorOther: return "otro";
		default: return "?";
	}
}

inline int VendorOf(uint32_t vendor_id) noexcept {
	switch (vendor_id) {
		case 0x10deu: return VendorNvidia;
		case 0x1002u: return VendorAmd;
		case 0x8086u: return VendorIntel;
		default: return VendorOther;
	}
}

constexpr int Causes = 8; // index 1..7

// What one GPU offers for the subgroup decisions (filled by VulkanInitSubgroupSizeControl).
struct SubgroupCaps {
	uint32_t vendor_id = 0, device_id = 0, api_version = 0, driver_version = 0;
	uint32_t default_size = 0, size_low = 0, size_high = 0; // subgroupSize, min/maxSubgroupSize
	uint32_t supported_stages = 0, supported_operations = 0, required_stages = 0;
	bool     size_control = false, full_subgroups = false, quad_all_stages = false;
	char     name[256] {}, driver_name[256] {}, driver_info[256] {};
};

struct State {
	bool         decided   = false;
	int          vendor    = VendorUnknown;
	int          mode      = -1; // -1 auto, 0 off, 1 all
	bool         alternate = false;
	bool         count_always = false; // KYTY_P7_CONTAR=1
	int          level[Causes] {};
	bool         by_hand[Causes] {};
	SubgroupCaps caps;
	// Pipeline creation (C3, C4): decided with the device, before any pipeline.
	uint32_t compute_required = 0;  // C3: the size every translated compute program requires (0: as before)
	uint32_t fragment_required = 0; // C4: the size every pixel shader requires (0: as before)
	bool     needs_size_control = false; // the device must enable subgroupSizeControl for C3/C4
	// Live values before the fixes (the GPU thread's switches applied), for the OFF windows.
	uint32_t base_chain = 1, base_image_dedupe = 0, base_global_dedupe = 0, base_readback = 0;
	bool     live_captured = false;
};

inline State& S() noexcept {
	static State state;
	return state;
}

// Live flags read on the hot paths (one relaxed load each).
inline std::atomic<uint32_t> g_args_barrier_in_chain {0}; // C1 == 3 (the p7 blanket version)
inline std::atomic<uint32_t> g_chain_precise {0};         // C1 == 1: a barrier where ChainStep finds a hazard
inline std::atomic<uint32_t> g_precise_barriers {0};      // C2: the image dedupe's redundancy proven
inline std::atomic<uint32_t> g_track {0};                 // C1/C2 hazard tracking and counts (see ApplyLive)
inline std::atomic<uint32_t> g_track_epoch {0};           // moves at every ApplyLive: a chain tracked across it is not proven
inline std::atomic<uint32_t> g_labels_at_completion {0};  // C5
inline std::atomic<uint32_t> g_clear_new_images {0};      // C7
inline std::atomic<bool>     g_window_on {true};          // KYTY_P7_ALTERNA: the window running now

[[nodiscard]] inline bool ArgsBarrierInChain() noexcept { return g_args_barrier_in_chain.load(std::memory_order_relaxed) != 0; }
[[nodiscard]] inline bool ChainPrecise() noexcept { return g_chain_precise.load(std::memory_order_relaxed) != 0; }
[[nodiscard]] inline bool PreciseBarriers() noexcept { return g_precise_barriers.load(std::memory_order_relaxed) != 0; }
[[nodiscard]] inline bool Tracking() noexcept { return g_track.load(std::memory_order_relaxed) != 0; }
[[nodiscard]] inline bool LabelsAtCompletion() noexcept { return g_labels_at_completion.load(std::memory_order_relaxed) != 0; }
[[nodiscard]] inline bool ClearNewImages() noexcept { return g_clear_new_images.load(std::memory_order_relaxed) != 0; }

// Counters of the "[p7]" line.
enum Counter : int {
	ArgsBarriers = 0,  // C1: indirect argument barriers recorded inside a chain
	LabelsDeferred,    // C5: labels written at GPU completion
	LabelsOrdered,     // C5: immediate writes that waited for a deferred label to the same address first
	LabelsImmediate,   // C5 on: labels written at once anyway (interrupt, speculation, another thread)
	ImagesCleared,     // C7: new images cleared before their first use
	UndefinedReads,    // any mode: first use of a new image that is not a transfer write (a read of undefined contents)
	UndefinedSkipped,  // C7 on: such first uses not cleared (format or use cannot be cleared)
	// C1 (p7b): chained dispatches (after the first of a chain) and their hazards against the chain's earlier ones.
	ChainChecked,      // dispatches checked (chained programs; any mode but 0)
	ChainHazards,      // dispatches with a hazard = the barriers precise C1 records (or would record)
	ChainBarriers,     // C1 = 1: barriers recorded
	HazArgs,           // its indirect arguments written earlier in the chain
	HazRaw,            // it reads a range written earlier in the chain
	HazWaw,            // it writes a range written earlier in the chain
	HazWar,            // it writes a range read earlier in the chain
	HazUnknown,        // flat/global reads (any address) after a write in the chain
	HazAtomic,         // every overlap found is between buffers both programs use atomically
	HazConservative,   // the chain's ranges overflowed or its state was lost: a barrier without a proven overlap
	// C2 (p7b): image barriers.
	DedupeDrops,       // repeated writes in one transit group dropped (the old rule, proven or not)
	DedupeUnproven,    // of those, not proven redundant (a write recorded between, a part of the image, a speculation)
	StageUncovered,    // same layout and access but the destination stages not covered by the last barrier's
	PreciseAdded,      // C2 on: image barriers recorded for the two above
	GlobalDedupeDrops, // full barriers dropped by KYTY_GLOBAL_BARRIER_DEDUPE (nothing recorded since the last)
	CounterCount
};
inline std::atomic<uint64_t> g_counter[CounterCount] {};
inline void Add(int counter, uint64_t value = 1) noexcept { g_counter[counter].fetch_add(value, std::memory_order_relaxed); }

[[nodiscard]] inline int Level(int cause) noexcept {
	return cause > 0 && cause < Causes && S().decided ? S().level[cause] : 0;
}

inline int ParseLevel(const char* text, int highest) noexcept {
	if (text == nullptr || text[0] < '0' || text[0] > '9' || text[1] != '\0') return -1;
	const int value = text[0] - '0';
	return value <= highest ? value : -1;
}

// The decision per cause, once (the first device: VulkanInitSubgroupSizeControl). Prints the "[p7] GPU" lines.
inline void Decide(const SubgroupCaps& caps) {
	auto& s = S();
	if (s.decided) return;
	s.caps   = caps;
	s.vendor = VendorOf(caps.vendor_id);
	const char* mode_text = std::getenv("KYTY_P7");
	s.mode = mode_text == nullptr || mode_text[0] == '\0' || std::strcmp(mode_text, "auto") == 0 ? -1
	         : std::strcmp(mode_text, "0") == 0                                                  ? 0
	         : std::strcmp(mode_text, "1") == 0                                                  ? 1
	                                                                                             : -2;
	if (s.mode == -2) {
		std::printf("[p7] KYTY_P7=%s no vale (auto, 0 o 1): auto\n", mode_text);
		s.mode = -1;
	}
	const char* alternate = std::getenv("KYTY_P7_ALTERNA");
	s.alternate           = alternate != nullptr && alternate[0] == '1';
	const char* count_env = std::getenv("KYTY_P7_CONTAR");
	s.count_always        = count_env != nullptr && count_env[0] == '1';
	const bool not_nvidia = s.vendor != VendorNvidia;
	for (int c = 1; c < Causes; ++c) {
		int level = 0;
		if (s.mode == 1) level = 1;
		else if (s.mode == -1) {
			switch (c) {
				case 3: level = 1; break;          // correctness: acts only where the size can vary below 32
				case 5: level = 0; break;          // latency: +52 ms a frame measured (RTX 5080), by hand or KYTY_P7=1
				case 6: level = 0; break;          // no readback hazard found; its cost measured with C7: by hand
				default: level = not_nvidia ? 1 : 0;
			}
		}
		char name[16];
		std::snprintf(name, sizeof(name), "KYTY_P7_C%d", c);
		const char* text = std::getenv(name);
		if (text != nullptr && text[0] != '\0') {
			const int parsed = ParseLevel(text, c == 1 ? 3 : c == 6 ? 2 : 1);
			if (parsed < 0)
				std::printf("[p7] %s=%s no vale (0%s): se ignora\n", name, text, c == 1 ? ", 1, 2 o 3" : c == 6 ? ", 1 o 2" : " o 1");
			else {
				level        = parsed;
				s.by_hand[c] = true;
			}
		}
		s.level[c] = level;
	}

	// C3: compute. Only where a pipeline cannot require 64 (else the size is required already: AMD), can require 32
	// and the GPU may pick fewer lanes than 32 (Intel); nothing on a GPU whose compute subgroups are always 32.
	const bool compute_requirable = caps.size_control && (caps.required_stages & 0x20u) != 0; // VK_SHADER_STAGE_COMPUTE_BIT
	const bool compute_64         = compute_requirable && caps.size_low <= 64u && caps.size_high >= 64u;
	const bool compute_32         = compute_requirable && caps.size_low <= 32u && caps.size_high >= 32u;
	const bool compute_varies     = caps.size_low != 0 && caps.size_low < 32u;
	std::string c3_why;
	if (s.level[3] != 0) {
		if (compute_64) c3_why = "no aplica: la GPU ya exige el tamano (64 posible)";
		else if (!compute_varies) c3_why = "no aplica: subgrupos de compute siempre >= 32";
		else if (!compute_32) c3_why = "AVISO: la GPU no deja exigir 32 en compute; el wave64 emulado puede fallar (no hay otra via)";
		else {
			s.compute_required   = 32;
			s.needs_size_control = true;
			c3_why               = "exige 32 en todos los compute traducidos";
		}
	} else if (compute_varies && !compute_64) {
		c3_why = "APAGADO: el compute wave64 emulado corre en subgrupos que el driver elige (puede ser < 32)";
	}

	// C4: pixel shaders. AMD: 64 (the translation's wave64); Intel: 32 (its largest, what NVIDIA runs). Only where a
	// pipeline can require a fragment size and the sizes vary.
	const bool fragment_requirable = caps.size_control && (caps.required_stages & 0x10u) != 0; // VK_SHADER_STAGE_FRAGMENT_BIT
	const bool sizes_vary          = caps.size_low != caps.size_high;
	std::string c4_why;
	if (s.level[4] != 0) {
		const uint32_t wanted = caps.size_low <= 64u && caps.size_high >= 64u ? 64u : caps.size_high >= 32u && caps.size_low <= 32u ? 32u : 0u;
		if (!sizes_vary) c4_why = "no aplica: un solo tamano de subgrupo";
		else if (!fragment_requirable)
			c4_why = "AVISO: la GPU no deja exigir tamano en pixel shaders; los wave64 con operaciones de carril corren al "
			         "tamano del driver (PS_WAVE_OR reduce sin depender del tamano)";
		else if (wanted == 0) c4_why = "AVISO: ni 32 ni 64 posibles";
		else {
			s.fragment_required  = wanted;
			s.needs_size_control = true;
			c4_why               = wanted == 64 ? "exige 64 en pixel shaders" : "exige 32 en pixel shaders";
		}
	}

	std::printf("[p7] GPU: %s (vendor 0x%04x, device 0x%04x) \"%s\", driver \"%s\" \"%s\" (0x%08x), Vulkan %u.%u.%u\n",
	            VendorName(s.vendor), caps.vendor_id, caps.device_id, caps.name, caps.driver_name, caps.driver_info,
	            caps.driver_version, (caps.api_version >> 22u) & 0x7fu, (caps.api_version >> 12u) & 0x3ffu,
	            caps.api_version & 0xfffu);
	std::printf("[p7] GPU subgrupos: por defecto %u, rango %u-%u, etapas con subgrupos 0x%x, operaciones 0x%x, etapas que "
	            "pueden exigir tamano 0x%x, subgroupSizeControl=%s, computeFullSubgroups=%s\n",
	            caps.default_size, caps.size_low, caps.size_high, caps.supported_stages, caps.supported_operations,
	            caps.required_stages, caps.size_control ? "si" : "no", caps.full_subgroups ? "si" : "no");
	static const char* const what[Causes] = {"",
	                                         "cadenas de dispatches (barrera solo con hazard)",
	                                         "dedupe de barreras de imagen (solo si es redundante)",
	                                         "subgrupo de compute",
	                                         "subgrupo de pixel shaders",
	                                         "etiquetas al completar",
	                                         "readback en la cola grafica",
	                                         "imagenes nuevas limpiadas"};
	std::printf("[p7] modo %s%s:", s.mode == -1 ? "auto" : s.mode == 0 ? "0 (todo apagado)" : "1 (todo forzado)",
	            s.alternate ? ", ALTERNA 5 s ON / 5 s OFF" : "");
	for (int c = 1; c < Causes; ++c) std::printf(" C%d=%d%s", c, s.level[c], s.by_hand[c] ? "(a mano)" : "");
	std::printf("\n");
	for (int c = 1; c < Causes; ++c) {
		std::printf("[p7]   C%d %s: %s", c, what[c], s.level[c] != 0 ? "ON" : "off");
		if (c == 1 && s.level[1] == 2) std::printf(" (nivel 2: sin cadenas, cuesta ~25-35%% de fps)");
		if (c == 1 && s.level[1] == 3) std::printf(" (nivel 3: la version p7, barrera de args en todo dispatch indirecto encadenado)");
		if (c == 6 && s.level[6] == 2) std::printf(" (nivel 2: KYTY_READBACK_QUEUE=3, cada copia comparada con la cola grafica)");
		if (c == 3 && !c3_why.empty()) std::printf(" - %s", c3_why.c_str());
		if (c == 4 && !c4_why.empty()) std::printf(" - %s", c4_why.c_str());
		if (c == 5 && s.level[5] == 0 && s.mode == -1 && !s.by_hand[5]) std::printf(" (auto: cuesta ~52 ms/frame medido; KYTY_P7_C5=1)");
		if (c == 6 && s.level[6] == 0 && s.mode == -1 && !s.by_hand[6]) std::printf(" (auto: solo a mano; KYTY_P7_C6=1 o 2)");
		if (s.level[c] == 0 && s.mode == -1 && !s.by_hand[c] && c != 3 && c != 5 && c != 6 && s.vendor == VendorNvidia)
			std::printf(" (auto: NVIDIA)");
		std::printf("\n");
	}
	if (s.mode != 0)
		std::printf("[p7] hazards de C1 (cadenas) y C2 (imagenes) contados %s: linea [p7] cada 5 s\n",
		            s.count_always ? "siempre (KYTY_P7_CONTAR=1), tambien con los arreglos apagados"
		                           : "donde C1=1 o C2 estan encendidos (KYTY_P7_CONTAR=1: siempre)");
	std::fflush(stdout);
	s.decided = true;
}

// The live fixes on (the ON window or no alternation) or back to the values before them (OFF window). The GPU thread,
// after its switches (InitializePerformanceSwitches), and the flip thread at each window.
inline void ApplyLive(bool on) noexcept {
	auto& s = S();
	if (!s.decided) return;
	if (!s.live_captured) {
		s.base_chain         = kyty_local_compute_chain_mode.load(std::memory_order_relaxed);
		s.base_image_dedupe  = kyty_local_image_barrier_dedupe.load(std::memory_order_relaxed);
		s.base_global_dedupe = kyty_local_global_barrier_dedupe.load(std::memory_order_relaxed);
		s.base_readback      = kyty_local_readback_queue_mode.load(std::memory_order_relaxed);
		s.live_captured      = true;
	}
	// Tracking (its CPU cost) inside the ON window of a fix that tracks - precise C1, precise C2 -, or always with
	// KYTY_P7_CONTAR=1 (the hazards with every fix off: NVIDIA's auto, an OFF window).
	g_track.store(s.mode != 0 && (s.count_always || (on && (s.level[1] == 1 || s.level[2] != 0))) ? 1u : 0u,
	              std::memory_order_relaxed);
	g_track_epoch.fetch_add(1, std::memory_order_relaxed);
	g_chain_precise.store(on && s.level[1] == 1 ? 1u : 0u, std::memory_order_relaxed);
	g_args_barrier_in_chain.store(on && s.level[1] == 3 ? 1u : 0u, std::memory_order_relaxed);
	kyty_local_compute_chain_mode.store(on && s.level[1] == 2 ? 0u : s.base_chain, std::memory_order_relaxed);
	// (p7b: C2 keeps both dedupes as they were and proves each image drop instead.)
	g_precise_barriers.store(on && s.level[2] != 0 ? 1u : 0u, std::memory_order_relaxed);
	g_labels_at_completion.store(on && s.level[5] != 0 ? 1u : 0u, std::memory_order_relaxed);
	kyty_local_readback_queue_mode.store(on && s.level[6] == 1   ? 0u
	                                     : on && s.level[6] == 2 ? 3u
	                                                             : s.base_readback,
	                                     std::memory_order_relaxed);
	g_clear_new_images.store(on && s.level[7] != 0 ? 1u : 0u, std::memory_order_relaxed);
	g_window_on.store(on, std::memory_order_relaxed);
}

// The GPU thread, right after InitializePerformanceSwitches.
inline void ApplyAtGpuStart() noexcept {
	auto& s = S();
	if (!s.decided) return;
	ApplyLive(true);
	std::printf("[p7] interruptores en vivo: cadenas %u->%u, dedupe imagen %u->%u, dedupe global %u->%u, readback cola %u->%u, "
	            "barrera de cadena solo con hazard %u, barrera de args en toda cadena %u, dedupe de imagen probado %u, "
	            "etiquetas al completar %u, limpiar imagenes nuevas %u\n",
	            s.base_chain, kyty_local_compute_chain_mode.load(std::memory_order_relaxed), s.base_image_dedupe,
	            kyty_local_image_barrier_dedupe.load(std::memory_order_relaxed), s.base_global_dedupe,
	            kyty_local_global_barrier_dedupe.load(std::memory_order_relaxed), s.base_readback,
	            kyty_local_readback_queue_mode.load(std::memory_order_relaxed),
	            g_chain_precise.load(std::memory_order_relaxed), g_args_barrier_in_chain.load(std::memory_order_relaxed),
	            g_precise_barriers.load(std::memory_order_relaxed), g_labels_at_completion.load(std::memory_order_relaxed),
	            g_clear_new_images.load(std::memory_order_relaxed));
	std::fflush(stdout);
}

[[nodiscard]] inline bool AnyLiveFix() noexcept {
	const auto& s = S();
	return s.decided && (s.level[1] != 0 || s.level[2] != 0 || s.level[5] != 0 || s.level[6] != 0 || s.level[7] != 0);
}

// ---- C1 (p7b): hazards inside a chain of dispatches -----------------------------------------------------------------
// Per command buffer being recorded (`owner`), on the recording thread: the guest ranges the chain's dispatches so far
// wrote and read. A chain's first dispatch (`continues` false: a barrier or a non-chained dispatch is before it) starts
// a new one. A hazard - the indirect arguments or a read over a written range (RAW), a write over a written (WAW) or a
// read range (WAR), flat/global reads after any write - is one barrier (compute -> compute | draw-indirect, every access),
// which orders everything before it: the ranges start again. Atomic overlaps are hazards too (counted apart: the shader
// info cannot tell an atomic-only buffer from one also written plainly).

struct ChainAccess {
	uint64_t begin = 0, end = 0;
	bool     read = false, write = false, atomic = false;
};

enum ChainHazardBit : uint32_t {
	ChainHazardArgs = 1, ChainHazardRaw = 2, ChainHazardWaw = 4, ChainHazardWar = 8, ChainHazardUnknown = 16,
	ChainHazardConservative = 32, ChainHazardAtomic = 64
};

struct ChainRange {
	uint64_t begin = 0, end = 0;
	bool     atomic = false;
};

struct ChainTrack {
	static constexpr uint32_t Capacity = 16;
	const void*                         owner = nullptr;
	uint32_t                            epoch = 0;
	uint32_t                            writes = 0, reads = 0;
	bool                                overflow = false;
	std::array<ChainRange, Capacity>    written {}, read {};
	void Clear() noexcept { writes = reads = 0, overflow = false; }
	// Merged with a range it overlaps or touches (same atomic use), else appended; full: overflow.
	static void Note(std::array<ChainRange, Capacity>& list, uint32_t& count, bool& full, uint64_t begin, uint64_t end,
	                 bool atomic) noexcept {
		for (uint32_t i = 0; i < count; ++i) {
			auto& r = list[i];
			if (begin <= r.end && r.begin <= end && r.atomic == atomic) {
				r.begin = std::min(r.begin, begin);
				r.end   = std::max(r.end, end);
				return;
			}
		}
		if (count == Capacity) {
			full = true;
			return;
		}
		list[count++] = {begin, end, atomic};
	}
};

// Overlap of [begin, end) with any range of the list: 0 none, 1 only atomic ones (and the access atomic), 2 any other.
// (Branch-free over the list: the ranges are data, a branch per range mispredicts.)
[[nodiscard]] inline int ChainOverlap(const std::array<ChainRange, ChainTrack::Capacity>& list, uint32_t count, uint64_t begin,
                                      uint64_t end, bool atomic) noexcept {
	uint32_t any = 0, plain = 0;
	for (uint32_t i = 0; i < count; ++i) {
		const uint32_t hit = static_cast<uint32_t>(begin < list[i].end) & static_cast<uint32_t>(list[i].begin < end);
		any |= hit;
		plain |= hit & static_cast<uint32_t>(!(atomic && list[i].atomic));
	}
	return plain != 0 ? 2 : any != 0 ? 1 : 0;
}

[[nodiscard]] inline ChainTrack& ChainOf(const void* owner, bool continues) noexcept {
	thread_local std::array<ChainTrack, 4> tracks {};
	thread_local uint32_t                  next = 0;
	const uint32_t epoch = g_track_epoch.load(std::memory_order_relaxed);
	for (auto& track: tracks)
		if (track.owner == owner) {
			if (track.epoch != epoch) {
				// (Tracking or the fixes switched since: the chain's dispatches meanwhile were not noted.)
				track.epoch = epoch;
				track.Clear();
				track.overflow = continues;
			}
			return track;
		}
	auto& track = tracks[next++ % tracks.size()];
	track.owner = owner;
	track.epoch = epoch;
	track.Clear();
	// (A chain whose state was not here - more command buffers than slots on this thread -: not proven, one barrier.)
	track.overflow = continues;
	return track;
}

// One dispatch of a chained program, before it is recorded: its hazards (ChainHazardBit; 0 none) against the chain's
// earlier dispatches, then its own accesses noted. `args`: its indirect arguments' guest address (0: direct);
// `any_read`: flat/global reads (addresses not known here). Counted (ChainChecked, ChainHazards, Haz*).
[[nodiscard]] inline uint32_t ChainStep(const void* owner, bool continues, uint64_t args, const ChainAccess* accesses,
                                        size_t count, bool any_read) noexcept {
	auto& t = ChainOf(owner, continues);
	if (!continues) t.Clear();
	uint32_t hazard = 0;
	if (continues) {
		Add(ChainChecked);
		bool only_atomic = true;
		const auto check = [&](int overlap, uint32_t bit) {
			if (overlap == 0) return;
			hazard |= bit;
			only_atomic = only_atomic && overlap == 1;
		};
		if (t.overflow) {
			hazard |= ChainHazardConservative;
			only_atomic = false;
		}
		if (args != 0) check(ChainOverlap(t.written, t.writes, args, args + 12u, false), ChainHazardArgs);
		if (any_read && t.writes != 0) {
			hazard |= ChainHazardUnknown;
			only_atomic = false;
		}
		for (size_t i = 0; i < count; ++i) {
			const auto& a       = accesses[i];
			const int   written = t.writes != 0 ? ChainOverlap(t.written, t.writes, a.begin, a.end, a.atomic) : 0;
			check(a.read ? written : 0, ChainHazardRaw);
			check(a.write ? written : 0, ChainHazardWaw);
			if (a.write && t.reads != 0) check(ChainOverlap(t.read, t.reads, a.begin, a.end, a.atomic), ChainHazardWar);
		}
		if (hazard != 0) {
			if (only_atomic) hazard |= ChainHazardAtomic;
			Add(ChainHazards);
			if ((hazard & ChainHazardArgs) != 0) Add(HazArgs);
			if ((hazard & ChainHazardRaw) != 0) Add(HazRaw);
			if ((hazard & ChainHazardWaw) != 0) Add(HazWaw);
			if ((hazard & ChainHazardWar) != 0) Add(HazWar);
			if ((hazard & ChainHazardUnknown) != 0) Add(HazUnknown);
			if ((hazard & ChainHazardAtomic) != 0) Add(HazAtomic);
			if ((hazard & ~uint32_t {ChainHazardConservative}) == 0) Add(HazConservative);
			t.Clear(); // (the barrier orders everything before it, recorded or - fix off - as if it were)
		}
	}
	if (args != 0) ChainTrack::Note(t.read, t.reads, t.overflow, args, args + 12u, false);
	for (size_t i = 0; i < count; ++i) {
		const auto& a = accesses[i];
		if (a.end <= a.begin) continue;
		if (a.write) ChainTrack::Note(t.written, t.writes, t.overflow, a.begin, a.end, a.atomic);
		if (a.read) ChainTrack::Note(t.read, t.reads, t.overflow, a.begin, a.end, a.atomic);
	}
	return hazard;
}

// ---- KYTY_VULKAN_SYNCVAL: the summary --------------------------------------------------------------------------------

struct Syncval {
	std::mutex                          mutex;
	uint64_t                            total = 0;
	std::map<std::string, uint64_t>     by_id;
	std::map<std::string, uint64_t>     by_pair; // "id | command <- prior command"
	uint64_t                            by_cause[Causes + 1] {}; // [0] other, [1..7], [8] unknown command
	std::chrono::steady_clock::time_point last_print = std::chrono::steady_clock::now();
};

// (Never destroyed: the summary at exit runs after static destructors may have.)
inline Syncval& SyncvalState() noexcept {
	static Syncval* state = new Syncval;
	return *state;
}

// The Vulkan commands named in a message, in order (with extra properties: the command, then the prior one).
inline std::vector<std::string> SyncvalCommands(const char* message) {
	std::vector<std::string> commands;
	for (const char* at = std::strstr(message, "vkCmd"); at != nullptr && commands.size() < 4; at = std::strstr(at + 5, "vkCmd")) {
		size_t length = 0;
		while ((at[length] >= 'a' && at[length] <= 'z') || (at[length] >= 'A' && at[length] <= 'Z') ||
		       (at[length] >= '0' && at[length] <= '9'))
			++length;
		commands.emplace_back(at, length);
	}
	return commands;
}

// Probable cause of a hazard: 1 dispatch after dispatch, 2 draws / rendering / image use between draws, 6 between
// queues (racing, submit time), 7 a layout transition from UNDEFINED, 0 other, 8 no command named.
inline int SyncvalCause(std::string_view id, const char* message, const std::vector<std::string>& commands) {
	const auto has = [&](const char* text) { return std::strstr(message, text) != nullptr; };
	if (id.find("RACING") != std::string_view::npos || has("vkQueueSubmit") || has("VkQueue ")) return 6;
	if (has("VK_IMAGE_LAYOUT_UNDEFINED") && has("transition")) return 7;
	if (commands.empty()) return 8;
	const auto dispatch = [](const std::string& c) { return c.rfind("vkCmdDispatch", 0) == 0; };
	const auto draw     = [](const std::string& c) {
        return c.rfind("vkCmdDraw", 0) == 0 || c.rfind("vkCmdBeginRendering", 0) == 0 || c.rfind("vkCmdEndRendering", 0) == 0 ||
               c.rfind("vkCmdClearAttachments", 0) == 0;
	};
	if (commands.size() >= 2 && dispatch(commands[0]) && dispatch(commands[1])) return 1;
	if (commands.size() >= 2 && draw(commands[0]) && draw(commands[1])) return 2;
	if (commands.size() >= 2 && (draw(commands[0]) || draw(commands[1])) && (has("VkImage") || has("image"))) return 2;
	return 0;
}

inline void SyncvalSummary(const char* when) {
	auto&           v = SyncvalState();
	std::lock_guard lock(v.mutex);
	static const char* const cause_names[Causes + 1] = {"otra",
	                                                    "C1 dispatch tras dispatch",
	                                                    "C2 entre draws/imagenes",
	                                                    "C3 -",
	                                                    "C4 -",
	                                                    "C5 -",
	                                                    "C6 entre colas/envio",
	                                                    "C7 transicion desde UNDEFINED",
	                                                    "sin comando"};
	std::printf("[p7-syncval] %s: %llu avisos de sincronizacion |", when, static_cast<unsigned long long>(v.total));
	for (int c = 0; c <= Causes; ++c)
		if (v.by_cause[c] != 0) std::printf(" %s=%llu", cause_names[c], static_cast<unsigned long long>(v.by_cause[c]));
	std::printf("\n");
	std::vector<std::pair<uint64_t, std::string>> ids;
	for (const auto& [id, count]: v.by_id) ids.emplace_back(count, id);
	std::sort(ids.begin(), ids.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	for (size_t i = 0; i < ids.size() && i < 12; ++i)
		std::printf("[p7-syncval]   id %s: %llu\n", ids[i].second.c_str(), static_cast<unsigned long long>(ids[i].first));
	std::vector<std::pair<uint64_t, std::string>> pairs;
	for (const auto& [pair, count]: v.by_pair) pairs.emplace_back(count, pair);
	std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	for (size_t i = 0; i < pairs.size() && i < 15; ++i)
		std::printf("[p7-syncval]   %s: %llu\n", pairs[i].second.c_str(), static_cast<unsigned long long>(pairs[i].first));
	std::fflush(stdout);
}

// Every synchronization message (the callback's), counted; a summary every 30 s, and at exit.
inline void SyncvalNote(const char* id_name, const char* message) {
	static const bool registered = [] {
		std::atexit([] { SyncvalSummary("al salir"); });
		return true;
	}();
	(void)registered;
	const std::string id       = id_name != nullptr ? id_name : "?";
	const auto        commands = SyncvalCommands(message);
	const int         cause    = SyncvalCause(id, message, commands);
	bool              print    = false;
	{
		auto&           v = SyncvalState();
		std::lock_guard lock(v.mutex);
		++v.total;
		++v.by_id[id];
		++v.by_cause[cause];
		std::string pair = id + " | " + (commands.empty() ? std::string("?") : commands[0]) + " <- " +
		                   (commands.size() >= 2 ? commands[1] : std::string("?"));
		++v.by_pair[pair];
		const auto now = std::chrono::steady_clock::now();
		if (now - v.last_print >= std::chrono::seconds(30)) {
			v.last_print = now;
			print        = true;
		}
	}
	if (print) SyncvalSummary("cada 30 s");
}

} // namespace P7
