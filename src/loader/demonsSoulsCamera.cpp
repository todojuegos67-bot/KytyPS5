// (10-25) Demon's Souls' free camera turn ramp, removed for keyboard and mouse (KYTY_KBM_NO_RAMP).
//
// The free camera's stick step (ChrExFollowCam, 01.007.000 eboot+0x18cf00; this = the follow camera, [this] = its
// camera data cd, dt at this+0x44, the right stick at this+0x120 (pitch) / +0x124 (yaw), its length at this+0x130):
//   if |stick| > 0.998 and the stick is mostly horizontal (|yaw| above a function of cd+0x220):
//     ramp = cd+0x128 += dt / cd+0x1d0 (towards +-1 with the stick's sign, from 0 on a reversal), and
//     yaw += sign(stick) * dt * lerp(cd+0x1f0, cd+0x208, |ramp|)     (the stick's size not used: full tilt)
//     pitch eased towards a position from the stick's vertical part
//   else:
//     ramp = 0
//     pitch += dt * cd+0x1ec * stick.pitch, yaw += dt * cd+0x1f0 * stick.yaw   (linear, no ramp)
// cd+0x1ec/0x1f0 (base) and 0x204/0x208 (top) are lerp(slowest, fastest, t^2) with t = the options' camera speed / 10,
// clamped to 1 (eboot+0x181b75 and eboot+0x192cc1: speeds above 10 turn as 10 does).
// So a full stick turns at the base rate and then speeds up for cd+0x1d0 s to the top rate, and stops dead when the
// stick leaves full tilt: the mouse saturates the stick at a quick swipe, so the camera accelerated over ~0.3 s (the
// video: 26, 27, 28 ... 35 px a frame) and stopped at once on release.
// The patch (only with keyboard and mouse): the full-tilt branch is never taken (jbe -> jmp to "ramp = 0"), so every
// stick takes the linear branch: an immediate yaw rate proportional to the stick. KYTY_KBM_NO_RAMP=1 (default) also
// makes the linear yaw use the top rate cd+0x208 (a full stick turns at the game's fastest at once), =2 keeps the base
// rate cd+0x1f0 (slower), =0 leaves the game's code.
//
// (11-10) Direct mouse camera (KYTY_KBM_DIRECT, default on with KYTY_KBM=1; =0 off), as the Bloodborne port's
// bb_kbmcam: the mouse turns the camera by its angle, no full-tilt cap. Two 5-byte jmp detours, both on the bytes
// above (the same patterns the no-ramp patch checks):
//   site A (FullTilt+0, the vmovss of |stick|): while mouse counts are pending, straight to the linear branch (no
//     0.998 full-tilt test, no ramp); otherwise the original test (the pad keeps the game's behaviour).
//   site B (Linear+0, "ramp = 0", rax = camera data, rbx = the camera): a host call that, with counts pending,
//     replaces the stick (this+0x124 yaw, +0x120 pitch) by angle / (rate * dt) with angle = counts * deg per count
//     (KYTY_KBM_DEG_PER_COUNT * the F2 sensitivity), the yaw rate the linear code multiplies by (its disp32 read back:
//     0x1f0 or 0x208 with KYTY_KBM_NO_RAMP=1), pitch cd+0x1ec, dt this+0x44: the game's own linear branch then turns
//     the camera by exactly that angle (and wraps / clamps it itself).
// The stick lanes' signs are learned on the first mouse motion (the mouse still drives the emulated stick until
// then), or set by KYTY_KBM_YAW_SIGN / KYTY_KBM_PITCH_SIGN (+1/-1). Counts go to the camera only while this free
// camera runs (its hook seen in the last 100 ms); otherwise (lock-on, menus) the mouse drives the stick as before.
// (11-10c) The mouse "disconnected" while running: with the counts going direct the emulated stick's X was centred, so
// in every frame without counts (~25% of hook frames at 1000 Hz mouse / 60 fps, all of them on a pause) the game saw no
// right-stick input and its follow camera swung back behind the running player, undoing the mouse's yaw. Now, while
// the mouse turned the camera in the last KYTY_KBM_HOLD_MS (400), the stick X is held off centre (KYTY_KBM_HOLD, 0.35 of
// full tilt, the last mouse direction) and the hook writes yaw stick 0 in frames without counts, so the held stick
// never turns the camera itself. Counts are never dropped: if the hook cannot convert them they stay for the next frame.
// (11-11) KYTY_KBM_EQUAL_AXES=1 (default): the horizontal no longer comes here as raw counts (no cap) but as the degrees
// the emulated stick path paid this game frame (DirectAddDegrees): the same drain, the same top rate (the curve's or the
// linear mapping's), the same carry as the vertical stick, so both axes turn alike; the hook still turns the camera by
// exactly that angle. =0: raw counts, no cap (the 11-10 behaviour).
#include "loader/demonsSoulsCamera.h"

#include "common/abi.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"

#include <xbyak/xbyak.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Loader::DemonsSoulsCamera {
namespace {

// vmovss xmm0, [rbx + 0x130]; vucomiss xmm0, [rip + 0.998f]; jbe <ramp = 0>; vbroadcastss xmm1, [rip + abs mask];
// vmovss xmm0, [rbx + 0x124].
constexpr const char* FullTilt =
    "c5 fa 10 83 30 01 00 00 c5 f8 2e 05 ?? ?? ?? ?? 0f 86 ?? ?? ?? ?? c4 e2 79 18 0d ?? ?? ?? ?? c5 fa 10 83 24 01 00 00";
// ramp = 0: mov dword [rax + 0x128], 0; vmovss xmm0, [rbx + 0x44]; vmulss xmm1, xmm0, [rax + 0x1ec]; lea rcx, [rip];
// vmulss xmm0, xmm0, [rax + 0x1f0]; lea rdi, [rip]; lea rsi, [rip]; vmulss xmm1, xmm1, [rbx + 0x120];
// vmulss xmm0, xmm0, [rbx + 0x124].
constexpr const char* Linear = "c7 80 28 01 00 00 00 00 00 00 c5 fa 10 43 44 c5 fa 59 88 ec 01 00 00 48 8d 0d ?? ?? ?? ?? "
                               "c5 fa 59 80 f0 01 00 00 48 8d 3d ?? ?? ?? ?? 48 8d 35 ?? ?? ?? ?? c5 f2 59 8b 20 01 00 00 "
                               "c5 fa 59 83 24 01 00 00";
constexpr uint64_t JbeOffset      = 16; // in FullTilt
constexpr uint64_t YawRateOffset  = 34; // in Linear: the disp32 of [rax + 0x1f0]
constexpr uint32_t BaseYawRate    = 0x1f0;
constexpr uint32_t TopYawRate     = 0x208;

bool Is1(const char* name) {
	const char* v = std::getenv(name);
	return v != nullptr && v[0] == '1' && v[1] == '\0';
}

// Direct mouse camera state.
constexpr uint64_t     CavePage = 0x4000, CaveOffset = 0x8000000 + 8 * CavePage;
std::atomic<bool>      g_direct {false}; // the detours installed
std::atomic<uint8_t>*  g_pending = nullptr; // in the cave: counts pending (site A takes the linear branch)
std::atomic<int64_t>   g_mx {0}, g_my {0};
std::atomic<int64_t>   g_udeg {0};          // (11-11) yaw degrees pending from the metered path, in 1e-6 deg
std::atomic<float>     g_rate_y_deg {0.0f}, g_rate_p_deg {0.0f}; // the game's yaw / pitch full-stick rates, deg/s
std::atomic<int>       g_yaw_sign {0}, g_pitch_sign {0};
std::atomic<int>       g_stick_x {128}, g_stick_y {128};
std::atomic<float>     g_dpc {0.05f};
std::atomic<int64_t>   g_hook_ns {0};
std::atomic<int64_t>   g_mouse_ns {0};      // last direct mouse counts
std::atomic<int>       g_last_dir {1};      // the last horizontal direction (+1/-1)
std::atomic<int64_t>   g_routed {0};        // counts routed to the stick path while the detours exist
std::atomic<int64_t>   g_kept {0};          // counts the hook could not convert (kept for the next frame)
uint64_t               g_linear = 0;
// Log accumulation (game thread only).
int64_t g_log_ns = 0;
int     g_frames = 0, g_injected = 0, g_held = 0;
double  g_deg = 0.0;

int64_t HoldNs() {
	static const int64_t ns = [] {
		const char* v = std::getenv("KYTY_KBM_HOLD_MS");
		const int   m = v != nullptr && *v != '\0' ? std::atoi(v) : 400;
		return static_cast<int64_t>(m < 0 ? 0 : m) * 1000000;
	}();
	return ns;
}

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int EnvSign(const char* n) {
	const char* v = std::getenv(n);
	if (v == nullptr || *v == '\0') return 0;
	return std::atoi(v) < 0 ? -1 : 1;
}

float F(const uint8_t* p, uint32_t off) {
	float f = 0.0f;
	std::memcpy(&f, p + off, sizeof(f));
	return f;
}

void KYTY_SYSV_ABI Hook(uint8_t* cam, const uint8_t* cd) {
	const int64_t now = NowNs();
	g_hook_ns.store(now, std::memory_order_relaxed);
	g_frames++;
	if (g_yaw_sign.load(std::memory_order_relaxed) == 0) {
		const int   sx = g_stick_x.load(std::memory_order_relaxed) - 128;
		const float v  = F(cam, 0x124);
		if (std::abs(sx) >= 40 && std::fabs(v) > 0.15f) {
			g_yaw_sign = (v > 0) == (sx > 0) ? 1 : -1;
			std::printf("Demon's Souls direct mouse camera: yaw sign learned %d (stick x %d, game yaw %.2f)\n",
			            g_yaw_sign.load(), sx, v);
		}
	}
	if (g_pitch_sign.load(std::memory_order_relaxed) == 0) {
		const int   sy = g_stick_y.load(std::memory_order_relaxed) - 128;
		const float v  = F(cam, 0x120);
		if (std::abs(sy) >= 40 && std::fabs(v) > 0.15f) {
			g_pitch_sign = (v > 0) == (sy > 0) ? 1 : -1;
			std::printf("Demon's Souls direct mouse camera: pitch sign learned %d (stick y %d, game pitch %.2f)\n",
			            g_pitch_sign.load(), sy, v);
		}
	}
	if (g_pending != nullptr) g_pending->store(0, std::memory_order_relaxed);
	const int64_t dx = g_mx.exchange(0, std::memory_order_acq_rel);
	const int64_t du = g_udeg.exchange(0, std::memory_order_acq_rel); // (11-11) metered degrees
	g_my.exchange(0, std::memory_order_acq_rel); // (the vertical is the emulated stick's)
	uint32_t yaw_off = BaseYawRate;
	std::memcpy(&yaw_off, reinterpret_cast<const void*>(g_linear + YawRateOffset), sizeof(yaw_off));
	const float dt = F(cam, 0x44), rate_y = F(cd, yaw_off), rate_p = F(cd, 0x1ec);
	if (std::isfinite(rate_y)) g_rate_y_deg.store(rate_y * 57.2957795f, std::memory_order_relaxed);
	if (std::isfinite(rate_p)) g_rate_p_deg.store(rate_p * 57.2957795f, std::memory_order_relaxed);
	const bool  ok = g_yaw_sign.load() != 0 && dt > 1e-4f && rate_y > 1e-4f && std::isfinite(dt) && std::isfinite(rate_y);
	const double deg = static_cast<double>(dx) * g_dpc.load(std::memory_order_relaxed) + static_cast<double>(du) * 1e-6;
	if ((dx != 0 || du != 0) && ok) {
		const float rad = static_cast<float>(deg) * 3.14159265f / 180.0f;
		const float yaw = static_cast<float>(g_yaw_sign.load()) * rad / (rate_y * dt);
		std::memcpy(cam + 0x124, &yaw, sizeof(yaw));
		g_injected++;
		g_deg += std::fabs(deg);
	} else if (dx != 0 || du != 0) {
		// Not convertible this frame: keep the counts / degrees (never lost), the camera turns by them next frame.
		g_mx.fetch_add(dx, std::memory_order_acq_rel);
		g_udeg.fetch_add(du, std::memory_order_acq_rel);
		g_kept.fetch_add(std::llabs(dx), std::memory_order_relaxed);
		if (g_pending != nullptr) g_pending->store(1, std::memory_order_release);
	} else if (now - g_mouse_ns.load(std::memory_order_relaxed) < HoldNs() + 50000000) {
		// The held stick (game sees camera input, no auto-follow) must not turn the camera by itself.
		const float zero = 0.0f;
		std::memcpy(cam + 0x124, &zero, sizeof(zero));
		g_held++;
	}
	if (now - g_log_ns > 2000000000 && (g_injected != 0 || g_held != 0 || g_kept.load() != 0 || g_routed.load() != 0)) {
		std::printf("Demon's Souls direct mouse camera: %d of %d frames turned by the mouse, %d held (stick held, yaw 0), "
		            "%.0f deg yaw asked, counts kept for a later frame %lld, counts routed to the stick path %lld, %.3f "
		            "deg/count, yaw rate %.2f (cd+0x%x, %.0f deg/s) pitch rate %.2f (cd+0x1ec, %.0f deg/s), dt %.4f\n",
		            g_injected, g_frames, g_held, g_deg, static_cast<long long>(g_kept.exchange(0)),
		            static_cast<long long>(g_routed.exchange(0)), g_dpc.load(), rate_y, yaw_off, rate_y * 57.2957795f, rate_p,
		            rate_p * 57.2957795f, dt);
		std::fflush(stdout);
		g_log_ns = now, g_injected = g_frames = g_held = 0, g_deg = 0.0;
	} else if (now - g_log_ns > 2000000000) {
		g_log_ns = now, g_injected = g_frames = g_held = 0, g_deg = 0.0;
	}
}

// The two caves; false (nothing patched) if the memory or code could not be made.
bool InstallDirect(Program* program, uint64_t tilt, uint64_t linear, const char** why) {
	const uint64_t requested = program->base_vaddr + CaveOffset;
	const uint64_t cave      = Libs::LibKernel::Memory::AllocateRuntimeMemory(
        requested, CavePage, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_kbm_camera", true);
	const auto is_near = [&](uint64_t a) {
		const int64_t d = static_cast<int64_t>(a) - static_cast<int64_t>(cave);
		return d > -0x70000000LL && d < 0x70000000LL;
	};
	if (cave == 0 || !is_near(tilt) || !is_near(linear)) {
		if (cave != 0) Libs::LibKernel::Memory::FreeGuestMemory(cave, CavePage);
		*why = "no executable memory near the eboot";
		return false;
	}
	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator c(CavePage, reinterpret_cast<void*>(cave));
	Xbyak::Label         flag;
	constexpr uint8_t    save_fp[] {0x48, 0x0f, 0xae, 0x04, 0x24}; // FXSAVE64 [rsp]
	// Site A: the original vmovss xmm0, [rbx+0x130]; counts pending -> the linear branch, else the 0.998 test.
	const uint8_t* site_a = c.getCurr();
	c.db(reinterpret_cast<const uint8_t*>(tilt), 8);
	c.cmp(byte[rip + flag], 0); // (flags are dead: the test sets them, the linear branch reads none)
	c.jne(reinterpret_cast<const void*>(linear));
	c.jmp(reinterpret_cast<const void*>(tilt + 8), Xbyak::CodeGenerator::T_NEAR);
	// Site B: registers, flags and FP state kept, below the red zone, on a realigned stack; then "ramp = 0" and on.
	c.align(16);
	const uint8_t* site_b = c.getCurr();
	c.lea(rsp, ptr[rsp - 128]);
	c.pushfq();
	for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, rbx})
		c.push(reg);
	c.mov(rdi, rbx);
	c.mov(rsi, rax);
	c.mov(rbx, rsp);
	c.and_(rsp, -16);
	c.sub(rsp, 512);
	c.db(save_fp, sizeof(save_fp));
	c.mov(rax, reinterpret_cast<uint64_t>(&Hook));
	c.call(rax);
	c.fxrstor64(ptr[rsp]);
	c.mov(rsp, rbx);
	for (const auto& reg: {rbx, r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
		c.pop(reg);
	c.popfq();
	c.lea(rsp, ptr[rsp + 128]);
	c.db(reinterpret_cast<const uint8_t*>(linear), 10); // mov dword [rax+0x128], 0
	c.jmp(reinterpret_cast<const void*>(linear + 10), Xbyak::CodeGenerator::T_NEAR);
	c.align(16);
	c.L(flag);
	const uint8_t* flag_at = c.getCurr();
	c.dd(0);
	c.ready();
	if (Xbyak::GetError() || !Common::VirtualMemory::FlushInstructionCache(cave, c.getSize())) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, CavePage);
		*why = "the hook code could not be generated";
		return false;
	}
	// The flag byte stays writable (the page is RWX).
	g_pending = reinterpret_cast<std::atomic<uint8_t>*>(const_cast<uint8_t*>(flag_at));
	g_pending->store(0);
	g_linear = linear;
	const auto jmp = [](uint64_t at, const uint8_t* to, size_t size) {
		uint8_t       j[10] {0xe9, 0, 0, 0, 0, 0x90, 0x90, 0x90, 0x90, 0x90};
		const int32_t rel = static_cast<int32_t>(reinterpret_cast<int64_t>(to) - static_cast<int64_t>(at + 5));
		std::memcpy(j + 1, &rel, sizeof(rel));
		std::memcpy(reinterpret_cast<void*>(at), j, size);
		Common::VirtualMemory::FlushInstructionCache(at, size);
	};
	jmp(linear, site_b, 10);
	jmp(tilt, site_a, 8);
	g_yaw_sign   = EnvSign("KYTY_KBM_YAW_SIGN");
	g_pitch_sign = EnvSign("KYTY_KBM_PITCH_SIGN");
	g_direct     = true;
	return true;
}

} // namespace

bool DirectWanted() {
	static const bool wanted = [] {
		if (!Is1("KYTY_KBM") || Is1("KYTY_KBM_CALIBRATE")) return false;
		const char* v = std::getenv("KYTY_KBM_DIRECT");
		return v == nullptr || v[0] != '0';
	}();
	return wanted;
}

bool DirectInstalled() {
	return g_direct.load(std::memory_order_relaxed);
}

bool DirectTakesMouse() {
	if (!g_direct.load(std::memory_order_relaxed) || g_yaw_sign.load(std::memory_order_relaxed) == 0 ||
	    g_pitch_sign.load(std::memory_order_relaxed) == 0)
		return false;
	const int64_t seen = g_hook_ns.load(std::memory_order_relaxed);
	return seen != 0 && NowNs() - seen < 100000000;
}

void DirectAddMouse(int dx, int dy) {
	g_mx.fetch_add(dx, std::memory_order_acq_rel);
	g_my.fetch_add(dy, std::memory_order_acq_rel);
	if (dx != 0) {
		g_mouse_ns.store(NowNs(), std::memory_order_relaxed);
		g_last_dir.store(dx > 0 ? 1 : -1, std::memory_order_relaxed);
	}
	if (g_pending != nullptr) g_pending->store(1, std::memory_order_release);
}

void DirectAddDegrees(double deg) {
	if (!std::isfinite(deg) || deg == 0.0) return;
	g_udeg.fetch_add(static_cast<int64_t>(std::llround(deg * 1e6)), std::memory_order_acq_rel);
	g_mouse_ns.store(NowNs(), std::memory_order_relaxed);
	g_last_dir.store(deg > 0 ? 1 : -1, std::memory_order_relaxed);
	if (g_pending != nullptr) g_pending->store(1, std::memory_order_release);
}

float DirectGameYawRateDeg() {
	return g_rate_y_deg.load(std::memory_order_relaxed);
}

float DirectGamePitchRateDeg() {
	return g_rate_p_deg.load(std::memory_order_relaxed);
}

int DirectHoldStickX() {
	if (!g_direct.load(std::memory_order_relaxed) || g_yaw_sign.load(std::memory_order_relaxed) == 0) return 128;
	const int64_t last = g_mouse_ns.load(std::memory_order_relaxed);
	const int64_t now = NowNs(), seen = g_hook_ns.load(std::memory_order_relaxed);
	// Only while the free camera runs (lock-on, menus: no held stick, it could switch targets / move a cursor).
	if (last == 0 || now - last >= HoldNs() || seen == 0 || now - seen >= 100000000) return 128;
	static const double amount = [] {
		const char* v = std::getenv("KYTY_KBM_HOLD");
		const double a = v != nullptr && *v != '\0' ? std::atof(v) : 0.35;
		return a > 0.0 && a <= 0.9 ? a : 0.35;
	}();
	return 128 + static_cast<int>(std::lround(g_last_dir.load(std::memory_order_relaxed) * amount * 127.0));
}

void DirectNoteStickRouted(int64_t counts) {
	if (g_direct.load(std::memory_order_relaxed)) g_routed.fetch_add(counts, std::memory_order_relaxed);
}

void DirectSetStick(int x, int y) {
	g_stick_x.store(x, std::memory_order_relaxed);
	g_stick_y.store(y, std::memory_order_relaxed);
}

void DirectSetDegPerCount(float deg) {
	if (std::isfinite(deg) && deg > 0.0f) g_dpc.store(deg, std::memory_order_relaxed);
}

int NoRampMode() {
	static const int mode = [] {
		if (!Is1("KYTY_KBM") && !Is1("KYTY_KBM_CALIBRATE")) return 0;
		const char* v = std::getenv("KYTY_KBM_NO_RAMP");
		if (v == nullptr || *v == '\0') return 1;
		const int m = std::atoi(v);
		return m == 0 || m == 2 ? m : 1;
	}();
	return mode;
}

void InstallNoRamp(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;
	const int  mode   = NoRampMode();
	const bool direct = DirectWanted();
	if (Is1("KYTY_KBM") && !direct)
		std::printf("Demon's Souls direct mouse camera: off (%s); the mouse drives the right stick\n",
		            Is1("KYTY_KBM_CALIBRATE") ? "KYTY_KBM_CALIBRATE=1" : "KYTY_KBM_DIRECT=0");
	if (mode == 0 && !direct) {
		if (Is1("KYTY_KBM") || Is1("KYTY_KBM_CALIBRATE"))
			std::printf("Demon's Souls camera: the game's full-tilt turn ramp kept (KYTY_KBM_NO_RAMP=0)\n");
		std::fflush(stdout);
		return;
	}
	const auto tilt   = GuestCode::FindUnique(*program, GuestCode::Pattern(FullTilt));
	const auto linear = GuestCode::FindUnique(*program, GuestCode::Pattern(Linear));
	bool       ok     = tilt.has_value() && linear.has_value();
	uint64_t   jbe    = 0;
	if (ok) {
		// The compare's constant is 0.998f, the jbe lands on the linear branch's "ramp = 0".
		int32_t disp = 0, rel = 0;
		std::memcpy(&disp, reinterpret_cast<const void*>(*tilt + 12), sizeof(disp));
		std::memcpy(&rel, reinterpret_cast<const void*>(*tilt + JbeOffset + 2), sizeof(rel));
		const uint64_t constant = *tilt + 16 + static_cast<int64_t>(disp);
		float          limit    = 0.0f;
		std::memcpy(&limit, reinterpret_cast<const void*>(constant), sizeof(limit));
		jbe = *tilt + JbeOffset;
		ok  = limit > 0.99f && limit < 1.0f && jbe + 6 + static_cast<int64_t>(rel) == *linear;
	}
	if (!ok) {
		std::printf("Demon's Souls camera: the free camera's turn ramp of a known build not found; the game's ramp kept "
		            "(KYTY_KBM_NO_RAMP)\n");
		if (direct)
			std::printf("Demon's Souls direct mouse camera: NOT applied (the free camera code of 01.007.000 not found / bytes "
			            "differ); the mouse drives the right stick\n");
		std::fflush(stdout);
		return;
	}
	if (direct) {
		const char* why = "";
		if (InstallDirect(program, *tilt, *linear, &why))
			std::printf("Demon's Souls direct mouse camera: applied (eboot+0x%llx full-tilt test skipped while the mouse "
			            "moves, eboot+0x%llx stick replaced by the mouse's angle / (rate * dt)): no speed cap; signs yaw %d "
			            "pitch %d (0 = learned on the first mouse motion); KYTY_KBM_DIRECT=0 turns it off\n",
			            static_cast<unsigned long long>(*tilt - program->base_vaddr),
			            static_cast<unsigned long long>(*linear - program->base_vaddr), g_yaw_sign.load(), g_pitch_sign.load());
		else
			std::printf("Demon's Souls direct mouse camera: NOT applied (%s); the mouse drives the right stick\n", why);
		std::fflush(stdout);
	}
	if (mode == 0) return;
	// jmp <ramp = 0>; nop (the 6-byte jbe).
	const int32_t rel = static_cast<int32_t>(static_cast<int64_t>(*linear) - static_cast<int64_t>(jbe + 5));
	uint8_t       jmp[6] {0xe9, 0, 0, 0, 0, 0x90};
	std::memcpy(jmp + 1, &rel, sizeof(rel));
	std::memcpy(reinterpret_cast<void*>(jbe), jmp, sizeof(jmp));
	Common::VirtualMemory::FlushInstructionCache(jbe, sizeof(jmp));
	if (mode == 1) {
		const uint64_t at = *linear + YawRateOffset;
		std::memcpy(reinterpret_cast<void*>(at), &TopYawRate, sizeof(TopYawRate));
		Common::VirtualMemory::FlushInstructionCache(at, sizeof(TopYawRate));
	}
	std::printf("Demon's Souls camera (keyboard and mouse): the free camera's full-tilt turn ramp removed (eboot+0x%llx: a "
	            "full stick no longer speeds up from the base turn rate over the camera's ramp time, nor stops dead when it "
	            "leaves full tilt); the yaw is linear in the stick at the game's %s turn rate (camera data +0x%x), pitch "
	            "at its base rate (KYTY_KBM_NO_RAMP=%d; 0 the game's ramp, 1 top rate, 2 base rate)\n",
	            static_cast<unsigned long long>(jbe - program->base_vaddr), mode == 1 ? "top" : "base",
	            mode == 1 ? TopYawRate : BaseYawRate, mode);
	std::fflush(stdout);
#endif
}

} // namespace Loader::DemonsSoulsCamera
