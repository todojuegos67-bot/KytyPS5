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

#include <algorithm>
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

// ---- (mouse1) Native mouse camera (KYTY_DS_MOUSE_NATIVE=1 with KYTY_KBM=1; off by default) ----
// What Ghidra shows of 01.007.000 (ghidra/*.c beside the notes; cd = the camera data, ctx = the frame's context):
//   eboot+0x191d70 builds ctx (0x188640: dt at ctx+0x44, flags at ctx+0x74) and runs the follow camera's steps
//   (0x196d40). ctx+0x74 bit 0x200 = the free camera (no lock-on cd+0x260, no reset cd+0x12c, no recentre cd+0x262/3).
//   0x18bd60 reads the right stick: cd+0x2dc == 1 ? (chr input)+0x50 {pitch, yaw} / +0x58 : a zero vector. A stick
//   other than zero with bit 0x200 is "camera input": cd+0x14c = 1, ctx+0x118 = 1; ctx+0x120 = pitch, +0x124 = yaw,
//   +0x130 = its length.
//   0x18cf00 (bit 0x200 only): with ctx+0x118 the angles are re-read from the camera's position, then the linear branch
//   pitch(cd+0x130) += dt * cd+0x1ec * ctx+0x120 (wrapped, clamped to cd+0x1d8..cd+0x1d4), yaw(cd+0x134) += dt *
//   cd+0x1f0 * ctx+0x124 (wrapped): no clamp of the stick itself; a stick longer than 0.998 and mostly horizontal
//   takes the ramp branch instead (yaw at a rate that ignores the stick's size). With ctx+0x118 it ends with cd+0x140 =
//   pitch and the follow timers cd+0x120/0x124 re-armed.
//   0x18d3e0 pulls the pitch back to cd+0x140: why a pitch written at ctx+0x120 alone (11-10b, the stick itself centred,
//   ctx+0x118 = 0) "did not look up or down", and why the yaw needed the held stick (11-10c).
//   The aim camera (bow, 0x178300 -> 0x176630): aim+0x180 == 1 ? the same (chr input)+0x50 : zero; pitch(aim+0x124) +=
//   dt([args+0x10]) * pow(|stick|, *[eboot+0x27a81f0]) * stick.pitch / |stick| * (aim+0x50 / aim+0x114) * aim+0xdc, the
//   yaw(aim+0x128) alike with aim+0xe0.
// So the mouse goes in where the game reads the stick, as a stick whose value is the angle it must turn this frame:
//   follow: {pitch, yaw} = sign * counts * rad per count / (|rate| * dt)   -> the linear branch turns exactly that angle
//           (a detour on the 0.998 test sends this frame to the linear branch; a pad frame keeps the game's test);
//   aim:    the vector whose pow(length, exponent) * direction is that angle / (dt * rate) (the zoom factor is the
//           game's: KYTY_DS_MOUSE_AIM_ZOOM=0 divides it out).
// Every other consequence of "the stick moved" (cd+0x14c, ctx+0x118, cd+0x140, the follow timers) is the game's own, as
// with a pad. Nothing is clamped or dropped on our side, the angle does not depend on dt (60 or 120 fps alike), and a
// frame without counts leaves the game's stick alone (the pad works as before). For KYTY_DS_MOUSE_HOLD_MS (100) after
// the last counts a frame without them reads a 1e-6 stick, so a slow hand (fewer counts than frames) does not flip the
// game between "stick moved" and "stick released" (cd+0x144: the pivot's follow factor ctx+0x90 changes with it).
// Not the free camera (lock-on, recentre, input off) or no camera hook in 100 ms (menus): the mouse drives the emulated
// stick as before (the lock-on's target switch).
constexpr const char* FollowStick =
    "80 b8 dc 02 00 00 01 75 13 48 8b 7b 08 e8 ?? ?? ?? ?? 48 8d 48 50 48 83 c0 58 eb 0b 48 8d 0d ?? ?? ?? ?? 48 8d 41 08 "
    "48 8b 11 48 8b 00 8b 7b 74 48 8b 33 48 89 d1 41 89 d0 48 c1 e9 20 41 09 c8 41 f7 c0 ff ff ff 7f";
constexpr uint64_t FollowSiteOffset = 39; // mov rdx, [rcx]; mov rax, [rax] (6 bytes)
constexpr uint64_t FollowZeroDisp   = 31; // the disp32 of lea rcx, [rip + zero vector]
// lea rcx, [rip + zero]; lea rax, [rcx + 8]; mov rcx, [rcx]; mov rax, [rax]; vmovd xmm0, ecx; shr rcx, 32; vpinsrd x3.
constexpr const char* AimStick = "48 8d 0d ?? ?? ?? ?? 48 8d 41 08 48 8b 09 48 8b 00 c5 f9 6e c1 48 c1 e9 20 c4 e3 79 22 c1 01 "
                                 "c4 e3 79 22 c0 02 48 c1 e8 20 c4 e3 79 22 d0 03";
constexpr uint64_t AimSiteOffset = 11;   // mov rcx, [rcx]; mov rax, [rax] (6 bytes)
constexpr uint64_t AimHeadBack   = 0x66; // cmp byte [rbx + 0x180], 1; jne <zero>; mov rdi, [r14]; call; lea; add; jmp <site>
constexpr uint64_t AimExpoOffset = 0xa1; // mov rcx, [rip + exponent's address]; ... vmovss xmm1, [rcx] at +0xbe
constexpr uint8_t  AimHead1[] {0x80, 0xbb, 0x80, 0x01, 0x00, 0x00, 0x01, 0x75, 0x5d, 0x49, 0x8b, 0x3e, 0xe8};
constexpr uint8_t  AimHead2[] {0x48, 0x8d, 0x48, 0x50, 0x48, 0x83, 0xc0, 0x58, 0xeb, 0x56};
constexpr double   RadPerDeg = 3.14159265358979323846 / 180.0;

struct NativeState {
	std::atomic<bool>     installed {false};
	std::atomic<bool>     aim {false};
	std::atomic<int64_t>  nx {0}, ny {0};           // counts pending (the window's thread adds, the hooks take)
	std::atomic<int64_t>  free_ns {0};              // the last frame of a camera that takes the mouse (0: none)
	std::atomic<int64_t>  mouse_ns {0};             // the last frame turned by the mouse
	std::atomic<int64_t>  rx {0}, ry {0};           // stats: counts received (absolute)
	std::atomic<int64_t>  routed {0};               // stats: counts that went to the emulated stick
	std::atomic<uint8_t>* linear_flag   = nullptr;  // in the cave: this frame's stick is the mouse's (linear branch)
	uint64_t              aim_expo_slot = 0;        // where the eboot keeps the address of the aim curve's exponent
	double                pitch_scale   = 1.0;      // KYTY_DS_MOUSE_PITCH_SCALE, negative with KYTY_DS_MOUSE_INVERT_Y=1
	int64_t               hold_ns       = 100000000;
	bool                  aim_zoom      = true;     // KYTY_DS_MOUSE_AIM_ZOOM
	bool                  stats         = false;    // KYTY_DS_MOUSE_STATS=1
	// The camera's thread only.
	int     kept_frames = 0;
	int64_t log_ns = 0;
	int64_t applied = 0, lost_not_free = 0, lost_aim = 0, lost_stale = 0, kept = 0, most = 0;
	int     frames = 0, frames_free = 0, frames_aim = 0, frames_mouse = 0, frames_held = 0;
	double  deg_yaw = 0.0, deg_yaw_net = 0.0, deg_pitch = 0.0, deg_pitch_net = 0.0, deg_pitch_clipped = 0.0;
	float   last_dt = 0.0f, last_ry = 0.0f, last_rp = 0.0f, last_zoom = 0.0f, last_expo = 0.0f;
	uint32_t last_yaw_off = 0;
	// (cam1) KYTY_CAM1=1 (implies the native mouse camera; off by default).
	bool    cam1 = false, cam1_comp = false;              // (raton2) COMP off by default: KYTY_CAM1_COMP=1 turns it on
	std::atomic<double> sens_x {1.0}, sens_y {1.0};    // KYTY_CAM1_SENS_X / _Y (raton2: changed live, read each frame)
	double  carry_y = 0.0, carry_p = 0.0;             // game radians the smoothing could not reach this frame
	bool    have_prev = false, prev_mouse = false;    // the last free frame's rendered angles / was it a mouse frame
	double  prev_ry = 0.0, prev_rp = 0.0;             // rendered yaw / pitch (atan2 of cd+0xd0 - cd+0x100)
	double  ask_y = 0.0, ask_p = 0.0;                 // the angle asked in the last mouse frame (game radians, signed)
	double  m_ask_y = 0.0, m_got_y = 0.0, m_ask_p = 0.0, m_got_p = 0.0; // 5 s sums: asked vs rendered next frame
	double  m_k = 0.0;                                // the last smoothing factor read
	int     m_pairs = 0, m_comp = 0, m_capped = 0;
};
NativeState g_native;

// (cam1) What Ghidra shows (01.007.000, base 0; disassembly read again for cam1):
//   0x18cf00+0x8c: with camera input (ctx+0x118 == 1) the pitch / yaw are RE-READ from the vector cd+0xd0 - cd+0x100
//     (0x182e40: -atan2(v.y, |v.xz|), 0x182ea0: atan2(v.x, v.z)) before the stick's step is added.
//   0x18f0d0: cd+0x100 (the camera drawn) moves towards the target cd+0xf0 by a FIXED per-frame factor k = cd+0x194
//     (cd+0x1b0 with ctx+0x74 bit 0x800), no dt, the distance blended by cd+0x1a0/0x1b4; no smoothing with cd+0x12c != 0
//     or ctx+0x74 bit 8.
// So in a frame with camera input the drawn camera turns k x the angle asked (target = drawn + asked, then a lerp of k):
// the rest is not delayed, it is lost (the next frame starts again from the drawn camera). A pad's rates were tuned with
// that; a mouse asks an exact angle. KYTY_CAM1 asks the angle theta whose chord lerp by k turns the drawn camera by
// exactly phi: theta = phi + asin(sin(phi) (1 - k) / k) (the drawn point k of the way from the drawn to the target
// direction, back on the sphere). theta is capped at 120 deg; what is beyond goes to the next frame (carry).
constexpr double Cam1MaxTheta = 2.0943951023931957; // 120 deg

double Cam1Theta(double phi, double k, double* carry) {
	*carry = 0.0;
	if (!(k > 1e-3) || k >= 0.999 || phi == 0.0) return phi;
	const double a    = std::fabs(phi);
	const double sgn  = phi < 0 ? -1.0 : 1.0;
	const double acap = std::atan2(k * std::sin(Cam1MaxTheta), 1.0 - k + k * std::cos(Cam1MaxTheta));
	if (a >= acap) {
		*carry = sgn * (a - acap);
		return sgn * Cam1MaxTheta;
	}
	const double s = std::min(1.0, std::sin(a) * (1.0 - k) / k);
	return sgn * std::min(Cam1MaxTheta, a + std::asin(s));
}

double WrapPi(double a) {
	while (a > 3.14159265358979323846) a -= 2.0 * 3.14159265358979323846;
	while (a < -3.14159265358979323846) a += 2.0 * 3.14159265358979323846;
	return a;
}

double EnvDouble(const char* name, double def, double lo, double hi) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0') return def;
	const double d = std::atof(v);
	return std::isfinite(d) && d >= lo && d <= hi ? d : def;
}

// The stick lanes' signs, from the game's own stick while the mouse still drives the emulated one (as the direct camera).
void NativeLearnSigns(float game_pitch, float game_yaw) {
	if (g_yaw_sign.load(std::memory_order_relaxed) == 0) {
		const int sx = g_stick_x.load(std::memory_order_relaxed) - 128;
		if (std::abs(sx) >= 40 && std::fabs(game_yaw) > 0.15f) {
			g_yaw_sign = (game_yaw > 0) == (sx > 0) ? 1 : -1;
			std::printf("Demon's Souls native mouse camera: yaw sign learned %d (stick x %d, game yaw %.2f)\n", g_yaw_sign.load(),
			            sx, game_yaw);
			std::fflush(stdout);
		}
	}
	if (g_pitch_sign.load(std::memory_order_relaxed) == 0) {
		const int sy = g_stick_y.load(std::memory_order_relaxed) - 128;
		if (std::abs(sy) >= 40 && std::fabs(game_pitch) > 0.15f) {
			g_pitch_sign = (game_pitch > 0) == (sy > 0) ? 1 : -1;
			std::printf("Demon's Souls native mouse camera: pitch sign learned %d (stick y %d, game pitch %.2f)\n",
			            g_pitch_sign.load(), sy, game_pitch);
			std::fflush(stdout);
		}
	}
}

// KYTY_DS_MOUSE_STATS=1: every 5 s, the counts' account (received = applied + lost + still pending) and the angle.
void NativeStats(int64_t now) {
	auto& n = g_native;
	if (!n.stats) return;
	if (n.log_ns == 0) n.log_ns = now;
	if (now - n.log_ns < 5000000000LL) return;
	const int64_t rx = n.rx.exchange(0, std::memory_order_relaxed), ry = n.ry.exchange(0, std::memory_order_relaxed);
	const int64_t pending = std::llabs(n.nx.load(std::memory_order_relaxed)) + std::llabs(n.ny.load(std::memory_order_relaxed));
	const int64_t lost    = n.lost_not_free + n.lost_aim + n.lost_stale;
	std::printf("Demon's Souls native mouse stats (%.1f s): counts received %lld (x %lld, y %lld), applied %lld, lost %lld "
	            "(camera not free %lld, aim %lld, stale %lld), kept for a later frame %lld, still pending %lld, routed to the "
	            "emulated stick %lld; angle applied: yaw %.1f deg (net %+.1f), pitch %.1f deg (net %+.1f; about %.1f deg asked "
	            "beyond the camera's pitch limit); camera frames %d (free %d, aim %d), turned by the mouse %d, held %d; most "
	            "counts in one frame %lld; %.4f deg per count; follow: dt %.4f, yaw rate %.3f rad/s (cd+0x%x), pitch rate "
	            "%.3f rad/s; aim: zoom factor %.3f, curve exponent %.3f; signs yaw %d pitch %d\n",
	            static_cast<double>(now - n.log_ns) * 1e-9, static_cast<long long>(rx + ry), static_cast<long long>(rx),
	            static_cast<long long>(ry), static_cast<long long>(n.applied), static_cast<long long>(lost),
	            static_cast<long long>(n.lost_not_free), static_cast<long long>(n.lost_aim), static_cast<long long>(n.lost_stale),
	            static_cast<long long>(n.kept), static_cast<long long>(pending),
	            static_cast<long long>(n.routed.exchange(0, std::memory_order_relaxed)), n.deg_yaw, n.deg_yaw_net, n.deg_pitch,
	            n.deg_pitch_net, n.deg_pitch_clipped, n.frames, n.frames_free, n.frames_aim, n.frames_mouse, n.frames_held,
	            static_cast<long long>(n.most), static_cast<double>(g_dpc.load(std::memory_order_relaxed)),
	            static_cast<double>(n.last_dt), static_cast<double>(n.last_ry), n.last_yaw_off, static_cast<double>(n.last_rp),
	            static_cast<double>(n.last_zoom), static_cast<double>(n.last_expo), g_yaw_sign.load(), g_pitch_sign.load());
	if (n.cam1)
		std::printf("Demon's Souls cam1 (5 s): smoothing compensation %s, k %.3f (cd+0x194/0x1b0, per frame), %d frames "
		            "compensated (%d capped, carried), sens x %.2f y %.2f; drawn camera the frame after a mouse frame: yaw asked "
		            "%.1f deg got %.1f deg (%.2f), pitch asked %.1f deg got %.1f deg (%.2f) over %d frames (stand still: 1.00 = "
		            "1:1)\n",
		            n.cam1_comp ? "ON" : "OFF (KYTY_CAM1_COMP=0)", n.m_k, n.m_comp, n.m_capped, n.sens_x.load(), n.sens_y.load(),
		            n.m_ask_y / RadPerDeg, n.m_got_y / RadPerDeg, n.m_ask_y > 0 ? n.m_got_y / n.m_ask_y : 0.0, n.m_ask_p / RadPerDeg,
		            n.m_got_p / RadPerDeg, n.m_ask_p > 0 ? n.m_got_p / n.m_ask_p : 0.0, n.m_pairs);
	n.m_ask_y = n.m_got_y = n.m_ask_p = n.m_got_p = 0.0;
	n.m_pairs = n.m_comp = n.m_capped = 0;
	std::fflush(stdout);
	n.log_ns  = now;
	n.applied = n.lost_not_free = n.lost_aim = n.lost_stale = n.kept = n.most = 0;
	n.frames = n.frames_free = n.frames_aim = n.frames_mouse = n.frames_held = 0;
	n.deg_yaw = n.deg_yaw_net = n.deg_pitch = n.deg_pitch_net = n.deg_pitch_clipped = 0.0;
}

// The counts pending, taken; false if there are none.
bool NativeTake(int64_t* dx, int64_t* dy) {
	*dx = g_native.nx.exchange(0, std::memory_order_acq_rel);
	*dy = g_native.ny.exchange(0, std::memory_order_acq_rel);
	return *dx != 0 || *dy != 0;
}

// The follow camera's stick read (eboot+0x18bd93): ctx = the frame's context, xy = the stick it just read {pitch, yaw}.
void KYTY_SYSV_ABI NativeFollowHook(uint8_t* ctx, uint64_t* xy) {
	auto&         n   = g_native;
	const int64_t now = NowNs();
	if (n.linear_flag != nullptr) n.linear_flag->store(0, std::memory_order_relaxed);
	uint8_t* cd = nullptr;
	std::memcpy(&cd, ctx, sizeof(cd));
	if (cd == nullptr) return;
	uint32_t flags = 0;
	std::memcpy(&flags, ctx + 0x74, sizeof(flags));
	const bool input_on = cd[0x2dc] == 1;
	float      game[2] {0.0f, 0.0f};
	std::memcpy(game, xy, sizeof(game));
	if (input_on) NativeLearnSigns(game[0], game[1]);
	n.frames++;
	int64_t dx = 0, dy = 0;
	if (!input_on || (flags & 0x200) == 0) {
		// Lock-on, recentre, reset, input off: the game does not turn this camera by the stick. The mouse goes back to
		// the emulated stick at once; what arrived since the last frame has no camera to turn.
		n.free_ns.store(0, std::memory_order_relaxed);
		if (NativeTake(&dx, &dy)) n.lost_not_free += std::llabs(dx) + std::llabs(dy);
		n.carry_y = n.carry_p = 0.0, n.have_prev = n.prev_mouse = false;
		NativeStats(now);
		return;
	}
	n.frames_free++;
	n.free_ns.store(now, std::memory_order_relaxed);
	if (n.cam1) {
		// The drawn camera's angles now (the last frame's result): measure what the last mouse frame turned.
		float v[4] {0, 0, 0, 0}, p[4] {0, 0, 0, 0};
		std::memcpy(v, cd + 0xd0, sizeof(v));
		std::memcpy(p, cd + 0x100, sizeof(p));
		const double x = static_cast<double>(v[0]) - p[0], y = static_cast<double>(v[1]) - p[1], z = static_cast<double>(v[2]) - p[2];
		if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && (x != 0.0 || z != 0.0)) {
			const double ryaw = std::atan2(x, z), rpitch = -std::atan2(y, std::hypot(x, z));
			if (n.have_prev && n.prev_mouse) {
				n.m_ask_y += std::fabs(n.ask_y), n.m_got_y += std::fabs(WrapPi(ryaw - n.prev_ry));
				n.m_ask_p += std::fabs(n.ask_p), n.m_got_p += std::fabs(rpitch - n.prev_rp);
				n.m_pairs++;
			}
			n.have_prev = true, n.prev_ry = ryaw, n.prev_rp = rpitch;
		} else {
			n.have_prev = false;
		}
		n.prev_mouse = false;
	}
	if (!NativeTake(&dx, &dy) && !(n.cam1 && (n.carry_y != 0.0 || n.carry_p != 0.0))) {
		// No counts this frame. A pad's stick is left as read; an idle stick shortly after the mouse stays "moved".
		if ((*xy & 0x7fffffff7fffffffULL) == 0 && n.hold_ns > 0 && now - n.mouse_ns.load(std::memory_order_relaxed) < n.hold_ns) {
			const float held[2] {0.0f, 1e-6f};
			std::memcpy(xy, held, sizeof(held));
			n.frames_held++;
		}
		NativeStats(now);
		return;
	}
	uint32_t yaw_off = BaseYawRate;
	std::memcpy(&yaw_off, reinterpret_cast<const void*>(g_linear + YawRateOffset), sizeof(yaw_off));
	const float dt = F(ctx, 0x44), rp = std::fabs(F(cd, 0x1ec)), ry = std::fabs(F(cd, yaw_off));
	const int   ys = g_yaw_sign.load(std::memory_order_relaxed), ps = g_pitch_sign.load(std::memory_order_relaxed);
	if (ys == 0 || ps == 0 || !std::isfinite(dt) || !std::isfinite(rp) || !std::isfinite(ry) || !(dt > 1e-4f) || !(rp > 1e-4f) ||
	    !(ry > 1e-4f)) {
		// Not convertible this frame (no dt, no rate yet): kept for the next one, a few frames at most (no late burst).
		if (++n.kept_frames <= 8) {
			n.nx.fetch_add(dx, std::memory_order_acq_rel);
			n.ny.fetch_add(dy, std::memory_order_acq_rel);
			n.kept += std::llabs(dx) + std::llabs(dy);
		} else {
			n.lost_stale += std::llabs(dx) + std::llabs(dy);
		}
		NativeStats(now);
		return;
	}
	n.kept_frames = 0;
	const double dpc   = static_cast<double>(g_dpc.load(std::memory_order_relaxed));
	const double deg_y = static_cast<double>(dx) * dpc * n.sens_x.load(std::memory_order_relaxed),
	             deg_p = static_cast<double>(dy) * dpc * n.pitch_scale * n.sens_y.load(std::memory_order_relaxed);
	// Game radians to turn the drawn camera by (signed as the game's stick lanes).
	double ang_p = ps * deg_p * RadPerDeg, ang_y = ys * deg_y * RadPerDeg;
	if (n.cam1) {
		ang_y += n.carry_y, ang_p += n.carry_p;
		n.carry_y = n.carry_p = 0.0;
		n.ask_y = ang_y, n.ask_p = ang_p, n.prev_mouse = true;
		uint32_t flags74 = 0;
		std::memcpy(&flags74, ctx + 0x74, sizeof(flags74));
		const bool  smoothed = cd[0x12c] == 0 && (flags74 & 8) == 0;
		const float k        = F(cd, (flags74 & 0x800) != 0 ? 0x1b0 : 0x194);
		n.m_k                = k;
		if (n.cam1_comp && smoothed && std::isfinite(k) && k > 1e-3f && k < 0.999f) {
			double cy = 0.0, cp = 0.0;
			ang_y = Cam1Theta(ang_y, k, &cy);
			ang_p = Cam1Theta(ang_p, k, &cp);
			n.carry_y = std::clamp(cy, -0.5236, 0.5236), n.carry_p = std::clamp(cp, -0.5236, 0.5236); // (30 deg at most)
			n.m_comp++;
			if (cy != 0.0 || cp != 0.0) n.m_capped++;
		}
	}
	const float  stick[2] {static_cast<float>(ang_p / (static_cast<double>(rp) * dt)),
	                       static_cast<float>(ang_y / (static_cast<double>(ry) * dt))};
	std::memcpy(xy, stick, sizeof(stick));
	if (n.linear_flag != nullptr) n.linear_flag->store(1, std::memory_order_relaxed);
	n.mouse_ns.store(now, std::memory_order_relaxed);
	if (n.stats) {
		const int64_t c = std::llabs(dx) + std::llabs(dy);
		n.applied += c;
		if (c > n.most) n.most = c;
		n.frames_mouse++;
		n.deg_yaw += std::fabs(deg_y), n.deg_yaw_net += deg_y, n.deg_pitch += std::fabs(deg_p), n.deg_pitch_net += deg_p;
		// The game clamps the pitch to cd+0x1d8..cd+0x1d4 (looking straight up or down): what was asked beyond it.
		const double p1 = static_cast<double>(F(cd, 0x130)) + static_cast<double>(dt) * F(cd, 0x1ec) * stick[0];
		const double lo = F(cd, 0x1d8), hi = F(cd, 0x1d4);
		if (std::isfinite(p1) && std::isfinite(lo) && std::isfinite(hi) && lo < hi) {
			if (p1 > hi) n.deg_pitch_clipped += (p1 - hi) / RadPerDeg;
			if (p1 < lo) n.deg_pitch_clipped += (lo - p1) / RadPerDeg;
		}
		n.last_dt = dt, n.last_ry = ry, n.last_rp = rp, n.last_yaw_off = yaw_off;
	}
	NativeStats(now);
}

// The aim camera's stick read (eboot+0x1766f7): aim = the camera, args = its frame arguments (dt at +0x10), xy / zw =
// the stick it just read ({pitch, yaw} and the two lanes that only count in its length).
void KYTY_SYSV_ABI NativeAimHook(const uint8_t* aim, const uint8_t* args, uint64_t* xy, uint64_t* zw) {
	auto&         n   = g_native;
	const int64_t now = NowNs();
	const bool    input_on = aim[0x180] == 1;
	float         game[2] {0.0f, 0.0f};
	std::memcpy(game, xy, sizeof(game));
	if (input_on) NativeLearnSigns(game[0], game[1]);
	n.frames++;
	const float* expo_at = nullptr;
	std::memcpy(&expo_at, reinterpret_cast<const void*>(n.aim_expo_slot), sizeof(expo_at));
	float expo = 0.0f;
	if (expo_at != nullptr) std::memcpy(&expo, expo_at, sizeof(expo));
	const float dt = F(args, 0x10), zoom = F(aim, 0x50) / F(aim, 0x114), rp = std::fabs(F(aim, 0xdc)), ry = std::fabs(F(aim, 0xe0));
	const int   ys = g_yaw_sign.load(std::memory_order_relaxed), ps = g_pitch_sign.load(std::memory_order_relaxed);
	const bool  ok = input_on && ys != 0 && ps != 0 && std::isfinite(expo) && expo > 0.05f && expo < 20.0f && std::isfinite(dt) &&
	                 dt > 1e-4f && std::isfinite(rp) && rp > 1e-4f && std::isfinite(ry) && ry > 1e-4f && std::isfinite(zoom) &&
	                 zoom > 1e-3f;
	int64_t dx = 0, dy = 0;
	if (!ok) {
		// Input off, or values this hook does not understand: the mouse drives the emulated stick, as before.
		n.free_ns.store(0, std::memory_order_relaxed);
		if (NativeTake(&dx, &dy)) n.lost_aim += std::llabs(dx) + std::llabs(dy);
		NativeStats(now);
		return;
	}
	n.frames_aim++;
	n.free_ns.store(now, std::memory_order_relaxed);
	if (!NativeTake(&dx, &dy)) {
		NativeStats(now);
		return;
	}
	const double dpc   = static_cast<double>(g_dpc.load(std::memory_order_relaxed));
	const double deg_y = static_cast<double>(dx) * dpc * n.sens_x.load(std::memory_order_relaxed),
	             deg_p = static_cast<double>(dy) * dpc * n.pitch_scale * n.sens_y.load(std::memory_order_relaxed);
	const double unzoom = n.aim_zoom ? 1.0 : 1.0 / static_cast<double>(zoom);
	const double a = ps * deg_p * RadPerDeg * unzoom / (static_cast<double>(dt) * rp);
	const double b = ys * deg_y * RadPerDeg * unzoom / (static_cast<double>(dt) * ry);
	const double m = std::hypot(a, b);
	if (!(m > 0.0) || !std::isfinite(m)) {
		n.lost_aim += std::llabs(dx) + std::llabs(dy);
		NativeStats(now);
		return;
	}
	// The game turns by pow(length, exponent) along the stick's direction: the length whose power is m.
	const double len = std::pow(m, 1.0 / static_cast<double>(expo));
	const float  stick[2] {static_cast<float>(a / m * len), static_cast<float>(b / m * len)};
	std::memcpy(xy, stick, sizeof(stick));
	*zw = 0;
	n.mouse_ns.store(now, std::memory_order_relaxed);
	if (n.stats) {
		const int64_t c = std::llabs(dx) + std::llabs(dy);
		n.applied += c;
		if (c > n.most) n.most = c;
		n.frames_mouse++;
		const double z = n.aim_zoom ? static_cast<double>(zoom) : 1.0;
		n.deg_yaw += std::fabs(deg_y) * z, n.deg_yaw_net += deg_y * z, n.deg_pitch += std::fabs(deg_p) * z, n.deg_pitch_net += deg_p * z;
		n.last_zoom = zoom, n.last_expo = expo;
	}
	NativeStats(now);
}

// The three detours (one cave, the direct camera's address: the two are never installed together). False, nothing
// patched, if the bytes differ or the cave cannot be made. *aim_at: the aim camera's site, 0 if only the follow camera.
bool InstallNative(Program* program, uint64_t tilt, uint64_t linear, uint64_t* follow_at, uint64_t* aim_at, const char** why) {
	auto& n     = g_native;
	*follow_at = *aim_at = 0;
	const auto follow = GuestCode::FindUnique(*program, GuestCode::Pattern(FollowStick));
	if (!follow.has_value()) {
		*why = "the follow camera's stick read of 01.007.000 (eboot+0x18bd6c) not found / bytes differ";
		return false;
	}
	{
		// The other source of that read is the eboot's zero vector.
		int32_t disp = 0;
		std::memcpy(&disp, reinterpret_cast<const void*>(*follow + FollowZeroDisp), sizeof(disp));
		const uint64_t zero = *follow + FollowZeroDisp + 4 + static_cast<int64_t>(disp);
		uint64_t       v[2] {0, 0};
		if (zero >= program->base_vaddr && zero + sizeof(v) <= program->base_vaddr + program->base_size)
			std::memcpy(v, reinterpret_cast<const void*>(zero), sizeof(v));
		if (v[0] != 0 || v[1] != 0) {
			*why = "the follow camera's stick read: its zero vector is not zero (another build?)";
			return false;
		}
	}
	const uint64_t site_f = *follow + FollowSiteOffset;
	uint64_t       site_a = 0;
	if (const auto aim = GuestCode::FindUnique(*program, GuestCode::Pattern(AimStick)); aim.has_value()) {
		// The block before it: "input on" (aim+0x180) -> (chr input)+0x50, else the zero vector; and the curve's exponent.
		const uint8_t* head = reinterpret_cast<const uint8_t*>(*aim - AimHeadBack);
		const uint8_t* expo = reinterpret_cast<const uint8_t*>(*aim + AimExpoOffset);
		if (std::memcmp(head, AimHead1, sizeof(AimHead1)) == 0 && std::memcmp(head + 17, AimHead2, sizeof(AimHead2)) == 0 &&
		    expo[0] == 0x48 && expo[1] == 0x8b && expo[2] == 0x0d && expo[0x1d] == 0xc5 && expo[0x1e] == 0xfa && expo[0x1f] == 0x10 &&
		    expo[0x20] == 0x09) {
			int32_t disp = 0;
			std::memcpy(&disp, expo + 3, sizeof(disp));
			n.aim_expo_slot = *aim + AimExpoOffset + 7 + static_cast<int64_t>(disp);
			site_a          = *aim + AimSiteOffset;
		}
	}
	const uint64_t requested = program->base_vaddr + CaveOffset;
	const uint64_t cave      = Libs::LibKernel::Memory::AllocateRuntimeMemory(
        requested, CavePage, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_kbm_camera", true);
	const auto in_reach = [&](uint64_t a) {
		const int64_t d = static_cast<int64_t>(a) - static_cast<int64_t>(cave);
		return d > -0x70000000LL && d < 0x70000000LL;
	};
	if (cave == 0 || !in_reach(tilt) || !in_reach(linear) || !in_reach(site_f) || (site_a != 0 && !in_reach(site_a))) {
		if (cave != 0) Libs::LibKernel::Memory::FreeGuestMemory(cave, CavePage);
		*why = "no executable memory near the eboot";
		return false;
	}
	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator c(CavePage, reinterpret_cast<void*>(cave));
	Xbyak::Label         flag;
	constexpr uint8_t    save_fp[] {0x48, 0x0f, 0xae, 0x04, 0x24}; // FXSAVE64 [rsp]
	// Registers, flags and FP state kept, below the red zone, on a realigned stack. After the pushes: [rsp] rbx, +8 r11,
	// +16 r10, +24 r9, +32 r8, +40 rdi, +48 rsi, +56 rdx, +64 rcx, +72 rax.
	const auto save = [&] {
		c.lea(rsp, ptr[rsp - 128]);
		c.pushfq();
		for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, rbx})
			c.push(reg);
	};
	const auto call_and_restore = [&](uint64_t host) {
		c.mov(rbx, rsp);
		c.and_(rsp, -16);
		c.sub(rsp, 512);
		c.db(save_fp, sizeof(save_fp));
		c.mov(rax, host);
		c.call(rax);
		c.fxrstor64(ptr[rsp]);
		c.mov(rsp, rbx);
		for (const auto& reg: {rbx, r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
			c.pop(reg);
		c.popfq();
		c.lea(rsp, ptr[rsp + 128]);
	};
	// The 0.998 full-tilt test: the original vmovss xmm0, [rbx+0x130]; a mouse frame -> the linear branch, else the test.
	const uint8_t* stub_t = c.getCurr();
	c.db(reinterpret_cast<const uint8_t*>(tilt), 8);
	c.cmp(byte[rip + flag], 0); // (flags are dead: the test sets them, the linear branch reads none)
	c.jne(reinterpret_cast<const void*>(linear));
	c.jmp(reinterpret_cast<const void*>(tilt + 8), Xbyak::CodeGenerator::T_NEAR);
	// The follow camera's stick read: the original two loads (rdx = {pitch, yaw}), then the hook may replace rdx.
	c.align(16);
	const uint8_t* stub_f = c.getCurr();
	c.db(reinterpret_cast<const uint8_t*>(site_f), 6);
	save();
	c.mov(rdi, rbx);           // ctx
	c.lea(rsi, ptr[rsp + 56]); // the saved rdx
	call_and_restore(reinterpret_cast<uint64_t>(&NativeFollowHook));
	c.jmp(reinterpret_cast<const void*>(site_f + 6), Xbyak::CodeGenerator::T_NEAR);
	// The aim camera's: the original two loads (rcx = {pitch, yaw}, rax = the other two lanes).
	c.align(16);
	const uint8_t* stub_a = c.getCurr();
	if (site_a != 0) {
		c.db(reinterpret_cast<const uint8_t*>(site_a), 6);
		save();
		c.mov(rdi, rbx);           // the aim camera
		c.mov(rsi, r14);           // its frame arguments
		c.lea(rdx, ptr[rsp + 64]); // the saved rcx
		c.lea(rcx, ptr[rsp + 72]); // the saved rax
		call_and_restore(reinterpret_cast<uint64_t>(&NativeAimHook));
		c.jmp(reinterpret_cast<const void*>(site_a + 6), Xbyak::CodeGenerator::T_NEAR);
	}
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
	n.linear_flag = reinterpret_cast<std::atomic<uint8_t>*>(const_cast<uint8_t*>(flag_at)); // (the page is RWX)
	n.linear_flag->store(0);
	g_linear      = linear;
	n.pitch_scale = EnvDouble("KYTY_DS_MOUSE_PITCH_SCALE", 1.0, 0.1, 4.0) * (Is1("KYTY_DS_MOUSE_INVERT_Y") ? -1.0 : 1.0);
	n.hold_ns     = static_cast<int64_t>(EnvDouble("KYTY_DS_MOUSE_HOLD_MS", 100.0, 0.0, 1000.0) * 1e6);
	if (const char* z = std::getenv("KYTY_DS_MOUSE_AIM_ZOOM"); z != nullptr && z[0] == '0') n.aim_zoom = false;
	n.cam1       = Is1("KYTY_CAM1");
	n.cam1_comp  = Is1("KYTY_CAM1_COMP"); // (raton2) off unless KYTY_CAM1_COMP=1 (on, it multiplied the turn by ~10)
	n.sens_x.store(n.cam1 ? EnvDouble("KYTY_CAM1_SENS_X", 1.0, 0.05, 20.0) : 1.0);
	n.sens_y.store(n.cam1 ? EnvDouble("KYTY_CAM1_SENS_Y", 1.0, 0.05, 20.0) : 1.0);
	n.stats      = Is1("KYTY_DS_MOUSE_STATS") || n.cam1;
	g_yaw_sign   = EnvSign("KYTY_KBM_YAW_SIGN");
	g_pitch_sign = EnvSign("KYTY_KBM_PITCH_SIGN");
	const auto jmp = [](uint64_t at, const uint8_t* to, size_t size) {
		uint8_t       j[8] {0xe9, 0, 0, 0, 0, 0x90, 0x90, 0x90};
		const int32_t rel = static_cast<int32_t>(reinterpret_cast<int64_t>(to) - static_cast<int64_t>(at + 5));
		std::memcpy(j + 1, &rel, sizeof(rel));
		std::memcpy(reinterpret_cast<void*>(at), j, size);
		Common::VirtualMemory::FlushInstructionCache(at, size);
	};
	jmp(site_f, stub_f, 6);
	if (site_a != 0) jmp(site_a, stub_a, 6);
	jmp(tilt, stub_t, 8);
	n.aim.store(site_a != 0);
	n.installed.store(true);
	*follow_at = site_f;
	*aim_at    = site_a;
	return true;
}

} // namespace

// (mouse1) Read at each call (the port sets its environment before the emulator starts; nothing is cached here).
bool NativeWanted() {
	return (Is1("KYTY_DS_MOUSE_NATIVE") || Is1("KYTY_CAM1")) && Is1("KYTY_KBM") && !Is1("KYTY_KBM_CALIBRATE");
}

bool NativeInstalled() {
	return g_native.installed.load(std::memory_order_relaxed);
}

bool NativeTakesMouse() {
	if (!g_native.installed.load(std::memory_order_relaxed) || g_yaw_sign.load(std::memory_order_relaxed) == 0 ||
	    g_pitch_sign.load(std::memory_order_relaxed) == 0)
		return false;
	const int64_t seen = g_native.free_ns.load(std::memory_order_relaxed);
	return seen != 0 && NowNs() - seen < 100000000;
}

void NativeAddMouse(int dx, int dy) {
	g_native.nx.fetch_add(dx, std::memory_order_acq_rel);
	g_native.ny.fetch_add(dy, std::memory_order_acq_rel);
	g_native.rx.fetch_add(dx < 0 ? -static_cast<int64_t>(dx) : dx, std::memory_order_relaxed);
	g_native.ry.fetch_add(dy < 0 ? -static_cast<int64_t>(dy) : dy, std::memory_order_relaxed);
}

void NativeNoteStickRouted(int64_t counts) {
	if (g_native.installed.load(std::memory_order_relaxed)) g_native.routed.fetch_add(counts, std::memory_order_relaxed);
}

bool DirectWanted() {
	static const bool wanted = [] {
		if (!Is1("KYTY_KBM") || Is1("KYTY_KBM_CALIBRATE")) return false;
		if (NativeWanted()) return false; // (mouse1) the native mouse camera replaces it
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

// (raton2) The native camera's per-axis multipliers, changed live (settings.ini raton_sens_x / raton_sens_y).
bool Cam1Active() {
	return g_native.cam1;
}

void Cam1SetSens(double x, double y) {
	if (std::isfinite(x) && x > 0.0) g_native.sens_x.store(std::clamp(x, 0.05, 20.0), std::memory_order_relaxed);
	if (std::isfinite(y) && y > 0.0) g_native.sens_y.store(std::clamp(y, 0.05, 20.0), std::memory_order_relaxed);
}

void Cam1GetSens(double* x, double* y) {
	*x = g_native.sens_x.load(std::memory_order_relaxed);
	*y = g_native.sens_y.load(std::memory_order_relaxed);
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
	const bool native = NativeWanted(); // (mouse1) KYTY_DS_MOUSE_NATIVE=1
	const bool direct = DirectWanted();
	if (Is1("KYTY_DS_MOUSE_NATIVE") && !native)
		std::printf("Demon's Souls native mouse camera: KYTY_DS_MOUSE_NATIVE=1 ignored (%s)\n",
		            Is1("KYTY_KBM") ? "KYTY_KBM_CALIBRATE=1" : "keyboard and mouse is off: teclado_raton / KYTY_KBM=1");
	if (Is1("KYTY_KBM") && !direct && !native)
		std::printf("Demon's Souls direct mouse camera: off (%s); the mouse drives the right stick\n",
		            Is1("KYTY_KBM_CALIBRATE") ? "KYTY_KBM_CALIBRATE=1" : "KYTY_KBM_DIRECT=0");
	// (mouse1) One line at start: the mouse mode in effect.
	const auto mode_line = [](const char* text) {
		std::printf("Demon's Souls mouse mode: %s\n", text);
		std::fflush(stdout);
	};
	if (mode == 0 && !direct && !native) {
		if (Is1("KYTY_KBM") || Is1("KYTY_KBM_CALIBRATE"))
			std::printf("Demon's Souls camera: the game's full-tilt turn ramp kept (KYTY_KBM_NO_RAMP=0)\n");
		mode_line(Is1("KYTY_KBM") || Is1("KYTY_KBM_CALIBRATE") ? "emulated right stick, both axes (no hook; KYTY_DS_MOUSE_NATIVE off)"
		                                                          : "keyboard and mouse off (pad only)");
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
		mode_line(native ? "NATIVE asked (KYTY_DS_MOUSE_NATIVE=1) but NOT applied: hook bytes DIFFER (the free camera's full-tilt "
		                   "test of 01.007.000 not found); emulated right stick, both axes"
		                 : "emulated right stick (hook bytes differ; KYTY_DS_MOUSE_NATIVE off)");
		return;
	}
	if (native) {
		const char* why    = "";
		uint64_t    follow = 0, aim = 0;
		if (InstallNative(program, *tilt, *linear, &follow, &aim, &why)) {
			char text[1024];
			std::snprintf(text, sizeof(text),
			              "NATIVE (KYTY_DS_MOUSE_NATIVE=1): hook bytes MATCH; follow camera stick read eboot+0x%llx, full-tilt "
			              "test eboot+0x%llx, aim camera %s%llx; counts -> angle at %.4f deg per count (x raton_sensibilidad, "
			              "live), no stick, no dead zone, no top rate; pitch scale %.2f%s, hold %lld ms, stats %s; signs yaw %d "
			              "pitch %d (0 = learned on the first mouse motion)",
			              static_cast<unsigned long long>(follow - program->base_vaddr),
			              static_cast<unsigned long long>(*tilt - program->base_vaddr),
			              aim != 0 ? "eboot+0x" : "NOT hooked (bytes differ: the bow keeps the stick) 0x",
			              static_cast<unsigned long long>(aim != 0 ? aim - program->base_vaddr : 0),
			              static_cast<double>(g_dpc.load()), std::fabs(g_native.pitch_scale),
			              g_native.pitch_scale < 0.0 ? " inverted (KYTY_DS_MOUSE_INVERT_Y=1)" : "",
			              static_cast<long long>(g_native.hold_ns / 1000000), g_native.stats ? "on (every 5 s)" : "off (KYTY_DS_MOUSE_STATS=1)",
			              g_yaw_sign.load(), g_pitch_sign.load());
			mode_line(text);
			if (g_native.cam1)
				std::printf("Demon's Souls cam1 (KYTY_CAM1=1): native mouse camera + the game's camera smoothing compensated "
				            "(%s), sens x %.2f y %.2f (KYTY_CAM1_SENS_X/_Y), a 'cam1' line every 5 s\n",
				            g_native.cam1_comp ? "on (KYTY_CAM1_COMP=1)" : "off", g_native.sens_x.load(), g_native.sens_y.load());
		} else {
			char text[384];
			std::snprintf(text, sizeof(text),
			              "NATIVE asked (KYTY_DS_MOUSE_NATIVE=1) but NOT applied (%s); emulated right stick, both axes", why);
			mode_line(text);
		}
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
		mode_line(DirectInstalled() ? "direct yaw hook (hook bytes match) + emulated right stick for the pitch (KYTY_DS_MOUSE_NATIVE off)"
		                            : "emulated right stick, both axes (direct hook not applied; KYTY_DS_MOUSE_NATIVE off)");
	} else if (!native) {
		mode_line("emulated right stick, both axes (turn ramp patch only; KYTY_DS_MOUSE_NATIVE off)");
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
