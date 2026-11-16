#pragma once
// (audit1) KYTY_DS_WRITE_AUDIT: the machine code of the write audit's hooks (loader/demonsSoulsWriteAudit.cpp), as
// plain functions over byte buffers so that they can be run in a test without the emulator.
//
// Every stub lives in a cave within +-2 GiB of the eboot and is reached either by a `call rel32` site of the game
// whose target it stands in for, or by a `jmp rel32` written over the first instructions of a game function. The
// host functions they call are SysV (KYTY_SYSV_ABI) and take the game's own argument registers; they may change every
// caller-saved register, so each stub keeps the ones the game still needs:
//   - at a function's entry and before the original call: the six integer argument registers, rax, r10, r11 (none
//     of the hooked functions takes floating-point or stack arguments);
//   - after the original call: rax and rdx (its result).
// The stack is 16-byte aligned at every call the stubs make (rsp = 8 mod 16 where they start: just after a call).
// Nothing below rsp is live at those points (a call was just made), so the pushes cannot hit a red zone.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace Loader::DemonsSoulsWriteAuditStubs {

// A cursor over the cave's bytes. `address` is where buffer[0] will run.
struct Emitter {
	uint8_t* buffer   = nullptr;
	size_t   capacity = 0;
	size_t   size     = 0;
	uint64_t address  = 0;
	bool     ok       = true;

	void Byte(uint8_t value) {
		if (size >= capacity) {
			ok = false;
			return;
		}
		buffer[size++] = value;
	}
	void Bytes(const uint8_t* data, size_t count) {
		for (size_t i = 0; i < count; ++i) Byte(data[i]);
	}
	void U64(uint64_t value) {
		for (int i = 0; i < 8; ++i) Byte(static_cast<uint8_t>(value >> (8 * i)));
	}
	// A rel32 operand whose instruction ends right after it.
	void Rel32(uint64_t target) {
		const uint64_t next       = address + size + 4;
		const int64_t  difference = static_cast<int64_t>(target) - static_cast<int64_t>(next);
		if (difference < INT32_MIN || difference > INT32_MAX) ok = false;
		const auto value = static_cast<uint32_t>(static_cast<int32_t>(difference));
		for (int i = 0; i < 4; ++i) Byte(static_cast<uint8_t>(value >> (8 * i)));
	}
	void CallHost(uint64_t function) { // mov rax, function; call rax
		Byte(0x48);
		Byte(0xb8);
		U64(function);
		Byte(0xff);
		Byte(0xd0);
	}
	[[nodiscard]] uint64_t Here() const { return address + size; }
	void                   Align16() {
        while ((size & 15u) != 0) Byte(0xcc);
	}
};

// The entry of a game function: host(rdi, rsi, rdx) with every argument register kept, then the instructions the
// jump replaced (`moved`, position-independent) and a jump to the rest of the function.
//   push rax, rdi, rsi, rdx, rcx, r8, r9, r10, r11   (9 pushes: rsp = 0 mod 16)
//   mov rax, host; call rax
//   pop r11, r10, r9, r8, rcx, rdx, rsi, rdi, rax
//   <moved>; jmp entry + moved_size
inline uint64_t EmitEntryStub(Emitter& e, uint64_t host, const uint8_t* moved, size_t moved_size, uint64_t entry) {
	e.Align16();
	const uint64_t                 stub      = e.Here();
	static constexpr uint8_t       Save[]    = {0x50, 0x57, 0x56, 0x52, 0x51, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53};
	static constexpr uint8_t       Restore[] = {0x41, 0x5b, 0x41, 0x5a, 0x41, 0x59, 0x41, 0x58, 0x59, 0x5a, 0x5e, 0x5f, 0x58};
	e.Bytes(Save, sizeof(Save));
	e.CallHost(host);
	e.Bytes(Restore, sizeof(Restore));
	e.Bytes(moved, moved_size);
	e.Byte(0xe9);
	e.Rel32(entry + moved_size);
	return stub;
}

// A `call original` site, the host told after the call: host(rdi as it was at the call), the call's rax/rdx kept.
//   push rdi; call original; pop rdi
//   push rax; push rdx; push rdi; mov rax, host; call rax; pop rdi; pop rdx; pop rax; ret
// (The original must take no stack arguments: its return address is the stub's and rdi sits above it.)
inline uint64_t EmitCallAfterStub(Emitter& e, uint64_t host, uint64_t original) {
	e.Align16();
	const uint64_t stub = e.Here();
	e.Byte(0x57);
	e.Byte(0xe8);
	e.Rel32(original);
	e.Byte(0x5f);
	e.Byte(0x50);
	e.Byte(0x52);
	e.Byte(0x57);
	e.CallHost(host);
	e.Byte(0x5f);
	e.Byte(0x5a);
	e.Byte(0x58);
	e.Byte(0xc3);
	return stub;
}

// A `call original` site of a three-argument function, the host told before and after:
//   push rdi; push rsi; push rdx; mov rax, before; call rax; pop rdx; pop rsi; pop rdi      before(rdi, rsi, rdx)
//   sub rsp, 8; call original; add rsp, 8
//   push rax; push rdx; sub rsp, 8; mov rax, after; call rax; add rsp, 8; pop rdx; pop rax; ret
// (The original must take its arguments in rdi, rsi, rdx only: rcx, r8-r11 are not kept across `before`.)
inline uint64_t EmitCallAroundStub(Emitter& e, uint64_t before, uint64_t after, uint64_t original) {
	e.Align16();
	const uint64_t           stub     = e.Here();
	static constexpr uint8_t Push3[]  = {0x57, 0x56, 0x52};
	static constexpr uint8_t Pop3[]   = {0x5a, 0x5e, 0x5f};
	static constexpr uint8_t SubRsp[] = {0x48, 0x83, 0xec, 0x08};
	static constexpr uint8_t AddRsp[] = {0x48, 0x83, 0xc4, 0x08};
	e.Bytes(Push3, sizeof(Push3));
	e.CallHost(before);
	e.Bytes(Pop3, sizeof(Pop3));
	e.Bytes(SubRsp, sizeof(SubRsp));
	e.Byte(0xe8);
	e.Rel32(original);
	e.Bytes(AddRsp, sizeof(AddRsp));
	e.Byte(0x50);
	e.Byte(0x52);
	e.Bytes(SubRsp, sizeof(SubRsp));
	e.CallHost(after);
	e.Bytes(AddRsp, sizeof(AddRsp));
	e.Byte(0x5a);
	e.Byte(0x58);
	e.Byte(0xc3);
	return stub;
}

// ---- (policy1) ---------------------------------------------------------------------------------------------------

// EmitCallAfterStub with the caller's frame pointer as the host's second argument: host(rdi as it was at the call,
// rbp). For a site inside a function that keeps rbp as its frame pointer (push rbp; mov rbp, rsp): [rbp + 8] is that
// function's own return address.
//   push rdi; call original; pop rdi
//   push rax; push rdx; push rdi; mov rsi, rbp; mov rax, host; call rax; pop rdi; pop rdx; pop rax; ret
inline uint64_t EmitCallAfterFrameStub(Emitter& e, uint64_t host, uint64_t original) {
	e.Align16();
	const uint64_t stub = e.Here();
	e.Byte(0x57);
	e.Byte(0xe8);
	e.Rel32(original);
	e.Byte(0x5f);
	e.Byte(0x50);
	e.Byte(0x52);
	e.Byte(0x57);
	e.Byte(0x48); // mov rsi, rbp
	e.Byte(0x89);
	e.Byte(0xee);
	e.CallHost(host);
	e.Byte(0x5f);
	e.Byte(0x5a);
	e.Byte(0x58);
	e.Byte(0xc3);
	return stub;
}

// A `call original` site of a function that may take stack arguments, the host told before and after. The original
// runs with the game's stack exactly as the game left it (its stack arguments where it expects them): only the return
// address is replaced by the stub's second half, and the game's own is kept by the host between the two calls.
//   push r9, r8, rcx, rdx, rsi, rdi                      (rsp = 8 mod 16)
//   sub rsp, 8; push rbp; lea rax, [rsp + 0x40]; push rax (rsp = 0 mod 16; rax = where the return address is)
//   mov rax, before; call rax        before(rdi, rsi, rdx, rcx, r8, r9, frame, rbp): frame[0] the game's return
//                                    address (to keep), frame[1...] the original's stack arguments
//   add rsp, 0x18; pop rdi, rsi, rdx, rcx, r8, r9
//   lea r11, [rip + back]; mov [rsp], r11; jmp original
// back:                                                   (the original returned here: rsp = 0 mod 16)
//   push rax; push rdx; mov rax, after; call rax          after(): returns the game's return address
//   mov r11, rax; pop rdx; pop rax; jmp r11
// The original must take no floating-point arguments and return in rax/rdx (xmm registers are not kept), and must not
// unwind through its caller (its return address is the cave's while it runs).
inline uint64_t EmitCallAroundStackStub(Emitter& e, uint64_t before, uint64_t after, uint64_t original) {
	e.Align16();
	const uint64_t           stub    = e.Here();
	static constexpr uint8_t Save[]  = {0x41, 0x51, 0x41, 0x50, 0x51, 0x52, 0x56, 0x57,       // push r9, r8, rcx, rdx, rsi, rdi
	                                    0x48, 0x83, 0xec, 0x08,                               // sub rsp, 8
	                                    0x55,                                                 // push rbp
	                                    0x48, 0x8d, 0x44, 0x24, 0x40,                         // lea rax, [rsp + 0x40]
	                                    0x50};                                                // push rax
	static constexpr uint8_t Rest[]  = {0x48, 0x83, 0xc4, 0x18,                               // add rsp, 0x18
	                                    0x5f, 0x5e, 0x5a, 0x59, 0x41, 0x58, 0x41, 0x59};      // pop rdi, rsi, rdx, rcx, r8, r9
	e.Bytes(Save, sizeof(Save));
	e.CallHost(before);
	e.Bytes(Rest, sizeof(Rest));
	// lea r11, [rip + disp32] (disp to `back`, patched below); mov [rsp], r11; jmp original
	e.Byte(0x4c);
	e.Byte(0x8d);
	e.Byte(0x1d);
	const size_t disp_at = e.size;
	for (int i = 0; i < 4; ++i) e.Byte(0);
	const uint64_t after_lea = e.Here();
	e.Byte(0x4c);
	e.Byte(0x89);
	e.Byte(0x1c);
	e.Byte(0x24);
	e.Byte(0xe9);
	e.Rel32(original);
	const uint64_t back = e.Here();
	if (e.ok) {
		const auto disp = static_cast<uint32_t>(back - after_lea);
		for (int i = 0; i < 4; ++i) e.buffer[disp_at + static_cast<size_t>(i)] = static_cast<uint8_t>(disp >> (8 * i));
	}
	e.Byte(0x50);
	e.Byte(0x52);
	e.CallHost(after);
	e.Byte(0x49); // mov r11, rax
	e.Byte(0x89);
	e.Byte(0xc3);
	e.Byte(0x5a);
	e.Byte(0x58);
	e.Byte(0x41); // jmp r11
	e.Byte(0xff);
	e.Byte(0xe3);
	return stub;
}

// (policy3) The entry of a game function, the host told before it runs and after it returns (with its result). As
// EmitCallAroundStackStub, at an entry instead of a call site: the function runs with the game's stack as it was (its
// stack arguments in place), only its return address replaced by the stub's second half.
//   push r9, r8, rcx, rdx, rsi, rdi; sub rsp, 8; push rbp; lea rax, [rsp + 0x40]; push rax
//   mov rax, before; call rax        before(rdi, rsi, rdx, rcx, r8, r9, frame, rbp): frame[0] the caller's return
//                                    address (to keep)
//   add rsp, 0x18; pop rdi, rsi, rdx, rcx, r8, r9
//   lea r11, [rip + back]; mov [rsp], r11; <moved>; jmp entry + moved_size
// back:
//   push rax; push rdx; mov rdi, rax; mov rax, after; call rax     after(result): returns the caller's return address
//   mov r11, rax; pop rdx; pop rax; jmp r11
// The function must not be variadic (rax is used before it runs), must take no floating-point arguments, return in
// rax/rdx, and not unwind through its caller. `moved`: position-independent whole instructions.
inline uint64_t EmitEntryAroundStub(Emitter& e, uint64_t before, uint64_t after, const uint8_t* moved, size_t moved_size, uint64_t entry) {
	e.Align16();
	const uint64_t           stub   = e.Here();
	static constexpr uint8_t Save[] = {0x41, 0x51, 0x41, 0x50, 0x51, 0x52, 0x56, 0x57, 0x48, 0x83, 0xec, 0x08, 0x55, 0x48, 0x8d, 0x44, 0x24, 0x40, 0x50};
	static constexpr uint8_t Rest[] = {0x48, 0x83, 0xc4, 0x18, 0x5f, 0x5e, 0x5a, 0x59, 0x41, 0x58, 0x41, 0x59};
	e.Bytes(Save, sizeof(Save));
	e.CallHost(before);
	e.Bytes(Rest, sizeof(Rest));
	e.Byte(0x4c); // lea r11, [rip + disp32]
	e.Byte(0x8d);
	e.Byte(0x1d);
	const size_t disp_at = e.size;
	for (int i = 0; i < 4; ++i) e.Byte(0);
	const uint64_t after_lea = e.Here();
	e.Byte(0x4c); // mov [rsp], r11
	e.Byte(0x89);
	e.Byte(0x1c);
	e.Byte(0x24);
	e.Bytes(moved, moved_size);
	e.Byte(0xe9);
	e.Rel32(entry + moved_size);
	const uint64_t back = e.Here();
	if (e.ok) {
		const auto disp = static_cast<uint32_t>(back - after_lea);
		for (int i = 0; i < 4; ++i) e.buffer[disp_at + static_cast<size_t>(i)] = static_cast<uint8_t>(disp >> (8 * i));
	}
	e.Byte(0x50);
	e.Byte(0x52);
	e.Byte(0x48); // mov rdi, rax
	e.Byte(0x89);
	e.Byte(0xc7);
	e.CallHost(after);
	e.Byte(0x49); // mov r11, rax
	e.Byte(0x89);
	e.Byte(0xc3);
	e.Byte(0x5a);
	e.Byte(0x58);
	e.Byte(0x41); // jmp r11
	e.Byte(0xff);
	e.Byte(0xe3);
	return stub;
}

// The bytes written over the game's code. `size`: 5, or more with NOPs after the instruction (an entry whose first
// instructions are longer than the jump).
inline bool EmitPatch(uint8_t* out, size_t size, uint8_t opcode, uint64_t site, uint64_t target) {
	if (size < 5) return false;
	const int64_t difference = static_cast<int64_t>(target) - static_cast<int64_t>(site + 5);
	if (difference < INT32_MIN || difference > INT32_MAX) return false;
	const auto value = static_cast<int32_t>(difference);
	out[0]           = opcode; // 0xe8 call, 0xe9 jmp
	std::memcpy(out + 1, &value, sizeof(value));
	for (size_t i = 5; i < size; ++i) out[i] = 0x90;
	return true;
}

// The target of the `call rel32` / `jmp rel32` at `site` (its 5 bytes at `code`).
inline uint64_t RelativeTarget(const uint8_t* code, uint64_t site) {
	int32_t displacement = 0;
	std::memcpy(&displacement, code + 1, sizeof(displacement));
	return site + 5 + static_cast<int64_t>(displacement);
}

} // namespace Loader::DemonsSoulsWriteAuditStubs
