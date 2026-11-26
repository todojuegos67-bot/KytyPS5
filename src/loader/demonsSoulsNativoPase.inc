// (F5 pase 1) KYTY_NATIVO_PASE_SSRCLEAR=1 / 2 (off by default; local/ds-nativo-pase.h). Included by
// demonsSoulsCopy.cpp inside its anonymous namespace, after the linear copy hook (AgcCommandBuffer, Port2::Suppress).
//
// The SSRClear pass's texture clear, 01.007.000 eboot+0x9ad560 (ctx, holder, texture): reached only from the
// dispatcher 0x9ad540 when the holder has a command buffer context (Ghidra: 0x9ad540 is called once, by SSRPreRender
// 0x9d6030 inside the "SSRClear" scope). Its entry jumps to NativoPaseClearHook, which announces the clear (the T# at
// texture + 0xbc) at the command buffer's cursor, runs the game's function whole through a trampoline, and seals the
// announcement when the function wrote only shader register packets and, last, its dispatch. The game's state and
// packets are what they are without the hook; only the announcement is added.

using NativoPaseClearFunction = KYTY_SYSV_ABI void (*)(uint64_t*, uint64_t*, uint64_t);
NativoPaseClearFunction g_nativo_pase_original   = nullptr;
uint64_t                g_nativo_pase_trampoline = 0;

constexpr uint64_t NativoPaseDispatcherOffset = 0x9ad540;
constexpr uint64_t NativoPaseClearOffset      = 0x9ad560;
// 0x9ad540..0x9ad5a0 as 01.007.000 has it: the dispatcher (mov rdx, rsi; mov rsi, rdi; mov rdi, [rdi]; test; je; jmp
// 0x9ad560 / mov rdi, [rsi + 8]; jmp), its padding, and the clear's first 0x40 bytes (to the width/height calls).
constexpr std::array<uint8_t, 96> NativoPaseCode {
    0x48, 0x89, 0xf2, 0x48, 0x89, 0xfe, 0x48, 0x8b, 0x3f, 0x48, 0x85, 0xff, 0x74, 0x05, 0xe9, 0x0d, 0x00, 0x00, 0x00, 0x48,
    0x8b, 0x7e, 0x08, 0xe9, 0x94, 0x01, 0x00, 0x00, 0xcc, 0xcc, 0xcc, 0xcc, 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
    0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x08, 0x02, 0x00, 0x00, 0x48, 0x8b, 0x0d, 0x45, 0x4d, 0xd2, 0x01, 0x48,
    0x8b, 0x01, 0x48, 0x89, 0x45, 0xd0, 0x48, 0x85, 0xd2, 0x0f, 0x84, 0x35, 0x01, 0x00, 0x00, 0x48, 0x89, 0xfb, 0x48, 0x89,
    0xd7, 0x49, 0x89, 0xd4, 0xe8, 0x77, 0x38, 0xf5, 0xff, 0x4c, 0x89, 0xe7, 0x41, 0x89, 0xc6, 0xe8};
// The 12 bytes the jump replaces at 0x9ad560: push rbp; mov rbp, rsp; push r15; push r14; push r13; push r12 (the
// linear copy's prologue: no jump lands inside them, none is position-dependent; rax is first written at +0x17).
constexpr size_t NativoPaseMoved = 12;
constexpr size_t NativoPaseClearInCode = NativoPaseClearOffset - NativoPaseDispatcherOffset; // 0x20

uint32_t* AnnounceNativoPase(AgcCommandBuffer& commands) {
	namespace H = DsNativoPase;
	uint32_t* const up   = commands.cursor_up;
	uint32_t* const down = commands.cursor_down;
	if (up == nullptr || down == nullptr || reinterpret_cast<uintptr_t>(down) <= reinterpret_cast<uintptr_t>(up))
		return nullptr;
	const uint64_t available = (reinterpret_cast<uintptr_t>(down) - reinterpret_cast<uintptr_t>(up)) / sizeof(uint32_t);
	if (available <= commands.reserved_dw || available - commands.reserved_dw < H::PacketDwords) return nullptr;
	up[0] = H::Header;
	up[1] = H::Magic0;
	up[2] = H::Magic1;
	for (uint32_t i = 3; i < H::PacketDwords; ++i) up[i] = 0;
	commands.cursor_up = up + H::PacketDwords;
	return up;
}

// The packets after the announcement, as 01.007.000 writes them (log 20261124-194508): IT_SET_SH_REG_INDIRECT (the
// program), IT_SET_SH_REG 0x240 with 8 dwords (COMPUTE_USER_DATA_0..7 = the T# itself), the dispatch of
// ((width + 7) / 8, (height + 7) / 8, 1) groups for that T#'s size, then EVENT_WRITEs only (0x407, 0x410). Sealed with
// the T# of the user data (what the GPU reads) and the dwords up to the dispatch (those =1 passes over).
// Where a refusal happened (the hook's thread prints it): the packet's dword offset in the span, its header, the span.
thread_local uint32_t g_np_at = 0, g_np_header = 0, g_np_span = 0;

bool SealNativoPase(const AgcCommandBuffer& commands, uint32_t* packet, const uint32_t* tsharp, const char** why) {
	namespace H = DsNativoPase;
	const uint32_t* const begin      = packet + H::PacketDwords;
	const auto            begin_addr = reinterpret_cast<uintptr_t>(begin);
	const auto            end_addr   = reinterpret_cast<uintptr_t>(commands.cursor_up);
	if (end_addr <= begin_addr || (end_addr - begin_addr) % sizeof(uint32_t) != 0 ||
	    (end_addr - begin_addr) / sizeof(uint32_t) > H::MaxSpanDwords)
		return *why = "the cursor is not just after its packets (another buffer?)", false;
	const auto span = static_cast<uint32_t>((end_addr - begin_addr) / sizeof(uint32_t));
	// (T# width / height: Width5 / Height5 + 1, shaderBindings.h ShaderTextureResource.)
	bool           dispatch = false;
	uint32_t       skip     = 0;
	uint32_t       user[8] {};
	bool           has_user = false;
	g_np_span = span;
	for (uint32_t at = 0; at < span;) {
		const uint32_t header = begin[at];
		g_np_at = at;
		g_np_header = header;
		// After the dispatch every packet runs as the game wrote it (=1 passes over the dwords up to the dispatch only):
		// whatever follows it (its EVENT_WRITEs, other functions' NOPs / markers) is accepted when it is plain type-3.
		if (dispatch) {
			if ((header >> 30u) != 3u || (header & 1u) != 0) return *why = "a packet after the dispatch that is not type-3", false;
			const uint32_t after = ((header >> 16u) & 0x3fffu) + 2u;
			if (after > span - at) return *why = "a packet past the cursor", false;
			at += after;
			continue;
		}
		if ((header >> 30u) != 3u || (header & 1u) != 0) return *why = "a packet that is not a plain type-3 one", false;
		const uint32_t length = ((header >> 16u) & 0x3fffu) + 2u;
		if (length > span - at) return *why = "a packet past the cursor", false;
		const uint32_t* const words = begin + at;
		switch ((header >> 8u) & 0xffu) {
			case 0x63: break; // IT_SET_SH_REG_INDIRECT: the shader's registers
			case 0x76: // IT_SET_SH_REG: COMPUTE_USER_DATA_0..7 = the T#
				if (dispatch) return *why = "a register write after the dispatch", false;
				if (length == 10 && words[1] == 0x240u) {
					for (uint32_t i = 0; i < 8; ++i) user[i] = words[2 + i];
					has_user = true;
				}
				break;
			case 0x46: // IT_EVENT_WRITE: only after the dispatch (its flushes)
				if (!dispatch || length != 2) return *why = "an event write before the dispatch", false;
				break;
			case 0x15: { // IT_DISPATCH_DIRECT: one, for the T#'s size
				const uint32_t* t = has_user ? user : tsharp;
				const uint32_t  w = (((t[1] >> 30u) & 0x3u) | ((t[2] & 0xFFFu) << 2u)) + 1u;
				const uint32_t  h = ((t[2] >> 14u) & 0x3FFFu) + 1u;
				if (dispatch || length != 5 || words[1] != (w + 7u) / 8u || words[2] != (h + 7u) / 8u || words[3] != 1u)
					return *why = "another dispatch", false;
				dispatch = true;
				skip     = at + length;
				break;
			}
			default: return *why = "a packet of another kind", false;
		}
		at += length;
	}
	if (!dispatch) return *why = "no dispatch", false;
	if (has_user) tsharp = user;
	const uint64_t hash = H::SpanHash(begin, span);
	for (uint32_t i = 0; i < 8; ++i) packet[H::TSharpWord + i] = tsharp[i];
	packet[H::HashWord]     = static_cast<uint32_t>(hash);
	packet[H::HashWord + 1] = static_cast<uint32_t>(hash >> 32u);
	packet[H::SkipWord]     = skip;
	packet[H::SpanWord]     = span; // (last: the packet is sealed)
	return true;
}

// 0x9ad560's entry jumps here (a guest thread recording the frame's command buffer).
void KYTY_SYSV_ABI NativoPaseClearHook(uint64_t* context, uint64_t* holder, uint64_t texture) {
	namespace H = DsNativoPase;
	H::Count(H::Calls);
	auto* const commands = context != nullptr ? reinterpret_cast<AgcCommandBuffer*>(context[0]) : nullptr;
	uint32_t*   packet   = nullptr;
	uint32_t    tsharp[8] {};
	if (commands != nullptr && texture != 0) {
		std::memcpy(tsharp, reinterpret_cast<const void*>(texture + 0xbc), sizeof(tsharp));
		packet = AnnounceNativoPase(*commands);
		if (packet == nullptr) H::Count(H::NoRoom);
	}
	{
		// (port2) No marker after the dispatch: the seal wants the function's own packets, the dispatch last.
		const Port2::Suppress no_port2_marker;
		g_nativo_pase_original(context, holder, texture);
	}
	if (packet == nullptr) return;
	const char* why = nullptr;
	if (reinterpret_cast<AgcCommandBuffer*>(context[0]) != commands) why = "another command buffer";
	if (why == nullptr && SealNativoPase(*commands, packet, tsharp, &why)) {
		H::Count(H::Sealed);
		return;
	}
	H::Count(H::OtherPackets);
	static std::atomic<uint32_t> printed {0};
	if (printed.fetch_add(1, std::memory_order_relaxed) < 8) {
		char       words[48 * 9 + 1] = "";
		const auto begin_addr        = reinterpret_cast<uintptr_t>(packet + H::PacketDwords);
		const auto end_addr          = reinterpret_cast<uintptr_t>(commands->cursor_up);
		if (reinterpret_cast<AgcCommandBuffer*>(context[0]) == commands && end_addr > begin_addr &&
		    (end_addr - begin_addr) / sizeof(uint32_t) <= H::MaxSpanDwords) {
			const auto count = (std::min<size_t>)((end_addr - begin_addr) / sizeof(uint32_t), 48);
			for (size_t i = 0; i < count; ++i)
				std::snprintf(words + i * 9, 10, " %08x", static_cast<unsigned>(packet[H::PacketDwords + i]));
		}
		std::printf("[nativo-pase] SSRClear no sellado (el pase sigue emulado): %s; paquete en dword %u cabecera %08x opcode 0x%02x, span %u dwords;%s\n",
		            why != nullptr ? why : "?", g_np_at, g_np_header, (g_np_header >> 8u) & 0xffu, g_np_span, words);
		std::fflush(stdout);
	}
}

void InstallNativoPaseHook(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (DsNativoPase::Mode() == 0 || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin")
		return;
	const auto refuse = [](const char* why) {
		std::printf("[nativo-pase] SSRClear: gancho NO instalado (KYTY_NATIVO_PASE_SSRCLEAR): %s; el pase sigue emulado\n", why);
		std::fflush(stdout);
	};
	if (program->mapped_size < NativoPaseDispatcherOffset + NativoPaseCode.size() ||
	    program->base_vaddr > UINT64_MAX - NativoPaseDispatcherOffset - NativoPaseCode.size())
		return refuse("the eboot is too short to be 01.007.000's");
	const auto  dispatcher = program->base_vaddr + NativoPaseDispatcherOffset;
	const auto  address    = program->base_vaddr + NativoPaseClearOffset;
	const auto* code       = reinterpret_cast<const uint8_t*>(dispatcher);
	if (!Libs::Graphics::HostMemoryRangeIsReadable(dispatcher, NativoPaseCode.size())) return refuse("its code is not readable");
	for (size_t i = 0; i < NativoPaseCode.size(); ++i) {
		if (code[i] == NativoPaseCode[i]) continue;
		std::printf("[nativo-pase] eboot+0x%llx holds 0x%02x where 01.007.000 has 0x%02x\n",
		            static_cast<unsigned long long>(NativoPaseDispatcherOffset + i), static_cast<unsigned>(code[i]),
		            static_cast<unsigned>(NativoPaseCode[i]));
		return refuse("eboot+0x9ad540 is not 01.007.000's texture clear (another build, or code already patched)");
	}
	if (g_nativo_pase_trampoline == 0)
		g_nativo_pase_trampoline = Libs::LibKernel::Memory::AllocateRuntimeMemory(
		    0, 0x1000, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_nativo_pase");
	if (g_nativo_pase_trampoline == 0) return refuse("no memory for its trampoline");
	// The trampoline: the moved bytes, then mov rax, <entry + moved>; jmp rax.
	std::array<uint8_t, NativoPaseMoved + 12> trampoline {};
	std::memcpy(trampoline.data(), NativoPaseCode.data() + NativoPaseClearInCode, NativoPaseMoved);
	const uint64_t resume            = address + NativoPaseMoved;
	trampoline[NativoPaseMoved]      = 0x48;
	trampoline[NativoPaseMoved + 1]  = 0xb8;
	std::memcpy(trampoline.data() + NativoPaseMoved + 2, &resume, sizeof(resume));
	trampoline[NativoPaseMoved + 10] = 0xff;
	trampoline[NativoPaseMoved + 11] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(g_nativo_pase_trampoline), trampoline.data(), trampoline.size());
	if (!Common::VirtualMemory::FlushInstructionCache(g_nativo_pase_trampoline, trampoline.size()))
		return refuse("its trampoline could not be flushed");
	g_nativo_pase_original = reinterpret_cast<NativoPaseClearFunction>(g_nativo_pase_trampoline);
	// The entry: mov rax, NativoPaseClearHook; jmp rax (before guest threads run; program memory is host-writable).
	std::array<uint8_t, NativoPaseMoved> entry {0x48, 0xb8};
	const auto                           hook = reinterpret_cast<uint64_t>(&NativoPaseClearHook);
	std::memcpy(entry.data() + 2, &hook, sizeof(hook));
	entry[10] = 0xff;
	entry[11] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(address), entry.data(), entry.size());
	if (!Common::VirtualMemory::FlushInstructionCache(address, entry.size())) {
		std::memcpy(reinterpret_cast<void*>(address), NativoPaseCode.data() + NativoPaseClearInCode, NativoPaseMoved);
		Common::VirtualMemory::FlushInstructionCache(address, NativoPaseMoved);
		return refuse("its entry could not be flushed");
	}
	std::printf("[nativo-pase] SSRClear: gancho instalado (KYTY_NATIVO_PASE_SSRCLEAR=%u; eboot+0x9ad560 de 01.007.000 anuncia "
	            "el borrado del SSR en el command buffer: %s)\n",
	            DsNativoPase::Mode(),
	            DsNativoPase::Mode() == 2 ? "verifica: borrado nativo y pase emulado, se ve el emulado"
	                                      : "el borrado lo hace Vulkan nativo y el pase del juego se salta");
	std::fflush(stdout);
#else
	(void)program;
#endif
}
