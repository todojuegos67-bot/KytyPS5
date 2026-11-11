#include "loader/demonsSoulsLanguages.h"

#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace Loader::DemonsSoulsLanguages {
namespace {

// The 32-byte constant a vmovups ymm0, [rip + disp32] at `load` reads (its disp32 after 4 bytes, from the end of its 8),
// if it lies in the program.
std::optional<uint64_t> Constant(const Program& program, uint64_t load) {
	int32_t displacement = 0;
	std::memcpy(&displacement, reinterpret_cast<const void*>(load + 4), sizeof(displacement));
	const uint64_t constant = load + 8 + static_cast<int64_t>(displacement);
	if (constant < program.base_vaddr || constant + 32 > program.base_vaddr + program.mapped_size) return std::nullopt;
	return constant;
}

uint32_t Immediate(uint64_t address) {
	uint32_t value = 0;
	std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
	return value;
}

} // namespace

void Install(Program* program) {
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;
	if (const char* value = std::getenv("KYTY_ALL_LANGUAGES"); value != nullptr && std::strcmp(value, "0") == 0) return;
	const auto site   = GuestCode::FindUnique(*program, GuestCode::Pattern(Sets));
	const auto first  = site ? Constant(*program, *site + FirstSets) : std::nullopt;
	const auto second = site ? Constant(*program, *site + SecondSets) : std::nullopt;
	if (!first || !second) {
		std::printf("Demon's Souls languages: the language sets of a known build (01.005.000) not found; the region's "
		            "kept\n");
		return;
	}
	std::array<uint64_t, 4> text {};  // 0, the Americas, Europe, Japan
	std::array<uint64_t, 4> mixed {}; // Asia's text, every text, the Americas' voices, Europe's
	std::memcpy(text.data(), reinterpret_cast<const void*>(*first), sizeof(text));
	std::memcpy(mixed.data(), reinterpret_cast<const void*>(*second), sizeof(mixed));
	const uint64_t all_text   = mixed[1];
	const uint64_t all_voices = Immediate(*site + AllVoices);
	const uint64_t voices     = mixed[2] | mixed[3] | Immediate(*site + JapanVoices) | Immediate(*site + AsiaVoices);
	// Each "every language" set holds the regions' (else these are other sets: nothing is changed).
	if (text[0] != 0 || ((text[1] | text[2] | text[3] | mixed[0]) & ~all_text) != 0 || (voices & ~all_voices) != 0 ||
	    all_voices > 0x7fffffff) {
		std::printf("Demon's Souls languages: the language sets are not as known (every text 0x%llx, every voice 0x%llx); "
		            "the region's kept\n",
		            static_cast<unsigned long long>(all_text), static_cast<unsigned long long>(all_voices));
		return;
	}
	// Installation occurs before module initializers or guest threads run; program memory is writable to the host.
	text  = {0, all_text, all_text, all_text};
	mixed = {all_text, all_text, all_voices, all_voices};
	std::memcpy(reinterpret_cast<void*>(*first), text.data(), sizeof(text));
	std::memcpy(reinterpret_cast<void*>(*second), mixed.data(), sizeof(mixed));
	const auto voice_immediate = static_cast<uint32_t>(all_voices); // sign-extended: below 2^31
	std::memcpy(reinterpret_cast<void*>(*site + JapanVoices), &voice_immediate, sizeof(voice_immediate));
	std::memcpy(reinterpret_cast<void*>(*site + AsiaVoices), &voice_immediate, sizeof(voice_immediate));
	Common::VirtualMemory::FlushInstructionCache(*site, GuestCode::Pattern(Sets).Size());
	std::printf("Demon's Souls languages: every region offers every language (text %d, mask 0x%llx; voices %d, mask "
	            "0x%llx; the sets' constructor at eboot+0x%llx; KYTY_ALL_LANGUAGES=0 keeps the region's)\n",
	            std::popcount(all_text), static_cast<unsigned long long>(all_text), std::popcount(all_voices),
	            static_cast<unsigned long long>(all_voices), static_cast<unsigned long long>(*site - program->base_vaddr));
}

} // namespace Loader::DemonsSoulsLanguages
