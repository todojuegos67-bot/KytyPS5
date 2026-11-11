#pragma once

#include <cstdint>
#include <string_view>

namespace Loader {
struct Program;
// Demon's Souls offers in its language settings only the languages of the console's region, though every package has
// them all: the text of 23 languages (text/common/_cmn, text/ds: ui.ru.ctxt and the others) and the voices of 9
// (sound/streams/02_dialogue and 06_cutscene_vo: en, fr, de, it, es, jp, br, la; British English speaks en). The
// languages are sets of bits (a language's bit is its ELanguage: Arabic 1, Chinese 2, Simplified Chinese 3, Danish 7,
// Dutch 8, English 9, British English 10, Finnish 11, French 12, German 14, Italian 18, Japanese 19, Korean 20,
// Norwegian 21, Polish 22, Portuguese 23, Brazilian Portuguese 25, Russian 26, Spanish 27, Latin American Spanish 28,
// Swedish 29, Turkish 30, Thai 31), which the language settings' constructor stores per region, the text's then the
// voices': the Americas, Europe, Japan, Asia and one with every language (US 01.005.000, PPSA01342):
//   text   0x12001200 (English, French, Brazilian Portuguese, Latin American Spanish), 0x6ce45d82 (15, Russian among
//          them), 0x80400 (British English, Japanese), 0x8010040c (Chinese, Simplified Chinese, British English,
//          Korean, Thai), 0xfefc5f8e (all 23)
//   voices 0x12001200, 0x8045400 (British English, French, German, Italian, Spanish), 0x80400, 0x200 (English),
//          0x1a0c5600 (all 9)
// Install gives every region the sets with every language, so a US game can show its menus and subtitles in Russian,
// Japanese or Korean and speak Japanese or German. KYTY_ALL_LANGUAGES=0 keeps the region's. (A GoldHEN-style patch of
// the Americas' text, 0x12001200 to 0x16001200, did the same for Russian alone in a user's US game: the menu offered
// it, and the menus and subtitles were Russian.)
namespace DemonsSoulsLanguages {
void Install(Program* program);

// The constructor's stores (01.005.000 0x145e4b8): vmovups ymm0, [rip + disp32] (the text of the first entry - not a
// region's, 0 -, the Americas, Europe and Japan); mov rcx, imm64; mov rbx, rdi; mov rax, [rax]; mov [rbp - 0x30], rax;
// mov [rdi], rcx; mov dword [rdi + 8], imm32; vmovups [rdi + 0x10], ymm0; vmovups ymm0, [rip + disp32] (Asia's text,
// the text with every language, the Americas' and Europe's voices); lea rax, [rip + disp32]; vmovups [rdi + 0x30],
// ymm0; mov qword [rdi + 0x50], imm32 (Japan's voices); mov qword [rdi + 0x58], imm32 (Asia's); mov qword [rdi + 0x60],
// imm32 (every voice).
inline constexpr std::string_view Sets =
    "c5 fc 10 05 ?? ?? ?? ?? 48 b9 ?? ?? ?? ?? ?? ?? ?? ?? 48 89 fb 48 8b 00 48 89 45 d0 48 89 0f c7 47 08 ?? ?? ?? ?? "
    "c5 fc 11 47 10 c5 fc 10 05 ?? ?? ?? ?? 48 8d 05 ?? ?? ?? ?? c5 fc 11 47 30 48 c7 47 50 ?? ?? ?? ?? 48 c7 47 58 ?? "
    "?? ?? ?? 48 c7 47 60 ?? ?? ?? ??";
inline constexpr uint32_t FirstSets = 0, SecondSets = 43;                   // the two vmovups loads
inline constexpr uint32_t JapanVoices = 67, AsiaVoices = 75, AllVoices = 83; // the immediates
} // namespace DemonsSoulsLanguages
} // namespace Loader
