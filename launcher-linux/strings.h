#ifndef KYTY_LAUNCHER_LINUX_STRINGS_H
#define KYTY_LAUNCHER_LINUX_STRINGS_H

#include <string>
#include <vector>

// The window's text in English, Simplified Chinese and Brazilian Portuguese (the launcher's own
// language, not the console language the game is told). A key missing in a language shows the
// English text. The keys follow launcher-windows' Strings.cs; the Linux ones (DLSS's runtime,
// MangoHud, the terminal) are this launcher's own.
namespace Strings {

enum class Language { English, Chinese, Portuguese };

extern Language Current;

// The language of the environment (LC_ALL, LC_MESSAGES, LANG): zh -> Chinese, pt -> Portuguese,
// else English.
Language SystemLanguage();
// "en", "zh", "pt" or "" (the system's).
Language FromCode(const std::string& code);

const char* T(const char* key);
// T(key) with {0}, {1}... replaced by the values in order.
std::string F(const char* key, const std::vector<std::string>& values);

// The PS5's console languages in its order (--console-language takes the index).
const std::vector<std::string>& ConsoleLanguages();

} // namespace Strings

#endif
