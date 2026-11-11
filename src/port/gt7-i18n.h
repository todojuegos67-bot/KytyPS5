#ifndef EMULATOR_SRC_PORT_GT7_I18N_H_
#define EMULATOR_SRC_PORT_GT7_I18N_H_

// "Gran Turismo 7" PC port: the launcher's language (as ds-i18n): every string is written in Spanish and looked up in
// one table; a string the table lacks stays Spanish.

#include <string>

namespace Port::I18n {

extern const char* const kMenuLanguages[];     // ids: es, en
extern const char* const kMenuLanguageNames[];
extern const int         kMenuLanguageCount;

void        SetLanguage(const std::string& id); // unknown: es
const char* T(const char* es);                  // the part from "##" on is kept
std::string DetectWindowsLanguage();            // es or en
int         GameLanguageFor(const std::string& id);

} // namespace Port::I18n

#endif // EMULATOR_SRC_PORT_GT7_I18N_H_
