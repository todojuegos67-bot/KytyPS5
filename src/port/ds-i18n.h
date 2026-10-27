#ifndef EMULATOR_SRC_PORT_DS_I18N_H_
#define EMULATOR_SRC_PORT_DS_I18N_H_

// "Demon's Souls" PC port (10-27b): the menus' language. Every string is written in Spanish in the code and looked up
// in one table (ds-i18n.cpp); a string the table lacks stays Spanish (the fallback).

#include <string>

namespace Port::I18n {

// es, en, pt-BR, fr, it, de.
extern const char* const kMenuLanguages[]; // ids
extern const char* const kMenuLanguageNames[];
extern const int         kMenuLanguageCount;

void        SetLanguage(const std::string& id); // unknown: es
const char* T(const char* es);              // the current language's text (the part from "##" on is kept)
std::string DetectWindowsLanguage();        // the Windows UI language among kMenuLanguages, else en
int         GameLanguageFor(const std::string& id); // the PS5 console language (--console-language) for a menu language

} // namespace Port::I18n

#endif // EMULATOR_SRC_PORT_DS_I18N_H_
