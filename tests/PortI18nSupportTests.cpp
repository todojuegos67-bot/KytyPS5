// English startup/support translations. Plain C++, linked with src/port/ds-i18n.cpp.
#include "port/ds-i18n.h"

#include <cstdio>
#include <string>

namespace {

struct Row {
	const char* t[6];
};

const Row kRows[] = {
#include "port/ds-startup-i18n.inc"
};

int failures = 0;

void Expect(const char* spanish, const char* expected) {
	const std::string actual = Port::I18n::T(spanish);
	if (actual != expected) {
		std::printf("FAIL: English translation of \"%s\"\n  expected: %s\n  actual:   %s\n", spanish, expected, actual.c_str());
		failures++;
	}
}

} // namespace

int main() {
	Port::I18n::SetLanguage("en");
	for (const auto& row: kRows) Expect(row.t[0], row.t[1]);

	// Spanish remains the source-language fallback after switching languages.
	Port::I18n::SetLanguage("es");
	for (const auto& row: kRows) Expect(row.t[0], row.t[0]);

	if (failures == 0) std::printf("All startup/support English UI translation checks passed.\n");
	return failures == 0 ? 0 : 1;
}
