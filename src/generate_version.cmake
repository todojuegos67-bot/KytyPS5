set(KYTY_GIT_VERSION "unknown")
set(KYTY_GIT_HASH "unknown")
set(KYTY_GIT_REVISION "unknown")
if(GIT_EXECUTABLE)
	execute_process(
		COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_RESULT EQUAL 0)
		set(KYTY_GIT_VERSION "unknown")
	endif()

	execute_process(
		COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_REVISION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_HASH_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_HASH_RESULT EQUAL 0)
		set(KYTY_GIT_HASH "unknown")
		set(KYTY_GIT_REVISION "unknown")
	else()
		string(SUBSTRING "${KYTY_GIT_REVISION}" 0 7 KYTY_GIT_HASH)
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD --
			WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
			RESULT_VARIABLE GIT_DIRTY_RESULT
			ERROR_QUIET
		)
		if(NOT GIT_DIRTY_RESULT EQUAL 0)
			string(APPEND KYTY_GIT_HASH "-dirty")
		endif()
	endif()
endif()

# Hash of the sources that decide what the guest->SPIR-V translation produces. The persistent
# translation cache (<title>.xlat) is keyed on this instead of the git revision, so a build that did
# not touch the shader translator keeps the translations of the previous builds (Yotei: ~100 ms per
# new shader, hundreds of them during the cinematic, all redone after every build).
set(KYTY_TRANSLATOR_HASH "unknown")
if(KYTY_SOURCE_ROOT AND EXISTS "${KYTY_SOURCE_ROOT}/graphics/shader")
	file(GLOB_RECURSE KYTY_TRANSLATOR_SOURCES LIST_DIRECTORIES false
		"${KYTY_SOURCE_ROOT}/graphics/shader/*.cpp" "${KYTY_SOURCE_ROOT}/graphics/shader/*.h"
		"${KYTY_SOURCE_ROOT}/graphics/shader/*.hpp" "${KYTY_SOURCE_ROOT}/graphics/shader/*.inl" "${KYTY_SOURCE_ROOT}/graphics/shader/*.inc")
	list(SORT KYTY_TRANSLATOR_SOURCES)
	set(KYTY_TRANSLATOR_ACC "")
	foreach(f IN LISTS KYTY_TRANSLATOR_SOURCES)
		file(SHA256 "${f}" file_hash)
		file(RELATIVE_PATH rel "${KYTY_SOURCE_ROOT}" "${f}")
		string(APPEND KYTY_TRANSLATOR_ACC "${rel}=${file_hash}\n")
	endforeach()
	string(SHA256 KYTY_TRANSLATOR_FULL "${KYTY_TRANSLATOR_ACC}")
	string(SUBSTRING "${KYTY_TRANSLATOR_FULL}" 0 32 KYTY_TRANSLATOR_HASH)
endif()
configure_file("${INPUT_FILE}" "${OUTPUT_FILE}")
