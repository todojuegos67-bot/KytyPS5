file(GLOB common_src CONFIGURE_DEPENDS
	"*.cpp"
	"logging/*.cpp"
	"platform/*.cpp"
)

file(GLOB_RECURSE common_headers CONFIGURE_DEPENDS "*.h")

# The memory guard (release on allocation failure, the clear out-of-memory message) lives with the fork's local code
# but the platform layer and the tests call it: it is part of this library.
list(APPEND common_src "${KYTY_SOURCE_DIR}/local/mem-guard.cpp")
list(APPEND common_headers "${KYTY_SOURCE_DIR}/local/mem-guard.h")

if(KYTY_CLANG_CL)
	set_source_files_properties(${common_src} PROPERTIES COMPILE_FLAGS "-Wno-pragma-pack")
endif()

add_library(common STATIC ${common_src} ${common_headers})

if(LINUX)
	set_property(SOURCE assert.cpp hostException.cpp APPEND PROPERTY COMPILE_OPTIONS -fexceptions)
endif()

target_include_directories(common
	PUBLIC ${KYTY_SOURCE_DIR}
	PRIVATE
		${KYTY_THIRD_PARTY_DIR}/cpuinfo/include
)

target_link_libraries(common
	PUBLIC
		fmt::fmt
		Tracy::TracyClient
	PRIVATE
		cpuinfo
		kyty_git_version
		spdlog::spdlog
		ZArchive::zarchive
)

get_property(inc_headers TARGET common PROPERTY INCLUDE_DIRECTORIES)
set(check_headers ${CMAKE_CURRENT_SOURCE_DIR})

clang_tidy_check(common "" "${check_headers}" "${inc_headers}")
include_what_you_use(common "${inc_headers}")
