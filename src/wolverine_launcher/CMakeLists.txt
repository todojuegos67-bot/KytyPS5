# The Wolverine launcher: a separate start window for Marvel's Wolverine (PPSA03671) with the shader precompile
# (session 12, Phase A). The official launcher (src/launcher) is left as upstream has it.
find_package(Qt6 6.8 COMPONENTS Widgets REQUIRED)

add_executable(wolverine_launcher WIN32
	main.cpp
	wolverineLauncher.cpp
	wolverineLauncher.h
	# The official launcher's game-file reads: a game folder or a ZArchive (.zar), through common.
	"${KYTY_SOURCE_DIR}/launcher/src/gameContent.cpp"
)
set_target_properties(wolverine_launcher PROPERTIES AUTOMOC ON)
target_include_directories(wolverine_launcher PRIVATE "${KYTY_SOURCE_DIR}/launcher/include")
target_link_libraries(wolverine_launcher Qt6::Widgets common)
if(WIN32)
	target_link_libraries(wolverine_launcher dxgi)
endif()

# Built with the official launcher (the build scripts name that target), installed next to the emulator, whose Qt
# runtime windeployqt puts there for the official launcher.
add_dependencies(launcher wolverine_launcher)
install(TARGETS wolverine_launcher RUNTIME DESTINATION .)
# Required: without the shipped hints the launcher has nothing to precompile.
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/assets/hints/PPSA03671.kyhints")
	message(FATAL_ERROR "assets/hints/PPSA03671.kyhints is missing: the release needs the shipped shader hints")
endif()
install(DIRECTORY "${CMAKE_SOURCE_DIR}/assets/hints/" DESTINATION hints)
