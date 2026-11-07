# kyty_launcher: the settings window for Linux (launcher-windows' counterpart). SDL2 (its static library, as the
# emulator links it) draws the window and Dear ImGui its controls, so the program needs nothing installed beyond what the
# emulator needs; it starts the game through run_ds.sh.
add_executable(kyty_launcher
	main.cpp
	ui.cpp
	commands.cpp
	game.cpp
	platform.cpp
	settings.cpp
	strings.cpp
	stb_image.cpp
	bytes.cpp
	keyboard_layout.cpp
	keyboard_layout_settings.cpp
	keyboard_prompts.cpp
	keyboard_editor.cpp
	prompt_assets.cpp
	prompt_texture.cpp
	"${KYTY_THIRD_PARTY_DIR}/imgui/backends/imgui_impl_sdl2.cpp"
	"${KYTY_THIRD_PARTY_DIR}/imgui/backends/imgui_impl_sdlrenderer2.cpp"
	"${KYTY_THIRD_PARTY_DIR}/imgui/misc/cpp/imgui_stdlib.cpp"
)
target_include_directories(kyty_launcher PRIVATE
	"${KYTY_THIRD_PARTY_DIR}/SDL2/include"
	"${KYTY_THIRD_PARTY_DIR}/imgui"
	"${KYTY_THIRD_PARTY_DIR}/imgui/backends"
	"${KYTY_THIRD_PARTY_DIR}/stb"
)
# The keyboard and mouse prompts' pack (the 22 textures, the blank template, the pixel map and the
# manifest) is the Windows launcher's own set of files, linked in as they are (prompt_assets.cpp's
# .incbin): one copy of the artwork for both launchers, nothing to install or find at run time.
set(KYTY_PROMPT_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../launcher-windows/KeyboardPrompts")
file(GLOB KYTY_PROMPT_ASSET_FILES "${KYTY_PROMPT_ASSETS_DIR}/*")
set_source_files_properties(prompt_assets.cpp PROPERTIES
	COMPILE_DEFINITIONS "KYTY_PROMPT_ASSETS_DIR=\"${KYTY_PROMPT_ASSETS_DIR}\""
	OBJECT_DEPENDS "${KYTY_PROMPT_ASSET_FILES}")
# The project compiles with -fno-exceptions; the keyboard prompts report what went wrong with
# exceptions (PromptError), as the repo's own exception tests do: -fexceptions for this target.
target_compile_options(kyty_launcher PRIVATE -fexceptions)
find_package(Threads REQUIRED)
target_link_libraries(kyty_launcher PRIVATE SDL2-static imgui nlohmann_json::nlohmann_json Vulkan::Headers Threads::Threads
	${CMAKE_DL_LIBS})
set_target_properties(kyty_launcher PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
# (No symbols in a release: a third of the size.)
target_link_options(kyty_launcher PRIVATE $<$<CONFIG:Release>:-s>)

# The keyboard prompts' tests (tests/KeyboardPromptsLinuxTests.cpp): no game files, SDL or ImGui needed.
#   cmake --build <build> --target kyty_launcher_prompts_tests && ctest -R launcher_keyboard_prompts
if(BUILD_TESTING)
	add_executable(kyty_launcher_prompts_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/KeyboardPromptsLinuxTests.cpp"
		bytes.cpp
		keyboard_layout.cpp
		keyboard_prompts.cpp
		prompt_assets.cpp
		prompt_texture.cpp
	)
	target_include_directories(kyty_launcher_prompts_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${KYTY_THIRD_PARTY_DIR}/stb")
	target_compile_options(kyty_launcher_prompts_tests PRIVATE -fexceptions)
	set_source_files_properties(prompt_assets.cpp PROPERTIES
		COMPILE_DEFINITIONS "KYTY_PROMPT_ASSETS_DIR=\"${KYTY_PROMPT_ASSETS_DIR}\""
		OBJECT_DEPENDS "${KYTY_PROMPT_ASSET_FILES}")
	add_test(NAME launcher_keyboard_prompts COMMAND $<TARGET_FILE:kyty_launcher_prompts_tests>)
endif()
