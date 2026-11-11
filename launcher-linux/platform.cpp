#include "platform.h"

#define VK_NO_PROTOTYPES
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <poll.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vulkan/vulkan_core.h>

namespace Platform {

fs::path ExecutableFolder() {
	std::error_code error;
	const auto      exe = fs::read_symlink("/proc/self/exe", error);
	return error ? fs::current_path(error) : exe.parent_path();
}

bool HasProgram(const std::string& name) {
	if (name.empty()) return false;
	if (name.find('/') != std::string::npos) return access(name.c_str(), X_OK) == 0;
	const char* path  = std::getenv("PATH");
	std::string paths = path != nullptr && *path != 0 ? path : "/usr/local/bin:/usr/bin:/bin";
	size_t      start = 0;
	while (start <= paths.size()) {
		const size_t end = std::min(paths.find(':', start), paths.size());
		const auto   dir = paths.substr(start, end - start);
		if (!dir.empty() && access((dir + "/" + name).c_str(), X_OK) == 0) return true;
		start = end + 1;
	}
	return false;
}

std::string Quote(const std::string& text) {
	std::string out = "'";
	for (const char c: text) {
		if (c == '\'')
			out += "'\\''";
		else
			out += c;
	}
	return out + "'";
}

namespace {

std::vector<char*> Argv(const std::vector<std::string>& argv) {
	std::vector<char*> out;
	out.reserve(argv.size() + 1);
	for (const auto& arg: argv)
		out.push_back(const_cast<char*>(arg.c_str()));
	out.push_back(nullptr);
	return out;
}

// In a child: no descriptors of the launcher's past stderr (its window's sockets, the pipes of
// other children).
void CloseOthers(int keep_from) {
	const long most = sysconf(_SC_OPEN_MAX);
	for (int fd = keep_from; fd < (most > 0 && most < 65536 ? most : 4096); ++fd)
		close(fd);
}

} // namespace

int Run(const std::vector<std::string>& argv, std::string* output, int timeout_ms,
        const fs::path& directory) {
	if (argv.empty()) return -1;
	int pipe_fds[2] {-1, -1};
	if (output != nullptr && pipe2(pipe_fds, O_CLOEXEC) != 0) return -1;
	const pid_t pid = fork();
	if (pid < 0) {
		if (output != nullptr) {
			close(pipe_fds[0]);
			close(pipe_fds[1]);
		}
		return -1;
	}
	if (pid == 0) {
		const int null = open("/dev/null", O_RDWR);
		dup2(null, 0);
		dup2(output != nullptr ? pipe_fds[1] : null, 1);
		dup2(null, 2);
		CloseOthers(3);
		if (!directory.empty() && chdir(directory.c_str()) != 0) _exit(127);
		auto args = Argv(argv);
		execvp(args[0], args.data());
		_exit(127);
	}
	if (output != nullptr) close(pipe_fds[1]);
	const auto deadline  = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
	bool       timed_out = false;
	if (output != nullptr) {
		char buffer[4096];
		for (;;) {
			const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
			                      deadline - std::chrono::steady_clock::now())
			                      .count();
			if (left <= 0) {
				timed_out = true;
				break;
			}
			pollfd    entry {pipe_fds[0], POLLIN, 0};
			const int ready = poll(&entry, 1, static_cast<int>(left));
			if (ready < 0 && errno == EINTR) continue;
			if (ready <= 0) {
				timed_out = ready == 0;
				break;
			}
			const ssize_t n = read(pipe_fds[0], buffer, sizeof(buffer));
			if (n < 0 && errno == EINTR) continue;
			if (n <= 0) break;
			output->append(buffer, static_cast<size_t>(n));
		}
		close(pipe_fds[0]);
	}
	int status = 0;
	for (;;) {
		const pid_t done = waitpid(pid, &status, timed_out ? 0 : WNOHANG);
		if (done == pid) break;
		if (done < 0 && errno != EINTR) return -1;
		if (timed_out || std::chrono::steady_clock::now() >= deadline) {
			kill(pid, SIGKILL);
			timed_out = true;
			continue;
		}
		usleep(10000);
	}
	if (timed_out) return -1;
	return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

bool Spawn(const std::vector<std::string>& argv, const fs::path& directory) {
	if (argv.empty() || !HasProgram(argv[0])) return false;
	// A grandchild in a session of its own: this process neither waits for it nor takes it down
	// when it exits.
	int report[2];
	if (pipe2(report, O_CLOEXEC) != 0) return false;
	const pid_t pid = fork();
	if (pid < 0) {
		close(report[0]);
		close(report[1]);
		return false;
	}
	if (pid == 0) {
		setsid();
		if (fork() != 0) _exit(0);
		const int null = open("/dev/null", O_RDWR);
		dup2(null, 0);
		dup2(null, 1);
		dup2(null, 2);
		if (!directory.empty() && chdir(directory.c_str()) != 0) {
			// (Started anyway, in the launcher's folder.)
		}
		// (The report pipe closes on exec: a byte written back means the exec failed.)
		const int keep = report[1];
		for (int fd = 3; fd < 4096; ++fd)
			if (fd != keep) close(fd);
		auto args = Argv(argv);
		execvp(args[0], args.data());
		const char failed = 1;
		(void)!write(keep, &failed, 1);
		_exit(127);
	}
	close(report[1]);
	int status = 0;
	waitpid(pid, &status, 0);
	char       failed = 0;
	const auto n      = read(report[0], &failed, 1);
	close(report[0]);
	return n <= 0;
}

void Open(const std::string& target) {
	if (!Spawn({"xdg-open", target})) Spawn({"gio", "open", target});
}

int ProcessorCount() {
	cpu_set_t set;
	CPU_ZERO(&set);
	if (sched_getaffinity(0, sizeof(set), &set) == 0) return std::max(1, CPU_COUNT(&set));
	return std::max(1L, sysconf(_SC_NPROCESSORS_ONLN)) > 0
	           ? static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN))
	           : 1;
}

fs::path WriteDesktopEntry(const fs::path& folder, const fs::path& icon, std::string* error) {
	const char*     data = std::getenv("XDG_DATA_HOME");
	const char*     home = std::getenv("HOME");
	const fs::path  base = data != nullptr && *data != 0
	                           ? fs::path(data)
	                           : fs::path(home != nullptr ? home : "/tmp") / ".local" / "share";
	const fs::path  file = base / "applications" / "kytyps5-launcher.desktop";
	std::error_code code;
	fs::create_directories(file.parent_path(), code);
	// (Exec's quoting: a path in double quotes, with ", `, $ and \ escaped.)
	auto quoted = [](const std::string& text) {
		std::string out = "\"";
		for (const char c: text) {
			if (c == '"' || c == '`' || c == '$' || c == '\\') out += "\\\\";
			out += c;
		}
		return out + "\"";
	};
	std::ofstream out(file, std::ios::binary | std::ios::trunc);
	out << "[Desktop Entry]\nType=Application\nName=KytyPS5\nComment=PS5 emulator (Demon's Souls)\n"
	    << "Exec=" << quoted((ExecutableFolder() / "kyty_launcher").string()) << "\n"
	    << "Path=" << folder.string() << "\n";
	if (!icon.empty()) out << "Icon=" << icon.string() << "\n";
	out << "Terminal=false\nCategories=Game;Emulator;\n";
	if (!out) {
		if (error != nullptr) *error = file.string();
		return {};
	}
	return file;
}

} // namespace Platform

namespace {

bool IsExecutable(const fs::path& path) {
	std::error_code error;
	return fs::is_regular_file(path, error) && access(path.c_str(), X_OK) == 0;
}

bool Exists(const fs::path& path) {
	std::error_code error;
	return fs::exists(path, error);
}

// The package folder (run_ds.sh next to the launcher) or the repository (tools/local/run_ds.sh),
// searched from the launcher's folder up.
fs::path FindRoot(const fs::path& start) {
	for (fs::path dir = start; !dir.empty(); dir = dir.parent_path()) {
		if (Exists(dir / "run_ds.sh") || Exists(dir / "tools" / "local" / "run_ds.sh")) return dir;
		if (dir == dir.parent_path()) break;
	}
	return start;
}

// The Vulkan devices (what the emulator itself sees), through the loader the emulator needs anyway.
std::vector<GpuInfo> QueryGpus() {
	std::vector<GpuInfo> gpus;
	void*                library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
	if (library == nullptr) library = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
	if (library == nullptr) return gpus;
	// (The loader stays loaded: some drivers do not survive being unloaded.)
	const auto get =
	    reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
	if (get == nullptr) return gpus;
	const auto create = reinterpret_cast<PFN_vkCreateInstance>(get(nullptr, "vkCreateInstance"));
	if (create == nullptr) return gpus;
	VkApplicationInfo app {};
	app.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app.pApplicationName = "kyty_launcher";
	app.apiVersion       = VK_API_VERSION_1_0;
	VkInstanceCreateInfo info {};
	info.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	info.pApplicationInfo = &app;
	VkInstance instance   = VK_NULL_HANDLE;
	if (create(&info, nullptr, &instance) != VK_SUCCESS) return gpus;
	const auto destroy =
	    reinterpret_cast<PFN_vkDestroyInstance>(get(instance, "vkDestroyInstance"));
	const auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
	    get(instance, "vkEnumeratePhysicalDevices"));
	const auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
	    get(instance, "vkGetPhysicalDeviceProperties"));
	const auto memory = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
	    get(instance, "vkGetPhysicalDeviceMemoryProperties"));
	uint32_t count = 0;
	if (enumerate != nullptr && properties != nullptr && memory != nullptr &&
	    enumerate(instance, &count, nullptr) == VK_SUCCESS) {
		std::vector<VkPhysicalDevice> devices(count);
		if (enumerate(instance, &count, devices.data()) == VK_SUCCESS) {
			for (uint32_t i = 0; i < count; ++i) {
				VkPhysicalDeviceProperties p {};
				properties(devices[i], &p);
				// (A CPU renderer such as llvmpipe is not a graphics card.)
				if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
				VkPhysicalDeviceMemoryProperties m {};
				memory(devices[i], &m);
				uint64_t largest = 0;
				for (uint32_t h = 0; h < m.memoryHeapCount; ++h)
					if ((m.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
						largest = std::max<uint64_t>(largest, m.memoryHeaps[h].size);
				GpuInfo gpu;
				gpu.name     = p.deviceName;
				gpu.memory   = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? largest : 0;
				gpu.vendor   = p.vendorID;
				gpu.discrete = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
				if (std::none_of(gpus.begin(), gpus.end(),
				                 [&](const GpuInfo& g) { return g.name == gpu.name; }))
					gpus.push_back(gpu);
			}
		}
	}
	if (destroy != nullptr) destroy(instance, nullptr);
	std::stable_sort(gpus.begin(), gpus.end(), [](const GpuInfo& a, const GpuInfo& b) {
		if (a.discrete != b.discrete) return a.discrete;
		return a.memory > b.memory;
	});
	return gpus;
}

bool HasWord(const std::string& text, const std::string& word) {
	for (size_t at = text.find(word); at != std::string::npos;
	     at        = text.find(word, at + word.size())) {
		const bool before = at == 0 || std::isalnum(static_cast<unsigned char>(text[at - 1])) == 0;
		const bool after  = at + word.size() >= text.size() ||
		                   std::isalnum(static_cast<unsigned char>(text[at + word.size()])) == 0;
		if (before && after) return true;
	}
	return false;
}

// Intel's hybrid processors list their efficiency cores as the cpu_atom PMU's CPUs ("16-23");
// others have one kind.
std::string EfficiencyCpus() {
	std::ifstream file("/sys/devices/cpu_atom/cpus");
	std::string   list;
	if (file && std::getline(file, list)) {
		while (!list.empty() && (list.back() == '\n' || list.back() == ' '))
			list.pop_back();
		return list;
	}
	return "";
}

// The terminal programs found, the desktop's own first (XDG_CURRENT_DESKTOP), each with how it runs
// a command.
std::vector<Terminal> FindTerminals() {
	const std::vector<Terminal> known {
	    {"ptyxis", {"--new-window", "--"}},
	    {"kgx", {"-e"}},
	    {"gnome-terminal", {"--"}},
	    {"konsole", {"-e"}},
	    {"xfce4-terminal", {"-x"}},
	    {"mate-terminal", {"-x"}},
	    {"lxterminal", {"-e"}},
	    {"qterminal", {"-e"}},
	    {"tilix", {"-e"}},
	    {"terminator", {"-x"}},
	    {"cosmic-term", {"-e"}},
	    {"alacritty", {"-e"}},
	    {"kitty", {}},
	    {"foot", {}},
	    {"wezterm", {"start", "--"}},
	    {"ghostty", {"-e"}},
	    {"x-terminal-emulator", {"-e"}},
	    {"xterm", {"-e"}},
	    {"urxvt", {"-e"}},
	    {"st", {"-e"}},
	};
	std::vector<std::string> preferred;
	std::string              desktop;
	if (const char* value = std::getenv("XDG_CURRENT_DESKTOP"); value != nullptr) desktop = value;
	for (auto& c: desktop)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	if (desktop.find("kde") != std::string::npos)
		preferred = {"konsole"};
	else if (desktop.find("gnome") != std::string::npos ||
	         desktop.find("unity") != std::string::npos ||
	         desktop.find("pantheon") != std::string::npos)
		preferred = {"ptyxis", "kgx", "gnome-terminal"};
	else if (desktop.find("xfce") != std::string::npos)
		preferred = {"xfce4-terminal"};
	else if (desktop.find("mate") != std::string::npos)
		preferred = {"mate-terminal"};
	else if (desktop.find("lxqt") != std::string::npos)
		preferred = {"qterminal"};
	else if (desktop.find("lxde") != std::string::npos)
		preferred = {"lxterminal"};
	else if (desktop.find("cosmic") != std::string::npos)
		preferred = {"cosmic-term"};
	std::vector<Terminal> found;
	auto                  add = [&](const Terminal& terminal) {
        if (std::any_of(found.begin(), found.end(),
		                                 [&](const Terminal& t) { return t.name == terminal.name; }))
            return;
        if (Platform::HasProgram(terminal.name)) found.push_back(terminal);
	};
	// $TERMINAL, a window manager user's choice (most take -e).
	if (const char* value = std::getenv("TERMINAL"); value != nullptr && *value != 0) {
		const std::string name = value;
		const auto        it   = std::find_if(known.begin(), known.end(),
		                                      [&](const Terminal& t) { return t.name == name; });
		add(it != known.end() ? *it : Terminal {name, {"-e"}});
	}
	for (const auto& name: preferred) {
		const auto it = std::find_if(known.begin(), known.end(),
		                             [&](const Terminal& t) { return t.name == name; });
		if (it != known.end()) add(*it);
	}
	for (const auto& terminal: known)
		add(terminal);
	return found;
}

} // namespace

LauncherContext::LauncherContext() {
	const auto here = Platform::ExecutableFolder();
	root            = FindRoot(here);
	// The package's programs executable again where an unpacker dropped the permission ("Permission
	// denied").
	for (const auto& file:
	     {here / "kyty_emulator", root / "kyty_emulator", root / "kyty_shader_precompile",
	      root / "run_ds.sh", root / "precompile-linux.sh", root / "get-dlss-linux.sh"}) {
		struct stat info {};
		if (stat(file.c_str(), &info) == 0 && S_ISREG(info.st_mode) &&
		    (info.st_mode & S_IXUSR) == 0)
			chmod(file.c_str(), info.st_mode | S_IXUSR |
			                        ((info.st_mode & S_IRGRP) != 0 ? S_IXGRP : 0) |
			                        ((info.st_mode & S_IROTH) != 0 ? S_IXOTH : 0));
	}
	run_script =
	    Exists(root / "run_ds.sh") ? root / "run_ds.sh" : root / "tools" / "local" / "run_ds.sh";
	if (const char* bin = std::getenv("KYTY_BIN");
	    bin != nullptr && *bin != 0 && IsExecutable(bin)) {
		emulator = fs::absolute(bin);
	} else {
		for (const auto& candidate:
		     {here / "kyty_emulator", root / "_Build" / "linux-nt" / "kyty_emulator",
		      root / "_Build" / "linux" / "kyty_emulator", root / "kyty_emulator"}) {
			if (IsExecutable(candidate)) {
				emulator = candidate;
				break;
			}
		}
	}
	for (const auto& candidate:
	     {emulator.empty() ? fs::path() : emulator.parent_path() / "kyty_shader_precompile",
	      root / "kyty_shader_precompile", root / "_Build" / "linux" / "kyty_shader_precompile"}) {
		if (!candidate.empty() && IsExecutable(candidate)) {
			precompile_tool = candidate;
			break;
		}
	}
	if (Exists(root / "precompile-linux.sh")) precompile_script = root / "precompile-linux.sh";
	for (const auto& candidate: {root / "get-dlss-linux.sh",
	                             root / "tools" / "local" / "dlss-bridge" / "get-dlss-linux.sh"}) {
		if (Exists(candidate)) {
			dlss_script = candidate;
			break;
		}
	}
	// The settings next to run_ds.sh (as the Windows launcher keeps them next to run-windows.ps1);
	// in the user's configuration folder when that folder is read-only (a system-wide install).
	fs::path settings_path = root / "launcher-settings.json";
	if (access(root.c_str(), W_OK) != 0) {
		const char*    config = std::getenv("XDG_CONFIG_HOME");
		const char*    home   = std::getenv("HOME");
		const fs::path base   = config != nullptr && *config != 0
		                            ? fs::path(config)
		                            : fs::path(home != nullptr ? home : "/tmp") / ".config";
		settings_path         = base / "KytyPS5" / "launcher-settings.json";
	}
	settings = std::make_unique<Settings>(settings_path);

	gpus = QueryGpus();
	for (const auto& gpu: gpus) {
		if (gpu.vendor == 0x10de && HasWord(gpu.name, "RTX")) rtx = true;
		if (gpu.vendor == 0x1002) amd = true;
	}
	struct sysinfo info {};
	if (sysinfo(&info) == 0) ram = static_cast<uint64_t>(info.totalram) * info.mem_unit;
	efficiency_cpus = EfficiencyCpus();
	terminals       = FindTerminals();
	mangohud_       = Platform::HasProgram("mangohud");
}

fs::path LauncherContext::DlssLibrary() const {
	std::vector<fs::path> folders;
	if (const char* dir = std::getenv("KYTY_DLSS_DIR"); dir != nullptr && *dir != 0) {
		folders.emplace_back(dir);
	} else if (!emulator.empty()) {
		// (The emulator looks next to its real file.)
		std::error_code error;
		const auto      real = fs::canonical(emulator, error);
		const auto      dir  = (error ? emulator : real).parent_path();
		folders.push_back(dir);
		folders.push_back(dir / "dlss");
	}
	for (const auto& folder: folders) {
		if (Exists(folder / "libkyty_dlss.so")) return folder / "libkyty_dlss.so";
	}
	return {};
}

const Terminal* LauncherContext::ChosenTerminal() const {
	const auto chosen = settings->Str("terminal");
	for (const auto& terminal: terminals)
		if (terminal.name == chosen) return &terminal;
	return terminals.empty() ? nullptr : &terminals.front();
}

fs::path LauncherContext::LogsFolder() const {
	return Exists(root / "_Build") ? root / "_Build" / "run-logs" : root / "logs";
}

fs::path LauncherContext::Readme() const {
	for (const auto& candidate: {root / "README.md", root / "docs" / "PORTABLE-README.md"})
		if (Exists(candidate)) return candidate;
	return {};
}
