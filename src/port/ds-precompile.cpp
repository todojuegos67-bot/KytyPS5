// "Demon's Souls" PC port: precompile-windows.ps1 in the exe (ds-precompile.h).
#include "port/ds-precompile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace Port::Precompile {

namespace fs = std::filesystem;

namespace {

Setup             g_setup;
std::mutex        g_mutex;
View              g_view;
std::atomic<bool> g_stop {false};
std::thread       g_worker;

void SetText(const std::string& text) {
	std::scoped_lock lock(g_mutex);
	g_view.text = text;
}

std::string ReadAll(const fs::path& p) {
	std::ifstream in(p, std::ios::binary);
	return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::wstring Wide(const std::string& s) {
	return fs::u8path(s).wstring();
}

std::wstring Quote(const fs::path& p) {
	return L"\"" + p.wstring() + L"\"";
}

#if defined(_WIN32)
struct Child {
	HANDLE   process = nullptr;
	fs::path nvidia_cache;
	fs::path log;
};

fs::path g_logs, g_nvidia;

// A precompile process: below normal priority, the launch's CPUs, its own empty NVIDIA disk cache (precompile-windows.ps1).
Child Launch(const std::string& name, const std::wstring& args, bool claims) {
	Child c;
	c.nvidia_cache = g_nvidia / name;
	c.log          = g_logs / (name + ".out.log");
	std::error_code ec;
	fs::create_directories(c.nvidia_cache, ec);
	SetEnvironmentVariableW(L"__GL_SHADER_DISK_CACHE_PATH", c.nvidia_cache.c_str());
	SetEnvironmentVariableW(L"KYTY_PRECOMPILE_CLAIMS", claims ? (g_nvidia / L"claims").c_str() : nullptr);
	// (10-27c) Launched from Steam: its Vulkan layers (Fossilize recording every pipeline into steamapps\shadercache,
	// the overlay) stay out of the precompile's processes; their caches are the port's own, in _PipelineCache.
	SetEnvironmentVariableW(L"DISABLE_VK_LAYER_VALVE_steam_fossilize_1", L"1");
	SetEnvironmentVariableW(L"DISABLE_VK_LAYER_VALVE_steam_overlay_1", L"1");
	std::wstring        command = Quote(g_setup.root / L"kyty_shader_precompile.exe") + L" --game \"" + Wide(g_setup.game) + L"\" " + args;
	SECURITY_ATTRIBUTES sa {sizeof(sa), nullptr, TRUE};
	HANDLE              out = CreateFileW(c.log.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, 0, nullptr);
	STARTUPINFOW        si {};
	si.cb         = sizeof(si);
	si.dwFlags    = STARTF_USESTDHANDLES;
	si.hStdOutput = out;
	si.hStdError  = out;
	PROCESS_INFORMATION pi {};
	const BOOL ok = CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED | BELOW_NORMAL_PRIORITY_CLASS,
	                               nullptr, g_setup.root.c_str(), &si, &pi);
	SetEnvironmentVariableW(L"__GL_SHADER_DISK_CACHE_PATH", nullptr);
	SetEnvironmentVariableW(L"KYTY_PRECOMPILE_CLAIMS", nullptr);
	if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
	if (!ok) return c;
	// (10-27c) In a job that ends with the port: a closed or killed game leaves no precompile running on (the next
	// launch merges the shard files it finished and goes on from there).
	static HANDLE job = [] {
		HANDLE h = CreateJobObjectW(nullptr, nullptr);
		JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits {};
		limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
		if (h != nullptr) SetInformationJobObject(h, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
		return h;
	}();
	if (job != nullptr) AssignProcessToJobObject(job, pi.hProcess);
	if (g_setup.affinity != 0) SetProcessAffinityMask(pi.hProcess, static_cast<DWORD_PTR>(g_setup.affinity));
	// (Windows 11: no power throttling for a windowless process, as precompile-windows.ps1's Qos.High.)
	PROCESS_POWER_THROTTLING_STATE state {PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0};
	SetProcessInformation(pi.hProcess, ProcessPowerThrottling, &state, sizeof(state));
	ResumeThread(pi.hThread);
	CloseHandle(pi.hThread);
	c.process = pi.hProcess;
	return c;
}

int Finish(Child& c) {
	DWORD code = 1;
	if (c.process != nullptr) {
		GetExitCodeProcess(c.process, &code);
		CloseHandle(c.process);
		c.process = nullptr;
	}
	std::error_code ec;
	fs::remove_all(c.nvidia_cache, ec);
	return static_cast<int>(code);
}

// Runs one to its end (or kills it on Stop, unless `stoppable` is false): its exit code, -1 when stopped.
int RunOne(const std::string& name, const std::wstring& args, bool claims = false, bool stoppable = true) {
	Child c = Launch(name, args, claims);
	if (c.process == nullptr) return -2;
	while (WaitForSingleObject(c.process, 250) == WAIT_TIMEOUT) {
		if (stoppable && g_stop) {
			TerminateProcess(c.process, 1);
			WaitForSingleObject(c.process, INFINITE);
			Finish(c);
			return -1;
		}
	}
	return Finish(c);
}

// (10-27c) The game's driver cache key (ds-settings.cpp KYTY_DRIVER_CACHE_KEY): the precompile warms the cache the game
// loads, _PipelineCache/local/<key>/<title>.bin, with every recipe of the static inputs (_PipelineCache/static/<id>.shaders,
// made at the splash: 78,428 pipelines for 1.07), Shards of them below: 256 processes of 3 threads, as many at once as
// the launch's CPUs make (the NVIDIA driver compiles nearly one pipeline at a time per process).
constexpr const wchar_t* kDriverCacheKey = L"1f3c29b70e536c8f5a5e812e793b33ac2eb6858b4810ed98ca2a51b9fa5eed97";
constexpr int            kShards         = 256;

std::wstring ShardsArg() {
	return L" --shard 0/" + std::to_wstring(kShards);
}

// The shards the driver cache holds already (--driver-cache-status), or all of them.
std::vector<bool> ShardsDone(const fs::path& log, bool* all) {
	std::vector<bool> done(kShards, false);
	const std::string text = ReadAll(log);
	*all                   = text.find("driver cache warm") != std::string::npos;
	if (*all) return std::vector<bool>(kShards, true);
	const auto at = text.find("driver cache shards done:");
	if (at != std::string::npos) {
		const auto end = text.find('\n', at);
		std::string list = text.substr(at + 25, end == std::string::npos ? std::string::npos : end - at - 25);
		size_t      pos  = 0;
		while (pos < list.size()) {
			while (pos < list.size() && list[pos] == ' ') pos++;
			size_t next = pos;
			while (next < list.size() && list[next] >= '0' && list[next] <= '9') next++;
			if (next == pos) break;
			const int i = std::stoi(list.substr(pos, next - pos));
			if (i >= 0 && i < kShards) done[i] = true;
			pos = next;
		}
	}
	return done;
}

void Work() {
	const fs::path& root  = g_setup.root;
	const fs::path& seeds = g_setup.seeds;
	// (10-27c) Bounded: a 32-thread PC ran 11 processes of 3 threads at once and rebooted (RAM, video memory or the
	// driver). Now at most 4 processes of 2 threads (one per 8 CPUs), a new one only with 8 GB of RAM and of commit
	// left, and one starting at a time (each creates its Vulkan device and translates its shaders first).
	const int threads     = 2;
	const int jobs        = std::clamp(g_setup.cpus / 8, 1, 4);
	const auto memory_ok  = [] {
		MEMORYSTATUSEX m {};
		m.dwLength = sizeof(m);
		if (!GlobalMemoryStatusEx(&m)) return true;
		constexpr DWORDLONG Need = DWORDLONG {8} << 30u;
		return m.ullAvailPhys >= Need && m.ullAvailPageFile >= Need;
	};
	auto      last_start = std::chrono::steady_clock::now() - std::chrono::seconds(60);
	const auto begin     = std::chrono::steady_clock::now();
	int        done_here = 0;
	wchar_t   temp[MAX_PATH + 1] = {};
	GetTempPathW(MAX_PATH, temp);
	const auto stamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
	g_logs           = root / L"logs" / fs::u8path(stamp + "-precompile");
	g_nvidia         = fs::path(temp) / fs::u8path("kyty-precompile-" + stamp);
	std::error_code ec;
	fs::create_directories(g_logs, ec);
	SetEnvironmentVariableW(L"KYTY_WARM_DRIVER_CACHE_KEY", kDriverCacheKey);
	const std::wstring base  = L"--seeds " + Quote(seeds);
	const auto         merge = [&](const std::string& name) {
        return RunOne(name, base + L" --merge-driver-cache" + ShardsArg(), false, false) == 0;
	};
	const auto fail = [&](const std::string& what) {
		std::scoped_lock lock(g_mutex);
		g_view.failed  = !g_stop;
		g_view.running = false;
		g_view.text    = g_stop ? "Preparacion de shaders detenida: lo hecho se guarda y sigue la proxima vez." : what;
	};

	// What an interrupted run finished (its shard files), then what is left.
	SetText("Preparando shaders: lo que ya estaba hecho...");
	if (!merge("merge-before")) return fail("No se pudo preparar los shaders (merge). Mira logs\\*-precompile.");
	if (RunOne("status-before", base + L" --driver-cache-status" + ShardsArg()) != 0) return fail("");
	bool              all  = false;
	std::vector<bool> done_shards = ShardsDone(g_logs / L"status-before.out.log", &all);
	std::deque<int>   pending;
	for (int i = 0; i < kShards; i++)
		if (!done_shards[i]) pending.push_back(i);
	int done = kShards - static_cast<int>(pending.size()), merged_at = done;
	const int resumed_pct = done * 100 / kShards;
	{
		std::scoped_lock lock(g_mutex);
		g_view.total = kShards;
		g_view.done  = done;
	}
	struct Running {
		int         shard;
		Child       c;
		std::string name;
	};
	std::vector<Running> running;
	const auto kill_all = [&] {
		for (auto& r: running) {
			TerminateProcess(r.c.process, 1);
			WaitForSingleObject(r.c.process, INFINITE);
			Finish(r.c);
		}
		running.clear();
	};
	while (!pending.empty() || !running.empty()) {
		if (g_stop) {
			kill_all();
			// What finished goes into the cache now: the game starts with it.
			SetText("Preparando shaders: guardando lo hecho...");
			(void)merge("merge-stopped");
			return fail("");
		}
		bool waiting_memory = false;
		while (!pending.empty() && static_cast<int>(running.size()) < jobs) {
			if (!memory_ok()) {
				waiting_memory = true;
				break;
			}
			// (One starting at a time: 4 s apart.)
			if (!running.empty() && std::chrono::steady_clock::now() - last_start < std::chrono::seconds(4)) break;
			last_start = std::chrono::steady_clock::now();
			const int i = pending.front();
			pending.pop_front();
			const std::string name = "warm" + std::to_string(i) + "of" + std::to_string(kShards);
			Child c = Launch(name, base + L" --warm-driver-cache --shard " + std::to_wstring(i) + L"/" + std::to_wstring(kShards) +
			                           L" --threads " + std::to_wstring(threads),
			                 false);
			if (c.process == nullptr) {
				kill_all();
				return fail("No se pudo arrancar kyty_shader_precompile.exe.");
			}
			running.push_back({i, c, name});
		}
		{
			std::scoped_lock lock(g_mutex);
			g_view.done = done;
			const int pct = done * 100 / kShards;
			std::string eta = "calculando el tiempo restante";
			if (done_here > 0) {
				const auto spent = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count();
				const long long left = spent * (kShards - done) / done_here;
				eta = "quedan unos " + (left >= 3600 ? std::to_string(left / 3600) + " h " + std::to_string(left % 3600 / 60) + " min"
				                                      : std::to_string(std::max<long long>(1, left / 60)) + " min");
			}
			g_view.text = "Ten paciencia: se están precompilando los shaders del juego (solo la primera vez).\n" + std::to_string(pct) +
			              " %  (" + std::to_string(done) + " / " + std::to_string(kShards) + " partes, " + eta + ")" +
			              (resumed_pct > 0 ? "  (retomada desde el " + std::to_string(resumed_pct) + " %)"
			                   : std::string()) +
			              (waiting_memory ? "  [esperando memoria libre]" : "");
		}
		Sleep(250);
		for (size_t k = 0; k < running.size();) {
			if (WaitForSingleObject(running[k].c.process, 0) == WAIT_TIMEOUT) {
				k++;
				continue;
			}
			Running r = running[k];
			running.erase(running.begin() + static_cast<std::ptrdiff_t>(k));
			if (Finish(r.c) != 0) {
				kill_all();
				(void)merge("merge-failed");
				return fail("Fallo un trozo de la preparacion de shaders (" + r.name + "). Mira logs\\*-precompile.");
			}
			done++;
			done_here++;
		}
		// Merged as it goes (every 16 shards): a crash or a closed window keeps what was made.
		if (done - merged_at >= 16) {
			if (!merge("merge-" + std::to_string(done))) {
				kill_all();
				return fail("No se pudo guardar la cache de shaders (merge). Mira logs\\*-precompile.");
			}
			merged_at = done;
		}
	}
	SetText("Preparando shaders: guardando la cache...");
	if (!merge("merge")) return fail("No se pudo guardar la cache de shaders (merge). Mira logs\\*-precompile.");
	fs::remove_all(g_nvidia, ec);
	std::scoped_lock lock(g_mutex);
	g_view.running  = false;
	g_view.finished = true;
	g_view.playable = true;
	g_view.done     = g_view.total;
	g_view.text     = "Shaders listos: todos los pipelines del juego estan en la cache de esta grafica.";
}
#endif

} // namespace

void Configure(const Setup& setup) {
	g_setup = setup;
}

bool CacheCurrent() {
#if defined(_WIN32)
	const fs::path out = g_setup.root / L"logs" / L"precompile-status.txt";
	std::error_code ec;
	fs::create_directories(out.parent_path(), ec);
	g_logs   = g_setup.root / L"logs";
	g_nvidia = g_setup.root / L"logs" / L"status-cache";
	SetEnvironmentVariableW(L"KYTY_WARM_DRIVER_CACHE_KEY", kDriverCacheKey);
	// (10-27c) Current: the game's driver cache holds every recipe of this GPU's static inputs (--driver-cache-status).
	const int code = RunOne("precompile-status", L"--seeds " + Quote(g_setup.seeds) + L" --driver-cache-status" + ShardsArg());
	const std::string text = ReadAll(g_logs / L"precompile-status.out.log");
	return code == 0 && text.find("driver cache warm") != std::string::npos;
#else
	return true;
#endif
}

void Start() {
#if defined(_WIN32)
	{
		std::scoped_lock lock(g_mutex);
		if (g_view.running) return;
		g_view          = {};
		g_view.running  = true;
		g_view.text     = "Preparando shaders...";
	}
	if (g_worker.joinable()) g_worker.join();
	g_stop   = false;
	g_worker = std::thread(Work);
#endif
}

void Stop() {
	g_stop = true;
	if (g_worker.joinable()) g_worker.join();
	std::scoped_lock lock(g_mutex);
	g_view.running = false;
}

View State() {
	std::scoped_lock lock(g_mutex);
	return g_view;
}

} // namespace Port::Precompile
