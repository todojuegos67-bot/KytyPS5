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

// Runs one to its end (or kills it on Stop): its exit code, -1 when stopped.
int RunOne(const std::string& name, const std::wstring& args, bool claims = false) {
	Child c = Launch(name, args, claims);
	if (c.process == nullptr) return -2;
	while (WaitForSingleObject(c.process, 250) == WAIT_TIMEOUT) {
		if (g_stop) {
			TerminateProcess(c.process, 1);
			WaitForSingleObject(c.process, INFINITE);
			Finish(c);
			return -1;
		}
	}
	return Finish(c);
}

int64_t Planned(const fs::path& log) {
	const std::string text = ReadAll(log);
	static const std::regex re(R"(\((\d+) pipelines, (\d+) of them)");
	int64_t last = 0;
	for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it) last = std::stoll((*it)[2]);
	return last;
}

void Work() {
	const fs::path& root  = g_setup.root;
	const fs::path& seeds = g_setup.seeds;
	const std::string& id = g_setup.id;
	const int threads     = 3;
	const int jobs        = std::max(1, (g_setup.cpus + threads - 1) / threads);
	wchar_t   temp[MAX_PATH + 1] = {};
	GetTempPathW(MAX_PATH, temp);
	const auto stamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
	g_logs           = root / L"logs" / fs::u8path(stamp + "-precompile");
	g_nvidia         = fs::path(temp) / fs::u8path("kyty-precompile-" + stamp);
	std::error_code ec;
	fs::create_directories(g_logs, ec);
	const auto fail = [&](const std::string& what) {
		std::scoped_lock lock(g_mutex);
		g_view.failed  = !g_stop;
		g_view.running = false;
		g_view.text    = g_stop ? "Preparacion de shaders detenida: lo hecho se guarda y sigue la proxima vez." : what;
	};

	// The specializations of recorded play (precompile-windows.ps1 -Recorded '*'): the release's hints (released-<id>.seeds,
	// made before the game), this PC's warmup recordings and recorded-<id>.seeds, together into hinted-<id>.seeds.
	SetText("Preparando shaders: especializaciones de partidas grabadas...");
	const fs::path released = root / fs::u8path("released-" + id + ".seeds");
	const fs::path manual   = root / fs::u8path("recorded-" + id + ".seeds");
	std::vector<fs::path> sources;
	if (fs::exists(released, ec)) sources.push_back(released);
	for (const auto& e: fs::directory_iterator(root / L"_PipelineCache" / L"warmup-v2", ec)) {
		const fs::path f = e.path() / fs::u8path(id + ".shaders");
		if (fs::is_regular_file(f, ec)) sources.push_back(f);
	}
	if (fs::exists(manual, ec)) sources.push_back(manual);
	fs::path recorded;
	if (sources.size() == 1 && sources[0] == released) {
		recorded = released;
	} else if (!sources.empty()) {
		const fs::path together = root / fs::u8path("recorded-play-" + id + ".hints");
		std::wstring   args     = L"--seeds " + Quote(seeds) + L" --make-hints " + Quote(together);
		for (const auto& s: sources) args += L" --recorded " + Quote(s);
		const fs::path hinted = root / fs::u8path("hinted-" + id + ".seeds");
		if (RunOne("make-hints", args) == 0 &&
		    RunOne("apply-hints", L"--seeds " + Quote(seeds) + L" --apply-hints " + Quote(together) + L" --out " + Quote(hinted)) == 0)
			recorded = hinted;
		else if (fs::exists(released, ec)) recorded = released;
	}
	if (g_stop) return fail("");

	// What an interrupted run finished first.
	SetText("Preparando shaders: lo que ya estaba hecho...");
	if (RunOne("merge-before", L"--merge") != 0) return fail("No se pudo preparar los shaders (merge). Mira logs\\*-precompile.");
	// The plan: every pipeline once, with the work item that makes it first.
	const fs::path plan = g_logs / L"plan.bin";
	const std::wstring cpus = std::to_wstring(std::max(1, g_setup.cpus));
	SetText("Preparando shaders: planificando (un par de minutos)...");
	if (!recorded.empty() && RunOne("plan-recorded", L"--seeds " + Quote(recorded) + L" --dry-run --plan " + Quote(plan) + L" --threads " + cpus) != 0)
		recorded.clear();
	std::wstring seed_plan = L"--seeds " + Quote(seeds) + L" --dry-run --plan " + Quote(plan) + L" --threads " + cpus + L" --static-inputs";
	if (!recorded.empty()) seed_plan += L" --covered-by " + Quote(recorded);
	if (RunOne("plan-seeds", seed_plan) != 0) return fail("No se pudo planificar los shaders. Mira logs\\*-precompile.");

	struct Shard {
		fs::path file;
		int64_t  i, n;
	};
	std::deque<Shard> first, second;
	if (!recorded.empty()) {
		const int64_t n = std::max<int64_t>(1, (Planned(g_logs / L"plan-recorded.out.log") + 99) / 100);
		for (int64_t i = 0; i < n; i++) first.push_back({recorded, i, n});
	}
	{
		const int64_t n = std::max<int64_t>(512, (Planned(g_logs / L"plan-seeds.out.log") + 99) / 100);
		for (int64_t i = 0; i < n; i++) second.push_back({seeds, i, n});
	}
	int total = static_cast<int>(first.size() + second.size()), done = 0;
	{
		std::scoped_lock lock(g_mutex);
		g_view.total = total;
		g_view.done  = 0;
	}
	struct Running {
		Shard s;
		Child c;
		std::string name;
	};
	std::vector<Running> running;
	bool tier_merged = first.empty();
	const auto kill_all = [&] {
		for (auto& r: running) {
			TerminateProcess(r.c.process, 1);
			WaitForSingleObject(r.c.process, INFINITE);
			Finish(r.c);
		}
		running.clear();
	};
	while (!first.empty() || !second.empty() || !running.empty()) {
		if (g_stop) {
			kill_all();
			return fail("");
		}
		if (!tier_merged && first.empty() && running.empty()) {
			tier_merged = true;
			SetText("Preparando shaders: guardando las de partidas grabadas...");
			if (RunOne("merge-recorded", L"--merge --keep-shards") != 0) return fail("No se pudo guardar los shaders (merge). Mira logs\\*-precompile.");
			std::scoped_lock lock(g_mutex);
			g_view.playable = true;
		}
		std::deque<Shard>* pending = !first.empty() ? &first : (tier_merged ? &second : nullptr);
		while (pending != nullptr && !pending->empty() && static_cast<int>(running.size()) < jobs) {
			const Shard s = pending->front();
			pending->pop_front();
			const std::string name = "shard" + std::to_string(s.i) + "of" + std::to_string(s.n) + "-" + s.file.stem().string();
			Child c = Launch(name, L"--seeds " + Quote(s.file) + L" --shard " + std::to_wstring(s.i) + L"/" + std::to_wstring(s.n) +
			                           L" --threads " + std::to_wstring(threads) + L" --plan " + Quote(plan),
			                 true);
			if (c.process == nullptr) {
				kill_all();
				return fail("No se pudo arrancar kyty_shader_precompile.exe.");
			}
			running.push_back({s, c, name});
		}
		{
			std::scoped_lock lock(g_mutex);
			g_view.done = done;
			g_view.text = std::string(g_view.playable ? "Preparando el resto de shaders del juego" : "Preparando shaders de partidas grabadas") +
			              " " + std::to_string(done) + "/" + std::to_string(total);
		}
		Sleep(250);
		for (size_t k = 0; k < running.size();) {
			if (WaitForSingleObject(running[k].c.process, 0) == WAIT_TIMEOUT) {
				k++;
				continue;
			}
			Running r = running[k];
			running.erase(running.begin() + static_cast<std::ptrdiff_t>(k));
			const int code = Finish(r.c);
			if (code == 3) {
				// NVIDIA's dictionary: run again as two halves (precompile-windows.ps1; from the first pipeline on: the
				// driver's own cache, which the user has to delete).
				if (ReadAll(r.c.log).find("dictionary from here on (after 0 captured") != std::string::npos || r.s.n >= 65536) {
					kill_all();
					return fail("El driver de NVIDIA comprime con su propio diccionario: borra las carpetas GLCache y DXCache de "
					            "%LOCALAPPDATA%\\NVIDIA y vuelve a abrir el juego (lo hecho se guarda).");
				}
				auto& q = r.s.file == recorded ? first : second;
				q.push_back({r.s.file, r.s.i, 2 * r.s.n});
				q.push_back({r.s.file, r.s.i + r.s.n, 2 * r.s.n});
				total++;
			} else if (code != 0) {
				kill_all();
				return fail("Fallo un trozo de la preparacion de shaders (" + r.name + "). Mira logs\\*-precompile.");
			} else {
				done++;
			}
		}
		std::scoped_lock lock(g_mutex);
		g_view.total = total;
	}
	SetText("Preparando shaders: guardando la cache...");
	if (RunOne("merge", L"--merge --stamp --prune") != 0) return fail("No se pudo guardar la cache de shaders (merge). Mira logs\\*-precompile.");
	fs::remove_all(g_nvidia, ec);
	std::scoped_lock lock(g_mutex);
	g_view.running  = false;
	g_view.finished = true;
	g_view.playable = true;
	g_view.done     = g_view.total;
	g_view.text     = "Shaders listos: el juego ya no compila shaders al jugar.";
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
	const int code = RunOne("precompile-status", L"--seeds " + Quote(g_setup.seeds) + L" --status");
	const std::string text = ReadAll(g_logs / L"precompile-status.out.log");
	return code == 0 && text.find("static cache current") != std::string::npos;
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
