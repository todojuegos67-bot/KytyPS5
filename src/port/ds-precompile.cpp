// "Demon's Souls" PC port: precompile-windows.ps1 in the exe (ds-precompile.h).
#include "port/ds-precompile.h"
#include "port/ds-i18n.h"
#include "port/ds-settings.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <dxgi.h>
#include <cwctype>
#endif

namespace Port::Precompile {

namespace fs = std::filesystem;

namespace {

Setup             g_setup;
std::mutex        g_mutex;
View              g_view;
std::atomic<bool> g_stop {false};
std::thread       g_worker;
std::atomic<bool> g_all_threads {false}; // (ds-pc-port13)
// Stop(false), the game closing: no merge at the stop. The merge loads, merges and writes the whole driver cache (GBs,
// minutes, not stoppable): with the window already closed the process lived on, and the next start found the
// "one game at a time" mutex taken and quit. The finished shard files stay on disk: the next start's merge-before
// takes them, and the precompile goes on from there.
std::atomic<bool> g_save_on_stop {true};

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

// (Its comment: below, at kShards.) The game's (Setup::cache_key, DriverCacheKey: one per "Bordes sombras" level).
std::wstring DriverKey() {
	return g_setup.cache_key.empty() ? std::wstring(L"1f3c29b70e536c8f5a5e812e793b33ac2eb6858b4810ed98ca2a51b9fa5eed97")
	                                 : std::wstring(g_setup.cache_key.begin(), g_setup.cache_key.end());
}

// The precompile runs under the game's executable name. AMD's Windows driver keys its pipelines (its pipeline cache,
// VK_KHR_pipeline_binary keys) by the executable's name: what "kyty_shader_precompile.exe" compiled, "Demon's
// Souls.exe" never found, and the game compiled those pipelines again while playing (issue #28, which
// precompile-windows.ps1 fixed for kyty_emulator.exe, but the port's own precompile did not). A hard link (a copy where
// the drive has no hard links) in _PipelineCache\precompile-as-game, with the DLLs beside it. Not in %TEMP% (issue #47:
// Windows Defender took it for a dropper). Made once a session (an update replaces the files the links point to).
fs::path PrecompileExe() {
	static const fs::path exe = [] {
		const fs::path source = g_setup.root / L"kyty_shader_precompile.exe";
		const fs::path dir    = g_setup.root / L"_PipelineCache" / L"precompile-as-game";
		const fs::path target = dir / L"Demon's Souls.exe";
		std::error_code ec;
		fs::create_directories(dir, ec);
		const auto link = [&](const fs::path& from, const fs::path& to) {
			std::error_code e;
			fs::remove(to, e);
			if (fs::exists(to, e)) {
				// In use (a precompile left running) or locked: kept when it is the same file.
				return fs::file_size(to, e) == fs::file_size(from, e) && fs::last_write_time(to, e) == fs::last_write_time(from, e);
			}
			if (CreateHardLinkW(to.c_str(), from.c_str(), nullptr)) return true;
			return CopyFileW(from.c_str(), to.c_str(), FALSE) != 0;
		};
		bool ok = link(source, target);
		for (const auto& entry: fs::directory_iterator(g_setup.root, ec)) {
			if (!ok) break;
			if (!entry.is_regular_file(ec)) continue;
			auto ext = entry.path().extension().wstring();
			std::transform(ext.begin(), ext.end(), ext.begin(), [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
			// (ds-pc-port46) Never NVIDIA's DLSS 5 DLL (the user's own nvngx_dlssnr.dll): the precompile does not use it, and it
			// is neither linked nor copied (it was hard-linked here with every DLL of the folder).
			auto name = entry.path().filename().wstring();
			std::transform(name.begin(), name.end(), name.begin(), [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
			if (name.starts_with(L"nvngx")) {
				fs::remove(dir / entry.path().filename(), ec);
				continue;
			}
			if (ext == L".dll") ok = link(entry.path(), dir / entry.path().filename());
		}
		// Without it, the old behavior (the precompile still runs; AMD GPUs compile again in play).
		return ok ? target : source;
	}();
	return exe;
}

// The primary adapter is AMD's (DXGI: no Vulkan instance in the launcher).
bool PrimaryGpuIsAmd() {
	IDXGIFactory1* factory = nullptr;
	if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) || factory == nullptr) return false;
	bool          amd     = false;
	IDXGIAdapter1* adapter = nullptr;
	if (SUCCEEDED(factory->EnumAdapters1(0, &adapter)) && adapter != nullptr) {
		DXGI_ADAPTER_DESC1 desc {};
		if (SUCCEEDED(adapter->GetDesc1(&desc))) amd = desc.VendorId == 0x1002u;
		adapter->Release();
	}
	factory->Release();
	return amd;
}

// Once: on an AMD GPU, a driver cache an earlier port warmed as "kyty_shader_precompile.exe" holds nothing the game
// finds (PrecompileExe). Its ".static-done" notes are dropped so every shard compiles again, now under the game's name.
void ForgetCachesWarmedUnderAnotherName() {
	static std::once_flag once;
	std::call_once(once, [] {
	std::error_code ec;
	const fs::path marker = g_setup.root / L"_PipelineCache" / L"precompile-as-game" / L"warmed-as-game-v1";
	if (fs::exists(marker, ec)) return;
	if (PrimaryGpuIsAmd()) {
		const fs::path local = g_setup.root / L"_PipelineCache" / L"local" / DriverKey();
		for (const auto& entry: fs::directory_iterator(local, ec)) {
			const auto name = entry.path().filename().wstring();
			if (name.size() > 12 && name.ends_with(L".static-done")) {
				std::error_code e;
				fs::remove(entry.path(), e);
			}
		}
	}
	fs::create_directories(marker.parent_path(), ec);
	std::ofstream(marker) << "precompile runs as \"Demon's Souls.exe\"\n";
	});
}

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
	std::wstring        command = Quote(PrecompileExe()) + L" --game \"" + Wide(g_setup.game) + L"\" " + args;
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
	// (ds-pc-port13: every thread: any CPU)
	if (false && g_setup.affinity != 0) SetProcessAffinityMask(pi.hProcess, static_cast<DWORD_PTR>(g_setup.affinity));
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

// (ds-pc-port37) The static store is "outdated" when the release's hints file is newer than it (StaticCacheCurrent): every
// update ships that file with the build's date, the same bytes or not, and every update precompiled everything again.
// The hints' content is remembered when a precompile ends (<store>.hints-sum); at the next check, the same content
// with a newer date gets the store's date back, so only new hints make a precompile.
fs::path HintsFile() {
	return g_setup.root / L"tools" / L"local" / L"static-precompile" / Wide("hints-" + g_setup.id + ".hints");
}
fs::path StoreFile() {
	fs::path best;
	fs::file_time_type at {};
	std::error_code ec;
	for (const wchar_t* ext: {L".binaries", L".blobs", L".bin"}) {
		const fs::path f = g_setup.root / L"_PipelineCache" / L"static" / (Wide(g_setup.id) + ext);
		if (!fs::exists(f, ec)) continue;
		const auto t = fs::last_write_time(f, ec);
		if (best.empty() || t > at) {
			best = f;
			at   = t;
		}
	}
	return best;
}
std::string HintsSum() {
	const std::string bytes = ReadAll(HintsFile());
	if (bytes.empty()) return {};
	uint64_t h = 1469598103934665603ull;
	for (const unsigned char c: bytes) h = (h ^ c) * 1099511628211ull;
	return std::to_string(bytes.size()) + ":" + std::to_string(h);
}
void RememberHints() {
	const fs::path store = StoreFile();
	if (store.empty()) return;
	std::ofstream(fs::path(store.wstring() + L".hints-sum"), std::ios::binary) << HintsSum();
}
// (ds-pc-port45) This PC's recorded play (warmup-v2's <id>.shaders, recorded-<id>.seeds) when the static store was last
// current: its bytes. The store's 63,849 pipelines against the emulator's 73,959 on the same PC (11-01): the port's
// precompile ran on a new install, before any play was recorded (the release's hints alone, no hinted seeds), and was
// never run again as play recorded the runtime variants (its specializations: the "not among the precompiled" ones).
uint64_t RecordedBytes() {
	uint64_t        total = 0;
	std::error_code ec;
	for (const auto& dir: fs::directory_iterator(g_setup.root / L"_PipelineCache" / L"warmup-v2", ec)) {
		const fs::path f = dir.path() / Wide(g_setup.id + ".shaders");
		if (fs::is_regular_file(f, ec)) total += fs::file_size(f, ec);
	}
	const fs::path manual = g_setup.root / Wide("recorded-" + g_setup.id + ".seeds");
	if (fs::is_regular_file(manual, ec)) total += fs::file_size(manual, ec);
	return total;
}
fs::path RecordedSumFile() {
	const fs::path store = StoreFile();
	return store.empty() ? fs::path {} : fs::path(store.wstring() + L".recorded-sum");
}
void RememberRecorded() {
	if (const auto f = RecordedSumFile(); !f.empty()) std::ofstream(f, std::ios::binary) << RecordedBytes();
}
// New recorded play since the store was made (2 % more and at least 256 KiB, or none noted while some exists): the
// precompile runs again, with it as hints (what the store holds is copied, not compiled again).
// (ds-pc-port48) The plan's options the store was made with (<store>.plan-options): a store of an older plan (without the
// native variants) is outdated once, its pipelines copied, the new ones compiled.
constexpr const char* kPlanOptions = "native-variants1";
bool PlanOptionsChanged() {
	const fs::path store = StoreFile();
	if (store.empty()) return false;
	const bool changed = ReadAll(fs::path(store.wstring() + L".plan-options")) != kPlanOptions;
	if (changed) std::printf("[port] precompile: the store's plan lacks the native variants: they are added\n");
	std::fflush(stdout);
	return changed;
}
void RememberPlanOptions() {
	if (const fs::path store = StoreFile(); !store.empty())
		std::ofstream(fs::path(store.wstring() + L".plan-options"), std::ios::binary) << kPlanOptions;
}

bool RecordedPlayGrew() {
	const auto f = RecordedSumFile();
	if (f.empty()) return false;
	const uint64_t now = RecordedBytes();
	if (now == 0) return false;
	const std::string saved = ReadAll(f);
	const uint64_t    then  = saved.empty() ? 0 : std::strtoull(saved.c_str(), nullptr, 10);
	const bool        grew  = now > then && (then == 0 || (now - then >= (256u << 10u) && (now - then) * 50 >= then));
	if (grew)
		std::printf("[port] precompile: this PC's recorded play grew (%llu -> %llu bytes): its variants into the static store\n",
		            static_cast<unsigned long long>(then), static_cast<unsigned long long>(now));
	std::fflush(stdout);
	return grew;
}

void KeepStoreCurrentForSameHints() {
	const fs::path store = StoreFile(), hints = HintsFile();
	std::error_code ec;
	if (store.empty() || !fs::exists(hints, ec)) return;
	const std::string saved = ReadAll(fs::path(store.wstring() + L".hints-sum"));
	if (saved.empty() || saved != HintsSum()) return;
	const auto store_time = fs::last_write_time(store, ec);
	if (ec || fs::last_write_time(hints, ec) <= store_time) return;
	fs::last_write_time(hints, store_time - std::chrono::seconds(1), ec);
	std::printf("[port] precompile: the release's hints are the same as the store's: no precompile for the update\n");
	std::fflush(stdout);
}

// The old (10-27c) way's driver cache, warmed with every static recipe: one multi-GB file the game loaded whole at every
// start, saved whole during play. Gone once the static store is current (the launcher's way below holds those pipelines).
void RemoveOldWarmedCache() {
	std::error_code ec;
	const fs::path local = g_setup.root / L"_PipelineCache" / L"local" / DriverKey();
	bool           warmed = false;
	for (const auto& entry: fs::directory_iterator(local, ec))
		if (entry.path().filename().wstring().ends_with(L".static-done")) warmed = true;
	if (!warmed) return;
	uint64_t freed = 0;
	for (const auto& entry: fs::directory_iterator(local, ec)) {
		const auto name = entry.path().filename().wstring();
		// (<title>.bin, its .static-done note, shard files; the table cache and played binaries are the game's own: kept.)
		if (name.find(L".table.") != std::wstring::npos || name.find(L".played.") != std::wstring::npos) continue;
		std::error_code e;
		const auto size = entry.is_regular_file(e) ? entry.file_size(e) : 0;
		if (fs::remove(entry.path(), e)) freed += size;
	}
	std::printf("[port] precompile: the old warmed driver cache removed (%llu MiB): the static store holds its pipelines\n",
	            static_cast<unsigned long long>(freed >> 20u));
	std::fflush(stdout);
}

// The pipelines a seed file's dry run planned as its own (precompile-windows.ps1 Get-PlannedPipelines), 0 without one.
long long PlannedPipelines(const fs::path& log) {
	const std::string text = ReadAll(log);
	const auto        at   = text.rfind("of them this seed file's");
	if (at == std::string::npos) return 0;
	const auto open = text.rfind('(', at);
	if (open == std::string::npos) return 0;
	std::smatch m;
	const std::string part = text.substr(open, at - open + 32);
	if (std::regex_search(part, m, std::regex(R"(\((\d+) pipelines, (\d+) of them)"))) return std::stoll(m[2].str());
	return 0;
}

// (ds-pc-port37) The emulator launcher's precompile (precompile-windows.ps1), the same steps: the static pipeline store
// (_PipelineCache/static/<title>_<version>.binaries or .blobs) the game reads one pipeline at a time as it needs it. The
// 10-27c way (every recipe warmed into one driver cache of several GB, loaded whole at each start and saved whole during
// play) is gone: the game started slowly, stalled while it was saved, and lost it all past 4 GiB.
//   1. recorded play: the release's hints, this PC's warmup recordings -> --make-hints, --apply-hints -> hinted seeds
//   2. --merge (what an interrupted run finished)
//   3. the plan: recorded first, then the seeds (--covered-by the recorded ones)
//   4. shards of ~100 planned pipelines: the recorded tier, --merge --keep-shards ("listo para jugar"), the seeds' tier
//   5. --merge --stamp --prune
void Work() {
	if (g_setup.refresh) {
		SetText("Preparando shaders: la carpeta del juego...");
		g_setup.refresh(g_setup);
	}
	const fs::path& root  = g_setup.root;
	const fs::path& seeds = g_setup.seeds;
	const std::string& id = g_setup.id;
	const bool all_cpu    = true; // (ds-pc-port47) every CPU thread of the PC, always (was the "relajado" choice too)
	const int  logical    = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
	// All CPU threads: one process of 3 threads per 3 CPUs (precompile-windows.ps1). Relaxed: the port's own bounds (10-27c,
	// a 32-thread PC rebooted with 11 processes): one process of 2 threads per 8 CPUs, at most 4; the plan on 4 threads.
	const int  threads    = all_cpu ? 3 : 2;
	const int  cpus       = all_cpu ? logical : std::max(1, std::min(g_setup.cpus, 4));
	int        jobs       = all_cpu ? std::max(1, (logical + threads - 1) / threads) : std::clamp(std::max(1, g_setup.cpus) / 8, 1, 4);
	std::printf("[port] precompile (as the launcher's): %s, %d processes of %d threads\n", all_cpu ? "all CPU threads" : "relaxed", jobs, threads);
	std::fflush(stdout);
	const auto memory_ok = [all_cpu] {
		MEMORYSTATUSEX m {};
		m.dwLength = sizeof(m);
		if (!GlobalMemoryStatusEx(&m)) return true;
		// (A process takes several GB while it compiles: 3 GB let them run the PC out of memory and fail.)
		const DWORDLONG Need = (all_cpu ? DWORDLONG {6} : DWORDLONG {8}) << 30u;
		return m.ullAvailPhys >= Need && m.ullAvailPageFile >= Need;
	};
	wchar_t temp[MAX_PATH + 1] = {};
	GetTempPathW(MAX_PATH, temp);
	const auto stamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
	g_logs           = root / L"logs" / fs::u8path(stamp + "-precompile");
	g_nvidia         = fs::path(temp) / fs::u8path("kyty-precompile-" + stamp);
	std::error_code ec;
	fs::create_directories(g_logs, ec);
	ForgetCachesWarmedUnderAnotherName();
	SetEnvironmentVariableW(L"KYTY_WARM_DRIVER_CACHE_KEY", nullptr);
	// The game's translation switches: a pipeline translated otherwise is another pipeline, never found by the game.
	SetEnvironmentVariableW(L"KYTY_SHADOW_FILTER", g_setup.shadow_filter.empty() ? nullptr : std::wstring(g_setup.shadow_filter.begin(), g_setup.shadow_filter.end()).c_str());
	const auto fail = [&](const std::string& what) {
		std::scoped_lock lock(g_mutex);
		g_view.failed  = !g_stop;
		g_view.running = false;
		g_view.text    = g_stop ? "Preparacion de shaders detenida: lo hecho se guarda y sigue la proxima vez." : what;
	};
	const auto run = [&](const std::string& name, const std::wstring& args, bool stoppable = true) {
		return RunOne(name, args, false, stoppable) == 0;
	};

	// 1. Recorded play (precompile-windows.ps1: the release's hints and this PC's warmup recordings, together).
	SetText("Preparando shaders: partidas grabadas...");
	fs::path recorded;
	{
		std::vector<fs::path> sources;
		const fs::path released = root / Wide("released-" + id + ".seeds"); // (made at the splash, ds-main PrepareShaders)
		if (fs::exists(released, ec)) sources.push_back(released);
		for (const auto& dir: fs::directory_iterator(root / L"_PipelineCache" / L"warmup-v2", ec)) {
			const fs::path f = dir.path() / Wide(id + ".shaders");
			if (fs::exists(f, ec)) sources.push_back(f);
		}
		const fs::path manual = root / Wide("recorded-" + id + ".seeds");
		if (fs::exists(manual, ec)) sources.push_back(manual);
		if (sources.size() == 1 && sources[0] == released) {
			recorded = released;
		} else if (!sources.empty()) {
			const fs::path together = root / Wide("recorded-play-" + id + ".hints");
			const fs::path hinted   = root / Wide("hinted-" + id + ".seeds");
			std::wstring   args     = L"--seeds " + Quote(seeds) + L" --make-hints " + Quote(together);
			for (const auto& f: sources) args += L" --recorded " + Quote(f);
			if (run("make-hints", args) &&
			    run("apply-hints", L"--seeds " + Quote(seeds) + L" --apply-hints " + Quote(together) + L" --out " + Quote(hinted)))
				recorded = hinted;
			else if (fs::exists(released, ec))
				recorded = released;
		}
		if (g_stop) return fail("");
	}
	const bool has_recorded = !recorded.empty() && fs::exists(recorded, ec);

	// 2. What an interrupted run finished.
	SetText("Preparando shaders: lo que ya estaba hecho...");
	if (!run("merge-before", L"--merge", false)) {
		std::printf("[port] precompile: merge-before failed; going on\n");
		std::fflush(stdout);
	}

	// 3. The plan (every pipeline once, with the work item that makes it first). (ds-pc-port37) Kept with the shards done in
	// _PipelineCache/static/precompile-resume while the inputs are the same (the seed file's and the recorded play's
	// bytes, the translation switches): a run that stopped at 50 % goes on at 50 %, without planning again.
	const fs::path resume_dir = root / L"_PipelineCache" / L"static" / L"precompile-resume";
	const auto     content    = [&](const fs::path& f) {
		const std::string b = ReadAll(f);
		uint64_t          h = 1469598103934665603ull;
		for (const unsigned char c: b) h = (h ^ c) * 1099511628211ull;
		return std::to_string(b.size()) + ":" + std::to_string(h);
	};
	const std::string resume_key = "v1|" + content(seeds) + "|" + (has_recorded ? content(recorded) : std::string("-")) + "|" +
	                               g_setup.shadow_filter + "|" + std::to_string(cpus > 0);
	bool resumed_plan = false;
	{
		std::error_code e;
		if (ReadAll(resume_dir / L"key.txt") == resume_key) {
			resumed_plan = true;
		} else {
			fs::remove_all(resume_dir, e);
			fs::create_directories(resume_dir, e);
		}
	}
	const fs::path plan_logs = resumed_plan ? resume_dir : g_logs;
	const fs::path plan = resume_dir / L"plan.bin";
	bool use_plan = true;
	if (!resumed_plan) {
	SetText("Preparando shaders: planificando (un minuto o dos)...");
	// (Not fatal: the first run's plan failed on players' PCs - every CPU thread translating every program at once, short of
	// memory - and only a second start went on. Tried again on 4 threads; without a plan the shards still make every
	// pipeline, some of them twice.)
	const auto plan_step = [&](const std::string& name, const std::wstring& args_no_threads) {
		if (run(name, args_no_threads + L" --threads " + std::to_wstring(cpus))) return true;
		if (g_stop) return false;
		std::error_code e;
		fs::remove(plan, e);
		return run(name + "-retry", args_no_threads + L" --threads " + std::to_wstring(std::min(cpus, 4)));
	};
	// (ds-pc-port48) The native XPR variants too (--native-variants; none by default): the 11-02 port47 log's runtime
	// misses ("not among the precompiled") were native and table variants while every shard of its precompile saved all it
	// planned (shards "N created from the store, 0 captured, 0 left out"): what the store lacks the plan never named. Same
	// flag in the plan and in every shard (a shard makes what the plan gives it).
	const std::wstring variants = L" --native-variants";
	if (has_recorded && !plan_step("plan-recorded", L"--seeds " + Quote(recorded) + L" --dry-run --plan " + Quote(plan) + variants)) use_plan = false;
	if (use_plan) {
		std::wstring args = L"--seeds " + Quote(seeds) + L" --dry-run --plan " + Quote(plan) + L" --static-inputs" + variants;
		if (has_recorded) args += L" --covered-by " + Quote(recorded);
		if (!plan_step("plan-seeds", args)) use_plan = false;
	}
	if (g_stop) return fail("");
	if (!use_plan) {
		std::error_code e;
		fs::remove(plan, e);
		std::printf("[port] precompile: no plan (logs\\*-precompile\\plan-*.out.log); the shards go on without one\n");
		std::fflush(stdout);
		// (The prefetch inputs the seeds' plan writes: made apart.)
		(void)run("inputs", L"--seeds " + Quote(seeds) + L" --no-pipelines --threads " + std::to_wstring(std::min(cpus, 4)) + L" --static-inputs");
	}

	{
		// (Kept for a resumed run: the plan and its counts when there is one, the key either way: without a plan the
		// shard counts are the defaults, the same next time.)
		std::error_code e;
		if (use_plan)
			for (const wchar_t* f: {L"plan-recorded.out.log", L"plan-seeds.out.log"})
				if (fs::exists(g_logs / f, e)) fs::copy_file(g_logs / f, resume_dir / f, fs::copy_options::overwrite_existing, e);
		std::ofstream(resume_dir / L"key.txt", std::ios::binary) << resume_key;
	}
	} else {
		std::error_code e;
		use_plan = fs::exists(plan, e);
		std::printf("[port] precompile: resuming (%s and the shards done of the last run)\n", use_plan ? "its plan" : "no plan");
		std::fflush(stdout);
	}

	// 4. The shards: ~100 planned pipelines each, the recorded tier first.
	struct Item {
		fs::path    file;
		int         i = 0, n = 1;
		bool        first = false;
	};
	std::deque<Item> first, second;
	if (has_recorded) {
		const long long planned = PlannedPipelines(plan_logs / L"plan-recorded.out.log");
		const int       count   = static_cast<int>(std::max<long long>(1, (std::max<long long>(planned, 100) + 99) / 100));
		for (int i = 0; i < count; i++) first.push_back({recorded, i, count, true});
	}
	{
		const long long planned = PlannedPipelines(plan_logs / L"plan-seeds.out.log");
		const int       count   = static_cast<int>(std::max<long long>(512, (planned + 99) / 100));
		for (int i = 0; i < count; i++) second.push_back({seeds, i, count, false});
	}
	// (ds-pc-port37) The shards an earlier run finished (resume_dir/done.txt; a split shard is done when both halves are).
	std::set<std::string> done_set;
	{
		std::istringstream lines(ReadAll(resume_dir / L"done.txt"));
		for (std::string line; std::getline(lines, line);)
			if (!line.empty()) done_set.insert(line);
	}
	const auto done_name = [](const Item& it) { return std::string(it.first ? "r " : "s ") + std::to_string(it.i) + " " + std::to_string(it.n); };
	const std::function<bool(const Item&, int)> is_done = [&](const Item& it, int depth) -> bool {
		if (done_set.contains(done_name(it))) return true;
		if (depth >= 8 || it.n >= 65536) return false;
		return is_done({it.file, it.i, 2 * it.n, it.first}, depth + 1) && is_done({it.file, it.i + it.n, 2 * it.n, it.first}, depth + 1);
	};
	int total = static_cast<int>(first.size() + second.size()), done = 0;
	for (auto* q: {&first, &second}) {
		std::deque<Item> left;
		for (const auto& it: *q)
			if (is_done(it, 0)) done++;
			else left.push_back(it);
		q->swap(left);
	}
	const bool resumed   = done > 0;
	const int  resumed_p = total > 0 ? done * 100 / total : 0;
	std::ofstream done_log(resume_dir / L"done.txt", std::ios::binary | std::ios::app);
	if (resumed) std::printf("[port] precompile: %d of %d shards done before: going on at %d %%\n", done, total, resumed_p);
	{
		std::scoped_lock lock(g_mutex);
		g_view.total       = total;
		g_view.done        = done;
		g_view.resumed_pct = resumed_p;
	}
	struct Running {
		Item        item;
		Child       c;
		std::string name;
	};
	std::vector<Running> running;
	std::map<std::string, int> retries;
	const auto kill_all = [&] {
		for (auto& r: running) {
			TerminateProcess(r.c.process, 1);
			WaitForSingleObject(r.c.process, INFINITE);
			Finish(r.c);
		}
		running.clear();
	};
	const auto begin      = std::chrono::steady_clock::now();
	auto       last_start = begin - std::chrono::seconds(60);
	bool       tier_merged = first.empty();
	while (!first.empty() || !second.empty() || !running.empty()) {
		if (g_stop) {
			kill_all();
			if (g_save_on_stop) {
				SetText("Preparando shaders: guardando lo hecho...");
				(void)run("merge-stopped", L"--merge", false);
			}
			return fail("");
		}
		// The recorded tier done: merged (shards kept for the final merge) so the game can use it now.
		if (!tier_merged && first.empty() && running.empty()) {
			tier_merged = true;
			SetText("Preparando shaders: guardando lo de las partidas grabadas...");
			(void)RunOne("merge-recorded", L"--merge --keep-shards", false, false);
			std::scoped_lock lock(g_mutex);
			g_view.playable = true;
		}
		std::deque<Item>* pending = !first.empty() ? &first : tier_merged ? &second : nullptr;
		bool waiting_memory = false;
		while (pending != nullptr && !pending->empty() && static_cast<int>(running.size()) < jobs) {
			if (!memory_ok()) {
				waiting_memory = true;
				break;
			}
			// (One starting at a time: each creates its Vulkan device and translates its shaders first.)
			if (!running.empty() && std::chrono::steady_clock::now() - last_start < std::chrono::milliseconds(all_cpu ? 1000 : 3000)) break;
			last_start = std::chrono::steady_clock::now();
			const Item item = pending->front();
			pending->pop_front();
			const std::string name = "shard" + std::to_string(item.i) + "of" + std::to_string(item.n) + (item.first ? "-recorded" : "-seeds");
			Child c = Launch(name, L"--seeds " + Quote(item.file) + L" --shard " + std::to_wstring(item.i) + L"/" + std::to_wstring(item.n) +
			                           L" --threads " + std::to_wstring(threads) + variants + (use_plan ? L" --plan " + Quote(plan) : std::wstring()),
			                 true);
			if (c.process == nullptr) {
				kill_all();
				return fail("No se pudo arrancar kyty_shader_precompile.exe.");
			}
			running.push_back({item, c, name});
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
			if (code == 3 && r.item.n < 65536) {
				// (Too many pipelines for one shard, NVIDIA's dictionary: split in two, as the launcher's.)
				auto& q = r.item.first ? first : second;
				q.push_back({r.item.file, r.item.i, 2 * r.item.n, r.item.first});
				q.push_back({r.item.file, r.item.i + r.item.n, 2 * r.item.n, r.item.first});
				total++;
				continue;
			}
			if (code != 0) {
				// (port15, kept) A shard that dies (the driver under many compiling processes at once, a process short of memory)
				// is tried again with one process fewer each time, instead of failing the whole run; 4 failures of the same
				// shard stop it.
				const int tries = ++retries[r.name];
				std::printf("[port] precompile: %s failed (exit %d, try %d); again with %d processes\n", r.name.c_str(), code, tries,
				            std::max(1, jobs - 1));
				std::fflush(stdout);
				if (tries < 4) {
					jobs = std::max(1, jobs - 1);
					(r.item.first ? first : second).push_front(r.item);
					continue;
				}
				kill_all();
				(void)run("merge-failed", L"--merge", false);
				return fail("Fallo un trozo de la preparacion de shaders (" + r.name + "). Lo hecho se guarda. Mira logs\\*-precompile.");
			}
			done++;
			done_log << done_name(r.item) << "\n";
			done_log.flush();
		}
		{
			std::scoped_lock lock(g_mutex);
			g_view.total       = total;
			g_view.done        = done;
			g_view.waiting_mem = waiting_memory;
			const int pct      = total > 0 ? done * 100 / total : 0;
			std::string eta    = "calculando el tiempo restante";
			g_view.eta_s       = -1;
			if (done - (total > 0 ? resumed_p * total / 100 : 0) > 0) {
				const int       here  = done - resumed_p * total / 100;
				const auto      spent = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count();
				const long long left  = spent * (total - done) / here;
				g_view.eta_s          = left;
				eta = "quedan unos " + (left >= 3600 ? std::to_string(left / 3600) + " h " + std::to_string(left % 3600 / 60) + " min"
				                                     : std::to_string(std::max<long long>(1, left / 60)) + " min");
			}
			g_view.text = "Ten paciencia: se estan precompilando los shaders del juego (solo la primera vez).\n" + std::to_string(pct) +
			              " %  (" + std::to_string(done) + " / " + std::to_string(total) + " partes, " + eta + ")" +
			              (resumed_p > 0 ? "  (retomada desde el " + std::to_string(resumed_p) + " %)" : std::string()) +
			              (waiting_memory ? "  [esperando memoria libre]" : "");
		}
	}

	// 5. The final merge: the store of this run's pipelines, stamped.
	SetText("Preparando shaders: guardando la cache...");
	// (Resumed: no --prune, which keeps only this run's pipelines: the earlier runs' shards were merged and are not made again.)
	if (!run("merge", resumed ? L"--merge --stamp" : L"--merge --stamp --prune", false))
		return fail("No se pudo guardar la cache de shaders (merge). Mira logs\\*-precompile.");
	done_log.close();
	fs::remove_all(resume_dir, ec);
	fs::remove_all(g_nvidia, ec);
	RememberHints();
	RememberRecorded(); // (ds-pc-port45: the recorded play this store holds)
	RememberPlanOptions(); // (ds-pc-port48)
	RemoveOldWarmedCache();
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

void SetAllThreads(bool all) {
	g_all_threads = all;
}
bool AllThreads() {
	return g_all_threads;
}

void Choose(const std::string& ini, Settings* settings) {
	if (settings == nullptr) return;
	if (settings->precompile_threads != "todos" && settings->precompile_threads != "relajado") {
		bool all = false;
#if defined(_WIN32)
		const std::string text = I18n::T("\xC2\xBFUsar todos los hilos de la CPU para precompilar m\xC3\xA1s r\xC3\xA1pido? (S\xC3\xAD = m\xC3\xA1xima "
		                                 "velocidad, el PC ir\xC3\xA1 cargado; No = modo relajado)");
		const std::string title = I18n::T("Preparar shaders");
		all = MessageBoxW(nullptr, Wide(text).c_str(), Wide(title).c_str(), MB_YESNO | MB_ICONQUESTION | MB_TOPMOST | MB_SETFOREGROUND) == IDYES;
#endif
		settings->precompile_threads = all ? "todos" : "relajado";
		// (only this key into the file: the menu's other edits stay unsaved until it saves them)
		Settings on_disk;
		if (!ini.empty() && LoadSettings(ini, &on_disk)) {
			on_disk.precompile_threads = settings->precompile_threads;
			(void)SaveSettings(ini, on_disk);
		} else if (!ini.empty()) {
			(void)SaveSettings(ini, *settings);
		}
	}
	SetAllThreads(settings->precompile_threads == "todos");
}

bool CacheCurrent() {
#if defined(_WIN32)
	if (g_setup.refresh) g_setup.refresh(g_setup);
	if (g_setup.id.empty()) return false;
	const fs::path out = g_setup.root / L"logs" / L"precompile-status.txt";
	std::error_code ec;
	fs::create_directories(out.parent_path(), ec);
	g_logs   = g_setup.root / L"logs";
	g_nvidia = g_setup.root / L"logs" / L"status-cache";
	ForgetCachesWarmedUnderAnotherName();
	SetEnvironmentVariableW(L"KYTY_WARM_DRIVER_CACHE_KEY", nullptr);
	SetEnvironmentVariableW(L"KYTY_SHADOW_FILTER", g_setup.shadow_filter.empty() ? nullptr : std::wstring(g_setup.shadow_filter.begin(), g_setup.shadow_filter.end()).c_str());
	KeepStoreCurrentForSameHints();
	// (ds-pc-port37) Current as the launcher's: the static store is this GPU's and driver's and not older than the seeds.
	const int code = RunOne("precompile-status", L"--seeds " + Quote(g_setup.seeds) + L" --status");
	const std::string text = ReadAll(g_logs / L"precompile-status.out.log");
	bool current = code == 0 && text.find("static cache current") != std::string::npos;
	if (current && RecordedPlayGrew()) current = false; // (ds-pc-port45)
	if (current && PlanOptionsChanged()) current = false; // (ds-pc-port48)
	if (current) {
		RememberHints(); // (for the next update: its hints compared by content)
		RememberRecorded();
		RemoveOldWarmedCache();
	}
	return current;
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

void Stop(bool save) {
	g_save_on_stop = save;
	g_stop         = true;
	if (g_worker.joinable()) g_worker.join();
	std::scoped_lock lock(g_mutex);
	g_view.running = false;
}

View State() {
	std::scoped_lock lock(g_mutex);
	return g_view;
}

} // namespace Port::Precompile
