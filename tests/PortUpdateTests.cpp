// The Demon's Souls PC port's in-place update (src/port/ds-update.cpp): copy, exclusions, settings.ini merge, backups,
// rollback, restore and the .old clean-up, on a temporary folder. Plain std::filesystem: runs on Linux too:
//   g++ -std=c++20 -Isrc tests/PortUpdateTests.cpp src/port/ds-update.cpp src/port/ds-settings.cpp -o port_update_tests
// Also BuildLaunch's environment against Launcher-souls.ps1 (10-27): the ray tracing rows ($hwTbl), the 16 GB profile.
#include "port/ds-settings.h"
#include "port/ds-update.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using namespace Port::Update;

namespace {

int failures = 0;

#define CHECK(cond)                                                                                                                        \
	do {                                                                                                                                   \
		if (!(cond)) {                                                                                                                     \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                                                    \
			failures++;                                                                                                                    \
		}                                                                                                                                  \
	} while (0)

void Put(const fs::path& p, const std::string& text) {
	fs::create_directories(p.parent_path());
	std::ofstream(p, std::ios::binary | std::ios::trunc) << text;
}

std::string Get(const fs::path& p) {
	std::ifstream in(p, std::ios::binary);
	return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

const char* kUserIni = "[Juego]\ncarpeta_juego=D:\\DS\nidioma=1\n\n[Graficos]\nmodo_souls=ultra\nsombras=bajo\n";
const char* kNewIni  = "[Juego]\ncarpeta_juego=\nidioma=3\n\n[Graficos]\nmodo_souls=ninguno\n; Sombras\nsombras=maximo\n"
                       "; Ray tracing: reflejos\nrt_reflejos=0\n\n[Actualizacion]\n; nueva\nnueva_clave=7\n";

// The installed port: v1 files plus everything that is the user's.
void MakeInstalled(const fs::path& root) {
	fs::remove_all(root);
	Put(root / kExeName, "exe v1");
	Put(root / "kyty_shader_precompile.exe", "pre v1");
	Put(root / "amd_fidelityfx_vk.dll", "ffx v1");
	Put(root / "libxess.dll", "xess v1");
	Put(root / "streamline" / "sl.interposer.dll", "sl v1");
	Put(root / "tools" / "local" / "static-precompile" / "pass-states.json", "states v1");
	Put(root / "LEEME.txt", "leeme v1");
	Put(root / "settings.ini", kUserIni);
	Put(root / "_PipelineCache" / "cache.bin", "my cache");
	Put(root / "_PipelineCache" / "LEEME.txt", "cache leeme");
	Put(root / "_SaveData" / "slot0", "my save");
	Put(root / "seeds-PPSA01341_01.007.000.seeds", "my seeds");
	Put(root / "seeds-PPSA01341_01.007.000.seeds.program", "stamp");
	Put(root / "logs" / "a.out.log", "log");
	Put(root / "kbm_curve.txt", "curve");
	Put(root / "Juego" / "eboot.bin", "the game");
}

// A release as the CI zips it: "Demon's Souls PC\..." with its own _PipelineCache\LEEME.txt and settings.ini.
fs::path MakeRelease(const fs::path& dir, const std::string& v) {
	fs::remove_all(dir);
	const fs::path r = dir / "Demon's Souls PC";
	Put(r / kExeName, "exe " + v);
	Put(r / "kyty_shader_precompile.exe", "pre " + v);
	Put(r / "amd_fidelityfx_vk.dll", "ffx " + v);
	Put(r / "libxess.dll", "xess " + v);
	Put(r / "nuevo.dll", "new " + v);
	Put(r / "streamline" / "sl.interposer.dll", "sl " + v);
	Put(r / "tools" / "local" / "static-precompile" / "pass-states.json", "states " + v);
	Put(r / "LEEME.txt", "leeme " + v);
	Put(r / "settings.ini", kNewIni);
	Put(r / "_PipelineCache" / "LEEME.txt", "shipped cache leeme");
	Put(r / "_SaveData" / "slot0", "shipped save");
	Put(r / "x.seeds", "shipped seeds");
	return dir;
}

void UserDataKept(const fs::path& root) {
	CHECK(Get(root / "_PipelineCache" / "cache.bin") == "my cache");
	CHECK(Get(root / "_PipelineCache" / "LEEME.txt") == "cache leeme");
	CHECK(Get(root / "_SaveData" / "slot0") == "my save");
	CHECK(Get(root / "seeds-PPSA01341_01.007.000.seeds") == "my seeds");
	CHECK(Get(root / "seeds-PPSA01341_01.007.000.seeds.program") == "stamp");
	CHECK(Get(root / "kbm_curve.txt") == "curve");
	CHECK(Get(root / "Juego" / "eboot.bin") == "the game");
	CHECK(!fs::exists(root / "x.seeds"));
}

void TestExcluded() {
	CHECK(IsExcluded("_PipelineCache/a"));
	CHECK(IsExcluded("_pipelinecache"));
	CHECK(IsExcluded("_SaveData/x/y"));
	CHECK(IsExcluded("_backup_update/v1/Demon's Souls.exe"));
	CHECK(IsExcluded("settings.ini"));
	CHECK(IsExcluded("a.seeds"));
	CHECK(IsExcluded("a.seeds.program"));
	CHECK(IsExcluded("Demon's Souls.exe.old"));
	CHECK(IsExcluded("../x.dll"));
	CHECK(IsExcluded("Juego/eboot.bin", "Juego"));
	CHECK(!IsExcluded("Juego2/eboot.bin", "Juego"));
	CHECK(!IsExcluded(kExeName));
	CHECK(!IsExcluded("streamline/sl.interposer.dll"));
	CHECK(!IsExcluded("tools/local/static-precompile/pass-states.json"));
}

void TestMerge() {
	const std::string m = MergeIni(kUserIni, kNewIni);
	CHECK(m.find("carpeta_juego=D:\\DS") != std::string::npos);
	CHECK(m.find("idioma=1") != std::string::npos);
	CHECK(m.find("idioma=3") == std::string::npos);
	CHECK(m.find("sombras=bajo") != std::string::npos);
	CHECK(m.find("sombras=maximo") == std::string::npos);
	CHECK(m.find("modo_souls=ultra") != std::string::npos);
	// New keys in their sections, with their comments.
	const auto graf = m.find("[Graficos]"), rt = m.find("rt_reflejos=0"), sec = m.find("[Actualizacion]");
	CHECK(rt != std::string::npos && rt > graf && rt < sec);
	CHECK(m.find("; Ray tracing: reflejos\nrt_reflejos=0") != std::string::npos);
	CHECK(m.find("[Actualizacion]\n; nueva\nnueva_clave=7") != std::string::npos);
	CHECK(MergeIni(m, kNewIni) == m); // (nothing more the second time)
}

void TestInstall(const fs::path& base) {
	const fs::path root = base / "port";
	MakeInstalled(root);
	const fs::path rel = FindReleaseRoot(MakeRelease(base / "rel2", "v2"));
	CHECK(!rel.empty());
	Options o;
	o.game_rel = "Juego";
	const Result r = Install(root, rel, "v1", "v2", o);
	CHECK(r.ok);
	CHECK(r.version == "v2");
	CHECK(r.backup == "v1");
	CHECK(Get(root / kExeName) == "exe v2");
	CHECK(Get(root / "amd_fidelityfx_vk.dll") == "ffx v2");
	CHECK(Get(root / "nuevo.dll") == "new v2");
	CHECK(Get(root / "streamline" / "sl.interposer.dll") == "sl v2");
	CHECK(Get(root / "tools" / "local" / "static-precompile" / "pass-states.json") == "states v2");
	UserDataKept(root);
	// settings.ini: the user's values, the new keys added.
	const std::string ini = Get(root / "settings.ini");
	CHECK(ini.find("idioma=1") != std::string::npos && ini.find("rt_reflejos=0") != std::string::npos && ini.find("nueva_clave=7") != std::string::npos);
	// The backup: the old engine, its version.
	const fs::path b = root / kBackupDir / "v1";
	CHECK(Get(b / kExeName) == "exe v1");
	CHECK(Get(b / "libxess.dll") == "xess v1");
	CHECK(Get(b / "streamline" / "sl.interposer.dll") == "sl v1");
	CHECK(Get(b / ".version") == "v1");
	CHECK(!fs::exists(b / "_SaveData"));
	CHECK(!fs::exists(b / "settings.ini"));
	// (Not in use here: the .old files were deleted at once.)
	CHECK(!fs::exists(root / (std::string(kExeName) + ".old")));
}

void TestRollback(const fs::path& base) {
	const fs::path root = base / "port";
	MakeInstalled(root);
	const fs::path rel = FindReleaseRoot(MakeRelease(base / "rel2", "v2"));
	Options        o;
	o.fail_on      = "streamline/sl.interposer.dll"; // after the exe and the DLLs of the folder were replaced
	const Result r = Install(root, rel, "v1", "v2", o);
	CHECK(!r.ok);
	CHECK(r.error.find("deshecho") != std::string::npos);
	CHECK(Get(root / kExeName) == "exe v1");
	CHECK(Get(root / "amd_fidelityfx_vk.dll") == "ffx v1");
	CHECK(Get(root / "kyty_shader_precompile.exe") == "pre v1");
	CHECK(Get(root / "streamline" / "sl.interposer.dll") == "sl v1");
	CHECK(!fs::exists(root / "nuevo.dll"));
	CHECK(Get(root / "settings.ini") == kUserIni);
	CHECK(!fs::exists(root / kBackupDir / "v1"));
	CHECK(!fs::exists(root / (std::string(kExeName) + ".old")));
	UserDataKept(root);
	// A release without the exe: nothing changes.
	const fs::path bad = base / "bad";
	fs::remove_all(bad);
	Put(bad / "x.dll", "x");
	const Result r2 = Install(root, bad, "v1", "v9");
	CHECK(!r2.ok);
	CHECK(Get(root / kExeName) == "exe v1");
	CHECK(FindReleaseRoot(bad).empty());
}

void TestKeepTwoAndRestore(const fs::path& base) {
	const fs::path root = base / "port";
	MakeInstalled(root);
	std::string cur = "v1";
	for (int v = 2; v <= 4; v++) {
		const std::string nv  = "v" + std::to_string(v);
		const fs::path    rel = FindReleaseRoot(MakeRelease(base / ("rel" + std::to_string(v)), nv));
		const Result      r   = Install(root, rel, cur, nv);
		CHECK(r.ok);
		cur = nv;
		std::this_thread::sleep_for(std::chrono::milliseconds(20)); // (distinct times)
	}
	auto list = ListBackups(root);
	CHECK(list.size() == 2);
	CHECK(list.size() == 2 && list[0].version == "v3" && list[1].version == "v2");
	CHECK(!fs::exists(root / kBackupDir / "v1"));
	CHECK(Get(root / kExeName) == "exe v4");
	// Back to v2: the current one (v4) is backed up first; v2 stays; v3 (the oldest other) goes.
	const Result r = Restore(root, "v2", cur);
	CHECK(r.ok);
	CHECK(r.version == "v2");
	CHECK(Get(root / kExeName) == "exe v2");
	CHECK(Get(root / "amd_fidelityfx_vk.dll") == "ffx v2");
	CHECK(Get(root / kBackupDir / "v4" / kExeName) == "exe v4");
	list = ListBackups(root);
	CHECK(list.size() == 2);
	CHECK(fs::exists(root / kBackupDir / "v2" / kExeName));
	CHECK(!fs::exists(root / kBackupDir / "v3"));
	UserDataKept(root);
	CHECK(Get(root / "settings.ini").find("idioma=1") != std::string::npos);
	// And undo it (back to v4).
	const Result r2 = Restore(root, "v4", "v2");
	CHECK(r2.ok && Get(root / kExeName) == "exe v4");
	CHECK(!Restore(root, "../x", "v4").ok);
	CHECK(!Restore(root, "nada", "v4").ok);
	// No backups at all.
	fs::remove_all(root / kBackupDir);
	CHECK(ListBackups(root).empty());
}

void TestOldFiles(const fs::path& base) {
	const fs::path root = base / "port";
	MakeInstalled(root);
	Put(root / (std::string(kExeName) + ".old"), "running exe");
	Put(root / "libxess.dll.old2", "x");
	Put(root / "streamline" / "sl.interposer.dll.old", "x");
	Put(root / "_SaveData" / "keep.old", "user");
	Put(root / "bold.txt", "not an .old");
	CHECK(RemoveOldFiles(root) == 3);
	CHECK(!fs::exists(root / (std::string(kExeName) + ".old")));
	CHECK(!fs::exists(root / "libxess.dll.old2"));
	CHECK(fs::exists(root / "_SaveData" / "keep.old"));
	CHECK(fs::exists(root / "bold.txt"));
	CHECK(Get(root / kExeName) == "exe v1");
}

std::string EnvOf(const Port::LaunchPlan& plan, const std::string& key, bool* has = nullptr) {
	for (const auto& [k, v]: plan.env)
		if (k == key) {
			if (has != nullptr) *has = true;
			return v;
		}
	if (has != nullptr) *has = false;
	return {};
}

void TestLaunch() {
	Port::PcInfo pc;
	pc.gpu    = "NVIDIA GeForce RTX 5080";
	pc.ram_gb = 32;
	Port::Settings s;
	// Defaults: no ray tracing, no GTAO / SSGI, no RAM flush, no 16 GB profile on 32 GB.
	auto p = Port::BuildLaunch(s, "PPSA01341", pc);
	bool has = false;
	EnvOf(p, "KYTY_HWRT", &has);
	CHECK(!has);
	EnvOf(p, "KYTY_GTAO", &has);
	CHECK(!has);
	EnvOf(p, "KYTY_SSGI", &has);
	CHECK(!has);
	CHECK(EnvOf(p, "KYTY_RAM_FLUSH_SECONDS").empty());
	EnvOf(p, "KYTY_LOWRAM", &has);
	CHECK(!has);
	EnvOf(p, "KYTY_RESIDENT_MB", &has);
	CHECK(!has);
	CHECK(EnvOf(p, "KYTY_TEXTURE_LOD_BIAS") == "auto"); // (default "nitidas")
	CHECK(EnvOf(p, "KYTY_ALIAS_STATS") == "0");
	CHECK(EnvOf(p, "KYTY_DS_FORCE_MODE") == "performance");
	// The 16 GB profile on a 32 GB PC: nothing; simulated: the launcher's -Ram16 with 1536.
	s.profile16 = true;
	p           = Port::BuildLaunch(s, "PPSA01341", pc);
	EnvOf(p, "KYTY_RESIDENT_MB", &has);
	CHECK(!has);
	s.simulate_ram16 = true;
	p                = Port::BuildLaunch(s, "PPSA01341", pc);
	CHECK(EnvOf(p, "KYTY_LOWRAM") == "1");
	CHECK(EnvOf(p, "KYTY_SIMULATE_RAM_MB") == "16384");
	CHECK(EnvOf(p, "KYTY_RESIDENT_MB") == "6144");
	CHECK(EnvOf(p, "KYTY_PREFETCH_MIN_AVAIL_MB") == "1536");
	s.simulate_ram16 = false;
	pc.ram_gb        = 16;
	p                = Port::BuildLaunch(s, "PPSA01341", pc);
	CHECK(EnvOf(p, "KYTY_PREFETCH_MIN_AVAIL_MB") == "1536");
	pc.ram_gb = 32;
	// (10-27b) Ray tracing rows of the s25 launcher: Reflejos, Sombras sol, Luces antorcha/lamparas; Luz volumetrica; Juego original.
	Port::Settings r;
	r.rt_fog = r.rt_metal = r.rt_ao = true; // (kept in settings.ini, never passed)
	p                     = Port::BuildLaunch(r, "PPSA01341", pc);
	EnvOf(p, "KYTY_HWRT_VOLUMETRIC", &has);
	CHECK(!has);
	r.rt_sun = r.rt_particles = r.rt_reflections = true;
	p                                            = Port::BuildLaunch(r, "PPSA01341", pc);
	CHECK(EnvOf(p, "KYTY_HWRT") == "1" && EnvOf(p, "KYTY_HWRT_SHADOW") == "2" && EnvOf(p, "KYTY_HWRT_LIGHTS") == "2" &&
	      EnvOf(p, "KYTY_HWRT_PARTICLES") == "2" && EnvOf(p, "KYTY_HWRT_REFL_MATERIAL") == "1");
	Port::Settings g;
	g.godrays = "3";
	p         = Port::BuildLaunch(g, "PPSA01341", pc);
	CHECK(EnvOf(p, "KYTY_GODRAYS") == "3" && EnvOf(p, "KYTY_RT_SHADOWS") == "1" && EnvOf(p, "KYTY_RT_SHADOWS_STRENGTH") == "0" &&
	      EnvOf(p, "KYTY_RT_SUN") == "0");
	g.pure_game = g.rt_reflections = true;
	p                              = Port::BuildLaunch(g, "PPSA01341", pc);
	CHECK(EnvOf(p, "KYTY_GODRAYS") == "0" && EnvOf(p, "KYTY_HWRT") == "0" && EnvOf(p, "KYTY_HWRT_REFL") == "0" && EnvOf(p, "KYTY_SSGI") == "0");
	CHECK(EnvOf(p, "KYTY_STATUS_HUD_SECONDS") == "0" && EnvOf(p, "KYTY_SHADER_WARMUP") == "1");
	Port::Settings a;
	a.shadows = "medio";
	p         = Port::BuildLaunch(a, "PPSA01341", pc);
	CHECK(EnvOf(p, "KYTY_GAME_CONVARS").find("sunShadowMaxDistance=125;sunShadowLightDistance=1000") != std::string::npos);
	{
		Port::Settings d;
		d.souls_mode = "ligero";
		d.res        = "dlss-calidad";
		d.fullscreen = false;
		d.height     = 1440;
		CHECK(Port::BuildLaunch(d, "PPSA01341", pc).ratio == 2.0); // (DLSS in Souls ligero: 720 input)
	}
	a.shadow_edges = "3";
	p              = Port::BuildLaunch(a, "PPSA01341", pc);
	CHECK(EnvOf(p, "KYTY_SHADOW_FILTER") == "3");
	// Every Souls mode: ray tracing off, and the fixed ones AMD x2 with our Anti-Lag in Ultra.
	for (const char* mode: {"ligero", "equilibrado", "ultra", "extremo", "low"}) {
		Port::Settings x = a;
		x.rt_reflections = x.rt_sun = x.rt_fog = x.rt_particles = x.rt_metal = true;
		x.souls_mode     = mode;
		Port::ApplyPreset(&x, mode);
		CHECK(!x.rt_reflections && !x.rt_sun && !x.rt_fog && !x.rt_particles && !x.rt_metal && !x.rt_ao);
		p = Port::BuildLaunch(x, "PPSA01341", pc);
		EnvOf(p, "KYTY_HWRT", &has);
		CHECK(!has);
		if (Port::FixedFg(x)) {
			CHECK(EnvOf(p, "KYTY_FSR3_FG") == "1" && EnvOf(p, "KYTY_LOWLAT_QUEUE") == "1");
			CHECK(EnvOf(p, "KYTY_FLIP_FPS") == (std::string(mode) == "equilibrado" ? "45" : "30"));
		}
	}
	// No DLSS on an AMD card.
	Port::PcInfo amd;
	amd.gpu    = "AMD Radeon RX 9070 XT";
	amd.nvidia = false;
	amd.amd    = true;
	Port::Settings d;
	d.res      = "dlss-calidad";
	d.framegen = "dlss-x3";
	p          = Port::BuildLaunch(d, "PPSA01341", amd);
	CHECK(EnvOf(p, "KYTY_UPSCALER") == "fsr");
	EnvOf(p, "KYTY_FRAMEGEN", &has);
	CHECK(!has);
	d.framegen = "amd-x2";
	p          = Port::BuildLaunch(d, "PPSA01341", amd);
	CHECK(EnvOf(p, "KYTY_ANTILAG") == "1"); // (auto: Anti-Lag 2 on AMD)
	// TU EQUIPO's table.
	Port::PcInfo t;
	t.gpu = "NVIDIA GeForce RTX 5080";
	Port::RecommendFor(&t);
	CHECK(t.recommended == "Modo Souls extremo");
	t.gpu = "AMD Radeon RX 6600";
	Port::RecommendFor(&t);
	CHECK(t.recommended == "Modo Souls ligero");
	t.gpu = "AMD Radeon RX 580";
	Port::RecommendFor(&t);
	CHECK(t.unsupported);
	t.gpu = "AMD Radeon RX 5700 XT";
	Port::RecommendFor(&t);
	CHECK(t.unsupported);
	t.gpu = "NVIDIA GeForce GTX 1660";
	Port::RecommendFor(&t);
	CHECK(t.recommended == "Modo Souls PC low profile");
}

} // namespace

int main() {
	TestLaunch();
	const fs::path base = fs::temp_directory_path() / ("ds-update-tests-" + std::to_string(std::rand()));
	TestExcluded();
	TestMerge();
	TestInstall(base);
	TestRollback(base);
	TestKeepTwoAndRestore(base);
	TestOldFiles(base);
	fs::remove_all(base);
	std::printf("%s (%d failures)\n", failures == 0 ? "PASS" : "FAILED", failures);
	return failures == 0 ? 0 : 1;
}
