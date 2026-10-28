// InjectorSetup.exe: pick a game's exe; its imports tell which graphics API it uses (DX11 / DX12 / Vulkan) and the
// matching files are copied next to it (files already there are kept as *.injector-bak). Uninstall removes exactly
// what the manifest (injector_install.txt next to the exe) lists and puts the backups back.
//   InjectorSetup.exe                       -> file dialog, then Install / Uninstall
//   InjectorSetup.exe install <game.exe>
//   InjectorSetup.exe uninstall <game.exe>
//   InjectorSetup.exe detect <game.exe>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {

std::wstring Lower(std::wstring s) {
	for (auto& c: s) c = static_cast<wchar_t>(towlower(c));
	return s;
}
std::string LowerA(std::string s) {
	for (auto& c: s) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
	return s;
}
std::wstring Dir(const std::wstring& path) {
	return path.substr(0, path.find_last_of(L"\\/") + 1);
}
std::wstring SelfDir() {
	wchar_t p[MAX_PATH] {};
	GetModuleFileNameW(nullptr, p, MAX_PATH);
	return Dir(p);
}
bool Exists(const std::wstring& p) {
	return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES;
}

struct PeInfo {
	bool ok = false, x64 = false;
	std::set<std::string> dlls;                       // imported DLLs (lower case), normal and delay-load
	std::set<std::string> vulkan_symbols;             // symbols imported from vulkan-1.dll
	std::string error;
};

PeInfo ReadImports(const std::wstring& exe) {
	PeInfo info;
	std::ifstream f(exe, std::ios::binary);
	if (!f) { info.error = "cannot open the file"; return info; }
	std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (data.size() < sizeof(IMAGE_DOS_HEADER)) { info.error = "not a PE file"; return info; }
	auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(data.data());
	if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew <= 0 || static_cast<size_t>(dos->e_lfanew) + 256 > data.size()) {
		info.error = "not a PE file";
		return info;
	}
	auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(data.data() + dos->e_lfanew);
	if (nt->Signature != IMAGE_NT_SIGNATURE) { info.error = "not a PE file"; return info; }
	info.x64 = nt->FileHeader.Machine == IMAGE_FILE_MACHINE_AMD64;
	IMAGE_DATA_DIRECTORY imports {}, delay {};
	if (nt->OptionalHeader.Magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC) {
		auto* o = reinterpret_cast<IMAGE_OPTIONAL_HEADER64*>(&nt->OptionalHeader);
		imports = o->DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
		delay   = o->DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
	} else {
		auto* o = reinterpret_cast<IMAGE_OPTIONAL_HEADER32*>(&nt->OptionalHeader);
		imports = o->DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
		delay   = o->DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
	}
	auto* sections = IMAGE_FIRST_SECTION(nt);
	const auto rva_to_off = [&](DWORD rva) -> size_t {
		for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
			const auto& s = sections[i];
			const DWORD size = std::max(s.Misc.VirtualSize, s.SizeOfRawData);
			if (rva >= s.VirtualAddress && rva < s.VirtualAddress + size) return rva - s.VirtualAddress + s.PointerToRawData;
		}
		return 0;
	};
	const auto str_at = [&](DWORD rva) -> std::string {
		const size_t off = rva_to_off(rva);
		if (off == 0 || off >= data.size()) return {};
		size_t end = off;
		while (end < data.size() && data[end] != 0 && end - off < 256) ++end;
		return std::string(data.data() + off, end - off);
	};
	const auto read_thunks = [&](DWORD rva, std::set<std::string>& out) {
		size_t off = rva_to_off(rva);
		for (int n = 0; off != 0 && off + 8 <= data.size() && n < 4096; ++n) {
			uint64_t thunk = 0;
			if (info.x64) { std::memcpy(&thunk, data.data() + off, 8); off += 8; }
			else { uint32_t t = 0; std::memcpy(&t, data.data() + off, 4); thunk = t; off += 4; }
			if (thunk == 0) break;
			const bool ordinal = info.x64 ? (thunk >> 63) != 0 : (thunk >> 31) != 0;
			if (ordinal) { out.insert("#ordinal"); continue; }
			out.insert(str_at(static_cast<DWORD>(thunk) + 2));
		}
	};
	if (imports.VirtualAddress != 0) {
		size_t off = rva_to_off(imports.VirtualAddress);
		for (; off != 0 && off + sizeof(IMAGE_IMPORT_DESCRIPTOR) <= data.size(); off += sizeof(IMAGE_IMPORT_DESCRIPTOR)) {
			IMAGE_IMPORT_DESCRIPTOR d {};
			std::memcpy(&d, data.data() + off, sizeof d);
			if (d.Name == 0) break;
			const std::string name = LowerA(str_at(d.Name));
			info.dlls.insert(name);
			if (name == "vulkan-1.dll") read_thunks(d.OriginalFirstThunk ? d.OriginalFirstThunk : d.FirstThunk, info.vulkan_symbols);
		}
	}
	if (delay.VirtualAddress != 0) {
		size_t off = rva_to_off(delay.VirtualAddress);
		for (; off != 0 && off + sizeof(IMAGE_DELAYLOAD_DESCRIPTOR) <= data.size(); off += sizeof(IMAGE_DELAYLOAD_DESCRIPTOR)) {
			IMAGE_DELAYLOAD_DESCRIPTOR d {};
			std::memcpy(&d, data.data() + off, sizeof d);
			if (d.DllNameRVA == 0) break;
			const std::string name = LowerA(str_at(d.DllNameRVA));
			info.dlls.insert(name);
			if (name == "vulkan-1.dll") read_thunks(d.ImportNameTableRVA, info.vulkan_symbols);
		}
	}
	info.ok = true;
	return info;
}

enum class Api { Unknown, DX11, DX12, Vulkan, DX9 };
const char* ApiName(Api a) {
	switch (a) {
	case Api::DX11: return "DirectX 11";
	case Api::DX12: return "DirectX 12";
	case Api::Vulkan: return "Vulkan";
	case Api::DX9: return "DirectX 9";
	default: return "unknown";
	}
}
Api Detect(const PeInfo& pe) {
	if (pe.dlls.count("d3d12.dll")) return Api::DX12;  // (DX12 games often import d3d11 too, for interop)
	if (pe.dlls.count("vulkan-1.dll")) return Api::Vulkan;
	if (pe.dlls.count("d3d11.dll")) return Api::DX11;
	if (pe.dlls.count("d3d9.dll")) return Api::DX9;
	if (pe.dlls.count("dxgi.dll")) return Api::DX11;  // dxgi only: most likely DX11 created through dxgi
	return Api::Unknown;
}

const std::wstring kManifest = L"injector_install.txt";

std::vector<std::wstring> FilesFor(Api api) {
	if (api == Api::DX11) return {L"d3d11.dll", L"dxgi.dll", L"winevulkan.dll", L"amd_fidelityfx_vk.dll", L"dxvk.conf", L"injector.ini"};
	if (api == Api::Vulkan) return {L"vulkan-1.dll", L"amd_fidelityfx_vk.dll", L"injector.ini"};
	return {};
}

bool AntiCheatPresent(const std::wstring& dir) {
	for (const wchar_t* name: {L"EasyAntiCheat", L"EasyAntiCheat_x64.dll", L"EasyAntiCheat_EOS_Setup.exe", L"BattlEye",
	                           L"EasyAntiCheat_launcher.exe", L"start_protected_game.exe"})
		if (Exists(dir + name)) return true;
	return false;
}

int Say(const std::wstring& text, UINT flags = MB_OK | MB_ICONINFORMATION) {
	if (GetConsoleWindow() != nullptr) std::wprintf(L"%ls\n", text.c_str());
	return MessageBoxW(nullptr, text.c_str(), L"Injector setup", flags);
}

bool Install(const std::wstring& exe) {
	const auto pe = ReadImports(exe);
	if (!pe.ok) { Say(L"No se pudo leer el exe: " + std::wstring(pe.error.begin(), pe.error.end()), MB_ICONERROR); return false; }
	const Api api = Detect(pe);
	const std::wstring dir = Dir(exe);
	if (!pe.x64) { Say(L"El exe es de 32 bits: solo se soportan juegos x64 en esta version.", MB_ICONERROR); return false; }
	if (Exists(dir + kManifest)) { Say(L"Ya instalado aqui (injector_install.txt). Desinstala primero.", MB_ICONWARNING); return false; }
	if (api == Api::DX12) {
		Say(L"DirectX 12 detectado: soporte previsto en la fase 2 (misma DLL via vkd3d-proton). Nada copiado.", MB_ICONWARNING);
		return false;
	}
	if (api == Api::Vulkan) {
		static const std::set<std::string> ours = {"vkGetInstanceProcAddr", "vkGetDeviceProcAddr", "vkCreateInstance",
		    "vkEnumerateInstanceExtensionProperties", "vkEnumerateInstanceLayerProperties", "vkEnumerateInstanceVersion"};
		for (const auto& s: pe.vulkan_symbols)
			if (!ours.count(s)) {
				Say(L"Juego Vulkan que importa " + std::wstring(s.begin(), s.end()) +
				    L" directamente: el proxy vulkan-1.dll de la fase 1 no lo exporta. No soportado aun.", MB_ICONWARNING);
				return false;
			}
	}
	if (api == Api::Unknown || api == Api::DX9) {
		Say(std::wstring(L"API detectada: ") + std::wstring(ApiName(api), ApiName(api) + std::strlen(ApiName(api))) +
		    L". No soportada.", MB_ICONWARNING);
		return false;
	}
	if (AntiCheatPresent(dir) &&
	    Say(L"Se detecto un anti-cheat (EasyAntiCheat / BattlEye) en la carpeta del juego.\n\n"
	        L"Usalo SOLO offline / un jugador. Inyectar DLLs en un juego online con anti-cheat puede llevar a un ban.\n\n"
	        L"Continuar?", MB_YESNO | MB_ICONWARNING) != IDYES)
		return false;
	const std::wstring payload = SelfDir() + L"payload\\";
	std::wofstream manifest(dir + kManifest);
	manifest << L"api=" << (api == Api::DX11 ? L"dx11" : L"vulkan") << L"\n";
	std::wstring done;
	// DX11: a vulkan-1.dll left next to the exe (an older install of ours) is moved aside; DXVK must find winevulkan.dll.
	if (api == Api::DX11 && Exists(dir + L"vulkan-1.dll")) {
		MoveFileExW((dir + L"vulkan-1.dll").c_str(), (dir + L"vulkan-1.dll.injector-bak").c_str(), MOVEFILE_REPLACE_EXISTING);
		manifest << L"backup=vulkan-1.dll\n";
		done += L"  (vulkan-1.dll -> vulkan-1.dll.injector-bak)\n";
	}
	for (const auto& name: FilesFor(api)) {
		const std::wstring src = payload + name, dst = dir + name;
		if (!Exists(src)) { Say(L"Falta " + src, MB_ICONERROR); return false; }
		if (name == L"injector.ini" && Exists(dst)) continue;  // keep the user's settings
		if (Exists(dst)) {
			MoveFileExW(dst.c_str(), (dst + L".injector-bak").c_str(), MOVEFILE_REPLACE_EXISTING);
			manifest << L"backup=" << name << L"\n";
		}
		if (!CopyFileW(src.c_str(), dst.c_str(), FALSE)) { Say(L"No se pudo copiar " + name + L" (permisos?)", MB_ICONERROR); return false; }
		manifest << L"file=" << name << L"\n";
		done += L"  " + name + L"\n";
	}
	Say(std::wstring(L"Instalado para ") + std::wstring(ApiName(api), ApiName(api) + std::strlen(ApiName(api))) + L" en\n" + dir +
	    L"\n\n" + done + L"\nInicia el juego y pulsa F2 (menu), F3 (frame generation). Log: injector.log junto al exe.");
	return true;
}

// Our DLL (as winevulkan.dll or vulkan-1.dll), recognized by a string only it contains: removed even when no manifest
// lists it (10-28: an older manifest without winevulkan.dll left it behind, and the user's own DXVK then loaded it).
bool IsOurDll(const std::wstring& path) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	const std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	return data.find("injector phase 1 (generic proxy") != std::string::npos;
}

bool Uninstall(const std::wstring& exe) {
	const std::wstring dir = Dir(exe);
	std::wstring removed, restored;
	std::set<std::wstring> backups;
	std::wifstream manifest(dir + kManifest);
	if (manifest) {
		std::wstring line;
		while (std::getline(manifest, line)) {
			if (line.rfind(L"file=", 0) == 0) {
				const std::wstring name = line.substr(5);
				if (name == L"injector.ini") continue;  // settings stay
				if (DeleteFileW((dir + name).c_str())) removed += L"  " + name + L"\n";
			} else if (line.rfind(L"backup=", 0) == 0) {
				backups.insert(line.substr(7));
			}
		}
		manifest.close();
	}
	// Every file the injector ever installs, whatever the manifest says: our DLL under either name.
	for (const wchar_t* name: {L"winevulkan.dll", L"vulkan-1.dll"})
		if (Exists(dir + name) && IsOurDll(dir + name) && DeleteFileW((dir + name).c_str())) removed += std::wstring(L"  ") + name + L"\n";
	// Backups: the manifest's, and any *.injector-bak left by an older setup, restored where the file is now missing.
	WIN32_FIND_DATAW fd {};
	HANDLE h = FindFirstFileW((dir + L"*.injector-bak").c_str(), &fd);
	if (h != INVALID_HANDLE_VALUE) {
		do {
			std::wstring name(fd.cFileName);
			backups.insert(name.substr(0, name.size() - std::wstring(L".injector-bak").size()));
		} while (FindNextFileW(h, &fd));
		FindClose(h);
	}
	for (const auto& name: backups) {
		const std::wstring bak = dir + name + L".injector-bak";
		if (!Exists(bak)) continue;
		if (MoveFileExW(bak.c_str(), (dir + name).c_str(), MOVEFILE_REPLACE_EXISTING)) restored += L"  " + name + L"\n";
	}
	DeleteFileW((dir + kManifest).c_str());
	if (removed.empty() && restored.empty()) { Say(L"No habia nada del injector en " + dir, MB_ICONINFORMATION); return true; }
	Say(L"Desinstalado de " + dir + L"\n\nBorrados:\n" + removed + L"\nRestaurados:\n" + restored +
	    L"\n(injector.ini, injector.log e injector_profiles se conservan.)");
	return true;
}

std::wstring PickExe() {
	wchar_t file[MAX_PATH] {};
	OPENFILENAMEW ofn {sizeof(ofn)};
	ofn.lpstrFilter = L"Ejecutable del juego\0*.exe\0";
	ofn.lpstrFile = file;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrTitle = L"Elige el exe del juego";
	ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
	return GetOpenFileNameW(&ofn) ? file : L"";
}

} // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
	int argc = 0;
	wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
	if (argc >= 3) {
		const std::wstring cmd = Lower(argv[1]), exe = argv[2];
		if (cmd == L"install") return Install(exe) ? 0 : 1;
		if (cmd == L"uninstall") return Uninstall(exe) ? 0 : 1;
		if (cmd == L"detect") {
			const auto pe = ReadImports(exe);
			std::wstring list;
			for (const auto& d: pe.dlls) list += std::wstring(d.begin(), d.end()) + L" ";
			Say(std::wstring(L"API: ") + std::wstring(ApiName(Detect(pe)), ApiName(Detect(pe)) + std::strlen(ApiName(Detect(pe)))) +
			    (pe.x64 ? L" (x64)" : L" (x86)") + L"\nImports: " + list);
			return pe.ok ? 0 : 1;
		}
	}
	const std::wstring exe = PickExe();
	if (exe.empty()) return 0;
	const auto pe = ReadImports(exe);
	const bool installed = Exists(Dir(exe) + kManifest) || (Exists(Dir(exe) + L"winevulkan.dll") && IsOurDll(Dir(exe) + L"winevulkan.dll")) ||
	                       (Exists(Dir(exe) + L"vulkan-1.dll") && IsOurDll(Dir(exe) + L"vulkan-1.dll"));
	std::wstring msg = L"Juego: " + exe + L"\nAPI detectada: ";
	const char* api = ApiName(Detect(pe));
	msg += std::wstring(api, api + std::strlen(api));
	msg += installed ? L"\n\nYa instalado. Si = desinstalar." : L"\n\nSi = instalar.";
	if (Say(msg, MB_YESNO | MB_ICONQUESTION) != IDYES) return 0;
	return (installed ? Uninstall(exe) : Install(exe)) ? 0 : 1;
}
