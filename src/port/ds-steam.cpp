// "Demon's Souls" PC port: "Anadir a Steam" (ds-steam.h).
#include "port/ds-steam.h"

#include "stb_image.h"

#include <windows.h>
#include <wincodec.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace Port::Steam {

namespace fs = std::filesystem;

namespace {

uint32_t Crc32(const std::string& s) {
	uint32_t table[256];
	for (uint32_t i = 0; i < 256; i++) {
		uint32_t c = i;
		for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
		table[i] = c;
	}
	uint32_t crc = 0xFFFFFFFFu;
	for (unsigned char ch: s) crc = table[(crc ^ ch) & 0xFFu] ^ (crc >> 8);
	return crc ^ 0xFFFFFFFFu;
}

std::string Utf8(const std::wstring& w) {
	if (w.empty()) return {};
	const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
	std::string s(static_cast<size_t>(n), '\0');
	WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
	return s;
}

std::string Lower(std::string s) {
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	s.erase(std::remove(s.begin(), s.end(), '"'), s.end());
	std::replace(s.begin(), s.end(), '/', '\\');
	return s;
}

fs::path SteamPath() {
	HKEY key = nullptr;
	if (RegOpenKeyExW(HKEY_CURRENT_USER, L"Software\\Valve\\Steam", 0, KEY_READ, &key) != ERROR_SUCCESS) return {};
	wchar_t buffer[MAX_PATH * 2] = {};
	DWORD   size = sizeof(buffer), type = 0;
	const auto ok = RegQueryValueExW(key, L"SteamPath", nullptr, &type, reinterpret_cast<BYTE*>(buffer), &size);
	RegCloseKey(key);
	if (ok != ERROR_SUCCESS || (type != REG_SZ && type != REG_EXPAND_SZ)) return {};
	return fs::path(buffer).make_preferred();
}

// ---- Binary VDF (shortcuts.vdf): 0x00 map, 0x01 string, 0x02 int32, 0x07 uint64, 0x08 end of map. ----

struct Reader {
	const std::string& d;
	size_t             p = 0;
	bool               ok = true;
	std::string        Str() {
		const auto end = d.find('\0', p);
		if (end == std::string::npos) {
			ok = false;
			return {};
		}
		std::string s = d.substr(p, end - p);
		p             = end + 1;
		return s;
	}
	// A map's entries until its 0x08; `exe` collects the values of keys named exe (any case).
	void Map(std::vector<std::string>* exes, int depth) {
		while (ok && p < d.size()) {
			const auto type = static_cast<unsigned char>(d[p++]);
			if (type == 0x08) return;
			const std::string k = Str();
			switch (type) {
				case 0x00: Map(exes, depth + 1); break;
				case 0x01: {
					const std::string v = Str();
					if (exes != nullptr && depth == 2 && Lower(k) == "exe") exes->push_back(v);
					break;
				}
				case 0x02: p += 4; break;
				case 0x07: p += 8; break;
				default: ok = false; return;
			}
			if (p > d.size()) ok = false;
		}
		ok = false;
	}
};

void PutStr(std::string& o, const char* key, const std::string& value) {
	o += '\x01';
	o += key;
	o += '\0';
	o += value;
	o += '\0';
}
void PutInt(std::string& o, const char* key, uint32_t value) {
	o += '\x02';
	o += key;
	o += '\0';
	o.append(reinterpret_cast<const char*>(&value), 4);
}

// ---- Images: fondo-souls.jpg scaled to cover each size and cropped to its centre, written by WIC. ----

struct Image {
	int                  w = 0, h = 0;
	std::vector<uint8_t> rgba;
};

Image Cover(const Image& src, int w, int h) {
	Image out;
	out.w = w;
	out.h = h;
	out.rgba.resize(static_cast<size_t>(w) * h * 4);
	const double scale = std::max(static_cast<double>(w) / src.w, static_cast<double>(h) / src.h);
	const double ox = (src.w * scale - w) / 2.0, oy = (src.h * scale - h) / 2.0;
	// (Box filter over the source pixels each output pixel covers.)
	for (int y = 0; y < h; y++) {
		const double sy0 = (y + oy) / scale, sy1 = (y + 1 + oy) / scale;
		for (int x = 0; x < w; x++) {
			const double sx0 = (x + ox) / scale, sx1 = (x + 1 + ox) / scale;
			double       acc[4] = {0, 0, 0, 0};
			int          n      = 0;
			for (int sy = std::max(0, static_cast<int>(sy0)); sy < std::min(src.h, std::max(static_cast<int>(sy0) + 1, static_cast<int>(sy1))); sy++)
				for (int sx = std::max(0, static_cast<int>(sx0)); sx < std::min(src.w, std::max(static_cast<int>(sx0) + 1, static_cast<int>(sx1))); sx++) {
					const uint8_t* p = &src.rgba[(static_cast<size_t>(sy) * src.w + sx) * 4];
					for (int c = 0; c < 4; c++) acc[c] += p[c];
					n++;
				}
			uint8_t* q = &out.rgba[(static_cast<size_t>(y) * w + x) * 4];
			for (int c = 0; c < 4; c++) q[c] = static_cast<uint8_t>(n > 0 ? acc[c] / n : 0);
		}
	}
	return out;
}

bool WriteImage(const Image& img, const fs::path& path, bool png) {
	IWICImagingFactory* factory = nullptr;
	if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))) return false;
	std::vector<uint8_t> bgra = img.rgba;
	for (size_t i = 0; i < bgra.size(); i += 4) std::swap(bgra[i], bgra[i + 2]);
	bool                     ok      = false;
	IWICStream*              stream  = nullptr;
	IWICBitmapEncoder*       encoder = nullptr;
	IWICBitmapFrameEncode*   frame   = nullptr;
	IPropertyBag2*           props   = nullptr;
	std::error_code          ec;
	fs::remove(path, ec);
	if (SUCCEEDED(factory->CreateStream(&stream)) && SUCCEEDED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE)) &&
	    SUCCEEDED(factory->CreateEncoder(png ? GUID_ContainerFormatPng : GUID_ContainerFormatJpeg, nullptr, &encoder)) &&
	    SUCCEEDED(encoder->Initialize(stream, WICBitmapEncoderNoCache)) && SUCCEEDED(encoder->CreateNewFrame(&frame, &props))) {
		if (!png && props != nullptr) {
			PROPBAG2 option {};
			option.pstrName = const_cast<LPOLESTR>(L"ImageQuality");
			VARIANT value;
			VariantInit(&value);
			value.vt     = VT_R4;
			value.fltVal = 0.92f;
			props->Write(1, &option, &value);
		}
		WICPixelFormatGUID format = png ? GUID_WICPixelFormat32bppBGRA : GUID_WICPixelFormat24bppBGR;
		std::vector<uint8_t> pixels;
		UINT                 stride = static_cast<UINT>(img.w) * 4;
		if (png) {
			pixels = bgra;
		} else {
			stride = static_cast<UINT>((img.w * 3 + 3) & ~3);
			pixels.assign(static_cast<size_t>(stride) * img.h, 0);
			for (int y = 0; y < img.h; y++)
				for (int x = 0; x < img.w; x++)
					std::memcpy(&pixels[static_cast<size_t>(y) * stride + x * 3], &bgra[(static_cast<size_t>(y) * img.w + x) * 4], 3);
		}
		if (SUCCEEDED(frame->Initialize(props)) && SUCCEEDED(frame->SetSize(static_cast<UINT>(img.w), static_cast<UINT>(img.h))) &&
		    SUCCEEDED(frame->SetPixelFormat(&format)) &&
		    IsEqualGUID(format, png ? GUID_WICPixelFormat32bppBGRA : GUID_WICPixelFormat24bppBGR) &&
		    SUCCEEDED(frame->WritePixels(static_cast<UINT>(img.h), stride, static_cast<UINT>(pixels.size()), pixels.data())) &&
		    SUCCEEDED(frame->Commit()) && SUCCEEDED(encoder->Commit()))
			ok = true;
	}
	if (props) props->Release();
	if (frame) frame->Release();
	if (encoder) encoder->Release();
	if (stream) stream->Release();
	factory->Release();
	return ok;
}

// The exe's icon (its resource 2: demons-souls.ico as it is) next to it.
bool WriteIcon(const fs::path& ico) {
	HMODULE module = GetModuleHandleW(nullptr);
	HRSRC   res    = FindResourceW(module, MAKEINTRESOURCEW(2), MAKEINTRESOURCEW(10)); // RT_RCDATA
	if (res == nullptr) return fs::exists(ico);
	HGLOBAL     data = LoadResource(module, res);
	const DWORD size = SizeofResource(module, res);
	const void* p    = data != nullptr ? LockResource(data) : nullptr;
	if (p == nullptr || size == 0) return false;
	std::ofstream out(ico, std::ios::binary | std::ios::trunc);
	out.write(static_cast<const char*>(p), size);
	return static_cast<bool>(out);
}

} // namespace

bool AddToSteam(const fs::path& exe, std::string* report) {
	std::string& r = *report;
	r.clear();
	const fs::path steam = SteamPath();
	std::error_code ec;
	if (steam.empty() || !fs::exists(steam / L"userdata", ec)) {
		r = "No se encontro Steam (HKCU\\Software\\Valve\\Steam SteamPath, o no tiene userdata). Instala Steam e inicia sesion una vez.";
		return false;
	}
	const fs::path    folder  = exe.parent_path();
	const std::string exe_q   = "\"" + Utf8(exe.wstring()) + "\"";
	const std::string dir_q   = "\"" + Utf8(folder.wstring()) + "\"";
	const std::string name    = "Demon's Souls";
	const fs::path    ico     = folder / L"Demon's Souls.ico";
	const bool        icon_ok = WriteIcon(ico);
	const uint32_t    appid   = Crc32(exe_q + name) | 0x80000000u;

	// The artwork, made once.
	Image src;
	{
		int        channels = 0;
		std::ifstream in(folder / L"fondo-souls.jpg", std::ios::binary);
		const std::string bytes {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
		if (auto* px = bytes.empty() ? nullptr
		                             : stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), static_cast<int>(bytes.size()),
		                                                     &src.w, &src.h, &channels, 4)) {
			src.rgba.assign(px, px + static_cast<size_t>(src.w) * src.h * 4);
			stbi_image_free(px);
		}
	}
	const bool com = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));

	int added = 0, present = 0, failed = 0, art = 0;
	for (const auto& user: fs::directory_iterator(steam / L"userdata", ec)) {
		const std::wstring id = user.path().filename().wstring();
		if (!user.is_directory() || id.empty() || id == L"0" ||
		    !std::all_of(id.begin(), id.end(), [](wchar_t c) { return c >= L'0' && c <= L'9'; }))
			continue;
		const fs::path config = user.path() / L"config";
		fs::create_directories(config, ec);
		const fs::path vdf = config / L"shortcuts.vdf";
		std::string    data;
		{
			std::ifstream in(vdf, std::ios::binary);
			data.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
		}
		// Its entries (a damaged file is left alone).
		size_t count = 0;
		bool   exists = false;
		if (data.empty()) {
			data = std::string("\0shortcuts\0\x08\x08", 13);
		}
		{
			if (data.size() < 13 || data[0] != '\0' || data.compare(1, 10, std::string("shortcuts\0", 10)) != 0 ||
			    data.substr(data.size() - 2) != "\x08\x08") {
				failed++;
				continue;
			}
			Reader                   rd {data, 11};
			std::vector<std::string> exes;
			// (Each entry: 0x00 "<n>" map.)
			while (rd.ok && rd.p < data.size() && data[rd.p] == '\0') {
				rd.p++;
				(void)rd.Str();
				rd.Map(&exes, 2);
				count++;
			}
			if (!rd.ok || rd.p != data.size() - 2) {
				failed++;
				continue;
			}
			for (const auto& e: exes)
				if (Lower(e) == Lower(exe_q)) exists = true;
		}
		if (exists) {
			present++;
		} else {
			std::string e;
			e += '\0';
			e += std::to_string(count);
			e += '\0';
			PutInt(e, "appid", appid);
			PutStr(e, "AppName", name);
			PutStr(e, "Exe", exe_q);
			PutStr(e, "StartDir", dir_q);
			PutStr(e, "icon", icon_ok ? Utf8(ico.wstring()) : Utf8(exe.wstring()));
			PutStr(e, "ShortcutPath", "");
			PutStr(e, "LaunchOptions", "");
			PutInt(e, "IsHidden", 0);
			PutInt(e, "AllowDesktopConfig", 1);
			PutInt(e, "AllowOverlay", 1);
			PutInt(e, "OpenVR", 0);
			PutInt(e, "Devkit", 0);
			PutStr(e, "DevkitGameID", "");
			PutInt(e, "DevkitOverrideAppID", 0);
			PutInt(e, "LastPlayTime", 0);
			PutStr(e, "FlatpakAppID", "");
			e += '\0';
			e += "tags";
			e += '\0';
			e += '\x08';
			e += '\x08';
			data.insert(data.size() - 2, e);
			// (A copy of the old one beside it, then the new one.)
			fs::copy_file(vdf, fs::path(vdf.wstring() + L".bak"), fs::copy_options::overwrite_existing, ec);
			const fs::path tmp = fs::path(vdf.wstring() + L".tmp");
			{
				std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
				out.write(data.data(), static_cast<std::streamsize>(data.size()));
				if (!out) {
					failed++;
					continue;
				}
			}
			fs::rename(tmp, vdf, ec);
			if (ec) {
				failed++;
				continue;
			}
			added++;
		}
		// Artwork (portrait 600x900, capsule 920x430, hero 1920x620, icon 256x256).
		if (!src.rgba.empty() && com) {
			const fs::path grid = config / L"grid";
			fs::create_directories(grid, ec);
			const std::wstring base = std::to_wstring(appid);
			bool ok = WriteImage(Cover(src, 600, 900), grid / (base + L"p.jpg"), false);
			ok &= WriteImage(Cover(src, 920, 430), grid / (base + L".jpg"), false);
			ok &= WriteImage(Cover(src, 1920, 620), grid / (base + L"_hero.jpg"), false);
			ok &= WriteImage(Cover(src, 256, 256), grid / (base + L"_icon.png"), true);
			if (ok) art++;
		}
	}
	if (com) CoUninitialize();
	if (added + present == 0) {
		r = failed > 0 ? "No se pudo escribir shortcuts.vdf de Steam (archivo danado o sin permiso). Cierra Steam y prueba otra vez."
		               : "Steam no tiene ningun usuario en userdata: inicia sesion en Steam una vez y prueba otra vez.";
		return false;
	}
	r = "Anadido a Steam: " + std::to_string(added) + " usuario(s)";
	if (present > 0) r += ", ya estaba en " + std::to_string(present);
	if (failed > 0) r += ", fallo en " + std::to_string(failed);
	r += art > 0 ? "; con sus imagenes." : "; sin imagenes (falta fondo-souls.jpg).";
	r += " REINICIA STEAM (cierralo del todo y abrelo) para que aparezca \"Demon's Souls\" en la biblioteca.";
	return true;
}

} // namespace Port::Steam
