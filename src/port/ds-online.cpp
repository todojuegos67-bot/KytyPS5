// (online4) The online server inside the port and its Windows side (tray icon, autostart, firewall, IP detection).
// The server itself (JSON, storage, commands, HTTP): ds-online-core.inc, the C++ port of tools/online/servidor.py.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>
#include <winhttp.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "port/ds-online.h"
#include "port/ds-i18n.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace Port::Online {
#if defined(KYTY_NO_ONLINE)
// (sin-online) Build sin el servidor online: solo cascarones. Sin sockets a la escucha, sin reglas de firewall, sin
// arranque con Windows, sin WinHTTP. Para una prueba A/B de antivirus: este .exe no contiene el codigo de red.
std::string ServerHost() { return "127.0.0.1"; }
bool        IsServerCommand(int /*argc*/, char* /*argv*/[]) { return false; }
int         ServerMain(int /*argc*/, char* /*argv*/[]) { return 0; }
bool        EnsureHost(int /*port*/) { return false; }
bool        StartServer(int /*port*/) { return false; }
void        StopServer() {}
bool        Probe(const std::string& /*host_port*/, std::string* info) { if (info != nullptr) info->clear(); return false; }
bool        Autostart() { return false; }
bool        SetAutostart(bool /*on*/, int /*port*/) { return false; }
bool        FirewallReady() { return false; }
bool        AddFirewallRule(int /*port*/) { return false; }
std::string PublicIp() { return {}; }
std::string LanIp() { return {}; }
std::string MakeCode(const std::string& /*ip_port*/) { return {}; }
std::string ReadCode(const std::string& /*text*/) { return {}; }
std::string DataFolder() { return {}; }
bool        CreateAccount(const std::string& /*host*/, const std::string& /*name*/, const std::string& /*password*/, std::string* error) {
	if (error != nullptr) *error = I18n::T("online no incluido en esta compilación");
	return false;
}
std::string ServerState(const std::string& /*host_port*/) { return {}; }
bool        Chat(const std::string& /*host_port*/, const std::string& /*name*/, const std::string& /*password*/, const std::string& /*to*/,
                 const std::string& /*text*/, std::vector<ChatMessage>* /*inbox*/, std::vector<std::string>* /*online*/, std::string* error) {
	if (error != nullptr) *error = I18n::T("online no incluido en esta compilación");
	return false;
}
} // namespace Port::Online
#else // real online implementation below

std::string ServerHost() { // (online10)
	std::string info;
	return Probe("127.0.0.1:" + std::to_string(kDefaultPort), &info) ? std::string("127.0.0.1") : std::string(kPublicServer);
}
#include "port/ds-online-core.inc"

namespace {

#if defined(_WIN32)
std::wstring Wide(const std::string& s) {
	if (s.empty()) return {};
	const int    n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
	std::wstring w(static_cast<size_t>(n), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
	return w;
}
std::string Narrow(const std::wstring& w) {
	if (w.empty()) return {};
	const int   n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
	std::string s(static_cast<size_t>(n), '\0');
	WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
	return s;
}
std::wstring ExeW() {
	wchar_t buf[MAX_PATH * 4];
	const DWORD n = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(std::size(buf)));
	return std::wstring(buf, n);
}
constexpr const wchar_t* kWindowClass = L"DemonsSoulsPC_ServidorOnline";
constexpr const wchar_t* kMutex       = L"Local\\DemonsSoulsPC_ServidorOnline";
constexpr const wchar_t* kRunKey      = L"Software\\Microsoft\\Windows\\CurrentVersion\\Run";
constexpr const wchar_t* kRunValue    = L"DemonsSoulsPC_ServidorOnline";
constexpr const wchar_t* kRuleTcp     = L"Demon's Souls PC - servidor online (TCP)";
constexpr const wchar_t* kRuleUdp     = L"Demon's Souls PC - relay P2P (UDP)";
#endif

// One small HTTP POST to host:port (a fresh connection; every wait bounded).
bool HttpPost(const std::string& host_port, const std::string& path, const std::string& body, int timeout_ms, int* status, std::string* reply) {
	*status = 0;
	reply->clear();
	const auto  colon = host_port.rfind(':');
	std::string host = colon == std::string::npos ? host_port : host_port.substr(0, colon);
	std::string port = colon == std::string::npos ? std::to_string(kDefaultPort) : host_port.substr(colon + 1);
#if defined(_WIN32)
	static const bool wsa = [] { WSADATA d {}; return WSAStartup(MAKEWORD(2, 2), &d) == 0; }();
	if (!wsa) return false;
#endif
	addrinfo hints {};
	hints.ai_family   = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	addrinfo* res     = nullptr;
	if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || res == nullptr) return false;
	OnlineCore::sock_t s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
	if (s == OnlineCore::kBadSock) { freeaddrinfo(res); return false; }
#if defined(_WIN32)
	u_long nb = 1;
	ioctlsocket(s, FIONBIO, &nb);
	DWORD tmo = static_cast<DWORD>(timeout_ms);
#else
	timeval tmo {timeout_ms / 1000, (timeout_ms % 1000) * 1000};
#endif
	bool connected = connect(s, res->ai_addr, static_cast<int>(res->ai_addrlen)) == 0;
	freeaddrinfo(res);
	if (!connected) {
		fd_set w, x;
		FD_ZERO(&w); FD_ZERO(&x);
		FD_SET(s, &w); FD_SET(s, &x);
		timeval wait {timeout_ms / 1000, (timeout_ms % 1000) * 1000};
		if (select(static_cast<int>(s + 1), nullptr, &w, &x, &wait) > 0 && FD_ISSET(s, &w)) {
			int err = 0;
			socklen_t len = sizeof(err);
			getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&err), &len);
			connected = err == 0;
		}
	}
#if defined(_WIN32)
	nb = 0;
	ioctlsocket(s, FIONBIO, &nb);
#endif
	if (!connected) { OnlineCore::CloseSock(s); return false; }
	setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tmo), sizeof(tmo));
	setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&tmo), sizeof(tmo));
	const std::string req = "POST " + path + " HTTP/1.1\r\nHost: " + host_port + "\r\nContent-Type: application/json\r\nContent-Length: " +
	                        std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
	if (send(s, req.data(), static_cast<int>(req.size()), 0) != static_cast<int>(req.size())) { OnlineCore::CloseSock(s); return false; }
	std::string raw;
	char        buf[8192];
	for (;;) {
		const int n = static_cast<int>(recv(s, buf, sizeof(buf), 0));
		if (n <= 0) break;
		raw.append(buf, static_cast<size_t>(n));
		if (raw.size() > (1u << 20)) break;
	}
	OnlineCore::CloseSock(s);
	const auto he = raw.find("\r\n\r\n");
	if (raw.compare(0, 5, "HTTP/") != 0 || he == std::string::npos) return false;
	*status = std::atoi(raw.c_str() + raw.find(' ') + 1);
	*reply  = raw.substr(he + 4);
	return true;
}

#if defined(_WIN32)
// The server process: its state for the tray menu.
std::atomic<bool>    g_stop {false};
OnlineCore::Server*  g_server = nullptr;
NOTIFYICONDATAW      g_nid {};
int                  g_port = kDefaultPort;
constexpr UINT       kTrayMsg = WM_APP + 1;
enum : UINT { kMenuFolder = 1, kMenuAutostart = 2, kMenuQuit = 3, kMenuLog = 4 };

// (online7) The server's own log: %LOCALAPPDATA%\Demon's Souls PC\servidor-online\servidor.log (start, every port it
// opens or fails to open, how it ends). Defined below (it needs DataFolder).
void ServerLog(const std::string& line);

LRESULT CALLBACK TrayProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
	if (msg == kTrayMsg && (LOWORD(lp) == WM_RBUTTONUP || LOWORD(lp) == WM_LBUTTONUP || LOWORD(lp) == WM_CONTEXTMENU)) {
		HMENU        menu = CreatePopupMenu();
		const std::wstring head = L"Servidor online de Demon's Souls: puerto " + std::to_wstring(g_port) + L" (" +
		                          std::to_wstring(g_server ? g_server->requests.load() : 0) + L" peticiones)";
		AppendMenuW(menu, MF_STRING | MF_GRAYED, 0, head.c_str());
		AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
		AppendMenuW(menu, MF_STRING | (Autostart() ? MF_CHECKED : 0), kMenuAutostart, L"Arrancar con Windows");
		AppendMenuW(menu, MF_STRING, kMenuFolder, L"Abrir la carpeta de datos");
		AppendMenuW(menu, MF_STRING, kMenuLog, L"Ver el registro de peticiones");
		AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
		AppendMenuW(menu, MF_STRING, kMenuQuit, L"Cerrar el servidor");
		POINT pt;
		GetCursorPos(&pt);
		SetForegroundWindow(hwnd);
		const UINT cmd = TrackPopupMenu(menu, TPM_RETURNCMD | TPM_RIGHTBUTTON, pt.x, pt.y, 0, hwnd, nullptr);
		DestroyMenu(menu);
		if (cmd == kMenuFolder) ShellExecuteW(nullptr, L"open", Wide(DataFolder()).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
		if (cmd == kMenuLog) ShellExecuteW(nullptr, L"open", Wide(DataFolder() + "\\peticiones.log").c_str(), nullptr, nullptr, SW_SHOWNORMAL);
		if (cmd == kMenuAutostart) SetAutostart(!Autostart(), g_port);
		if (cmd == kMenuQuit) {
			ServerLog("cerrado desde el icono (Cerrar el servidor)");
			DestroyWindow(hwnd);
		}
		return 0;
	}
	if (msg == WM_CLOSE) {
		ServerLog("cerrado a peticion (boton Apagar el servidor del juego o Windows al salir)");
		DestroyWindow(hwnd);
		return 0;
	}
	if (msg == WM_ENDSESSION && wp) ServerLog("Windows cierra la sesion");
	if (msg == WM_DESTROY) {
		Shell_NotifyIconW(NIM_DELETE, &g_nid);
		g_stop = true;
		PostQuitMessage(0);
		return 0;
	}
	return DefWindowProcW(hwnd, msg, wp, lp);
}
#endif

} // namespace

std::string MakeCode(const std::string& ip_port) {
	return OnlineCore::MakeCode(ip_port);
}
std::string ReadCode(const std::string& text) {
	return OnlineCore::ReadCode(text);
}

std::string DataFolder() {
#if defined(_WIN32)
	wchar_t* base = nullptr;
	std::wstring dir;
	if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &base)) && base != nullptr) dir = base;
	CoTaskMemFree(base);
	if (dir.empty()) dir = std::filesystem::temp_directory_path().wstring();
	dir += L"\\Demon's Souls PC\\servidor-online";
	std::error_code ec;
	std::filesystem::create_directories(dir, ec);
	return Narrow(dir);
#else
	return "/tmp/ds-servidor-online";
#endif
}

#if defined(_WIN32)
namespace {
void ServerLog(const std::string& line) {
	static std::mutex m;
	std::lock_guard   lock(m);
	SYSTEMTIME        t;
	GetLocalTime(&t);
	char when[40];
	std::snprintf(when, sizeof(when), "%04u-%02u-%02u %02u:%02u:%02u ", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
	std::ofstream out(Wide(DataFolder() + "\\servidor.log"), std::ios::binary | std::ios::app);
	out << when << "[pid " << GetCurrentProcessId() << "] " << line << "\r\n";
}
} // namespace
#endif

// (online7) The accounts port (TCP 31413) answers HTTP (any status: it is our server).
bool ProbeAccounts(const std::string& host) {
	int         status = 0;
	std::string reply;
	return HttpPost(host + ":" + std::to_string(kAccountPort), "/cuenta/estado", "{}", 3000, &status, &reply) && status != 0;
}

bool Probe(const std::string& host_port, std::string* info) {
	int         status = 0;
	std::string reply;
	if (!HttpPost(host_port, "/dsmaster/ss", "{\"cmd\":\"getServerState\"}", 3000, &status, &reply)) {
		if (info) *info = I18n::T("no responde (apagado, puerto cerrado o dirección mal escrita)");
		return false;
	}
	OnlineCore::J j;
	if (status != 200 || !OnlineCore::Parse(reply, &j) || j.Get("ServerState") == nullptr || j.Get("ServerState")->AsInt(-1) != 0) {
		if (info) *info = std::string(I18n::T("responde, pero no es un servidor de Demon's Souls (HTTP ")) + std::to_string(status) + ")";
		return false;
	}
	if (info) *info = I18n::T("conectado");
	return true;
}

bool IsServerCommand(int argc, char* argv[]) {
	for (int i = 1; i < argc; i++)
		if (std::strcmp(argv[i], "--servidor") == 0) return true;
	return false;
}

int ServerMain(int argc, char* argv[]) {
	int port = kDefaultPort;
	for (int i = 1; i + 1 < argc; i++)
		if (std::strcmp(argv[i], "--servidor") == 0 && std::atoi(argv[i + 1]) > 0) port = std::atoi(argv[i + 1]);
#if defined(_WIN32)
	HANDLE mutex = CreateMutexW(nullptr, TRUE, kMutex);
	if (mutex != nullptr && GetLastError() == ERROR_ALREADY_EXISTS) { // one server per user session
		ServerLog("otro servidor ya esta abierto en esta sesion de Windows: este sale");
		return 0;
	}
	ServerLog("arranca: TCP " + std::to_string(port) + " (juego + WebAPI), TCP " + std::to_string(kAccountPort) + " (cuentas), UDP " +
	          std::to_string(kRelayPort) + " (relay); datos en " + DataFolder());
	static OnlineCore::Server server;
	server.port     = port;
	server.store.Open(DataFolder());
	server.LoadPsn(); // (11-21) the accounts saved before
	server.log_path = DataFolder() + "\\peticiones.log";
	g_server        = &server;
	g_port          = port;
	std::string         error, error2, error3;
	std::atomic<bool>   failed {false};
	// (online5) TCP 31415 (game server + WebAPI), TCP 31413 (accounts), UDP 3758 (P2P relay).
	std::thread listener([&] {
		if (!OnlineCore::Listen(&server, &g_stop, &error, port)) failed = true;
	});
	std::thread accounts([&] {
		if (!OnlineCore::Listen(&server, &g_stop, &error2, kAccountPort)) failed = true;
	});
	std::thread relay([&] {
		if (!OnlineCore::Relay(&server, &g_stop, &error3, kRelayPort)) failed = true;
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(400));
	if (failed.load()) {
		g_stop = true;
		listener.join();
		accounts.join();
		relay.join();
		server.Log("No se pudo abrir el servidor: " + error + " " + error2 + " " + error3);
		ServerLog("NO ARRANCA: TCP " + std::to_string(port) + ": " + (error.empty() ? "bien" : error) + "; TCP " + std::to_string(kAccountPort) +
		          ": " + (error2.empty() ? "bien" : error2) + "; UDP " + std::to_string(kRelayPort) + ": " + (error3.empty() ? "bien" : error3) +
		          " (WSA " + std::to_string(WSAGetLastError()) + ")");
		return 1;
	}
	{ // (online7) each port checked from this same PC, written to servidor.log
		std::string info;
		const bool  game = Probe("127.0.0.1:" + std::to_string(port), &info);
		const bool  acc  = ProbeAccounts("127.0.0.1");
		ServerLog(std::string("escuchando: TCP ") + std::to_string(port) + (game ? " responde" : " NO RESPONDE (" + info + ")") + ", TCP " +
		          std::to_string(kAccountPort) + (acc ? " responde" : " NO RESPONDE") + ", UDP " + std::to_string(kRelayPort) + " abierto");
	}
	WNDCLASSW wc {};
	wc.lpfnWndProc   = TrayProc;
	wc.hInstance     = GetModuleHandleW(nullptr);
	wc.lpszClassName = kWindowClass;
	RegisterClassW(&wc);
	HWND hwnd = CreateWindowW(kWindowClass, L"Servidor online de Demon's Souls", 0, 0, 0, 0, 0, nullptr, nullptr, wc.hInstance, nullptr);
	g_nid.cbSize           = sizeof(g_nid);
	g_nid.hWnd             = hwnd;
	g_nid.uID              = 1;
	g_nid.uFlags           = NIF_ICON | NIF_MESSAGE | NIF_TIP;
	g_nid.uCallbackMessage = kTrayMsg;
	g_nid.hIcon            = ExtractIconW(wc.hInstance, ExeW().c_str(), 0);
	if (g_nid.hIcon == nullptr || g_nid.hIcon == reinterpret_cast<HICON>(1)) g_nid.hIcon = LoadIconW(nullptr, MAKEINTRESOURCEW(32512));
	const std::wstring tip = L"Demon's Souls: servidor online (puerto " + std::to_wstring(port) + L")";
	wcsncpy_s(g_nid.szTip, tip.c_str(), _TRUNCATE);
	Shell_NotifyIconW(NIM_ADD, &g_nid);
	server.Log("Servidor arrancado en el puerto " + std::to_string(port));
	MSG msg;
	while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
		TranslateMessage(&msg);
		DispatchMessageW(&msg);
	}
	g_stop = true;
	listener.join();
	accounts.join();
	relay.join();
	server.Log("Servidor cerrado");
	ServerLog("cerrado (" + std::to_string(server.requests.load()) + " peticiones atendidas)");
	if (mutex != nullptr) CloseHandle(mutex);
	return 0;
#else
	(void)argc; (void)argv;
	static OnlineCore::Server server;
	server.port = port;
	server.store.Open(DataFolder());
	server.LoadPsn(); // (11-21) the accounts saved before
	server.log_path = DataFolder() + "/peticiones.log";
	static std::atomic<bool> stop {false};
	static std::string       e1, e2, e3;
	std::thread a([] { OnlineCore::Listen(&server, &stop, &e2, kAccountPort); });
	std::thread r([] { OnlineCore::Relay(&server, &stop, &e3, kRelayPort); });
	const bool ok = OnlineCore::Listen(&server, &stop, &e1, port);
	stop = true;
	a.join();
	r.join();
	return ok ? 0 : 1;
#endif
}

bool StartServer(int port) {
#if defined(_WIN32)
	std::wstring        cmd = L"\"" + ExeW() + L"\" --servidor " + std::to_wstring(port);
	STARTUPINFOW        si {};
	PROCESS_INFORMATION pi {};
	si.cb = sizeof(si);
	// (online7) out of the game's job (if it has one: a launcher's or Steam's), so the server is not closed with it
	if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB, nullptr,
	                    nullptr, &si, &pi) &&
	    !CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, nullptr, nullptr, &si, &pi))
		return false;
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
	return true;
#else
	(void)port;
	return false;
#endif
}

bool EnsureHost(int port) {
	// (online7) "up" = both TCP ports answer (31415 game + WebAPI, 31413 accounts)
	const std::string local = "127.0.0.1:" + std::to_string(port);
	if (Probe(local, nullptr) && ProbeAccounts("127.0.0.1")) return true;
	if (!StartServer(port)) return false;
	for (int i = 0; i < 30; i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(150));
		if (Probe(local, nullptr) && ProbeAccounts("127.0.0.1")) return true;
	}
	return false;
}

void StopServer() {
#if defined(_WIN32)
	if (HWND w = FindWindowW(kWindowClass, nullptr); w != nullptr) PostMessageW(w, WM_CLOSE, 0, 0);
#endif
}

bool Autostart() {
#if defined(_WIN32)
	wchar_t buf[2048];
	DWORD   size = sizeof(buf);
	return RegGetValueW(HKEY_CURRENT_USER, kRunKey, kRunValue, RRF_RT_REG_SZ, nullptr, buf, &size) == ERROR_SUCCESS;
#else
	return false;
#endif
}

bool SetAutostart(bool on, int port) {
#if defined(_WIN32)
	HKEY key = nullptr;
	if (RegOpenKeyExW(HKEY_CURRENT_USER, kRunKey, 0, KEY_SET_VALUE, &key) != ERROR_SUCCESS) return false;
	LONG r;
	if (on) {
		const std::wstring cmd = L"\"" + ExeW() + L"\" --servidor " + std::to_wstring(port);
		r = RegSetValueExW(key, kRunValue, 0, REG_SZ, reinterpret_cast<const BYTE*>(cmd.c_str()), static_cast<DWORD>((cmd.size() + 1) * sizeof(wchar_t)));
	} else {
		r = RegDeleteValueW(key, kRunValue);
		if (r == ERROR_FILE_NOT_FOUND) r = ERROR_SUCCESS;
	}
	RegCloseKey(key);
	return r == ERROR_SUCCESS;
#else
	(void)on; (void)port;
	return false;
#endif
}

#if defined(_WIN32)
namespace {
DWORD RunHidden(const std::wstring& exe, const std::wstring& params, bool elevated) {
	SHELLEXECUTEINFOW sei {};
	sei.cbSize       = sizeof(sei);
	sei.fMask        = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_FLAG_NO_UI;
	sei.lpVerb       = elevated ? L"runas" : L"open";
	sei.lpFile       = exe.c_str();
	sei.lpParameters = params.c_str();
	sei.nShow        = SW_HIDE;
	if (!ShellExecuteExW(&sei) || sei.hProcess == nullptr) return static_cast<DWORD>(-1);
	WaitForSingleObject(sei.hProcess, 60000);
	DWORD code = static_cast<DWORD>(-1);
	GetExitCodeProcess(sei.hProcess, &code);
	CloseHandle(sei.hProcess);
	return code;
}
} // namespace
#endif

bool FirewallReady() {
#if defined(_WIN32)
	return RunHidden(L"netsh.exe", std::wstring(L"advfirewall firewall show rule name=\"") + kRuleTcp + L"\"", false) == 0;
#else
	return false;
#endif
}

bool AddFirewallRule(int port) {
#if defined(_WIN32)
	// One elevated cmd (one UAC prompt): the old rules out, TCP 31413 + <port> (31415) and UDP 3758 for this exe.
	const std::wstring exe  = ExeW();
	const std::wstring args = std::wstring(L"/c netsh advfirewall firewall delete rule name=\"") + kRuleTcp + L"\" >nul & " +
	                          L"netsh advfirewall firewall delete rule name=\"" + kRuleUdp + L"\" >nul & " +
	                          L"netsh advfirewall firewall add rule name=\"" + kRuleTcp + L"\" dir=in action=allow protocol=TCP localport=" +
	                          std::to_wstring(kAccountPort) + L"," + std::to_wstring(port) + L" program=\"" + exe + L"\" enable=yes && " +
	                          L"netsh advfirewall firewall add rule name=\"" + kRuleUdp + L"\" dir=in action=allow protocol=UDP localport=" +
	                          std::to_wstring(kRelayPort) + L" program=\"" + exe + L"\" enable=yes";
	return RunHidden(L"cmd.exe", args, true) == 0;
#else
	(void)port;
	return false;
#endif
}

std::string PublicIp() {
#if defined(_WIN32)
	std::string out;
	HINTERNET   s = WinHttpOpen(L"DemonsSoulsPC/online", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
	if (s == nullptr) return out;
	WinHttpSetTimeouts(s, 3000, 3000, 3000, 3000);
	for (const wchar_t* host: {L"api.ipify.org", L"icanhazip.com"}) {
		HINTERNET c = WinHttpConnect(s, host, INTERNET_DEFAULT_HTTPS_PORT, 0);
		HINTERNET r = c ? WinHttpOpenRequest(c, L"GET", L"/", nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE) : nullptr;
		if (r && WinHttpSendRequest(r, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) && WinHttpReceiveResponse(r, nullptr)) {
			char  buf[128];
			DWORD got = 0;
			if (WinHttpReadData(r, buf, sizeof(buf) - 1, &got) && got > 0) {
				buf[got] = 0;
				out      = buf;
				out.erase(std::remove_if(out.begin(), out.end(), [](char ch) { return !(std::isdigit(static_cast<unsigned char>(ch)) || ch == '.'); }), out.end());
			}
		}
		if (r) WinHttpCloseHandle(r);
		if (c) WinHttpCloseHandle(c);
		if (!out.empty()) break;
	}
	WinHttpCloseHandle(s);
	return out;
#else
	return {};
#endif
}

std::string LanIp() {
#if defined(_WIN32)
	static const bool wsa = [] { WSADATA d {}; return WSAStartup(MAKEWORD(2, 2), &d) == 0; }();
	(void)wsa;
#endif
	// The address Windows would use to reach the internet (no packet is sent: a UDP "connect" only picks the route).
	OnlineCore::sock_t s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (s == OnlineCore::kBadSock) return {};
	sockaddr_in to {};
	to.sin_family = AF_INET;
	to.sin_port   = htons(53);
	inet_pton(AF_INET, "8.8.8.8", &to.sin_addr);
	std::string out;
	if (connect(s, reinterpret_cast<sockaddr*>(&to), sizeof(to)) == 0) {
		sockaddr_in me {};
		socklen_t   len = sizeof(me);
		if (getsockname(s, reinterpret_cast<sockaddr*>(&me), &len) == 0) {
			char ip[64] = "";
			inet_ntop(AF_INET, &me.sin_addr, ip, sizeof(ip));
			out = ip;
		}
	}
	OnlineCore::CloseSock(s);
	return out;
}

bool CreateAccount(const std::string& host, const std::string& name, const std::string& password, std::string* error) {
	OnlineCore::J body = OnlineCore::J::Obj();
	body.Set("name", OnlineCore::J::Str(name));
	body.Set("password", OnlineCore::J::Str(password));
	int         status = 0;
	std::string reply;
	if (!HttpPost(host + ":" + std::to_string(kAccountPort), "/cuenta/registro", OnlineCore::Dump(body), 4000, &status, &reply)) {
		if (error) *error = std::string(I18n::T("el servidor no responde (")) + host + ":" + std::to_string(kAccountPort) + ")";
		return false;
	}
	OnlineCore::J j;
	if (!OnlineCore::Parse(reply, &j) || j.Get("ok") == nullptr || j.Get("ok")->t != OnlineCore::J::T::Bool || !j.Get("ok")->b) {
		if (error) *error = (j.Get("error") ? j.Get("error")->AsStr() : I18n::T("respuesta no válida"));
		return false;
	}
	return true;
}

std::string ServerState(const std::string& host_port) {
	int         status = 0;
	std::string reply;
	if (!HttpPost(host_port, "/kyty/estado", "{}", 2500, &status, &reply) || status != 200) return {};
	return reply;
}

bool Chat(const std::string& host_port, const std::string& name, const std::string& password, const std::string& to, const std::string& text,
          std::vector<ChatMessage>* inbox, std::vector<std::string>* online, std::string* error) {
	using OnlineCore::J;
	J body = J::Obj();
	body.Set("name", J::Str(name));
	body.Set("password", J::Str(password));
	body.Set("op", J::Str(to.empty() ? "leer" : "enviar"));
	if (!to.empty()) {
		body.Set("to", J::Str(to));
		body.Set("text", J::Str(text));
	}
	int         status = 0;
	std::string reply;
	if (!HttpPost(host_port, "/kyty/chat", OnlineCore::Dump(body), 4000, &status, &reply) || status == 0) {
		if (error) *error = I18n::T("el servidor no responde");
		return false;
	}
	J j;
	if (!OnlineCore::Parse(reply, &j) || j.t != J::T::Obj) {
		if (error) *error = I18n::T(status == 404 ? "el servidor no tiene chat (actualízalo)" : "respuesta del servidor no válida");
		return false;
	}
	if (const J* ok = j.Get("ok"); ok == nullptr || !ok->b) {
		if (error) *error = j.Get("error") ? j.Get("error")->AsStr() : I18n::T("error del servidor");
		return false;
	}
	if (inbox)
		if (const J* m = j.Get("mensajes"))
			for (const auto& e: m->a)
				inbox->push_back({e.Get("de") ? e.Get("de")->AsStr() : "", e.Get("texto") ? e.Get("texto")->AsStr() : "", e.Get("t") ? e.Get("t")->AsInt() : 0});
	if (online)
		if (const J* c = j.Get("conectados"))
			for (const auto& e: c->a) online->push_back(e.AsStr());
	return true;
}

// ---- (chat-tr) Automatic translation of the chat messages received (see ds-online.h).
std::string UrlEncode(const std::string& s) {
	static const char* hex = "0123456789ABCDEF";
	std::string        out;
	for (unsigned char c: s) {
		if (std::isalnum(c) != 0 || c == '-' || c == '_' || c == '.' || c == '~') out += static_cast<char>(c);
		else {
			out += '%';
			out += hex[c >> 4];
			out += hex[c & 15];
		}
	}
	return out;
}
std::string GoogleTranslatePath(const std::string& text, const std::string& lang) {
	return "/translate_a/single?client=gtx&sl=auto&tl=" + UrlEncode(lang) + "&dt=t&ie=UTF-8&oe=UTF-8&q=" + UrlEncode(text);
}
std::string MyMemoryPath(const std::string& text, const std::string& lang) {
	return "/get?q=" + UrlEncode(text) + "&langpair=" + UrlEncode("autodetect|" + lang);
}
static std::string BaseLang(std::string l) { // "en-GB" -> "en", "pt-BR" -> "pt"
	if (auto d = l.find_first_of("-_"); d != std::string::npos) l.resize(d);
	for (auto& c: l) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return l;
}
bool ParseGoogleTranslate(const std::string& json, std::string* out, std::string* src) {
	using OnlineCore::J;
	J j;
	if (!OnlineCore::Parse(json, &j) || j.t != J::T::Arr || j.a.empty() || j.a[0].t != J::T::Arr) return false;
	std::string t;
	for (const auto& seg: j.a[0].a)
		if (seg.t == J::T::Arr && !seg.a.empty() && seg.a[0].t == J::T::Str) t += seg.a[0].s;
	if (t.empty()) return false;
	if (out) *out = t;
	if (src) *src = j.a.size() > 2 && j.a[2].t == J::T::Str ? BaseLang(j.a[2].s) : "";
	return true;
}
bool ParseMyMemory(const std::string& json, std::string* out, std::string* src) {
	using OnlineCore::J;
	J j;
	if (!OnlineCore::Parse(json, &j) || j.t != J::T::Obj) return false;
	const J* st = j.Get("responseStatus");
	const J* d  = j.Get("responseData");
	if (st == nullptr || st->AsInt() != 200 || d == nullptr || d->t != J::T::Obj) return false;
	const J* t = d->Get("translatedText");
	if (t == nullptr || t->t != J::T::Str || t->s.empty() || t->s.rfind("MYMEMORY WARNING", 0) == 0) return false;
	if (out) *out = t->s;
	if (src) *src = d->Get("detectedLanguage") && d->Get("detectedLanguage")->t == J::T::Str ? BaseLang(d->Get("detectedLanguage")->s) : "";
	return true;
}
#if defined(_WIN32)
static bool HttpsGet(const wchar_t* host, const std::string& path, std::string* body) {
	bool      ok = false;
	HINTERNET s  = WinHttpOpen(L"Mozilla/5.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
	if (s == nullptr) return false;
	WinHttpSetTimeouts(s, 3000, 3000, 3000, 3000);
	const std::wstring wpath(path.begin(), path.end()); // (ASCII: the path is percent-encoded)
	HINTERNET          c = WinHttpConnect(s, host, INTERNET_DEFAULT_HTTPS_PORT, 0);
	HINTERNET r = c ? WinHttpOpenRequest(c, L"GET", wpath.c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE) : nullptr;
	if (r && WinHttpSendRequest(r, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) && WinHttpReceiveResponse(r, nullptr)) {
		DWORD code = 0, len = sizeof(code);
		WinHttpQueryHeaders(r, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &code, &len, WINHTTP_NO_HEADER_INDEX);
		char  buf[4096];
		DWORD got = 0;
		while (WinHttpReadData(r, buf, sizeof(buf), &got) && got > 0 && body->size() < (1u << 20)) body->append(buf, got);
		ok = code == 200 && !body->empty();
	}
	if (r) WinHttpCloseHandle(r);
	if (c) WinHttpCloseHandle(c);
	WinHttpCloseHandle(s);
	return ok;
}
#endif
namespace {
struct TrEntry {
	int         state = 0; // 0 queued, 1 done, 2 failed
	std::string text;      // the translation (empty: same language)
	double      failed_at = 0;
};
std::mutex                        g_tr_m;
std::condition_variable           g_tr_cv;
std::map<std::string, TrEntry>    g_tr_cache; // lang + '\n' + text
std::deque<std::string>           g_tr_queue;
bool                              g_tr_started = false;
double TrNow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// One translation: Google, then MyMemory. false: both failed.
bool TranslateNow(const std::string& text, const std::string& lang, std::string* out) {
#if defined(_WIN32)
	std::string body, t, src;
	if (HttpsGet(L"translate.googleapis.com", GoogleTranslatePath(text, lang), &body) && ParseGoogleTranslate(body, &t, &src)) {
		*out = (src == lang || t == text) ? "" : t;
		return true;
	}
	body.clear();
	if (HttpsGet(L"api.mymemory.translated.net", MyMemoryPath(text, lang), &body) && ParseMyMemory(body, &t, &src)) {
		*out = (src == lang || t == text) ? "" : t;
		return true;
	}
	if (body.find("DISTINCT LANGUAGES") != std::string::npos) { // MyMemory: the text is already in `lang`
		out->clear();
		return true;
	}
#else
	(void)text, (void)lang, (void)out;
#endif
	return false;
}
void TrWorker() {
	for (;;) {
		std::string key;
		{
			std::unique_lock lock(g_tr_m);
			g_tr_cv.wait(lock, [] { return !g_tr_queue.empty(); });
			key = g_tr_queue.front();
			g_tr_queue.pop_front();
		}
		const auto        nl   = key.find('\n');
		const std::string lang = key.substr(0, nl), text = key.substr(nl + 1);
		std::string       out;
		const bool        ok = TranslateNow(text, lang, &out);
		std::lock_guard   lock(g_tr_m);
		TrEntry&          e = g_tr_cache[key];
		e.state             = ok ? 1 : 2;
		e.text              = ok ? out : "";
		e.failed_at         = ok ? 0 : TrNow();
	}
}
} // namespace

std::string TranslatedText(const std::string& text, const std::string& lang) {
	if (text.empty() || lang.empty() || text.size() > 1000) return {};
	const std::string key = lang + "\n" + text;
	std::lock_guard   lock(g_tr_m);
	auto              it = g_tr_cache.find(key);
	if (it != g_tr_cache.end()) {
		if (it->second.state == 1) return it->second.text;
		if (it->second.state == 0 || TrNow() - it->second.failed_at < 120.0) return {};
	}
	g_tr_cache[key].state = 0;
	g_tr_queue.push_back(key);
	if (!g_tr_started) {
		g_tr_started = true;
		std::thread(TrWorker).detach();
	}
	g_tr_cv.notify_one();
	return {};
}

} // namespace Port::Online
#endif // KYTY_NO_ONLINE
