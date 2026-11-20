#pragma once
// (online4/online5) The online of the Demon's Souls port: our own server inside the port ("Demon's Souls.exe" --servidor: in the
// background, icon by the clock, optionally started with Windows) and what the F2 > Online page needs. ds-online.cpp.
#include <string>

namespace Port::Online {

constexpr int kDefaultPort = 31415; // (online5) TCP: the game server (/dsmaster, /dsgame) and the PSN-like WebAPI
constexpr int kAccountPort = 31413; // TCP: accounts (register / login)
constexpr int kRelayPort   = 3758;  // UDP: the P2P relay

bool IsServerCommand(int argc, char* argv[]); // --servidor [port]
int  ServerMain(int argc, char* argv[]);      // the server process (tray icon); returns at "Cerrar el servidor"

// The host's server, started in the background if it does not answer (launch, "Ser anfitrion").
bool EnsureHost(int port);
bool StartServer(int port);
void StopServer();
// One getServerState to host:port (timeout ~3 s): true if a Demon's Souls server answered. *info: a short text.
bool Probe(const std::string& host_port, std::string* info);
// Windows: start the server with the user's session (HKCU Run, no admin).
bool Autostart();
bool SetAutostart(bool on, int port);
// Windows firewall: inbound TCP 31413 + 31415 and UDP 3758 for this exe (one UAC prompt). FirewallReady: the rule is there.
bool FirewallReady();
bool AddFirewallRule(int port);
std::string PublicIp(); // blocking (~2 s): from the internet (empty: unknown)
std::string LanIp();    // this PC on the local network
std::string MakeCode(const std::string& ip_port);
std::string ReadCode(const std::string& text); // a DS- code, IP:port, host:port or bare IP -> host:port ("" invalid)
std::string DataFolder();
// (online5) Account on the server (port 31413): register or log in. true: ok; *error: the server's reason.
bool CreateAccount(const std::string& host, const std::string& name, const std::string& password, std::string* error);
// (online5) The server's state (/kyty/estado) as JSON text ("" if it does not answer).
std::string ServerState(const std::string& host_port);

} // namespace Port::Online
