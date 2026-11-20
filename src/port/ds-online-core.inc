// (online4) Demon's Souls online server, inside the port: the same API as tools/online/servidor.py (online2/3), formats
// from the eboot (01.007.000) with Ghidra:
//   envelope {"ServerState":0,"data":...} (DSNetworkClient::_AnalyzeMessage @0x6605b0)
//   /dsmaster/ss getServerState (@0x6614a0) - /dsgame/playerdata login, addNewAccount, initializeMultiplay,
//   getMultiplayGrade, getTimeMessage - /dsgame/bloodmessages addBloodMessage, getBloodMessage (@0x670040),
//   updateBloodMessageGrade, getBloodMessageGrade, deleteBloodMessage - /dsgame/replay addReplayData,
//   getReplayList (@0x66eff0), getReplayData (@0x671120) - /dsgame/character setWanderingGhost, getWanderingGhost
//   (@0x66ebf0) - /dsgame/qwc getQWCData (@0x672e40) - /dsgame/sos addSummonSign (case 6 of @0x6605b0),
//   deleteSummonSign, getSummonSigns, getForceSummons. Anything else: logged, answered {} (a valid empty answer).
// Portable part (JSON, storage, commands, the HTTP loop over sockets); ds-online.inc has the Windows side.
// Data: one JSON file per table in the data folder (written whole on each change: small tables, capped).
namespace OnlineCore {

// ---- JSON ---------------------------------------------------------------------------------------------------------
struct J {
	enum class T { Null, Bool, Num, Str, Arr, Obj };
	T                                   t = T::Null;
	bool                                b = false;
	double                              n = 0.0;
	bool                                is_int = false;
	long long                           i = 0;
	std::string                         s;
	std::vector<J>                      a;
	std::vector<std::pair<std::string, J>> o;

	static J Int(long long v) { J j; j.t = T::Num; j.is_int = true; j.i = v; j.n = static_cast<double>(v); return j; }
	static J Num(double v) { J j; j.t = T::Num; j.n = v; return j; }
	static J Str(std::string v) { J j; j.t = T::Str; j.s = std::move(v); return j; }
	static J Bool(bool v) { J j; j.t = T::Bool; j.b = v; return j; }
	static J Arr() { J j; j.t = T::Arr; return j; }
	static J Obj() { J j; j.t = T::Obj; return j; }
	const J* Get(const std::string& k) const {
		if (t != T::Obj) return nullptr;
		for (const auto& kv: o)
			if (kv.first == k) return &kv.second;
		return nullptr;
	}
	J& Set(const std::string& k, J v) {
		for (auto& kv: o)
			if (kv.first == k) return kv.second = std::move(v);
		o.emplace_back(k, std::move(v));
		return o.back().second;
	}
	void Push(J v) { a.push_back(std::move(v)); }
	long long AsInt(long long def = 0) const {
		if (t == T::Num) return is_int ? i : static_cast<long long>(n);
		if (t == T::Str) { char* e = nullptr; const long long v = std::strtoll(s.c_str(), &e, 10); return (e != s.c_str()) ? v : def; }
		if (t == T::Bool) return b ? 1 : 0;
		return def;
	}
	double AsNum(double def = 0.0) const {
		if (t == T::Num) return is_int ? static_cast<double>(i) : n;
		if (t == T::Str) { char* e = nullptr; const double v = std::strtod(s.c_str(), &e); return (e != s.c_str()) ? v : def; }
		return def;
	}
	std::string AsStr() const {
		if (t == T::Str) return s;
		if (t == T::Num) return is_int ? std::to_string(i) : std::to_string(n);
		return {};
	}
};

struct Parser {
	const char* p;
	const char* e;
	bool        ok = true;
	void ws() { while (p < e && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) p++; }
	static void utf8(std::string& out, unsigned cp) {
		if (cp < 0x80) out += static_cast<char>(cp);
		else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
		else if (cp < 0x10000) { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
		else { out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
	}
	unsigned hex4() {
		unsigned v = 0;
		for (int k = 0; k < 4; k++) {
			if (p >= e) { ok = false; return 0; }
			const char c = *p++;
			v <<= 4;
			if (c >= '0' && c <= '9') v |= c - '0';
			else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
			else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
			else ok = false;
		}
		return v;
	}
	std::string str() {
		std::string out;
		p++; // "
		while (p < e && *p != '"') {
			if (*p == '\\') {
				p++;
				if (p >= e) break;
				const char c = *p++;
				switch (c) {
				case 'n': out += '\n'; break;
				case 't': out += '\t'; break;
				case 'r': out += '\r'; break;
				case 'b': out += '\b'; break;
				case 'f': out += '\f'; break;
				case 'u': {
					unsigned cp = hex4();
					if (cp >= 0xD800 && cp < 0xDC00 && p + 1 < e && p[0] == '\\' && p[1] == 'u') {
						p += 2;
						const unsigned lo = hex4();
						cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
					}
					utf8(out, cp);
					break;
				}
				default: out += c;
				}
			} else {
				out += *p++;
			}
		}
		if (p < e) p++; else ok = false;
		return out;
	}
	J val(int depth = 0) {
		ws();
		J j;
		if (p >= e || depth > 64) { ok = false; return j; }
		const char c = *p;
		if (c == '{') {
			p++;
			j.t = J::T::Obj;
			ws();
			if (p < e && *p == '}') { p++; return j; }
			while (ok) {
				ws();
				if (p >= e || *p != '"') { ok = false; break; }
				std::string k = str();
				ws();
				if (p >= e || *p != ':') { ok = false; break; }
				p++;
				j.o.emplace_back(std::move(k), val(depth + 1));
				ws();
				if (p < e && *p == ',') { p++; continue; }
				if (p < e && *p == '}') { p++; break; }
				ok = false;
			}
		} else if (c == '[') {
			p++;
			j.t = J::T::Arr;
			ws();
			if (p < e && *p == ']') { p++; return j; }
			while (ok) {
				j.a.push_back(val(depth + 1));
				ws();
				if (p < e && *p == ',') { p++; continue; }
				if (p < e && *p == ']') { p++; break; }
				ok = false;
			}
		} else if (c == '"') {
			j.t = J::T::Str;
			j.s = str();
		} else if (c == 't' && e - p >= 4 && std::strncmp(p, "true", 4) == 0) {
			p += 4; j = J::Bool(true);
		} else if (c == 'f' && e - p >= 5 && std::strncmp(p, "false", 5) == 0) {
			p += 5; j = J::Bool(false);
		} else if (c == 'n' && e - p >= 4 && std::strncmp(p, "null", 4) == 0) {
			p += 4;
		} else {
			const char* s = p;
			bool        real = false;
			while (p < e && (std::isdigit(static_cast<unsigned char>(*p)) || *p == '-' || *p == '+' || *p == '.' || *p == 'e' || *p == 'E')) {
				if (*p == '.' || *p == 'e' || *p == 'E') real = true;
				p++;
			}
			if (s == p) { ok = false; return j; }
			const std::string num(s, p);
			if (real) j = J::Num(std::strtod(num.c_str(), nullptr));
			else j = J::Int(std::strtoll(num.c_str(), nullptr, 10));
		}
		return j;
	}
};

inline bool Parse(const std::string& text, J* out) {
	Parser ps {text.data(), text.data() + text.size()};
	*out = ps.val();
	ps.ws();
	return ps.ok;
}

inline void Dump(const J& j, std::string& out) {
	switch (j.t) {
	case J::T::Null: out += "null"; break;
	case J::T::Bool: out += j.b ? "true" : "false"; break;
	case J::T::Num: {
		if (j.is_int) { out += std::to_string(j.i); break; }
		char buf[64];
		std::snprintf(buf, sizeof(buf), "%.9g", j.n);
		out += buf;
		if (std::strpbrk(buf, ".eEn") == nullptr) out += ".0"; // a float stays a float (pos_x...: the game reads doubles)
		break;
	}
	case J::T::Str:
		out += '"';
		for (unsigned char c: j.s) {
			if (c == '"') out += "\\\"";
			else if (c == '\\') out += "\\\\";
			else if (c == '\n') out += "\\n";
			else if (c == '\r') out += "\\r";
			else if (c == '\t') out += "\\t";
			else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
			else out += static_cast<char>(c);
		}
		out += '"';
		break;
	case J::T::Arr:
		out += '[';
		for (size_t k = 0; k < j.a.size(); k++) { if (k) out += ','; Dump(j.a[k], out); }
		out += ']';
		break;
	case J::T::Obj:
		out += '{';
		for (size_t k = 0; k < j.o.size(); k++) {
			if (k) out += ',';
			Dump(J::Str(j.o[k].first), out);
			out += ':';
			Dump(j.o[k].second, out);
		}
		out += '}';
		break;
	}
}
inline std::string Dump(const J& j) { std::string s; Dump(j, s); return s; }

// ---- storage ------------------------------------------------------------------------------------------------------
struct Store {
	std::filesystem::path dir;
	std::mutex            m;
	J                     cuentas = J::Obj(); // account -> {"grade_pendiente": n, ...}
	J                     mensajes = J::Arr(), replays = J::Arr(), fantasmas = J::Arr(), senales = J::Arr();

	J Load(const char* name, J def) {
		std::ifstream in(dir / name, std::ios::binary);
		if (!in) return def;
		std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		J j;
		return (Parse(text, &j) && j.t == def.t) ? j : def;
	}
	void Save(const char* name, const J& j) {
		std::error_code ec;
		std::filesystem::create_directories(dir, ec);
		const auto tmp = dir / (std::string(name) + ".tmp");
		{
			std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
			out << Dump(j);
		}
		std::filesystem::rename(tmp, dir / name, ec);
		if (ec) { std::filesystem::remove(dir / name, ec); std::filesystem::rename(tmp, dir / name, ec); }
	}
	void Open(const std::filesystem::path& d) {
		dir       = d;
		cuentas   = Load("cuentas.json", J::Obj());
		mensajes  = Load("mensajes.json", J::Arr());
		replays   = Load("manchas.json", J::Arr());
		fantasmas = Load("fantasmas.json", J::Arr());
		senales   = Load("senales.json", J::Arr());
	}
	static void Cap(J& arr, size_t max) {
		if (arr.a.size() > max) arr.a.erase(arr.a.begin(), arr.a.begin() + static_cast<long>(arr.a.size() - max));
	}
};

inline std::string Uuid() {
	static std::mt19937_64 rng {std::random_device {}() ^ static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())};
	static std::mutex      m;
	std::lock_guard        lock(m);
	const uint64_t a = rng(), b = rng();
	char           buf[48];
	std::snprintf(buf, sizeof(buf), "%08X-%04X-%04X-%04X-%012llX", static_cast<unsigned>(a >> 32), static_cast<unsigned>((a >> 16) & 0xFFFF),
	              static_cast<unsigned>(0x4000 | (a & 0x0FFF)), static_cast<unsigned>(0x8000 | (b >> 50)), static_cast<unsigned long long>(b & 0xFFFFFFFFFFFFull));
	return buf;
}
inline double Now() { return static_cast<double>(std::time(nullptr)); }
inline std::string Upper(std::string s) { for (auto& c: s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c))); return s; }

struct Server {
	Store       store;
	int         port = 8090;
	bool        see_own = false;     // own messages / stains listed too (one-PC tests)
	bool        show_signs = false;  // other players' SOS signs (summoning needs P2P: off)
	std::string log_path;
	std::mutex  log_m;
	std::atomic<long long> requests {0};

	static long long Block(const J& p) {
		for (const char* k: {"blockId", "block_id", "blockID"})
			if (const J* v = p.Get(k)) return v->AsInt();
		return 0;
	}
	static std::string Account(const J& p) {
		for (const char* k: {"AccountId", "accountId"})
			if (const J* v = p.Get(k)) { auto s = v->AsStr(); if (!s.empty()) return s; }
		return "0";
	}
	static void Vec(const J* v, double* x, double* y, double* z) {
		*x = *y = *z = 0.0;
		if (v == nullptr) return;
		if (v->t == J::T::Obj) {
			if (const J* a = v->Get("x")) *x = a->AsNum();
			if (const J* a = v->Get("y")) *y = a->AsNum();
			if (const J* a = v->Get("z")) *z = a->AsNum();
		} else if (v->t == J::T::Arr && v->a.size() >= 3) {
			*x = v->a[0].AsNum(); *y = v->a[1].AsNum(); *z = v->a[2].AsNum();
		} else if (v->t == J::T::Num) {
			*y = v->AsNum();
		}
	}
	static void PutPos(J& row, const J& p) {
		double x, y, z;
		Vec(p.Get("pos"), &x, &y, &z);
		row.Set("pos_x", J::Num(x)); row.Set("pos_y", J::Num(y)); row.Set("pos_z", J::Num(z));
		Vec(p.Get("ang"), &x, &y, &z);
		row.Set("ang_x", J::Num(x)); row.Set("ang_y", J::Num(y)); row.Set("ang_z", J::Num(z));
	}
	static void CopyPos(J& to, const J& row) {
		for (const char* k: {"pos_x", "pos_y", "pos_z", "ang_x", "ang_y", "ang_z"}) {
			const J* v = row.Get(k);
			to.Set(k, J::Num(v ? v->AsNum() : 0.0));
		}
	}
	template <class F>
	static std::vector<const J*> Pick(const J& arr, F keep, size_t num) {
		std::vector<const J*> all;
		for (auto it = arr.a.rbegin(); it != arr.a.rend() && all.size() < 500; ++it)
			if (keep(*it)) all.push_back(&*it);
		static std::mt19937 rng {std::random_device {}()};
		std::shuffle(all.begin(), all.end(), rng);
		if (all.size() > num) all.resize(num);
		return all;
	}

	J State(const std::string& host) {
		J d = J::Obj();
		d.Set("serverState", J::Int(0)); // a number, else the game reads nothing more (@0x6614a0)
		d.Set("serverStateMsg", J::Str(""));
		d.Set("serverSandbox", J::Str("kyty"));
		d.Set("gameServerUrl", J::Str("http://" + host));
		d.Set("gameServerStatusMessageInterval", J::Int(600));
		d.Set("enableOnlineMultiplay", J::Bool(true)); // (online5) co-op / invasions through our sessions + relay
		d.Set("enableWanderingGhost", J::Bool(true));
		d.Set("getWanderingGhostInterval", J::Int(30)); // the game clamps 20..48
		d.Set("setWanderingGhostInterval", J::Int(30));
		d.Set("getBloodMessageNum", J::Int(50)); // 50..100
		d.Set("getReplayListNum", J::Int(50));
		for (const char* k: {"getBluePhantomNum", "getRedPhantomNum", "getBlackPhantomNum", "getOrangePhantomNum"}) d.Set(k, J::Int(0));
		d.Set("getSOSQueryInternvalMin", J::Int(30));
		d.Set("getSOSRefreshInternval", J::Int(30));
		d.Set("getSOSQueryInternvalMax", J::Int(60));
		return d;
	}

	// One command; *known false: answered {} (unknown command).
	J Command(const std::string& cmd, const J& p, const std::string& host, bool* known) {
		*known = true;
		const std::string acc = Account(p);
		if (cmd == "getServerState") return State(host);
		std::lock_guard lock(store.m);
		J& user = [&]() -> J& {
			for (auto& kv: store.cuentas.o)
				if (kv.first == acc) return kv.second;
			store.cuentas.o.emplace_back(acc, J::Obj());
			return store.cuentas.o.back().second;
		}();
		if (cmd == "login") {
			if (!user.Get("creada")) user.Set("creada", J::Num(Now()));
			user.Set("ultimo", J::Num(Now()));
			store.Save("cuentas.json", store.cuentas);
			J d = J::Obj();
			d.Set("loginState", J::Int(1)); // 1: in (0 would ask to accept "agreement")
			return d;
		}
		if (cmd == "addNewAccount") {
			const bool fresh = user.Get("creada") == nullptr;
			if (fresh) user.Set("creada", J::Num(Now()));
			store.Save("cuentas.json", store.cuentas);
			J d = J::Obj();
			d.Set("createdNewAccount", J::Bool(fresh));
			return d;
		}
		if (cmd == "initializeMultiplay" || cmd == "initializeCharacter") return J::Obj();
		if (cmd == "getMultiplayGrade") { J d = J::Obj(); d.Set("multiPlayCount", J::Int(0)); return d; }
		if (cmd == "getTimeMessage") { J d = J::Obj(); d.Set("timeMessages", J::Arr()); d.Set("timeMessageIds", J::Arr()); return d; }
		if (cmd == "getQWCData") { // data: a list of 7 {wb, lr} (@0x672e40); no global tendency kept: neutral
			J d = J::Arr();
			for (int k = 0; k < 7; k++) { J w = J::Obj(); w.Set("wb", J::Int(0)); w.Set("lr", J::Int(0)); d.Push(w); }
			return d;
		}
		if (cmd == "addBloodMessage") {
			J row = J::Obj();
			row.Set("id", J::Str(Uuid())); row.Set("account", J::Str(acc)); row.Set("block", J::Int(Block(p)));
			PutPos(row, p);
			for (const char* k: {"messageId", "mainMsgId", "addMsgCateId"}) { const J* v = p.Get(k); row.Set(k, J::Int(v ? v->AsInt() : 0)); }
			row.Set("up", J::Int(0)); row.Set("down", J::Int(0)); row.Set("t", J::Num(Now()));
			store.mensajes.Push(row);
			Store::Cap(store.mensajes, 20000);
			store.Save("mensajes.json", store.mensajes);
			return J::Obj();
		}
		if (cmd == "getBloodMessage") {
			const J*  n   = p.Get("messageNum");
			long long num = n ? n->AsInt(50) : 50;
			num = std::clamp(num <= 0 ? 50 : num, 1LL, 100LL);
			const long long blk = Block(p);
			auto rows = Pick(store.mensajes, [&](const J& r) { return r.Get("block")->AsInt() == blk && (see_own || r.Get("account")->AsStr() != acc); }, static_cast<size_t>(num));
			J list = J::Arr();
			for (const J* r: rows) {
				J m = J::Obj(); // the keys and types the game reads (@0x670040)
				m.Set("blood_msg_id", J::Str(r->Get("id")->AsStr()));
				m.Set("account_id", J::Str(r->Get("account")->AsStr()));
				m.Set("block_id", J::Int(r->Get("block")->AsInt()));
				CopyPos(m, *r);
				m.Set("message_id", J::Int(r->Get("messageId")->AsInt()));
				m.Set("main_msg_id", J::Int(r->Get("mainMsgId")->AsInt()));
				m.Set("add_msg_cate_id", J::Int(r->Get("addMsgCateId")->AsInt()));
				m.Set("grade_up", J::Int(r->Get("up")->AsInt()));
				m.Set("grade_down", J::Int(r->Get("down")->AsInt()));
				list.Push(m);
			}
			J d = J::Obj();
			d.Set("bloodMsgs", list);
			return d;
		}
		if (cmd == "updateBloodMessageGrade") {
			const J*          idv = p.Get("bloodMsgId");
			const std::string id  = Upper(idv ? idv->AsStr() : "");
			const long long   up = p.Get("upgrade") ? p.Get("upgrade")->AsInt() : 0, down = p.Get("downgrade") ? p.Get("downgrade")->AsInt() : 0;
			for (auto& r: store.mensajes.a)
				if (Upper(r.Get("id")->AsStr()) == id) {
					r.Set("up", J::Int(r.Get("up")->AsInt() + up));
					r.Set("down", J::Int(r.Get("down")->AsInt() + down));
					const std::string owner = r.Get("account")->AsStr();
					if (up > 0)
						for (auto& kv: store.cuentas.o)
							if (kv.first == owner) {
								const J* g = kv.second.Get("grade_pendiente");
								kv.second.Set("grade_pendiente", J::Int((g ? g->AsInt() : 0) + up));
							}
					store.Save("mensajes.json", store.mensajes);
					store.Save("cuentas.json", store.cuentas);
					break;
				}
			return J::Obj();
		}
		if (cmd == "getBloodMessageGrade") { // new ratings of my messages since the last ask (the game heals for them)
			const J*        g = user.Get("grade_pendiente");
			const long long n = g ? g->AsInt() : 0;
			user.Set("grade_pendiente", J::Int(0));
			if (n) store.Save("cuentas.json", store.cuentas);
			J d = J::Obj();
			d.Set("blood_message_grade", J::Int(n));
			return d;
		}
		if (cmd == "deleteBloodMessage") {
			const J*          idv = p.Get("bloodMsgId");
			const std::string id  = Upper(idv ? idv->AsStr() : "");
			auto& v = store.mensajes.a;
			const auto before = v.size();
			v.erase(std::remove_if(v.begin(), v.end(), [&](const J& r) { return Upper(r.Get("id")->AsStr()) == id && r.Get("account")->AsStr() == acc; }), v.end());
			if (v.size() != before) store.Save("mensajes.json", store.mensajes);
			return J::Obj();
		}
		if (cmd == "addReplayData") {
			J row = J::Obj();
			row.Set("id", J::Str(Uuid())); row.Set("account", J::Str(acc)); row.Set("block", J::Int(Block(p)));
			PutPos(row, p);
			const J* data = p.Get("replayData");
			row.Set("data", J::Str(data == nullptr ? std::string() : data->t == J::T::Str ? data->s : Dump(*data)));
			row.Set("t", J::Num(Now()));
			store.replays.Push(row);
			Store::Cap(store.replays, 5000);
			store.Save("manchas.json", store.replays);
			return J::Obj();
		}
		if (cmd == "getReplayList") {
			long long num = 50;
			for (const char* k: {"replayNum", "maxReplayNum"})
				if (const J* v = p.Get(k)) num = v->AsInt(50);
			num = std::clamp(num <= 0 ? 50 : num, 1LL, 100LL);
			const long long blk = Block(p);
			auto rows = Pick(store.replays, [&](const J& r) { return r.Get("block")->AsInt() == blk && (see_own || r.Get("account")->AsStr() != acc); }, static_cast<size_t>(num));
			J list = J::Arr();
			for (const J* r: rows) {
				J m = J::Obj(); // @0x66eff0
				m.Set("replay_id", J::Str(r->Get("id")->AsStr()));
				m.Set("account_id", J::Str(r->Get("account")->AsStr()));
				m.Set("block_id", J::Int(r->Get("block")->AsInt()));
				CopyPos(m, *r);
				list.Push(m);
			}
			J d = J::Obj();
			d.Set("replays", list);
			return d;
		}
		if (cmd == "getReplayData") {
			const J*          idv = p.Get("replayMsgId");
			const std::string id  = Upper(idv ? idv->AsStr() : "");
			for (const auto& r: store.replays.a)
				if (Upper(r.Get("id")->AsStr()) == id) {
					J d = J::Obj(); // @0x671120
					d.Set("replayData", J::Str(r.Get("data")->AsStr()));
					d.Set("replayMsgId", J::Str(r.Get("id")->AsStr()));
					return d;
				}
			return J::Obj();
		}
		if (cmd == "setWanderingGhost") {
			const J* data = p.Get("replayData");
			if (data == nullptr || data->t != J::T::Str || data->s.empty()) return J::Obj();
			const long long blk = Block(p);
			auto& v = store.fantasmas.a;
			v.erase(std::remove_if(v.begin(), v.end(), [&](const J& r) { return r.Get("account")->AsStr() == acc && r.Get("block")->AsInt() == blk; }), v.end());
			J row = J::Obj();
			row.Set("account", J::Str(acc)); row.Set("block", J::Int(blk)); row.Set("data", J::Str(data->s)); row.Set("t", J::Num(Now()));
			store.fantasmas.Push(row);
			Store::Cap(store.fantasmas, 5000);
			store.Save("fantasmas.json", store.fantasmas);
			return J::Obj();
		}
		if (cmd == "getWanderingGhost") {
			const J*        n   = p.Get("maxGhostNum");
			const long long num = std::clamp(n ? n->AsInt(10) : 10LL, 0LL, 50LL);
			const long long blk = Block(p);
			const double    since = Now() - 30 * 60; // the last half hour
			J               list  = J::Arr();
			for (auto it = store.fantasmas.a.rbegin(); it != store.fantasmas.a.rend() && static_cast<long long>(list.a.size()) < num; ++it)
				if (it->Get("block")->AsInt() == blk && it->Get("t")->AsNum() > since && (see_own || it->Get("account")->AsStr() != acc))
					list.Push(J::Str(it->Get("data")->AsStr()));
			J d = J::Obj(); // base64 strings (@0x66ebf0)
			d.Set("sosWandering", J::Arr());
			d.Set("wandering", list);
			return d;
		}
		if (cmd == "addSummonSign") { // the game reads its own sign back: newSigns{signType, signList[{accountId, pos}]}
			const J* st = p.Get("signType");
			J        sign = J::Obj();
			sign.Set("accountId", J::Str(acc));
			sign.Set("platform", J::Int(p.Get("platform") ? p.Get("platform")->AsInt(1) : 1));
			double x, y, z;
			Vec(p.Get("pos"), &x, &y, &z);
			J pos = J::Obj();
			pos.Set("x", J::Num(x)); pos.Set("y", J::Num(y)); pos.Set("z", J::Num(z));
			sign.Set("pos", pos);
			if (const J* pd = p.Get("playerData")) sign.Set("playerData", *pd);
			auto& v = store.senales.a;
			v.erase(std::remove_if(v.begin(), v.end(), [&](const J& r) { return r.Get("account")->AsStr() == acc; }), v.end());
			J row = J::Obj();
			row.Set("account", J::Str(acc)); row.Set("type", J::Int(st ? st->AsInt() : 0)); row.Set("sign", sign); row.Set("t", J::Num(Now()));
			store.senales.Push(row);
			store.Save("senales.json", store.senales);
			J ns = J::Obj();
			ns.Set("signType", J::Int(st ? st->AsInt() : 0));
			J list = J::Arr();
			list.Push(sign);
			ns.Set("signList", list);
			J d = J::Obj();
			d.Set("newSigns", ns);
			return d;
		}
		if (cmd == "deleteSummonSign") {
			auto& v = store.senales.a;
			v.erase(std::remove_if(v.begin(), v.end(), [&](const J& r) { return r.Get("account")->AsStr() == acc; }), v.end());
			store.Save("senales.json", store.senales);
			return J::Obj();
		}
		if (cmd == "getSummonSigns" || cmd == "getForceSummons") {
			J list = J::Arr();
			if (cmd == "getSummonSigns" && show_signs) {
				std::map<long long, J> groups;
				for (const auto& r: store.senales.a)
					if (r.Get("account")->AsStr() != acc && r.Get("t")->AsNum() > Now() - 600) {
						const long long t = r.Get("type")->AsInt();
						auto&           g = groups[t];
						if (g.t != J::T::Obj) { g = J::Obj(); g.Set("signType", J::Int(t)); g.Set("ignoredPassword", J::Int(0)); g.Set("signList", J::Arr()); }
						for (auto& kv: g.o) if (kv.first == "signList") kv.second.Push(*r.Get("sign"));
					}
				for (auto& kv: groups) list.Push(kv.second);
			}
			J d = J::Obj();
			d.Set("newSigns", list);
			return d;
		}
		*known = false;
		return J::Obj();
	}

	// ---- (online5) Accounts (port 31413), PSN WebAPI of PS5 used by the game (port 31415), push events ---------------
	// Exactly the calls of the eboot's NexusRevolution2 PS4_PS5_Crossgen matching (Ghidra, see INFORME.md):
	//   GET    /v1/users/me/gameSessions?fields=sessionId&joinStateFilter=RESERVED          (@0x13a7b20)
	//   GET    /v1/gameSessions?fields=customData1,matchmaking,sessionId  + X-PSN-SESSION-MANAGER-SESSION-IDS (@0x13a7b20)
	//   GET    /v1/gameSessions?fields=member(players),member(players(customData1)) + ids header (@0x13a92f0)
	//   GET    /v1/gameSessions?fields=createdTimestamp + ids header (@0x13ba4d0, after a create)
	//   POST   /v1/gameSessions  {"gameSessions":[{"maxPlayers":..,"member":{"players":[{"accountId":"me",...}]},...}]} (@0x13b0f20)
	//   PATCH  /v1/gameSessions/{id}  {"customData1":...}                                 (@0x13b8830)
	//   POST   /v1/gameSessions/{id}/member/players {"players":[{accountId,platform,pushContexts,customData1,joinState}]}
	//   DELETE /v1/gameSessions/{id}/members/me                                            (@0x13b9f90)
	//   POST   /v1/gameSessions/{id}/sessionMessage {"to":[{accountId,platform}],"payload":"kick"} (@0x13b6540)
	//   POST   /v1/tickets {"rulesetName":..,"playerAttributes":[..]} -> {"ticketId": 36-char UUID} (@0x13a8d50)
	//   DELETE /v1/tickets/{id}                                                            (@0x13aa870)
	// Push (psn:sessionManager:gs:sessionMessage:created, ...): queued per account, fetched by the emulator with
	// GET /kyty/push (long poll). Members for the emulator's session signaling: the same member GET.
	struct Psn {
		J        accounts = J::Obj(); // name -> {accountId, salt, hash}
		std::map<std::string, std::string> tokens; // token -> accountId
		std::map<std::string, std::string> names;  // accountId -> name
		std::map<std::string, J>           sessions; // id -> session
		std::map<std::string, J>           tickets;  // id -> {account, body, t}
		std::map<std::string, std::vector<J>> push;  // accountId -> events
		std::map<std::string, double>      seen;     // accountId -> last request time
	} psn;
	std::condition_variable push_cv;

	static uint64_t Fnv(const std::string& s, uint64_t h = 1469598103934665603ull) {
		for (unsigned char c: s) { h ^= c; h *= 1099511628211ull; }
		return h;
	}
	static std::string PassHash(const std::string& salt, const std::string& pw) {
		uint64_t h = Fnv(salt + ":" + pw);
		for (int k = 0; k < 20000; k++) h = Fnv(std::to_string(h) + salt + pw, h);
		char b[32];
		std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(h));
		return b;
	}
	std::string AccountOf(const std::map<std::string, std::string>& headers) {
		auto it = headers.find("authorization");
		if (it == headers.end()) return {};
		std::string v = it->second;
		if (v.rfind("Bearer ", 0) == 0) v = v.substr(7);
		auto t = psn.tokens.find(v);
		return t == psn.tokens.end() ? std::string() : t->second;
	}
	void SavePsn() {
		store.Save("cuentas_ps5.json", psn.accounts);
	}
	J Account(const std::string& path, const J& p, int* status) {
		J          out  = J::Obj();
		const std::string name = p.Get("name") ? p.Get("name")->AsStr() : "";
		const std::string pw   = p.Get("password") ? p.Get("password")->AsStr() : "";
		bool       valid = !name.empty() && name.size() <= 16 && !pw.empty();
		for (char c: name) if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) valid = false;
		if (!valid) { out.Set("ok", J::Bool(false)); out.Set("error", J::Str("nombre (1-16 letras/numeros/_/-) y contrasena obligatorios")); *status = 400; return out; }
		const J* acc = psn.accounts.Get(name);
		if (path == "/cuenta/registro") {
			if (acc != nullptr) { out.Set("ok", J::Bool(false)); out.Set("error", J::Str("ese nombre ya existe")); *status = 409; return out; }
			J a = J::Obj();
			const std::string salt = Uuid();
			a.Set("salt", J::Str(salt));
			a.Set("hash", J::Str(PassHash(salt, pw)));
			a.Set("accountId", J::Str(std::to_string((Fnv(name) & 0x0fffffffffffffffull) | 0x1000000000000000ull)));
			psn.accounts.Set(name, a);
			SavePsn();
			acc = psn.accounts.Get(name);
		} else if (acc == nullptr || PassHash(acc->Get("salt")->AsStr(), pw) != acc->Get("hash")->AsStr()) {
			out.Set("ok", J::Bool(false)); out.Set("error", J::Str("nombre o contrasena incorrectos")); *status = 401; return out;
		}
		const std::string id = acc->Get("accountId")->AsStr(), token = Uuid() + Uuid();
		psn.tokens[token] = id;
		psn.names[id]     = name;
		psn.seen[id]      = Now();
		out.Set("ok", J::Bool(true)); out.Set("accountId", J::Str(id)); out.Set("onlineId", J::Str(name)); out.Set("token", J::Str(token));
		return out;
	}
	void Push(const std::string& to, const std::string& service, const std::string& type, const J& data, const std::string& from) {
		J e = J::Obj();
		e.Set("npServiceName", J::Str(service)); e.Set("dataType", J::Str(type)); e.Set("data", J::Str(Dump(data)));
		e.Set("from", J::Str(from)); e.Set("fromOnlineId", J::Str(psn.names.count(from) ? psn.names[from] : ""));
		auto& q = psn.push[to];
		q.push_back(e);
		if (q.size() > 200) q.erase(q.begin());
		push_cv.notify_all();
	}
	J Player(const J& in, const std::string& me, const char* default_state) {
		J pl = J::Obj();
		std::string id = in.Get("accountId") ? in.Get("accountId")->AsStr() : me;
		if (id == "me" || id.empty()) id = me;
		pl.Set("accountId", J::Str(id));
		pl.Set("platform", J::Str(in.Get("platform") ? in.Get("platform")->AsStr() : "PS5"));
		pl.Set("onlineId", J::Str(psn.names.count(id) ? psn.names[id] : ""));
		pl.Set("joinState", J::Str(in.Get("joinState") ? in.Get("joinState")->AsStr() : default_state));
		if (const J* cd = in.Get("customData1")) pl.Set("customData1", *cd);
		if (const J* pc = in.Get("pushContexts")) pl.Set("pushContexts", *pc);
		pl.Set("t", J::Num(Now()));
		return pl;
	}
	static J* Players(J& s) {
		for (auto& kv: s.o) if (kv.first == "member") for (auto& m: kv.second.o) if (m.first == "players") return &m.second;
		return nullptr;
	}
	J View(J s) { // a session as the game reads it (any fields asked: all are given)
		J out = J::Obj();
		for (const char* k: {"sessionId", "createdTimestamp", "maxPlayers", "customData1", "matchmaking", "representative", "member", "joinDisabled"})
			if (const J* v = s.Get(k)) out.Set(k, *v);
		return out;
	}
	static std::vector<std::string> SplitIds(std::string v) {
		std::vector<std::string> ids;
		size_t at = 0;
		while (at <= v.size()) {
			const size_t e = std::min(v.find(',', at), v.size());
			std::string id = v.substr(at, e - at);
			while (!id.empty() && id.front() == ' ') id.erase(id.begin());
			while (!id.empty() && id.back() == ' ') id.pop_back();
			if (!id.empty()) ids.push_back(id);
			at = e + 1;
		}
		return ids;
	}
	void TryMatch(const std::string& new_ticket) { // tickets of the same ruleset: both reserved into one new session
		J&                tk  = psn.tickets[new_ticket];
		const std::string rs  = tk.Get("ruleset")->AsStr();
		const std::string acc = tk.Get("account")->AsStr();
		for (auto& [id, other]: psn.tickets) {
			if (id == new_ticket || other.Get("ruleset")->AsStr() != rs || other.Get("account")->AsStr() == acc) continue;
			const std::string host = other.Get("account")->AsStr(), sid = Uuid();
			J s = J::Obj();
			s.Set("sessionId", J::Str(sid));
			s.Set("createdTimestamp", J::Str(std::to_string(static_cast<long long>(Now() * 1000))));
			s.Set("maxPlayers", J::Int(4));
			J mm = J::Obj(); mm.Set("ticketIds", J::Arr()); s.Set("matchmaking", mm);
			J rep = J::Obj(); rep.Set("accountId", J::Str(host)); rep.Set("platform", J::Str("PS5")); s.Set("representative", rep);
			J players = J::Arr();
			for (const std::string& a: {host, acc}) { J in = J::Obj(); in.Set("accountId", J::Str(a)); players.Push(Player(in, a, "RESERVED")); }
			J member = J::Obj(); member.Set("players", players); s.Set("member", member);
			psn.sessions[sid] = s;
			Log("matchmaking: tickets " + id + " + " + new_ticket + " (" + rs + ") -> sesion " + sid);
			psn.tickets.erase(id);
			psn.tickets.erase(new_ticket);
			return;
		}
	}
	// Routes of the PSN-like WebAPI and the emulator helpers; *status: the HTTP status. known: a route matched.
	J WebApi(const std::string& method, const std::string& full_path, const std::map<std::string, std::string>& headers, const J& p, int* status, bool* known) {
		*known = true;
		*status = 200;
		const std::string me   = AccountOf(headers);
		if (!me.empty()) psn.seen[me] = Now();
		const auto        q    = full_path.find('?');
		const std::string path = full_path.substr(0, q);
		const std::string query = q == std::string::npos ? "" : full_path.substr(q + 1);
		const auto ids_header = [&]() {
			auto it = headers.find("x-psn-session-manager-session-ids");
			return it == headers.end() ? std::vector<std::string>() : SplitIds(it->second);
		};
		if (path == "/v1/users/me/gameSessions" && method == "GET") {
			const bool reserved_only = query.find("joinStateFilter=RESERVED") != std::string::npos;
			J list = J::Arr();
			for (auto& [id, s]: psn.sessions)
				if (J* pl = Players(s))
					for (const auto& m: pl->a)
						if (m.Get("accountId")->AsStr() == me && (!reserved_only || m.Get("joinState")->AsStr() == "RESERVED")) {
							J e = J::Obj(); e.Set("sessionId", J::Str(id)); list.Push(e);
							break;
						}
			J out = J::Obj(); out.Set("gameSessions", list); return out;
		}
		if (path == "/v1/gameSessions" && method == "GET") {
			J list = J::Arr();
			for (const auto& id: ids_header())
				if (auto it = psn.sessions.find(id); it != psn.sessions.end()) list.Push(View(it->second));
			J out = J::Obj(); out.Set("gameSessions", list); return out;
		}
		if (path == "/v1/gameSessions" && method == "POST") {
			J list = J::Arr();
			const J* in = p.Get("gameSessions");
			const size_t n = (in && in->t == J::T::Arr && !in->a.empty()) ? in->a.size() : 1;
			for (size_t k = 0; k < n; k++) {
				const J     req = (in && in->t == J::T::Arr && k < in->a.size()) ? in->a[k] : J::Obj();
				const std::string sid = Uuid();
				J s = J::Obj();
				s.Set("sessionId", J::Str(sid));
				s.Set("createdTimestamp", J::Str(std::to_string(static_cast<long long>(Now() * 1000))));
				s.Set("maxPlayers", J::Int(req.Get("maxPlayers") ? req.Get("maxPlayers")->AsInt(4) : 4));
				s.Set("joinDisabled", J::Bool(req.Get("joinDisabled") && req.Get("joinDisabled")->t == J::T::Bool && req.Get("joinDisabled")->b));
				if (const J* cd = req.Get("customData1")) s.Set("customData1", *cd);
				J rep = J::Obj(); rep.Set("accountId", J::Str(me)); rep.Set("platform", J::Str("PS5")); s.Set("representative", rep);
				J players = J::Arr();
				const J* mem = req.Get("member");
				const J* pls = mem ? mem->Get("players") : nullptr;
				if (pls && pls->t == J::T::Arr) for (const auto& pl: pls->a) players.Push(Player(pl, me, "JOINED"));
				if (players.a.empty()) players.Push(Player(J::Obj(), me, "JOINED"));
				J member = J::Obj(); member.Set("players", players); s.Set("member", member);
				psn.sessions[sid] = s;
				J e = J::Obj(); e.Set("sessionId", J::Str(sid)); list.Push(e);
				Log("sesion creada " + sid + " por " + me);
			}
			J out = J::Obj(); out.Set("gameSessions", list); return out;
		}
		if (path == "/v1/tickets" && method == "POST") {
			const std::string tid = Uuid();
			J t = J::Obj();
			t.Set("account", J::Str(me)); t.Set("ruleset", J::Str(p.Get("rulesetName") ? p.Get("rulesetName")->AsStr() : "")); t.Set("body", p); t.Set("t", J::Num(Now()));
			psn.tickets[tid] = t;
			Log("ticket " + tid + " de " + me + ": " + Dump(p).substr(0, 300));
			TryMatch(tid);
			J out = J::Obj(); out.Set("ticketId", J::Str(tid)); return out;
		}
		if (path.rfind("/v1/tickets/", 0) == 0) {
			const std::string tid = path.substr(12);
			if (method == "DELETE") { psn.tickets.erase(tid); *status = 200; return J::Obj(); }
			auto it = psn.tickets.find(tid);
			if (it == psn.tickets.end()) { *status = 404; return J::Obj(); }
			J out = J::Obj(); out.Set("ticketId", J::Str(tid)); out.Set("status", J::Str("SEARCHING")); return out;
		}
		if (path.rfind("/v1/gameSessions/", 0) == 0) {
			std::string rest = path.substr(17);
			const auto  sl   = rest.find('/');
			const std::string sid = rest.substr(0, sl), sub = sl == std::string::npos ? "" : rest.substr(sl);
			auto it = psn.sessions.find(sid);
			if (it == psn.sessions.end()) { *status = 404; J e = J::Obj(); e.Set("error", J::Str("sesion no encontrada")); return e; }
			J& s = it->second;
			if (sub.empty() && method == "PATCH") {
				for (const auto& kv: p.o) s.Set(kv.first, kv.second);
				*status = 200;
				return J::Obj();
			}
			if (sub.empty() && method == "GET") return View(s);
			if (sub.empty() && method == "DELETE") { psn.sessions.erase(it); *status = 200; return J::Obj(); }
			if (sub == "/member/players" && method == "POST") {
				J* pl = Players(s);
				const J* in = p.Get("players");
				J added = J::Arr();
				if (pl && in && in->t == J::T::Arr)
					for (const auto& one: in->a) {
						J np = Player(one, me, "JOINED");
						const std::string id = np.Get("accountId")->AsStr();
						bool replaced = false;
						for (auto& m: pl->a) if (m.Get("accountId")->AsStr() == id) { m = np; replaced = true; }
						if (!replaced) pl->Push(np);
						J a = J::Obj(); a.Set("accountId", J::Str(id)); added.Push(a);
						Log("sesion " + sid + ": jugador " + id + " " + np.Get("joinState")->AsStr());
					}
				J out = J::Obj(); out.Set("players", added); return out;
			}
			if ((sub == "/members/me" || sub == "/member/players/me") && method == "DELETE") {
				if (J* pl = Players(s)) {
					auto& v = pl->a;
					v.erase(std::remove_if(v.begin(), v.end(), [&](const J& m) { return m.Get("accountId")->AsStr() == me; }), v.end());
					if (v.empty()) psn.sessions.erase(it);
				}
				*status = 200;
				return J::Obj();
			}
			if (sub == "/sessionMessage" && method == "POST") {
				const J* to = p.Get("to");
				J data = J::Obj();
				data.Set("gameSessionId", J::Str(sid));
				if (const J* pl = p.Get("payload")) data.Set("payload", *pl);
				J from = J::Obj(); from.Set("accountId", J::Str(me)); from.Set("platform", J::Str("PS5")); data.Set("from", from);
				if (to && to->t == J::T::Arr)
					for (const auto& t: to->a)
						if (const J* a = t.Get("accountId")) Push(a->AsStr(), "sessionManager", "psn:sessionManager:gs:sessionMessage:created", data, me);
				*status = 200;
				return J::Obj();
			}
		}
		*known = false;
		*status = 200;
		return J::Obj();
	}

	void Log(const std::string& line) {
		std::lock_guard lock(log_m);
		std::ofstream   out(log_path, std::ios::binary | std::ios::app);
		out << line << "\n";
	}

	// One HTTP request (body JSON) -> the response body.
	std::string Handle(const std::string& method, const std::string& path, const std::string& host, const std::string& client,
	                   const std::string& body) {
		requests++;
		J p;
		if (!Parse(body, &p) || p.t != J::T::Obj) p = J::Obj();
		std::string cmd = p.Get("cmd") ? p.Get("cmd")->AsStr() : "";
		if (cmd.empty() && path.find("/dsmaster/ss") != std::string::npos) cmd = "getServerState";
		bool known = false;
		J    data;
		try {
			data = Command(cmd, p, host, &known);
		} catch (...) {
			data = J::Obj();
		}
		char when[32];
		const std::time_t now = std::time(nullptr);
		std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
		Log(std::string(when) + " " + client + " " + method + " " + path + " cmd=" + (cmd.empty() ? "-" : cmd) +
		    (known ? "" : " (desconocido: respuesta vacia)") + " bytes=" + std::to_string(body.size()));
		J env = J::Obj();
		env.Set("ServerState", J::Int(0));
		env.Set("data", data);
		return Dump(env);
	}
};

// ---- HTTP over sockets --------------------------------------------------------------------------------------------
#if defined(_WIN32)
using sock_t = SOCKET;
inline void CloseSock(sock_t s) { closesocket(s); }
constexpr sock_t kBadSock = INVALID_SOCKET;
#else
using sock_t = int;
inline void CloseSock(sock_t s) { close(s); }
constexpr sock_t kBadSock = -1;
#endif

inline void ServeClient(Server* srv, sock_t c, std::string client) {
#if defined(_WIN32)
	DWORD tmo = 30000;
#else
	timeval tmo {30, 0};
#endif
	setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tmo), sizeof(tmo));
	setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&tmo), sizeof(tmo));
	std::string raw;
	char        buf[16384];
	size_t      header_end = std::string::npos;
	size_t      need       = 0;
	for (;;) {
		const int n = static_cast<int>(recv(c, buf, sizeof(buf), 0));
		if (n <= 0) break;
		raw.append(buf, static_cast<size_t>(n));
		if (raw.size() > 64u * 1024u * 1024u) break;
		if (header_end == std::string::npos) {
			header_end = raw.find("\r\n\r\n");
			if (header_end != std::string::npos) {
				std::string low = raw.substr(0, header_end);
				for (auto& ch: low) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
				const auto cl = low.find("\ncontent-length:");
				need          = header_end + 4 + (cl != std::string::npos ? static_cast<size_t>(std::strtoull(low.c_str() + cl + 16, nullptr, 10)) : 0);
			}
		}
		if (header_end != std::string::npos && raw.size() >= need) break;
	}
	if (header_end != std::string::npos) {
		const std::string head = raw.substr(0, header_end);
		const auto        sp1  = head.find(' ');
		const auto        sp2  = head.find(' ', sp1 + 1);
		const std::string method = head.substr(0, sp1);
		const std::string path   = (sp1 != std::string::npos && sp2 != std::string::npos) ? head.substr(sp1 + 1, sp2 - sp1 - 1) : "/";
		std::map<std::string, std::string> headers;
		for (size_t at = head.find("\r\n"); at != std::string::npos && at < head.size();) {
			const size_t e    = head.find("\r\n", at + 2);
			const std::string line = head.substr(at + 2, (e == std::string::npos ? head.size() : e) - at - 2);
			const auto colon = line.find(':');
			if (colon != std::string::npos) {
				std::string k = line.substr(0, colon), v = line.substr(colon + 1);
				for (auto& ch: k) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
				while (!v.empty() && v.front() == ' ') v.erase(v.begin());
				headers[k] = v;
			}
			at = e;
		}
		std::string host = headers.count("host") ? headers["host"] : "127.0.0.1:" + std::to_string(srv->port);
		const std::string body = raw.size() > header_end + 4 ? raw.substr(header_end + 4) : std::string();
		int         status = 200;
		std::string out;
		const std::string bare = path.substr(0, path.find('?'));
		if (bare.rfind("/cuenta/", 0) == 0 || bare.rfind("/v1/", 0) == 0 || bare == "/kyty/push" || bare == "/kyty/estado" || bare == "/kyty/miembros") {
			J p;
			if (!Parse(body, &p) || p.t != J::T::Obj) p = J::Obj();
			J    res;
			bool known = true;
			{
				std::unique_lock lock(srv->store.m);
				if (bare.rfind("/cuenta/", 0) == 0) {
					res = srv->Account(bare, p, &status);
				} else if (bare == "/kyty/push") { // long poll: the push events of the caller (the emulator's thread)
					const std::string me = srv->AccountOf(headers);
					if (!me.empty()) srv->psn.seen[me] = Now();
					int espera = 20;
					if (const auto q = path.find("espera="); q != std::string::npos) espera = std::clamp(std::atoi(path.c_str() + q + 7), 0, 20);
					srv->push_cv.wait_for(lock, std::chrono::seconds(espera), [&] { return me.empty() || !srv->psn.push[me].empty(); });
					J list = J::Arr();
					if (!me.empty()) {
						for (auto& e: srv->psn.push[me]) list.Push(e);
						srv->psn.push[me].clear();
					}
					res = J::Obj();
					res.Set("events", list);
				} else if (bare == "/kyty/miembros") { // the players of a session, flat (the emulator's session signaling)
					const auto q = path.find("sesion=");
					const std::string sid = q == std::string::npos ? "" : path.substr(q + 7, 36);
					J list = J::Arr();
					if (auto it = srv->psn.sessions.find(sid); it != srv->psn.sessions.end())
						if (J* pl = Server::Players(it->second))
							for (const auto& m: pl->a) {
								J e = J::Obj();
								e.Set("accountId", J::Str(m.Get("accountId")->AsStr()));
								e.Set("joinState", J::Str(m.Get("joinState")->AsStr()));
								list.Push(e);
							}
					res = J::Obj();
					res.Set("jugadores", list);
				} else if (bare == "/kyty/estado") {
					res = J::Obj();
					J players = J::Arr();
					for (const auto& [id, t]: srv->psn.seen)
						if (Now() - t < 90) players.Push(J::Str(srv->psn.names.count(id) ? srv->psn.names[id] : id));
					res.Set("jugadores", players);
					J ses = J::Arr();
					for (const auto& [id, s]: srv->psn.sessions) {
						J e = J::Obj();
						e.Set("sessionId", J::Str(id));
						int n = 0;
						if (const J* m = s.Get("member")) if (const J* pl = m->Get("players")) n = static_cast<int>(pl->a.size());
						e.Set("jugadores", J::Int(n));
						ses.Push(e);
					}
					res.Set("sesiones", ses);
					res.Set("mensajes", J::Int(static_cast<long long>(srv->store.mensajes.a.size())));
					res.Set("manchas", J::Int(static_cast<long long>(srv->store.replays.a.size())));
					res.Set("peticiones", J::Int(srv->requests.load()));
				} else {
					res = srv->WebApi(method, path, headers, p, &status, &known);
				}
			}
			srv->requests++;
			if (bare != "/kyty/push")
				srv->Log(client + " " + method + " " + path + " -> " + std::to_string(status) + (known ? "" : " (ruta desconocida: respuesta vacia)") + " " + body.substr(0, 400));
			out = (status == 204) ? std::string() : Dump(res);
		} else {
			out = srv->Handle(method, path, host, client, body);
		}
		const char* text = status == 200 ? "OK" : status == 201 ? "Created" : status == 204 ? "No Content" : status == 404 ? "Not Found" : "Error";
		const std::string resp = "HTTP/1.1 " + std::to_string(status) + " " + text + "\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: " +
		                         std::to_string(out.size()) + "\r\nConnection: close\r\n\r\n" + out;
		size_t sent = 0;
		while (sent < resp.size()) {
			const int n = static_cast<int>(send(c, resp.data() + sent, static_cast<int>(resp.size() - sent), 0));
			if (n <= 0) break;
			sent += static_cast<size_t>(n);
		}
	}
	CloseSock(c);
}

// Listens on 0.0.0.0:port until *stop. false: the port could not be opened.
inline bool Listen(Server* srv, std::atomic<bool>* stop, std::string* error, int port = 0) {
	if (port == 0) port = srv->port;
#if defined(_WIN32)
	WSADATA wsa {};
	WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
	sock_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (s == kBadSock) { *error = "socket"; return false; }
#if !defined(_WIN32)
	int yes = 1;
	setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
#endif
	sockaddr_in addr {};
	addr.sin_family      = AF_INET;
	addr.sin_port        = htons(static_cast<uint16_t>(port));
	addr.sin_addr.s_addr = htonl(INADDR_ANY);
	if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(s, 64) != 0) {
		*error = "el puerto " + std::to_string(port) + " ya esta en uso";
		CloseSock(s);
		return false;
	}
	while (!stop->load()) {
		fd_set rs;
		FD_ZERO(&rs);
		FD_SET(s, &rs);
		timeval wait {0, 300000};
		if (select(static_cast<int>(s + 1), &rs, nullptr, nullptr, &wait) <= 0) continue;
		sockaddr_in from {};
		socklen_t   len = sizeof(from);
		sock_t      c   = accept(s, reinterpret_cast<sockaddr*>(&from), &len);
		if (c == kBadSock) continue;
		char ip[64] = "?";
		inet_ntop(AF_INET, &from.sin_addr, ip, sizeof(ip));
		std::thread(ServeClient, srv, c, std::string(ip)).detach();
	}
	CloseSock(s);
	return true;
}


// ---- (online5) UDP relay (port 3758): the P2P packets of the game between players, no port open at the players'.
//   up:   "DSR\x01" + token              (hello, every few seconds: this endpoint is that account)
//   up:   "DSR\x02" + u64 dst account + payload
//   down: "DSR\x02" + u64 src account + payload
inline bool Relay(Server* srv, std::atomic<bool>* stop, std::string* error, int port) {
#if defined(_WIN32)
	{ WSADATA wsa {}; WSAStartup(MAKEWORD(2, 2), &wsa); } // (online8) this thread may run before the TCP ones start Winsock
#endif
	sock_t s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (s == kBadSock) { *error = "socket udp"; return false; }
	sockaddr_in addr {};
	addr.sin_family      = AF_INET;
	addr.sin_port        = htons(static_cast<uint16_t>(port));
	addr.sin_addr.s_addr = htonl(INADDR_ANY);
	if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) { *error = "el puerto UDP " + std::to_string(port) + " ya esta en uso"; CloseSock(s); return false; }
	std::map<uint64_t, sockaddr_in> where;   // account -> endpoint
	std::map<uint64_t, uint64_t>    who;     // endpoint key -> account
	const auto key = [](const sockaddr_in& a) { return (static_cast<uint64_t>(a.sin_addr.s_addr) << 16) | a.sin_port; };
	std::vector<char> buf(65536 + 16);
	long long         forwarded = 0, dropped = 0;
	double            last_log  = Now();
	while (!stop->load()) {
		fd_set rs;
		FD_ZERO(&rs);
		FD_SET(s, &rs);
		timeval wait {0, 300000};
		if (select(static_cast<int>(s + 1), &rs, nullptr, nullptr, &wait) <= 0) continue;
		sockaddr_in from {};
		socklen_t   len = sizeof(from);
		const int   n   = static_cast<int>(recvfrom(s, buf.data(), static_cast<int>(buf.size()), 0, reinterpret_cast<sockaddr*>(&from), &len));
		if (n < 4 || std::memcmp(buf.data(), "DSR", 3) != 0) continue;
		if (buf[3] == 1) {
			const std::string token(buf.data() + 4, static_cast<size_t>(n - 4));
			std::lock_guard lock(srv->store.m);
			auto it = srv->psn.tokens.find(token);
			if (it == srv->psn.tokens.end()) continue;
			const uint64_t acc = std::strtoull(it->second.c_str(), nullptr, 10);
			where[acc]         = from;
			who[key(from)]     = acc;
			srv->psn.seen[it->second] = Now();
			char ack[4] = {'D', 'S', 'R', 1};
			sendto(s, ack, 4, 0, reinterpret_cast<sockaddr*>(&from), sizeof(from));
		} else if (buf[3] == 2 && n >= 12) {
			auto src = who.find(key(from));
			uint64_t dst = 0;
			std::memcpy(&dst, buf.data() + 4, 8);
			auto to = where.find(dst);
			if (src == who.end() || to == where.end()) { dropped++; continue; }
			std::memcpy(buf.data() + 4, &src->second, 8);
			sendto(s, buf.data(), n, 0, reinterpret_cast<sockaddr*>(&to->second), sizeof(to->second));
			forwarded++;
		}
		if (Now() - last_log > 60 && (forwarded || dropped)) {
			srv->Log("relay: " + std::to_string(forwarded) + " paquetes reenviados, " + std::to_string(dropped) + " descartados (destino sin conectar)");
			forwarded = dropped = 0;
			last_log = Now();
		}
	}
	CloseSock(s);
	return true;
}

// ---- share code: "DS-" + 10 base-32 characters of the IPv4 address and the port ------------------------------------
inline std::string MakeCode(const std::string& ip_port) {
	unsigned a, b, c, d, port;
	if (std::sscanf(ip_port.c_str(), "%u.%u.%u.%u:%u", &a, &b, &c, &d, &port) != 5 || a > 255 || b > 255 || c > 255 || d > 255 || port > 65535) return {};
	uint64_t           v      = (static_cast<uint64_t>(a) << 40) | (static_cast<uint64_t>(b) << 32) | (c << 24) | (d << 16) | port;
	const char*        digits = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; // no 0/O, 1/I
	std::string        out;
	for (int k = 0; k < 10; k++) { out.insert(out.begin(), digits[v & 31]); v >>= 5; }
	return "DS-" + out.substr(0, 5) + "-" + out.substr(5);
}
// A code, or "IP:port", or "host:port", or a bare IP (port 8090) -> "host:port" (empty: not valid).
inline std::string ReadCode(std::string text) {
	text.erase(std::remove_if(text.begin(), text.end(), [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }), text.end());
	std::string up = Upper(text);
	if (up.rfind("DS-", 0) == 0) {
		up.erase(std::remove(up.begin(), up.end(), '-'), up.end());
		up = up.substr(2);
		if (up.size() != 10) return {};
		const char* digits = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
		uint64_t    v      = 0;
		for (char ch: up) {
			const char* f = std::strchr(digits, ch);
			if (f == nullptr) return {};
			v = (v << 5) | static_cast<uint64_t>(f - digits);
		}
		char buf[32];
		std::snprintf(buf, sizeof(buf), "%u.%u.%u.%u:%u", static_cast<unsigned>((v >> 40) & 255), static_cast<unsigned>((v >> 32) & 255),
		              static_cast<unsigned>((v >> 24) & 255), static_cast<unsigned>((v >> 16) & 255), static_cast<unsigned>(v & 0xFFFF));
		return buf;
	}
	if (text.rfind("http://", 0) == 0) text = text.substr(7);
	while (!text.empty() && text.back() == '/') text.pop_back();
	if (text.empty() || text.find_first_of("+=;\"") != std::string::npos) return {};
	if (text.find(':') == std::string::npos) text += ":8090";
	return text;
}

} // namespace OnlineCore
