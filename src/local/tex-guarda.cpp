#include "local/tex-guarda.h"

#include "common/hostException.h"

#include <cstdio>
#include <mutex>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#endif

namespace TexGuarda {

namespace {

constexpr uint64_t kPage  = 0x1000;
constexpr uint64_t kPages = (kEnd - kBegin) >> 12u; // 4 Mi paginas
constexpr uint64_t kWords = kPages / 64u;

std::atomic<uint64_t> g_guarded[kWords] {}; // pagina vigilada
std::atomic<uint64_t> g_down[kWords] {};    // y su escritura la quitamos nosotros (hay que devolverla)
std::mutex            g_mutex;              // cambios de bits + proteccion (hoja: no se toma otro candado dentro)
std::atomic<OnWrite>  g_on_write {nullptr};
std::once_flag        g_install;

inline uint64_t Idx(uint64_t a) { return (a - kBegin) >> 12u; }
inline bool     Get(const std::atomic<uint64_t>* bits, uint64_t i) {
    return ((bits[i >> 6u].load(std::memory_order_acquire) >> (i & 63u)) & 1u) != 0;
}
inline void Set(std::atomic<uint64_t>* bits, uint64_t i, bool v) {
	const uint64_t m = uint64_t {1} << (i & 63u);
	if (v) bits[i >> 6u].fetch_or(m, std::memory_order_acq_rel);
	else bits[i >> 6u].fetch_and(~m, std::memory_order_acq_rel);
}
// Recorta [b, e) a paginas de la ventana; false si queda vacio.
inline bool Clamp(uint64_t& b, uint64_t& e) {
	b = b & ~(kPage - 1);
	e = (e + kPage - 1) & ~(kPage - 1);
	if (b < kBegin) b = kBegin;
	if (e > kEnd) e = kEnd;
	return b < e;
}
bool AnyGuarded(uint64_t b, uint64_t e) { // [b, e) ya recortado
	uint64_t i = Idx(b);
	const uint64_t last = Idx(e);
	while (i < last) {
		const uint64_t w = g_guarded[i >> 6u].load(std::memory_order_acquire);
		if (w == 0) {
			i = (i | 63u) + 1;
			continue;
		}
		if (((w >> (i & 63u)) & 1u) != 0) return true;
		++i;
	}
	return false;
}

#if defined(_WIN32)
bool Protect(uint64_t b, uint64_t e, DWORD prot) {
	DWORD old = 0;
	return VirtualProtect(reinterpret_cast<void*>(b), e - b, prot, &old) != 0;
}

// Fault de escritura de la CPU sobre una pagina vigilada. true: la instruccion se repite.
bool OnWriteFault(uint64_t address) {
	if (g_pages.load(std::memory_order_acquire) <= 0 || address < kBegin || address >= kEnd) return false;
	const uint64_t pb = address & ~(kPage - 1), pe = pb + kPage, i = Idx(pb);
	if (!Get(g_guarded, i)) {
		// Otro hilo la acaba de abrir (mismo fault a la vez): si ya tiene escritura, repetir; si no, no es nuestra.
		thread_local uint64_t last = 0;
		thread_local uint32_t again = 0;
		again = (last == pb) ? again + 1 : 0;
		last  = pb;
		MEMORY_BASIC_INFORMATION mbi {};
		if (again < 8 && VirtualQuery(reinterpret_cast<void*>(pb), &mbi, sizeof(mbi)) == sizeof(mbi) &&
		    mbi.State == MEM_COMMIT && (mbi.Protect & PAGE_GUARD) == 0 &&
		    ((mbi.Protect & 0xffu) == PAGE_READWRITE || (mbi.Protect & 0xffu) == PAGE_EXECUTE_READWRITE))
			return true;
		return false;
	}
	const bool ours = Get(g_down, i);
	if (t_emulator > 0) {
		// El emulador escribe bytes de la VRAM bajo OwnedMutex (y retira la propiedad justo despues): solo abrir.
		c_emulator.fetch_add(1, std::memory_order_relaxed);
		Unguard(pb, pe);
		return ours;
	}
	// El juego (o una copia del emulador en su nombre) escribe: la RAM vuelve a ser suya ANTES de la escritura.
	c_writes.fetch_add(1, std::memory_order_relaxed);
	if (auto f = g_on_write.load(std::memory_order_acquire); f != nullptr) f(pb, pe); // RemoveOwned -> Unguard
	if (Get(g_guarded, i)) Unguard(pb, pe); // (sin propiedad en el registro: igual se abre)
	// Escritura quitada por nosotros: devuelta, repetir. Si la proteccion era del gestor de paginas, su manejador sigue.
	return ours;
}

void Install() {
	std::call_once(g_install, [] {
		Common::HostException::SetWriteGuardHandler(&OnWriteFault);
		std::printf("[tex-nativa] paginas solo en la GPU protegidas contra escritura de la CPU (TEX-GUARDA): una escritura "
		            "del juego retira su propiedad antes de escribir\n");
		std::fflush(stdout);
	});
}
#endif

} // namespace

void SetOnWrite(OnWrite f) { g_on_write.store(f, std::memory_order_release); }

void Guard(uint64_t b, uint64_t e) {
#if defined(_WIN32)
	if (!Clamp(b, e)) return;
	Install();
	std::lock_guard<std::mutex> lock(g_mutex);
	uint64_t p = b;
	while (p < e) {
		if (Get(g_guarded, Idx(p))) {
			p += kPage;
			continue;
		}
		uint64_t q = p + kPage; // [p, q): paginas seguidas aun sin vigilar
		while (q < e && !Get(g_guarded, Idx(q))) q += kPage;
		for (uint64_t at = p; at < q;) {
			MEMORY_BASIC_INFORMATION mbi {};
			if (VirtualQuery(reinterpret_cast<void*>(at), &mbi, sizeof(mbi)) != sizeof(mbi)) {
				c_no_commit.fetch_add((q - at) >> 12u, std::memory_order_relaxed);
				break;
			}
			uint64_t       r_end = reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize;
			if (r_end > q) r_end = q;
			if (r_end <= at) r_end = at + kPage;
			const DWORD prot  = mbi.Protect & 0xffu;
			const DWORD extra = mbi.Protect & ~DWORD {0xff};
			DWORD       down  = 0;
			bool        ok    = mbi.State == MEM_COMMIT && (mbi.Protect & PAGE_GUARD) == 0;
			if (ok) {
				if (prot == PAGE_READWRITE) down = PAGE_READONLY;
				else if (prot == PAGE_EXECUTE_READWRITE) down = PAGE_EXECUTE_READ;
				else ok = prot == PAGE_READONLY || prot == PAGE_NOACCESS || prot == PAGE_EXECUTE_READ || prot == PAGE_EXECUTE;
			}
			if (ok && down != 0) ok = Protect(at, r_end, down | extra);
			if (ok) {
				for (uint64_t x = at; x < r_end; x += kPage) {
					Set(g_guarded, Idx(x), true);
					Set(g_down, Idx(x), down != 0);
				}
				g_pages.fetch_add(static_cast<int64_t>((r_end - at) >> 12u), std::memory_order_acq_rel);
			} else {
				c_no_commit.fetch_add((r_end - at) >> 12u, std::memory_order_relaxed);
			}
			at = r_end;
		}
		p = q;
	}
#else
	(void)b, (void)e;
#endif
}

void Unguard(uint64_t pb, uint64_t pe) {
#if defined(_WIN32)
	if (g_pages.load(std::memory_order_acquire) <= 0 || !Clamp(pb, pe) || !AnyGuarded(pb, pe)) return;
	std::lock_guard<std::mutex> lock(g_mutex);
	for (uint64_t p = pb; p < pe;) {
		const uint64_t i = Idx(p);
		if (!Get(g_guarded, i)) {
			p += kPage;
			continue;
		}
		const bool down = Get(g_down, i);
		uint64_t   q    = p + kPage; // [p, q): vigiladas seguidas con el mismo "down"
		while (q < pe && Get(g_guarded, Idx(q)) && Get(g_down, Idx(q)) == down) q += kPage;
		for (uint64_t x = p; x < q; x += kPage) {
			Set(g_guarded, Idx(x), false);
			Set(g_down, Idx(x), false);
		}
		g_pages.fetch_sub(static_cast<int64_t>((q - p) >> 12u), std::memory_order_acq_rel);
		if (down) {
			// Solo lo que sigue como lo dejamos (solo-lectura); lo que el emulador cambio despues es suyo.
			for (uint64_t at = p; at < q;) {
				MEMORY_BASIC_INFORMATION mbi {};
				if (VirtualQuery(reinterpret_cast<void*>(at), &mbi, sizeof(mbi)) != sizeof(mbi)) break;
				uint64_t r_end = reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize;
				if (r_end > q) r_end = q;
				if (r_end <= at) r_end = at + kPage;
				const DWORD prot = mbi.Protect & 0xffu, extra = mbi.Protect & ~DWORD {0xff};
				if (mbi.State == MEM_COMMIT) {
					if (prot == PAGE_READONLY) (void)Protect(at, r_end, PAGE_READWRITE | extra);
					else if (prot == PAGE_EXECUTE_READ) (void)Protect(at, r_end, PAGE_EXECUTE_READWRITE | extra);
				}
				at = r_end;
			}
		}
		p = q;
	}
#else
	(void)pb, (void)pe;
#endif
}

bool Intact(uint64_t b, uint64_t e) {
#if defined(_WIN32)
	if (!Clamp(b, e)) return false;
	for (uint64_t p = b; p < e; p += kPage)
		if (!Get(g_guarded, Idx(p))) return false;
	// Y de verdad sin escritura ahora (ninguna via que no pase por aqui - un commit nuevo, otra proteccion - la abrio).
	for (uint64_t at = b; at < e;) {
		MEMORY_BASIC_INFORMATION mbi {};
		if (VirtualQuery(reinterpret_cast<void*>(at), &mbi, sizeof(mbi)) != sizeof(mbi) || mbi.State != MEM_COMMIT) return false;
		const DWORD prot = mbi.Protect & 0xffu;
		if (prot != PAGE_READONLY && prot != PAGE_NOACCESS && prot != PAGE_EXECUTE_READ && prot != PAGE_EXECUTE) return false;
		const uint64_t r_end = reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize;
		at = r_end > at ? r_end : at + kPage;
	}
	return true;
#else
	(void)b, (void)e;
	return true; // (sin VirtualProtect propio: como antes)
#endif
}

void AfterProtect(uint64_t vaddr, uint64_t size, bool write, bool exec) {
#if defined(_WIN32)
	if (g_pages.load(std::memory_order_acquire) <= 0) return;
	uint64_t b = vaddr, e = vaddr + size;
	if (!Clamp(b, e) || !AnyGuarded(b, e)) return;
	std::lock_guard<std::mutex> lock(g_mutex);
	for (uint64_t p = b; p < e;) {
		if (!Get(g_guarded, Idx(p))) {
			p += kPage;
			continue;
		}
		uint64_t q = p + kPage;
		while (q < e && Get(g_guarded, Idx(q))) q += kPage;
		bool down = false;
		if (write) {
			down = Protect(p, q, exec ? PAGE_EXECUTE_READ : PAGE_READONLY);
			if (down) c_reprotect.fetch_add(1, std::memory_order_relaxed);
		}
		if (write && !down) {
			// Sin poder quitarle la escritura: ya no esta vigilada (Intact = false: ninguna descarga la escribe).
			for (uint64_t x = p; x < q; x += kPage) Set(g_guarded, Idx(x), false);
			g_pages.fetch_sub(static_cast<int64_t>((q - p) >> 12u), std::memory_order_acq_rel);
			c_no_commit.fetch_add((q - p) >> 12u, std::memory_order_relaxed);
		}
		for (uint64_t x = p; x < q; x += kPage) Set(g_down, Idx(x), down);
		p = q;
	}
#else
	(void)vaddr, (void)size, (void)write, (void)exec;
#endif
}

} // namespace TexGuarda
