// Banco offline (tej-coste): trabajo de tej en el hilo de render en B, antes/despues. Numeros del log a280f07.
#include <x86intrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>
#include <cstdlib>
using R = std::pair<uint64_t, uint64_t>;
constexpr size_t STATE = 24 * 1024; // (contexto ~ m_ctx+m_sh_ctx+m_ucfg+saved: cota alta)
struct St { std::vector<uint32_t> w = std::vector<uint32_t>(STATE / 4); };
static bool Eq(const St& a, const St& b, size_t from, size_t n) { return std::memcmp(&a.w[from], &b.w[from], n * 4) == 0; }
std::vector<std::atomic<uint64_t>> regions(1 << 18);           // BufferCache 4 MiB
std::vector<uint64_t> pages(1 << 22);                           // TextureCache: MultiLevelPageTable (indice directo, como Find)
std::unordered_map<uint64_t, std::vector<uint64_t>> unmap;     // GpuResourceManager hojas
std::mutex mu; std::unordered_map<const char*, uint64_t> any_;
static const char* names[512];
bool RegSince(uint64_t a, uint64_t s, uint64_t ep) { for (auto r = a >> 22, l = (a + s - 1) >> 22; r <= l; ++r) if (regions[r & ((1<<18)-1)].load(std::memory_order_acquire) > ep) return true; return false; }
bool TexSince(uint64_t a, uint64_t s, uint64_t ep) { for (auto p = a >> 16, l = (a + s - 1) >> 16; p <= l; ++p) { if (pages[p & ((1<<22)-1)] > ep) return true; } return false; }
bool UnmSince(uint64_t ep, uint64_t a, uint64_t s) { for (auto g = a >> 14, l = (a + s - 1) >> 14; g <= l;) { auto st = std::min(l, g | 0xffff); auto it = unmap.find(g >> 16); if (it != unmap.end()) for (auto i = g & 0xffff; i <= (st & 0xffff); ++i) if (it->second[i] > ep) return true; g = st + 1; } return false; }
void Merge(std::vector<R>& v, uint64_t g) { size_t k=0; for (auto [a,e]: v){ if(e<=a)continue; if(g)a-=a%g,e=(e+g-1)/g*g; v[k++]={a,e};} v.resize(k); std::sort(v.begin(),v.end()); k=0; for(size_t i=0;i<v.size();++i){ if(k&&v[i].first<=v[k-1].second)v[k-1].second=std::max(v[k-1].second,v[i].second); else v[k++]=v[i];} v.resize(k);}
int main() {
	std::mt19937_64 rng(1);
	for (auto& r : regions) r = 1;
	for (int i = 0; i < 20000; ++i) pages[((0x200000000ull >> 16) + rng() % 200000) & ((1<<22)-1)] = 1;
	const int PK = 49, NR = 788, FR = 1000;
	St render, work[PK]; for (auto& w : work) w.w = render.w; for (int i = 0; i < PK; i += 3) work[i].w[rng() % (STATE / 4)] ^= 1; // 1/3 falla regs
	std::vector<std::vector<R>> bufs(PK), all(PK), rb(PK), rx(PK);
	for (int p = 0; p < PK; ++p) { // rangos repetidos (Note solo quita los 8 ultimos): ~120 distintos de 788
		std::vector<R> base; for (int i = 0; i < 120; ++i) { uint64_t a = 0x200000000ull + (rng() % (1ull << 32)); base.push_back({a, a + 4096 + rng() % 262144}); }
		for (int i = 0; i < NR; ++i) { auto r = base[rng() % base.size()]; (i < 300 ? bufs[p] : all[p]).push_back(r); }
		all[p].insert(all[p].end(), bufs[p].begin(), bufs[p].end());
		rb[p] = bufs[p]; Merge(rb[p], 1ull << 22); rx[p] = all[p]; Merge(rx[p], 0);
	}
	size_t RS = 0; for (int p = 0; p < PK; ++p) RS += rb[p].size() + rx[p].size();
	volatile uint64_t sink = 0;
	auto run = [&](bool after) {
		auto t0 = std::chrono::steady_clock::now(); uint32_t k = 0;
		for (int f = 0; f < FR; ++f) for (int p = 0; p < PK; ++p) {
			const St& w = work[p];
			bool same = Eq(render, w, 0, STATE / 4), gfx = Eq(render, w, 0, STATE / 5), cs = Eq(render, w, STATE / 5, STATE / 4 - STATE / 5);
			if (after) { same = gfx && cs; }
			bool ok = true;
			if (!after) { for (auto& [a,e] : bufs[p]) ok &= !RegSince(a,e-a,5); for (auto& [a,e] : all[p]) { ok &= !TexSince(a,e-a,5); ok &= !UnmSince(5,a,e-a);} }
			else if (gfx && getenv("NORANGE") == nullptr) { for (auto& [a,e] : rb[p]) if (RegSince(a,e-a,5)) {ok=false;break;} if (ok) for (auto& [a,e] : rx[p]) if (TexSince(a,e-a,5)||UnmSince(5,a,e-a)) {ok=false;break;} }
			if (!same && (!after || k++ % 64 == 0)) { // DiffNames: campo a campo (~600 campos) + mutex + mapa
				uint32_t n = 0; for (size_t i = 0; i < STATE / 4; i += 10) if (!Eq(render, w, i, 10) && n < 512) names[n++] = "x";
				std::lock_guard g(mu); for (uint32_t i = 0; i < n; ++i) ++any_[names[i]];
			}
			sink += ok + same;
		}
		return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / FR;
	};
	run(false);
	std::printf("rangos compactos medios/paquete %.1f (de %d)\nANTES  %.3f ms/fot\nDESPUES %.3f ms/fot\n", double(RS) / PK, NR, run(false), run(true));
}
