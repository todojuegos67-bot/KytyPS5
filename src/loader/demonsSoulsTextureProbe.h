#pragma once
// KYTY_TEXTURE_PROBE (on by default; =0 off): measurement only, nothing the game does is changed. Counts the mip chunks
// CTexture::SetCurrentStreamingChunk (eboot+0x901330, 01.007.000) copies into a texture's memory with memcpy (the call
// at eboot+0x901459, return address eboot+0x90145e), keeps the destination ranges as "live" until the game unmaps
// that memory (sceKernelMunmap / BatchMap unmap), and prints one line every 5 s (demonsSoulsTextureProbe.cpp).
// KYTY_TEXTURE_RELEASE (Windows; off by default; =1 on): those live ranges, ~120 vblanks after
// their copy, are taken out of the working set (VirtualUnlock: contents kept, read back from the page file on use).
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "local/buf-nativo.h" // (BUF-NATIVO)

namespace Loader {
struct Program;
namespace DemonsSoulsTextureProbe {
// From DemonsSoulsCopy::Install (each program; only eboot.bin is looked at).
void Install(Program* program);
// The absolute return address of the audited call; 0 while the probe is off (the only cost then: this load).
inline std::atomic<uint64_t> g_return_address {0};
// KYTY_BUFFER_PROBE (on by default; =0 off; measurement only): CBuffer::Create (eboot+0x909460) copies a new buffer's
// initial data into its fresh memory with memcpy (call at eboot+0x90954b, return eboot+0x909550): those destination
// ranges are kept as live game buffers until the game unmaps the memory. Buffers created without initial data and
// later CPU writes into buffers are not seen. 0 while off.
inline std::atomic<uint64_t> g_buffer_return_address {0};
// (BUF-NATIVO) eboot+0x909550 absoluto con KYTY_BUF_NATIVO y los ganchos de 0xaa06a0 / 0xaa1d10 puestos; 0 si no.
inline std::atomic<uint64_t> g_bufnativo_return_address {0};
void RecordBuffer(uint64_t destination, uint64_t size);
// (Fault report) Bytes of [begin, end) inside live CBuffer ranges; UINT64_MAX if the buffer probe is busy or off.
uint64_t BufferOverlap(uint64_t begin, uint64_t end);
void Record(uint64_t destination, uint64_t size);
void Unmapped(uint64_t vaddr, uint64_t size);
// (Fault report) Bytes of [begin, end) inside live texture ranges; UINT64_MAX if the probe is busy or off.
uint64_t LiveOverlap(uint64_t begin, uint64_t end);
// From the guest's memcpy (libc.prx's patched body: DemonsSoulsCopy's CoherentMove; Kyty's LibcInternal::memcpy).
inline void OnCopy(uint64_t return_address, const void* destination, size_t size) {
	const uint64_t wanted = g_return_address.load(std::memory_order_relaxed);
	if (wanted != 0 && return_address == wanted && size != 0) Record(reinterpret_cast<uint64_t>(destination), size);
	const uint64_t buffer = g_buffer_return_address.load(std::memory_order_relaxed);
	if (buffer != 0 && return_address == buffer && size != 0) RecordBuffer(reinterpret_cast<uint64_t>(destination), size);
	// (BUF-NATIVO) El memcpy de CBuffer::Create (vuelta eboot+0x909550) dentro de 0xaa06a0: datos de una malla.
	const uint64_t mesh = g_bufnativo_return_address.load(std::memory_order_relaxed);
	if (mesh != 0 && return_address == mesh && size != 0) ::BufNativo::OnMeshCopy(reinterpret_cast<uint64_t>(destination), size);
}
// From KernelMunmap, after a successful unmap.
inline void OnUnmap(uint64_t vaddr, uint64_t size) {
	if ((g_return_address.load(std::memory_order_relaxed) != 0 ||
	     g_buffer_return_address.load(std::memory_order_relaxed) != 0) && size != 0)
		Unmapped(vaddr, size);
}
} // namespace DemonsSoulsTextureProbe
} // namespace Loader
