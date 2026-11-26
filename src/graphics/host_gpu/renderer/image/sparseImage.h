#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SPARSEIMAGE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SPARSEIMAGE_H_

// (ram1b) KYTY_RAM_NATIVE_TEX=1|2 (local/ram-native-tex.h): a texture array of the game's VirtualMemoryGPU zone as a
// sparse (partially resident) image, video memory bound to the layers the game has committed only. See sparseImage.cpp.

#include "graphics/host_gpu/graphicContext.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
struct ImageInfo;

// Whether an image of `info`, created as `create`, is made sparse here: the switch on, the device able, a sampled 2D
// colour array of the zone (no depth/stencil, not multisampled, its guest layers of equal stride), made on the GPU thread
// (the bindings wait on the GPU thread's scheduler), and no ImageForceDense scope open.
[[nodiscard]] bool SparseImageWanted(const GraphicContext& graphics, const ImageInfo& info, const vk::ImageCreateInfo& create);

// The image made sparse, no memory bound yet (false: nothing made; the caller makes it dense). `image` filled as
// GraphicContext::CreateImage fills it, its `sparse` set.
[[nodiscard]] bool CreateSparseImage(GraphicContext& graphics, const vk::ImageCreateInfo& create, VulkanImage& image);

// The image and all its video memory freed (GraphicContext::DeleteImage: the GPU is done with it).
void DestroySparseImage(GraphicContext& graphics, VulkanImage& image);

// The layers of `image` whose guest slices [address + layer * stride, + stride) are committed now - minus
// [gone_begin, gone_end), about to be unmapped - bound, the others unbound, for the layers whose slices meet
// [begin, end). The binding is queued on the graphics queue; the next submission of `scheduler` waits for it.
void SparseUpdateLayers(GraphicContext& graphics, CommandScheduler& scheduler, VulkanImage& image, uint64_t address,
                        uint64_t stride, uint64_t begin, uint64_t end, uint64_t gone_begin, uint64_t gone_end);

// (N1, KYTY_TEX_NATIVA) A native texture's levels (guest range [address + offsets[i], + sizes[i])) bound while
// committed, unbound otherwise, for the levels meeting [begin, end); levels below min_level stay unbound.
void SparseUpdateLevels(GraphicContext& graphics, CommandScheduler& scheduler, VulkanImage& image, uint64_t address,
                        const uint64_t* offsets, const uint64_t* sizes, uint32_t levels, uint32_t min_level, uint64_t begin,
                        uint64_t end, uint64_t gone_begin, uint64_t gone_end);
// (P5) Los ligados que SparseUpdateLevels dejo pendientes para `scheduler`, en un solo vkQueueBindSparse; su siguiente
// entrega los espera. CommandScheduler::SubmitBuffers lo llama antes de cada entrega.
void SparseFlushPendingBinds(GraphicContext& graphics, CommandScheduler& scheduler);

// (NEGRAS) Si el mip `level` de una textura nativa tiene memoria ligada (true si no es nativa). Una subida a un mip sin
// ligar se pierde (sparse: se lee como cero = textura negra).
[[nodiscard]] bool SparseLevelBound(const VulkanImage& image, uint32_t level);
// (N1) Whether `image` is a native texture (memory per level).
[[nodiscard]] bool SparseIsPerLevel(const VulkanImage& image);

// (=2) The layers bound now (ascending).
void SparseBoundLayers(const VulkanImage& image, std::vector<uint32_t>* layers);

// (VRAM-PRES) KYTY_TEX_NATIVA_PRESUPUESTO (por defecto 1): el recolector cuenta y devuelve la VRAM de las texturas nativas.
struct SparseNativeStats {
	uint64_t bound = 0, retired = 0, pool = 0; // ligada a mips, desligada pendiente de la GPU, libre en el pool
};
[[nodiscard]] bool              SparsePresupuestoOn();
[[nodiscard]] SparseNativeStats SparseNativeStatsNow();
// (VRAM-PRES) Bytes de video ligados ahora a la imagen nativa (0 si no lo es).
[[nodiscard]] uint64_t SparseBoundBytes(const VulkanImage& image);
// (VRAM-PRES) Devuelve lo retirado cuya GPU acabo (todas las imagenes); con `pressure` libera el pool entero y el hilo
// deja de rellenarlo hasta la siguiente llamada sin presion. Devuelve los bytes liberados a Windows aqui. Hilo de la GPU.
uint64_t SparsePresupuestoReclaim(GraphicContext& graphics, bool pressure);

// (W14, SPARSE-COMPACTAR; KYTY_SPARSE_COMPACTAR=0: apagado) Traslada mips de los bloques de 16 MiB menos llenos a los
// mas llenos (copia GPU de bufer a bufer sobre la memoria de los bloques + nuevo ligado cuando la copia acabo), solo de
// las imagenes de `sin_uso` (nativas sin usar en este fotograma ni en los anteriores). Los bloques que quedan vacios los
// devuelve FreeEmptyBlocks. Hilo de la GPU, fuera de un render pass (graba en scheduler.Current()).
void SparseCompactar(GraphicContext& graphics, CommandScheduler& scheduler, const std::vector<VulkanImage*>& sin_uso);
// (W14) La imagen se va a usar: un traslado suyo en curso se descarta (sigue en su memoria de antes).
void SparseMoverCancelar(const VulkanImage& image);

// While one lives on this thread, SparseImageWanted says no (the verification's dense reference image).
struct ImageForceDense {
	ImageForceDense();
	~ImageForceDense();
	ImageForceDense(const ImageForceDense&)            = delete;
	ImageForceDense& operator=(const ImageForceDense&) = delete;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SPARSEIMAGE_H_ */
