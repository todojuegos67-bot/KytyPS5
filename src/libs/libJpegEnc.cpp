#include "common/abi.h"
#include "common/logging/log.h"
#include "kernel/memory.h"
#include "libs/libs.h"
#include "loader/symbolDatabase.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STBIWDEF static inline
#define STBI_WRITE_NO_STDIO
#include "stb_image_write.h"

// Round 124: libSceJpegEnc. Its four functions were unresolved imports, stubs that returned 0
// and wrote nothing: no handle, no JPEG, no size. The structures follow the PS4 toolchain's
// jpeg.h (48-byte encode parameters); the encoding itself is stb_image_write's baseline JPEG. As
// with PngEnc (and the decoders), a successful encode returns the size of the JPEG and also
// reports it in the output info. The first calls are logged with all their fields.
namespace Libs {

LIB_VERSION("JpegEnc", 1, "JpegEnc", 1, 1);

namespace JpegEnc {

constexpr int32_t JPEG_ENC_ERROR_INVALID_ADDR   = -2140864255; // 0x80650101
constexpr int32_t JPEG_ENC_ERROR_INVALID_SIZE   = -2140864254; // 0x80650102
constexpr int32_t JPEG_ENC_ERROR_INVALID_PARAM  = -2140864253; // 0x80650103
constexpr int32_t JPEG_ENC_ERROR_INVALID_HANDLE = -2140864252; // 0x80650104

constexpr uint32_t JPEG_ENC_MEMORY_SIZE   = 0x800;
constexpr uint32_t JPEG_ENC_MAX_DIMENSION = 0xffff;

constexpr uint16_t JPEG_ENC_PIXEL_FORMAT_R8G8B8A8 = 0;
constexpr uint16_t JPEG_ENC_PIXEL_FORMAT_B8G8R8A8 = 1;
constexpr uint16_t JPEG_ENC_PIXEL_FORMAT_Y8U8Y8V8 = 10;
constexpr uint16_t JPEG_ENC_PIXEL_FORMAT_Y8       = 11;
constexpr uint16_t JPEG_ENC_ENCODE_MODE_NORMAL    = 0;
constexpr uint16_t JPEG_ENC_ENCODE_MODE_MJPEG     = 1;
constexpr uint16_t JPEG_ENC_COLOR_SPACE_YCC       = 1;
constexpr uint16_t JPEG_ENC_COLOR_SPACE_GRAYSCALE = 2;
constexpr uint8_t  JPEG_ENC_SAMPLING_TYPE_420     = 2;

struct JpegEncCreateParam {
	uint32_t this_size;
	uint32_t attribute;
};

struct JpegEncEncodeParam {
	const uint8_t* image_mem_addr;
	uint8_t*       jpeg_mem_addr;
	uint32_t       image_mem_size;
	uint32_t       jpeg_mem_size;
	uint32_t       image_width;
	uint32_t       image_height;
	uint32_t       image_pitch;
	uint16_t       pixel_format;
	uint16_t       encode_mode;
	uint16_t       color_space;
	uint8_t        sampling_type;
	uint8_t        compression_ratio;
	int32_t        restart_interval;
};

struct JpegEncOutputInfo {
	uint32_t data_size;
	uint32_t processed_height;
};

static_assert(sizeof(JpegEncCreateParam) == 8);
static_assert(sizeof(JpegEncEncodeParam) == 48);
static_assert(offsetof(JpegEncEncodeParam, pixel_format) == 36);
static_assert(offsetof(JpegEncEncodeParam, compression_ratio) == 43);
static_assert(offsetof(JpegEncEncodeParam, restart_interval) == 44);
static_assert(sizeof(JpegEncOutputInfo) == 8);

// In the memory the game gives Create, at its first 32-byte boundary.
struct JpegEncContext {
	uint64_t magic;
	uint64_t self;
};

constexpr uint64_t JPEG_ENC_CONTEXT_MAGIC = 0x4b5954594a504745ull; // KYTYJPGE
constexpr uint32_t JPEG_ENC_LOGGED_CALLS  = 16;

static_assert(sizeof(JpegEncContext) + 31u <= JPEG_ENC_MEMORY_SIZE);

static bool LogCall(std::atomic<uint32_t>* counter) {
	return counter->fetch_add(1, std::memory_order_relaxed) < JPEG_ENC_LOGGED_CALLS;
}

static JpegEncContext* ContextOf(void* handle) {
	auto* ctx = static_cast<JpegEncContext*>(handle);
	if (ctx == nullptr || (reinterpret_cast<uintptr_t>(ctx) & 31u) != 0 ||
	    ctx->magic != JPEG_ENC_CONTEXT_MAGIC || ctx->self != reinterpret_cast<uintptr_t>(ctx)) {
		return nullptr;
	}
	return ctx;
}

static int32_t ValidateCreateParam(const JpegEncCreateParam* param) {
	if (param == nullptr) {
		return JPEG_ENC_ERROR_INVALID_ADDR;
	}
	if (param->this_size != sizeof(JpegEncCreateParam)) {
		return JPEG_ENC_ERROR_INVALID_SIZE;
	}
	if (param->attribute != 0) {
		return JPEG_ENC_ERROR_INVALID_PARAM;
	}
	return 0;
}

static int32_t KYTY_SYSV_ABI JpegEncQueryMemorySize(const JpegEncCreateParam* param) {
	PRINT_NAME();

	if (const int32_t result = ValidateCreateParam(param); result != 0) {
		LOGF("JpegEnc: query memory size refused: 0x%08" PRIx32 "\n",
		     static_cast<uint32_t>(result));
		return result;
	}
	return static_cast<int32_t>(JPEG_ENC_MEMORY_SIZE);
}

static int32_t KYTY_SYSV_ABI JpegEncCreate(const JpegEncCreateParam* param, void* memory_address,
                                           uint32_t memory_size, void** handle) {
	PRINT_NAME();

	int32_t result = ValidateCreateParam(param);
	if (result == 0 && (memory_address == nullptr || handle == nullptr)) {
		result = JPEG_ENC_ERROR_INVALID_ADDR;
	}
	const auto address = reinterpret_cast<uintptr_t>(memory_address);
	const auto aligned = (address + 31u) & ~uintptr_t {31u};
	if (result == 0 && aligned + sizeof(JpegEncContext) > address + memory_size) {
		result = JPEG_ENC_ERROR_INVALID_SIZE;
	}
	static std::atomic<uint32_t> created {0};
	if (LogCall(&created)) {
		LOGF("JpegEnc: create in %p, %" PRIu32 " bytes: 0x%08" PRIx32 "\n", memory_address,
		     memory_size, static_cast<uint32_t>(result));
	}
	if (result != 0) {
		return result;
	}
	auto* ctx  = reinterpret_cast<JpegEncContext*>(aligned);
	ctx->magic = JPEG_ENC_CONTEXT_MAGIC;
	ctx->self  = aligned;
	*handle    = ctx;
	return 0;
}

static int32_t KYTY_SYSV_ABI JpegEncDelete(void* handle) {
	PRINT_NAME();

	auto* ctx = ContextOf(handle);
	if (ctx == nullptr) {
		return JPEG_ENC_ERROR_INVALID_HANDLE;
	}
	ctx->magic = 0;
	ctx->self  = 0;
	return 0;
}

static int32_t ValidateEncodeParam(const JpegEncEncodeParam* param, uint32_t* bytes_per_pixel) {
	if (param == nullptr || param->image_mem_addr == nullptr || param->jpeg_mem_addr == nullptr) {
		return JPEG_ENC_ERROR_INVALID_ADDR;
	}
	switch (param->pixel_format) {
		case JPEG_ENC_PIXEL_FORMAT_R8G8B8A8:
		case JPEG_ENC_PIXEL_FORMAT_B8G8R8A8: *bytes_per_pixel = 4; break;
		case JPEG_ENC_PIXEL_FORMAT_Y8U8Y8V8: *bytes_per_pixel = 2; break;
		case JPEG_ENC_PIXEL_FORMAT_Y8: *bytes_per_pixel = 1; break;
		default: return JPEG_ENC_ERROR_INVALID_PARAM;
	}
	if (param->encode_mode != JPEG_ENC_ENCODE_MODE_NORMAL &&
	    param->encode_mode != JPEG_ENC_ENCODE_MODE_MJPEG) {
		return JPEG_ENC_ERROR_INVALID_PARAM;
	}
	if (param->color_space != JPEG_ENC_COLOR_SPACE_YCC &&
	    param->color_space != JPEG_ENC_COLOR_SPACE_GRAYSCALE) {
		return JPEG_ENC_ERROR_INVALID_PARAM;
	}
	// The restart interval is not checked: stb_image_write writes no restart markers.
	if (param->sampling_type > JPEG_ENC_SAMPLING_TYPE_420) {
		return JPEG_ENC_ERROR_INVALID_PARAM;
	}
	if (param->image_width == 0 || param->image_height == 0 ||
	    param->image_width > JPEG_ENC_MAX_DIMENSION ||
	    param->image_height > JPEG_ENC_MAX_DIMENSION) {
		return JPEG_ENC_ERROR_INVALID_PARAM;
	}
	// A Y8U8Y8V8 row holds pairs of pixels.
	const uint64_t row_pixels = param->pixel_format == JPEG_ENC_PIXEL_FORMAT_Y8U8Y8V8
	                                ? (uint64_t {param->image_width} + 1u) & ~uint64_t {1u}
	                                : uint64_t {param->image_width};
	if (param->image_mem_size == 0 || param->jpeg_mem_size == 0 ||
	    uint64_t {param->image_pitch} < row_pixels * *bytes_per_pixel ||
	    uint64_t {param->image_pitch} * param->image_height > param->image_mem_size) {
		return JPEG_ENC_ERROR_INVALID_SIZE;
	}
	return 0;
}

static uint8_t ClampByte(int32_t value) {
	return static_cast<uint8_t>(std::clamp(value, 0, 255));
}

// Whether the JPEG holds one gray byte a pixel (else RGB).
static bool GrayJpeg(const JpegEncEncodeParam& param) {
	return param.color_space == JPEG_ENC_COLOR_SPACE_GRAYSCALE ||
	       param.pixel_format == JPEG_ENC_PIXEL_FORMAT_Y8;
}

// The image as packed RGB (or one gray byte a pixel), what stb_image_write takes, into `pixels`
// (width * height * components bytes).
static void PackPixels(const JpegEncEncodeParam& param, uint8_t* pixels) {
	const uint32_t width      = param.image_width;
	const uint32_t height     = param.image_height;
	const bool     gray       = GrayJpeg(param);
	const size_t   components = gray ? 1u : 3u;
	for (uint32_t y = 0; y < height; y++) {
		const uint8_t* src = param.image_mem_addr + size_t {y} * param.image_pitch;
		uint8_t*       dst = pixels + size_t {y} * width * components;
		for (uint32_t x = 0; x < width; x++) {
			int32_t r    = 0;
			int32_t g    = 0;
			int32_t b    = 0;
			int32_t luma = -1;
			switch (param.pixel_format) {
				case JPEG_ENC_PIXEL_FORMAT_R8G8B8A8:
					r = src[x * 4u];
					g = src[x * 4u + 1u];
					b = src[x * 4u + 2u];
					break;
				case JPEG_ENC_PIXEL_FORMAT_B8G8R8A8:
					b = src[x * 4u];
					g = src[x * 4u + 1u];
					r = src[x * 4u + 2u];
					break;
				case JPEG_ENC_PIXEL_FORMAT_Y8U8Y8V8: {
					// Y0 U Y1 V for each pair of pixels, JPEG's full-range YCbCr.
					const uint8_t* pair       = src + (x / 2u) * 4u;
					const int32_t  luma_value = pair[(x & 1u) != 0 ? 2u : 0u];
					const int32_t  cb         = pair[1] - 128;
					const int32_t  cr         = pair[3] - 128;
					luma                      = luma_value;
					r                         = luma_value + ((91881 * cr) >> 16);
					g                         = luma_value - ((22554 * cb + 46802 * cr) >> 16);
					b                         = luma_value + ((116130 * cb) >> 16);
					break;
				}
				default: luma = src[x]; break;
			}
			if (gray) {
				if (luma < 0) {
					luma = (19595 * r + 38470 * g + 7471 * b + 32768) >> 16;
				}
				dst[x] = ClampByte(luma);
			} else {
				dst[x * 3u]      = ClampByte(r);
				dst[x * 3u + 1u] = ClampByte(g);
				dst[x * 3u + 2u] = ClampByte(b);
			}
		}
	}
}

// The compression ratio, 1 (least compressed) to 255, as stb's quality, 100 to 1; 0: the
// default, 90.
static int Quality(uint8_t compression_ratio) {
	if (compression_ratio == 0) {
		return 90;
	}
	return std::clamp(100 - (static_cast<int>(compression_ratio) - 1) * 99 / 254, 1, 100);
}

// stb_image_write's output, into a buffer the size of the game's: what does not fit is only
// counted (the JPEG is made again with more compression). Nothing is allocated as it is written.
struct JpegOutput {
	uint8_t* bytes    = nullptr;
	uint64_t capacity = 0;
	uint64_t size     = 0;
};

static void AppendBytes(void* context, void* data, int size) {
	auto* out = static_cast<JpegOutput*>(context);
	if (size <= 0) {
		return;
	}
	const auto count = static_cast<uint64_t>(size);
	if (out->size <= out->capacity && count <= out->capacity - out->size) {
		std::memcpy(out->bytes + out->size, data, count);
	}
	out->size += count;
}

// stb_image_write keeps its options in globals.
static std::mutex g_jpeg_enc_mutex;

static int32_t KYTY_SYSV_ABI JpegEncEncode(void* handle, const JpegEncEncodeParam* param,
                                           JpegEncOutputInfo* output_info) {
	PRINT_NAME();

	int32_t  result          = ContextOf(handle) == nullptr ? JPEG_ENC_ERROR_INVALID_HANDLE : 0;
	uint32_t bytes_per_pixel = 0;
	if (result == 0) {
		result = ValidateEncodeParam(param, &bytes_per_pixel);
	}
	int                        quality    = 0;
	int                        components = 0;
	JpegOutput                 output;
	std::unique_ptr<uint8_t[]> pixels;
	std::unique_ptr<uint8_t[]> jpeg;
	if (result == 0) {
		// Without exceptions (the emulator is built without them): an image or buffer too large
		// for the host's memory is an error code.
		components = GrayJpeg(*param) ? 1 : 3;
		pixels.reset(new (std::nothrow) uint8_t[size_t {param->image_width} * param->image_height *
		                                        static_cast<size_t>(components)]);
		jpeg.reset(new (std::nothrow) uint8_t[param->jpeg_mem_size]);
		if (pixels == nullptr || jpeg == nullptr) {
			result = JPEG_ENC_ERROR_INVALID_SIZE;
		}
	}
	if (result == 0) {
		PackPixels(*param, pixels.get());
		output.bytes    = jpeg.get();
		output.capacity = param->jpeg_mem_size;
		// A JPEG larger than the game's buffer is made again with more compression.
		for (quality = Quality(param->compression_ratio);; quality = std::max(quality - 15, 1)) {
			output.size = 0;
			int ok      = 0;
			{
				std::scoped_lock lock(g_jpeg_enc_mutex);
				ok = stbi_write_jpg_to_func(
				    AppendBytes, &output, static_cast<int>(param->image_width),
				    static_cast<int>(param->image_height), components, pixels.get(), quality);
			}
			if (ok == 0) {
				result = JPEG_ENC_ERROR_INVALID_PARAM;
				break;
			}
			if (output.size <= output.capacity) {
				break;
			}
			if (quality == 1) {
				result = JPEG_ENC_ERROR_INVALID_SIZE;
				break;
			}
		}
	}
	if (result == 0) {
		std::memcpy(param->jpeg_mem_addr, jpeg.get(), output.size);
		result = static_cast<int32_t>(output.size);
		if (output_info != nullptr) {
			output_info->data_size        = static_cast<uint32_t>(output.size);
			output_info->processed_height = param->image_height;
		}
	}

	static std::atomic<uint32_t> encoded {0};
	static std::atomic<uint32_t> failed {0};
	if (LogCall(result < 0 ? &failed : &encoded)) {
		if (param != nullptr) {
			LOGF("JpegEnc: encode %p, %" PRIu32 " bytes -> %p, %" PRIu32 " bytes: %" PRIu32
			     "x%" PRIu32 " pitch %" PRIu32 ", pixel format %u, encode mode %u, colour space "
			     "%u, sampling %u, compression ratio %u (quality %d), restart interval %" PRId32
			     ": result %" PRId32 " (0x%08" PRIx32 ")\n",
			     static_cast<const void*>(param->image_mem_addr), param->image_mem_size,
			     static_cast<void*>(param->jpeg_mem_addr), param->jpeg_mem_size, param->image_width,
			     param->image_height, param->image_pitch,
			     static_cast<unsigned>(param->pixel_format),
			     static_cast<unsigned>(param->encode_mode),
			     static_cast<unsigned>(param->color_space),
			     static_cast<unsigned>(param->sampling_type),
			     static_cast<unsigned>(param->compression_ratio), quality, param->restart_interval,
			     result, static_cast<uint32_t>(result));
		} else {
			LOGF("JpegEnc: encode without parameters: 0x%08" PRIx32 "\n",
			     static_cast<uint32_t>(result));
		}
	}
	return result;
}

} // namespace JpegEnc

LIB_DEFINE(InitJpegEnc_1) {
	PRINT_NAME_ENABLE(true);

	LIB_FUNC("o6ZgXfFdWXQ", JpegEnc::JpegEncQueryMemorySize);
	LIB_FUNC("K+rocojkr-I", JpegEnc::JpegEncCreate);
	LIB_FUNC("j1LyMdaM+C0", JpegEnc::JpegEncDelete);
	LIB_FUNC("QbrU0cUghEM", JpegEnc::JpegEncEncode);
}

} // namespace Libs
