#include "common/abi.h"
#include "common/logging/log.h"
#include "libs/libs.h"
#include "loader/symbolDatabase.h"
#include "stb_image.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>

// Round 114: libSceJpegDec, which a game's car dealer uses for its photos. Unimplemented, its calls
// returned 0 and wrote nothing, so every picture was 0x0 and none showed. The structures follow
// the PS4 toolchain's jpeg.h; the decode itself is stb_image's. The first calls are logged with
// all their fields, so the layouts can be checked in a log.
namespace Libs {

LIB_VERSION("JpegDec", 1, "JpegDec", 1, 1);

namespace JpegDec {

// The PngDec codes: no reference lists JpegDec's own.
constexpr int32_t JPEG_DEC_ERROR_INVALID_ADDR        = -2140602367; // 0x80690001
constexpr int32_t JPEG_DEC_ERROR_INVALID_SIZE        = -2140602366; // 0x80690002
constexpr int32_t JPEG_DEC_ERROR_INVALID_PARAM       = -2140602365; // 0x80690003
constexpr int32_t JPEG_DEC_ERROR_INVALID_HANDLE      = -2140602364; // 0x80690004
constexpr int32_t JPEG_DEC_ERROR_INVALID_WORK_MEMORY = -2140602363; // 0x80690005
constexpr int32_t JPEG_DEC_ERROR_INVALID_DATA        = -2140602352; // 0x80690010
constexpr int32_t JPEG_DEC_ERROR_DECODE_ERROR        = -2140602350; // 0x80690012

constexpr uint16_t JPEG_DEC_COLOR_SPACE_GRAYSCALE = 2;
constexpr uint16_t JPEG_DEC_COLOR_SPACE_RGB       = 3;
constexpr uint16_t JPEG_DEC_PIXEL_FORMAT_R8G8B8A8 = 0;
constexpr uint16_t JPEG_DEC_PIXEL_FORMAT_B8G8R8A8 = 1;
constexpr uint32_t JPEG_DEC_CSC_ORDINARY          = 1;

struct JpegDecCreateParam {
	uint32_t this_size;
	uint32_t attribute;
	uint32_t max_image_width;
};

struct JpegDecParseParam {
	const void* jpeg_mem_addr;
	uint32_t    jpeg_mem_size;
	uint16_t    decode_mode;
	uint16_t    down_scale;
};

struct JpegDecDecodeParam {
	const void* jpeg_mem_addr;
	void*       image_mem_addr;
	void*       coefficient_mem_addr;
	uint32_t    jpeg_mem_size;
	uint32_t    image_mem_size;
	uint32_t    coefficient_mem_size;
	uint16_t    decode_mode;
	uint16_t    down_scale;
	uint16_t    pixel_format;
	uint16_t    alpha_value;
	uint32_t    image_pitch;
};

struct JpegDecImageInfo {
	uint32_t image_width;
	uint32_t image_height;
	uint16_t color_space;
	uint16_t component_number;
	uint8_t  component_id[4];
	uint8_t  sampling_factor[4];
	uint32_t coefficient_mem_size;
	uint32_t suitable_csc_attribute;
	uint32_t output_image_width;
	uint32_t output_image_height;
};

static_assert(sizeof(JpegDecCreateParam) == 0xc);
static_assert(sizeof(JpegDecParseParam) == 0x10);
static_assert(sizeof(JpegDecDecodeParam) == 0x30);
static_assert(sizeof(JpegDecImageInfo) == 0x24);

struct JpegDecContext {
	uint64_t magic;
	uint32_t attribute;
	uint32_t max_image_width;
};

constexpr uint64_t JPEG_DEC_CONTEXT_MAGIC = 0x4b59545944474a50ull; // PJGDYTYK
constexpr uint32_t JPEG_DEC_MEMORY_SIZE   = 0x40;
constexpr uint32_t JPEG_DEC_LOGGED_CALLS  = 16;

struct JpegHeader {
	uint32_t width       = 0;
	uint32_t height      = 0;
	uint32_t components  = 0;
	bool     progressive = false;
	uint8_t  component_id[4] {};
	uint8_t  sampling_factor[4] {};
};

// The frame header (SOF0-SOF3, SOF5-SOF7, SOF9-SOF11, SOF13-SOF15) of a JPEG.
static bool ParseJpegHeader(const void* data, uint32_t size, JpegHeader* header) {
	if (data == nullptr || header == nullptr || size < 4) {
		return false;
	}
	const auto* bytes = static_cast<const uint8_t*>(data);
	if (bytes[0] != 0xff || bytes[1] != 0xd8) {
		return false;
	}
	uint32_t offset = 2;
	while (offset + 4 <= size) {
		if (bytes[offset] != 0xff) {
			return false;
		}
		const uint8_t marker = bytes[offset + 1];
		if (marker == 0xff) {
			offset++; // fill byte
			continue;
		}
		if (marker == 0xd8 || marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
			offset += 2; // no length
			continue;
		}
		if (marker == 0xd9 || marker == 0xda) {
			return false; // end of image or start of scan before a frame header
		}
		const uint32_t length =
		    (static_cast<uint32_t>(bytes[offset + 2]) << 8u) | bytes[offset + 3];
		if (length < 2 || offset + 2 + length > size) {
			return false;
		}
		const bool is_frame =
		    marker >= 0xc0 && marker <= 0xcf && marker != 0xc4 && marker != 0xc8 && marker != 0xcc;
		if (is_frame) {
			const auto* frame = bytes + offset + 4;
			if (length < 8) {
				return false;
			}
			header->height     = (static_cast<uint32_t>(frame[1]) << 8u) | frame[2];
			header->width      = (static_cast<uint32_t>(frame[3]) << 8u) | frame[4];
			header->components = frame[5];
			header->progressive =
			    marker == 0xc2 || marker == 0xc6 || marker == 0xca || marker == 0xce;
			if (header->width == 0 || header->height == 0 || header->components == 0 ||
			    length < 8 + 3 * header->components) {
				return false;
			}
			for (uint32_t i = 0; i < std::min<uint32_t>(header->components, 4); i++) {
				header->component_id[i]    = frame[6 + i * 3];
				header->sampling_factor[i] = frame[7 + i * 3];
			}
			return true;
		}
		offset += 2 + length;
	}
	return false;
}

static uint32_t ScaledSize(uint32_t size, uint32_t scale) {
	return (size + scale - 1) / scale;
}

static bool ValidScale(uint16_t scale) {
	return scale == 1 || scale == 2 || scale == 4 || scale == 8;
}

static void FillImageInfo(const JpegHeader& header, uint32_t scale, JpegDecImageInfo* info) {
	if (info == nullptr) {
		return;
	}
	*info              = JpegDecImageInfo {};
	info->image_width  = header.width;
	info->image_height = header.height;
	info->color_space =
	    header.components == 1 ? JPEG_DEC_COLOR_SPACE_GRAYSCALE : JPEG_DEC_COLOR_SPACE_RGB;
	info->component_number = static_cast<uint16_t>(header.components);
	std::memcpy(info->component_id, header.component_id, sizeof(info->component_id));
	std::memcpy(info->sampling_factor, header.sampling_factor, sizeof(info->sampling_factor));
	// The host decoder needs no coefficient buffer.
	info->coefficient_mem_size   = 0;
	info->suitable_csc_attribute = JPEG_DEC_CSC_ORDINARY;
	info->output_image_width     = ScaledSize(header.width, scale);
	info->output_image_height    = ScaledSize(header.height, scale);
}

static JpegDecContext* ToContext(void* handle) {
	if (handle == nullptr || (reinterpret_cast<uintptr_t>(handle) & 7u) != 0) {
		return nullptr;
	}
	auto* ctx = static_cast<JpegDecContext*>(handle);
	return ctx->magic == JPEG_DEC_CONTEXT_MAGIC ? ctx : nullptr;
}

static bool LogCall(std::atomic<uint32_t>* counter) {
	return counter->fetch_add(1, std::memory_order_relaxed) < JPEG_DEC_LOGGED_CALLS;
}

static int32_t KYTY_SYSV_ABI JpegDecQueryMemorySize(const JpegDecCreateParam* param) {
	PRINT_NAME();

	if (param == nullptr) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}

	LOGF("\t this_size       = %" PRIu32 "\n", param->this_size);
	LOGF("\t attribute       = %" PRIu32 "\n", param->attribute);
	LOGF("\t max_image_width = %" PRIu32 "\n", param->max_image_width);

	return static_cast<int32_t>(JPEG_DEC_MEMORY_SIZE);
}

static int32_t KYTY_SYSV_ABI JpegDecCreate(const JpegDecCreateParam* param, void* memory_address,
                                           uint32_t memory_size, void** handle) {
	PRINT_NAME();

	if (param == nullptr) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}

	LOGF("\t this_size       = %" PRIu32 "\n", param->this_size);
	LOGF("\t attribute       = %" PRIu32 "\n", param->attribute);
	LOGF("\t max_image_width = %" PRIu32 "\n", param->max_image_width);
	LOGF("\t memory_address  = %p\n", memory_address);
	LOGF("\t memory_size     = %" PRIu32 "\n", memory_size);

	if (memory_address == nullptr || handle == nullptr) {
		return JPEG_DEC_ERROR_INVALID_ADDR;
	}

	const auto address = reinterpret_cast<uintptr_t>(memory_address);
	const auto aligned = (address + 7u) & ~uintptr_t {7};
	if (aligned - address + sizeof(JpegDecContext) > memory_size) {
		return JPEG_DEC_ERROR_INVALID_WORK_MEMORY;
	}

	auto* ctx            = reinterpret_cast<JpegDecContext*>(aligned);
	ctx->magic           = JPEG_DEC_CONTEXT_MAGIC;
	ctx->attribute       = param->attribute;
	ctx->max_image_width = param->max_image_width;
	*handle              = ctx;

	return 0;
}

static int32_t KYTY_SYSV_ABI JpegDecDelete(void* handle) {
	PRINT_NAME();

	auto* ctx = ToContext(handle);
	if (ctx == nullptr) {
		return JPEG_DEC_ERROR_INVALID_HANDLE;
	}
	ctx->magic = 0;
	return 0;
}

static int32_t KYTY_SYSV_ABI JpegDecParseHeader(const JpegDecParseParam* param,
                                                JpegDecImageInfo*        image_info) {
	if (param == nullptr) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}
	if (param->jpeg_mem_addr == nullptr || image_info == nullptr) {
		return JPEG_DEC_ERROR_INVALID_ADDR;
	}

	JpegHeader    header {};
	const bool    parsed = ParseJpegHeader(param->jpeg_mem_addr, param->jpeg_mem_size, &header);
	const auto    scale  = ValidScale(param->down_scale) ? param->down_scale : uint16_t {1};
	const int32_t result = !parsed ? JPEG_DEC_ERROR_INVALID_DATA : 0;
	if (parsed) {
		FillImageInfo(header, scale, image_info);
	}

	static std::atomic<uint32_t> calls {0};
	if (LogCall(&calls) || result != 0) {
		LOGF("JpegDec: parse header %p, %" PRIu32 " bytes, decode mode %u, down scale %u: %" PRIu32
		     "x%" PRIu32 ", %" PRIu32 " components%s, result 0x%08" PRIx32 "\n",
		     param->jpeg_mem_addr, param->jpeg_mem_size, static_cast<unsigned>(param->decode_mode),
		     static_cast<unsigned>(param->down_scale), header.width, header.height,
		     header.components, header.progressive ? " (progressive)" : "",
		     static_cast<uint32_t>(result));
	}
	return result;
}

static int32_t JpegDecDecodeImage(void* handle, const JpegDecDecodeParam* param,
                                  JpegDecImageInfo* image_info, JpegHeader* header) {
	if (ToContext(handle) == nullptr) {
		return JPEG_DEC_ERROR_INVALID_HANDLE;
	}
	if (param == nullptr) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}
	if (param->jpeg_mem_addr == nullptr || param->image_mem_addr == nullptr) {
		return JPEG_DEC_ERROR_INVALID_ADDR;
	}
	if (param->pixel_format != JPEG_DEC_PIXEL_FORMAT_R8G8B8A8 &&
	    param->pixel_format != JPEG_DEC_PIXEL_FORMAT_B8G8R8A8) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}
	if (!ParseJpegHeader(param->jpeg_mem_addr, param->jpeg_mem_size, header)) {
		return JPEG_DEC_ERROR_INVALID_DATA;
	}

	const uint32_t scale      = ValidScale(param->down_scale) ? param->down_scale : 1u;
	const uint32_t out_width  = ScaledSize(header->width, scale);
	const uint32_t out_height = ScaledSize(header->height, scale);
	const uint64_t min_pitch  = static_cast<uint64_t>(out_width) * 4u;
	const uint64_t pitch      = param->image_pitch == 0 ? min_pitch : param->image_pitch;
	if (pitch < min_pitch) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}
	if (pitch * (out_height - 1u) + min_pitch > param->image_mem_size) {
		return JPEG_DEC_ERROR_INVALID_SIZE;
	}

	int   width      = 0;
	int   height     = 0;
	int   components = 0;
	auto* rgba       = stbi_load_from_memory(static_cast<const stbi_uc*>(param->jpeg_mem_addr),
	                                         static_cast<int>(param->jpeg_mem_size), &width, &height,
	                                         &components, 4);
	if (rgba == nullptr || static_cast<uint32_t>(width) != header->width ||
	    static_cast<uint32_t>(height) != header->height) {
		if (rgba != nullptr) {
			stbi_image_free(rgba);
		}
		return JPEG_DEC_ERROR_DECODE_ERROR;
	}

	// A JPEG has no alpha: every pixel gets the alpha value asked for.
	const auto alpha   = static_cast<uint8_t>(std::min<uint16_t>(param->alpha_value, 255u));
	const bool swap_rb = param->pixel_format == JPEG_DEC_PIXEL_FORMAT_B8G8R8A8;
	auto*      dst_row = static_cast<uint8_t*>(param->image_mem_addr);
	for (uint32_t y = 0; y < out_height; y++, dst_row += pitch) {
		auto* dst = dst_row;
		for (uint32_t x = 0; x < out_width; x++, dst += 4) {
			// With a down scale, the average of the block the output pixel covers.
			uint32_t sum[3] {};
			uint32_t count = 0;
			for (uint32_t sy = y * scale; sy < std::min(header->height, (y + 1) * scale); sy++) {
				for (uint32_t sx = x * scale; sx < std::min(header->width, (x + 1) * scale); sx++) {
					const auto* src = rgba + (static_cast<size_t>(sy) * header->width + sx) * 4u;
					sum[0] += src[0];
					sum[1] += src[1];
					sum[2] += src[2];
					count++;
				}
			}
			dst[0] = static_cast<uint8_t>((sum[0] + count / 2) / count);
			dst[1] = static_cast<uint8_t>((sum[1] + count / 2) / count);
			dst[2] = static_cast<uint8_t>((sum[2] + count / 2) / count);
			dst[3] = alpha;
			if (swap_rb) {
				std::swap(dst[0], dst[2]);
			}
		}
	}
	stbi_image_free(rgba);

	FillImageInfo(*header, scale, image_info);

	// As PngDec: the output's width and height packed, when they fit.
	return out_width > 32767u || out_height > 32767u
	           ? 0
	           : static_cast<int32_t>((out_width << 16u) | out_height);
}

static int32_t KYTY_SYSV_ABI JpegDecDecode(void* handle, const JpegDecDecodeParam* param,
                                           JpegDecImageInfo* image_info) {
	JpegHeader    header {};
	const int32_t result = JpegDecDecodeImage(handle, param, image_info, &header);

	static std::atomic<uint32_t> decoded {0};
	static std::atomic<uint32_t> failed {0};
	if (LogCall(result < 0 ? &failed : &decoded)) {
		if (param != nullptr) {
			LOGF("JpegDec: decode %p, %" PRIu32 " bytes -> %p, %" PRIu32 " bytes, pitch %" PRIu32
			     ", coefficients %p, %" PRIu32
			     " bytes, decode mode %u, down scale %u, pixel format "
			     "%u, alpha %u: %" PRIu32 "x%" PRIu32 ", %" PRIu32
			     " components%s, result 0x%08" PRIx32 "\n",
			     param->jpeg_mem_addr, param->jpeg_mem_size, param->image_mem_addr,
			     param->image_mem_size, param->image_pitch, param->coefficient_mem_addr,
			     param->coefficient_mem_size, static_cast<unsigned>(param->decode_mode),
			     static_cast<unsigned>(param->down_scale),
			     static_cast<unsigned>(param->pixel_format),
			     static_cast<unsigned>(param->alpha_value), header.width, header.height,
			     header.components, header.progressive ? " (progressive)" : "",
			     static_cast<uint32_t>(result));
		} else {
			LOGF("JpegDec: decode without parameters, result 0x%08" PRIx32 "\n",
			     static_cast<uint32_t>(result));
		}
	}
	return result;
}

} // namespace JpegDec

LIB_DEFINE(InitJpegDec_1) {
	PRINT_NAME_ENABLE(true);

	LIB_FUNC("uNAUmANZMEw", JpegDec::JpegDecQueryMemorySize);
	LIB_FUNC("JPh3Zgg0Zwc", JpegDec::JpegDecCreate);
	LIB_FUNC("1kzQRoWEgSA", JpegDec::JpegDecDecode);
	LIB_FUNC("Hwh11+m5KoI", JpegDec::JpegDecDelete);
	LIB_FUNC("LSinoSQH790", JpegDec::JpegDecParseHeader);
}

} // namespace Libs
