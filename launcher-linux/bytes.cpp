#include "bytes.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

namespace Bytes {

namespace {

constexpr std::array<uint32_t, 64> K = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

uint32_t Rotr(uint32_t x, unsigned n) {
	return (x >> n) | (x << (32u - n));
}

void Block(std::array<uint32_t, 8>& h, const uint8_t* p) {
	std::array<uint32_t, 64> w {};
	for (size_t i = 0; i < 16; ++i)
		w[i] = static_cast<uint32_t>(p[i * 4]) << 24u | static_cast<uint32_t>(p[i * 4 + 1]) << 16u |
		       static_cast<uint32_t>(p[i * 4 + 2]) << 8u | static_cast<uint32_t>(p[i * 4 + 3]);
	for (size_t i = 16; i < 64; ++i) {
		const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3u);
		const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10u);
		w[i]              = w[i - 16] + s0 + w[i - 7] + s1;
	}
	uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
	for (size_t i = 0; i < 64; ++i) {
		const uint32_t s1    = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
		const uint32_t ch    = (e & f) ^ (~e & g);
		const uint32_t temp1 = hh + s1 + ch + K[i] + w[i];
		const uint32_t s0    = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
		const uint32_t maj   = (a & b) ^ (a & c) ^ (b & c);
		const uint32_t temp2 = s0 + maj;
		hh                   = g;
		g                    = f;
		f                    = e;
		e                    = d + temp1;
		d                    = c;
		c                    = b;
		b                    = a;
		a                    = temp1 + temp2;
	}
	h[0] += a;
	h[1] += b;
	h[2] += c;
	h[3] += d;
	h[4] += e;
	h[5] += f;
	h[6] += g;
	h[7] += hh;
}

const std::array<uint32_t, 256>& CrcTable() {
	static const auto table = [] {
		std::array<uint32_t, 256> t {};
		for (uint32_t i = 0; i < 256; ++i) {
			uint32_t c = i;
			for (int k = 0; k < 8; ++k)
				c = (c & 1u) != 0 ? 0xedb88320u ^ (c >> 1u) : c >> 1u;
			t[i] = c;
		}
		return t;
	}();
	return table;
}

// ---- inflate (RFC 1951), in the manner of zlib's puff.c: small, and checked against zlib in
// tests. (stb_image's own inflate refuses some valid streams.)

struct Huffman {
	std::array<uint16_t, 16>  count {};
	std::array<uint16_t, 288> symbol {};
};

// false: over-subscribed or unusable lengths. An incomplete code is allowed (deflate permits one
// for the distances).
bool Build(Huffman& h, const uint8_t* lengths, size_t n) {
	h.count.fill(0);
	for (size_t i = 0; i < n; ++i)
		++h.count[lengths[i]];
	int left = 1;
	for (int len = 1; len < 16; ++len) {
		left <<= 1;
		left -= h.count[static_cast<size_t>(len)];
		if (left < 0) return false;
	}
	std::array<uint16_t, 16> offs {};
	for (int len = 1; len < 15; ++len)
		offs[static_cast<size_t>(len) + 1] = static_cast<uint16_t>(offs[static_cast<size_t>(len)] + h.count[static_cast<size_t>(len)]);
	for (size_t i = 0; i < n; ++i)
		if (lengths[i] != 0) h.symbol[offs[lengths[i]]++] = static_cast<uint16_t>(i);
	return true;
}

struct Inflater {
	const uint8_t* in   = nullptr;
	size_t         size = 0;
	size_t         at   = 0;
	uint32_t       bits = 0;
	int            have = 0;
	Buffer         out;
	bool           failed = false;

	int Bit() {
		if (have == 0) {
			if (at >= size) {
				failed = true;
				return 0;
			}
			bits = in[at++];
			have = 8;
		}
		const int b = static_cast<int>(bits & 1u);
		bits >>= 1u;
		--have;
		return b;
	}
	uint32_t Bits(int n) {
		uint32_t v = 0;
		for (int i = 0; i < n; ++i)
			v |= static_cast<uint32_t>(Bit()) << static_cast<unsigned>(i);
		return v;
	}
	int Decode(const Huffman& h) {
		int code = 0, first = 0, index = 0;
		for (int len = 1; len < 16; ++len) {
			code |= Bit();
			const int count = h.count[static_cast<size_t>(len)];
			if (code - count < first) return h.symbol[static_cast<size_t>(index + (code - first))];
			index += count;
			first += count;
			first <<= 1;
			code <<= 1;
			if (failed) return -1;
		}
		return -1;
	}
	bool Codes(const Huffman& lit, const Huffman& dist) {
		static const uint16_t lbase[] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
		static const uint16_t lext[]  = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
		static const uint16_t dbase[] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
		static const uint16_t dext[]  = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
		for (;;) {
			int symbol = Decode(lit);
			if (symbol < 0 || failed) return false;
			if (symbol < 256) {
				out.push_back(static_cast<uint8_t>(symbol));
			} else if (symbol == 256) {
				return true;
			} else {
				symbol -= 257;
				if (symbol >= 29) return false;
				const size_t length = lbase[symbol] + Bits(lext[symbol]);
				const int    d      = Decode(dist);
				if (d < 0 || d >= 30 || failed) return false;
				const size_t distance = dbase[d] + Bits(dext[d]);
				if (failed || distance > out.size()) return false;
				for (size_t i = 0; i < length; ++i)
					out.push_back(out[out.size() - distance]);
			}
			if (out.size() > (256u << 20u)) return false; // (nothing here is that big)
		}
	}
	bool Run() {
		int last = 0;
		do {
			last           = Bit();
			const int type = static_cast<int>(Bits(2));
			if (failed) return false;
			if (type == 0) {
				have = 0; // (to the byte boundary)
				if (at + 4 > size) return false;
				const size_t len  = static_cast<size_t>(in[at]) | static_cast<size_t>(in[at + 1]) << 8u;
				const size_t nlen = static_cast<size_t>(in[at + 2]) | static_cast<size_t>(in[at + 3]) << 8u;
				at += 4;
				if (len != (~nlen & 0xffffu) || at + len > size) return false;
				out.insert(out.end(), in + at, in + at + len);
				at += len;
			} else if (type == 1) {
				static const Huffman fixed_lit = [] {
					Huffman h;
					uint8_t l[288];
					for (int i = 0; i < 144; ++i) l[i] = 8;
					for (int i = 144; i < 256; ++i) l[i] = 9;
					for (int i = 256; i < 280; ++i) l[i] = 7;
					for (int i = 280; i < 288; ++i) l[i] = 8;
					Build(h, l, 288);
					return h;
				}();
				static const Huffman fixed_dist = [] {
					Huffman h;
					uint8_t l[30];
					for (auto& v: l) v = 5;
					Build(h, l, 30);
					return h;
				}();
				if (!Codes(fixed_lit, fixed_dist)) return false;
			} else if (type == 2) {
				static const uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
				const size_t nlen  = Bits(5) + 257;
				const size_t ndist = Bits(5) + 1;
				const size_t ncode = Bits(4) + 4;
				if (failed || nlen > 286 || ndist > 30) return false;
				uint8_t lengths[320] = {};
				for (size_t i = 0; i < ncode; ++i)
					lengths[order[i]] = static_cast<uint8_t>(Bits(3));
				Huffman lencode;
				if (!Build(lencode, lengths, 19)) return false;
				size_t index = 0;
				uint8_t table[320] = {};
				while (index < nlen + ndist) {
					const int symbol = Decode(lencode);
					if (symbol < 0 || failed) return false;
					if (symbol < 16) {
						table[index++] = static_cast<uint8_t>(symbol);
					} else {
						uint8_t previous = 0;
						size_t  repeat   = 0;
						if (symbol == 16) {
							if (index == 0) return false;
							previous = table[index - 1];
							repeat   = 3 + Bits(2);
						} else if (symbol == 17) {
							repeat = 3 + Bits(3);
						} else {
							repeat = 11 + Bits(7);
						}
						if (index + repeat > nlen + ndist) return false;
						while (repeat-- != 0) table[index++] = previous;
					}
				}
				if (table[256] == 0) return false; // (no end of block)
				Huffman lit, dist;
				if (!Build(lit, table, nlen) || !Build(dist, table + nlen, ndist)) return false;
				if (!Codes(lit, dist)) return false;
			} else {
				return false;
			}
		} while (last == 0);
		return !failed;
	}
};

void Put32(Buffer& out, uint32_t value) {
	for (unsigned shift = 0; shift < 32; shift += 8)
		out.push_back(static_cast<uint8_t>(value >> shift));
}

} // namespace

std::string Sha256Hex(const uint8_t* data, size_t size) {
	std::array<uint32_t, 8> h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
	size_t                  at = 0;
	for (; at + 64 <= size; at += 64)
		Block(h, data + at);
	std::array<uint8_t, 128> tail {};
	const size_t             rest = size - at;
	if (rest != 0) std::memcpy(tail.data(), data + at, rest);
	tail[rest]         = 0x80;
	const size_t total = rest + 1 + 8 <= 64 ? 64 : 128;
	const uint64_t bits = static_cast<uint64_t>(size) * 8u;
	for (size_t i = 0; i < 8; ++i)
		tail[total - 1 - i] = static_cast<uint8_t>(bits >> (8u * i));
	for (size_t i = 0; i < total; i += 64)
		Block(h, tail.data() + i);
	static const char digits[] = "0123456789abcdef";
	std::string       text;
	for (const uint32_t word: h)
		for (int shift = 28; shift >= 0; shift -= 4)
			text += digits[(word >> static_cast<unsigned>(shift)) & 0xfu];
	return text;
}

uint32_t Crc32(const uint8_t* data, size_t size) {
	const auto& table = CrcTable();
	uint32_t    crc   = 0xffffffffu;
	for (size_t i = 0; i < size; ++i)
		crc = table[(crc ^ data[i]) & 0xffu] ^ (crc >> 8u);
	return ~crc;
}

Buffer Gunzip(const uint8_t* data, size_t size, std::string* error) {
	auto fail = [&](const char* why) {
		if (error != nullptr) *error = why;
		return Buffer {};
	};
	if (error != nullptr) error->clear();
	if (size < 18 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 8) return fail("not a gzip file");
	const uint8_t flags = data[3];
	size_t        at    = 10;
	if ((flags & 4u) != 0) { // FEXTRA
		if (at + 2 > size) return fail("damaged gzip header");
		at += 2u + (static_cast<size_t>(data[at]) | static_cast<size_t>(data[at + 1]) << 8u);
	}
	for (const unsigned bit: {8u, 16u}) { // FNAME, FCOMMENT: zero-terminated
		if ((flags & bit) == 0) continue;
		while (at < size && data[at] != 0)
			++at;
		++at;
	}
	if ((flags & 2u) != 0) at += 2; // FHCRC
	if (at + 8 > size) return fail("damaged gzip header");
	const size_t deflated = size - 8 - at;
	if (deflated > 0x7fffffffu) return fail("gzip file too large");
	Inflater inflater;
	inflater.in   = data + at;
	inflater.size = deflated;
	if (!inflater.Run()) return fail("damaged gzip data");
	Buffer result = std::move(inflater.out);
	const uint8_t* trailer = data + size - 8;
	const uint32_t crc     = static_cast<uint32_t>(trailer[0]) | static_cast<uint32_t>(trailer[1]) << 8u |
	                         static_cast<uint32_t>(trailer[2]) << 16u |
	                         static_cast<uint32_t>(trailer[3]) << 24u;
	const uint32_t length = static_cast<uint32_t>(trailer[4]) | static_cast<uint32_t>(trailer[5]) << 8u |
	                        static_cast<uint32_t>(trailer[6]) << 16u |
	                        static_cast<uint32_t>(trailer[7]) << 24u;
	if (length != static_cast<uint32_t>(result.size())) return fail("gzip length mismatch");
	if (crc != Crc32(result.data(), result.size())) return fail("gzip checksum mismatch");
	return result;
}

Buffer GzipStored(const Buffer& data) {
	Buffer out = {0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff};
	size_t at  = 0;
	do {
		const size_t   n     = std::min<size_t>(65535, data.size() - at);
		const bool     final = at + n == data.size();
		out.push_back(final ? 1 : 0);
		out.push_back(static_cast<uint8_t>(n));
		out.push_back(static_cast<uint8_t>(n >> 8u));
		out.push_back(static_cast<uint8_t>(~n));
		out.push_back(static_cast<uint8_t>(~n >> 8u));
		out.insert(out.end(), data.begin() + static_cast<std::ptrdiff_t>(at),
		           data.begin() + static_cast<std::ptrdiff_t>(at + n));
		at += n;
	} while (at < data.size());
	Put32(out, Crc32(data.data(), data.size()));
	Put32(out, static_cast<uint32_t>(data.size()));
	return out;
}

} // namespace Bytes
