vgmstream ATRAC9 decoder (internal, with ATRAC9 v3 vibration blocks)

From https://github.com/vgmstream/vgmstream at commit 24ffd6ce3e1f34fdebf9f5316d18ad2e162838ce (2026-11-10):
src/coding/libs/atrac9_dec.c, atrac9_dec.h, atrac9_internal.c, atrac9_internal.h, atrac9_internal_data.h and
src/util/bitstream_msb.h, unchanged, at the same relative paths. Licence: COPYING (vgmstream, ISC-style).

Kyty uses it only for ATRAC9 channel configs 6 and 7 (vibration, DualSense haptics), which LibAtrac9 does not decode.
