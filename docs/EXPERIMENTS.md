# Experiments (Wolverine, PPSA03671)

| Date | Build | Change | Measure | Result |
|---|---|---|---|---|
| 2026-10-14 | 4f7b4bf | live A/B, still camera, 15-20 s per segment: KYTY_IMAGE_RECYCLE_MB=0, KYTY_FREE_WRITEBACK=2, KYTY_BINDLESS_SLOT_REUSE=0 (vs base between each) | median mean-abs diff of consecutive distinct frames, 480x230 gray (video 19.09.26) | 0.48 / 0.45 / 0.46 vs base 0.48-0.52: no effect (±7 %), discarded |
| 2026-10-14 | 4f7b4bf | same run, not recorded: ASYNC_READBACK=0, COPY_QUEUE_READBACK=0, REGION_BITMAP=0, LABEL_FLUSH_US=0, ALIAS_TILE_COPY=1 | - | applied (log), no crash; visual not measured |
| 2026-10-14 | 4f7b4bf | brightness jumps 33 -> 78 in the still video | period | every 56.4 s, same pattern: game lightning, not a bug |
| 2026-10-14 | 4f7b4bf | KYTY_DRAW_DIFF=1, still camera | fresh image reads / 5 s | c821afdb reads R8 416x234 @0x1163060000 ~2/frame, its GPU contents dropped 0-2 frames before (evicted by an unwritten RGBA8 832x468 at the same address); 1374f1a5 reads RGBA8 2560x1440 @0x114c6c0000 1/frame, re-created over unwritten B10G11R11 832x468/1664x936 |
| 2026-10-14 | 2141c25 | live A/B KYTY_FREE_WRITEBACK 0/3/0/3, 25 s each, still camera (video 19.37.26), DRAW_DIFF on | fresh reads / 5 s; median frame diff | writebacks happen (~109/5 s at 0x1163060000 and 0x114c6c0000) but stale R8 reads by c821afdb unchanged (209 + 109 per 5 s in both modes); frame diff 1.35/1.51 (m0) vs 1.44/1.53 (m3): no effect. Reverted to 4f7b4bf |
| 2026-10-14 | 4f7b4bf | --rd (RenderDoc 2-frame capture) | fps | 0.6 fps (1.7 s/frame) with RenderDoc injected: unusable in Wolverine, mode removed |
| 2026-10-14 | 4f7b4bf | normal mode, 19:50 run | crash | ErrorDeviceLost at 28 fps after ~70 s (vulkanCommon.cpp:134 submit readback), not reproduced yet |
| 2026-10-14 | ff9d3a2/2f5dbdb | KYTY_DUMP_IMAGE=0x1163060000, FREE_WRITEBACK 0 and 3 | bytes read by c821afdb (R8 416x234) vs RGBA8 832x468 written by 2feed4c0 the frame before | the R8 covers the first 128 KiB of that memory, which the RGBA8 leaves at 0 (its data starts at +180 KiB): the read gets zeros in both modes, as the console would. Not a bug, discarded |
| 2026-10-14 | 2f5dbdb | KYTY_DUMP_IMAGE=0x114c6c0000 | input of 73cfdf01 (RGBA8 2560x1440, read then written by itself, then read by 1374f1a5) | input zeros (console: last B10G11R11 832x468 bloom-chain data at that address); output 0.76 % non-zero, sparse = likely HUD layer. Not the ground/vegetation flicker |
| 2026-10-15 | db853ee | KYTY_DRAW_DIFF history reads, still camera | sampled images written in an earlier frame | histories are kept (same image across frames, not re-created): cebf8908 reads 2560x1440 A2B10G10R10 @0x116fa20000 written by f1cdc44f (alive since frame 1714); c7c9148c reads its own three 1664x936 B10G11R11 (addresses rotate); 4efbcf79 reads 1664x936 written 2 frames before; a0c0e96d/ba80f372/b28b058c read their own |
| 2026-10-15 | db853ee | KYTY_DUMP_IMAGE=0x116fa20000, two consecutive frames | change of the output-res history, still camera | mean 5.9/1023 (0.6 %), 2.5 % of pixels change > 40/1023; read == contents before the write (no loss). History is not lost: the temporal-history-loss hypothesis is not supported |
| 2026-10-15 | db853ee | launch without --skip-shaders (the 6 preset hashes run) | user, still camera, same spot | ground/vegetation boil the same (13 fps vs 14). The skipped shaders are not the cause |
| 2026-10-15 | db853ee | skip c821afdb (A) / skip 2feed4c0 (B, confirmed absent in the log) | user, still camera | ground flicker unchanged in both: neither shadow/AO nor the half-res temporal pass |
| 2026-10-15 | db853ee | KYTY_IMAGE_STATS still camera | image creations | ~110/s, all the 3 aliased targets at 0x114c6c0000 (HUD/bloom); terrain textures are not re-created |
| 2026-10-15 | db853ee | live visual A/B: ASYNC_READ_SNAPSHOT=1, ASYNC_READBACK=0, DIRECT_READBACK=1 (also COPY_QUEUE_READBACK=0, LABEL_WAIT_FORWARD=1 with draw diff) | user, still camera | no visible change ("como siempre") |
| 2026-10-15 | db853ee | KYTY_DRAW_DIFF still camera | per-frame draw keys | ground draws 33d13a84/e6234173 and 4efbcf79/4a49ed80 change index count every frame (e.g. 3085-4604, 1050-12000) with the camera still: geometry submitted differs frame to frame |
| 2026-10-15 | 1f477d1 | draw diff with count source tags, still camera | where varying ground counts come from | 33d13a84/e6234173 (553-8006 per frame) are direct PM4 counts written by the game's CPU; 4efbcf79/4a49ed80 (11100-14100) are DRAW_INDEX_INDIRECT_MULTI args read by the CPU, never GPU-dirty at read time. The emulator does not read stale counts: the game itself submits different geometry each frame (stormy scene: rain/splash-type dynamic draws). Hypothesis discarded |

## 2026-10-15 04:24 — Wolverine on the DS engine: is VRSQRTPS emulation (`--amd-cpu`) the missing piece?
- Evidence: the DS engine (7b660856) has no unresolved imports left, yet still faults at eboot+0x403fb16 (null entry in the component table at obj+0x388). The DS engine has no VRSQRTPS emulation.
- Test (switch only): the Wolverine engine 1f477d1 without `--amd-cpu`.
- Result: the game boots. VRSQRTPS emulation is **not** the cause. Discarded.
- Next: run the Wolverine engine with `--printf-direction Console` and diff its kernel call sequence against the DS engine's log (20261015-041840) up to the fault.
