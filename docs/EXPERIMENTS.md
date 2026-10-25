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

## 2026-10-15 04:35 — Wolverine on the DS engine: linear-image readback off
- Test (switch only): DS engine 7b660856 with `--readback-linear-images false`.
- Result: same fault at eboot+0x403fb16. Discarded.
- New evidence (static + logs): the three objects registered in the table (eboot+0x40411f0, called from 0x4012737/58/c0) are static globals (eboot+0x9b8ae20, 0xbeda190, 0x9b8b230), never null. Yet table[0] reads 0 at the fault and the mask still says 3 entries (r15=0xe000…). So the table's heap memory (InitAllocator heap, 0x1010000000, 650 MiB) was overwritten after registration. File I/O and memory mappings are identical between both engines up to the fault.
- Next: `KYTY_GPU_WRITE_WATCH=1010000000-1038c00000` on the DS engine to see whether a GPU write-back lands in that heap.

## 2026-10-15 05:42 — Moon repeated again, in a cutscene only
- User report: the repeated moon is back in one cinematic; in gameplay it is fine (engine 1f477d1 with KYTY_CLAMP_SAMPLER_SHADERS=59bdf02950d06cb6).
- Likely: the cutscene uses another shader not in the clamp list. Pending: capture which shader (dump during that cutscene).

## 2026-10-15 05:55 — Ready for tomorrow (FPS, Demon's Souls method)
- Built ab36e30 (KytyPS5-2026-10-15-ab36e30, Windows exe md5 12935aa510f820c441c1fae6b466983f): the profiler no longer kills the process when it unwinds the guest render thread (the vectored handler ran before the sampler's __except; 3 crashes at 05:11, 05:13, 05:20).
- Not installed: the PC went offline. To do: install as engines/wolverine/kyty_emulator-dev-ab36e30.exe, add a mode for it, keep "Normal" on 1f477d1.
- Known (logs 20261015-022155, 051709): ~19 fps, 50-60 ms frames; Guest_RenderThread 82-98 % of a core is the limit; Thread_Gpu 52-62 %; GPU not the limit.
- Method step 2: KYTY_PROF=1 via live.txt once in game (never at launch), same spot, 2 min -> split Guest_RenderThread time (guest code vs HLE/fault handlers by first exe caller), then attack the largest block with the Demon's Souls fixes 1-6 (METODO-WOLVERINE.md), each behind a switch with in-process A/B.

## 2026-10-15 12:47-14:00 — First real profile of the render thread (engine ab36e30)
- KYTY_PROF (guest thread sampled, fix ab36e30 works: no crash). Guest_RenderThread, 18 reports: guest code 63 %; BufferCache::ReadMemory from page faults (RenderContext::HandleFault) waiting for the GPU ~24 % (SendCommandSync 12.4, MasterSemaphore::Wait 7.1, WaitPriorityOperations 3.8); page protection/invalidation ~7 %; VideoOutGetFlipStatus 2.3 %. Ceiling if all of that went: ~+40-45 % (19 -> ~27 fps).
- KYTY_READBACK_STATS + KYTY_FAULT_STATS (overhead: 6-13 fps while on, not a measurement): 120k-430k write faults and 2-7k read faults per 5 s on 25-31k pages (the same pages re-fault every frame); readbacks 233-673 ms per 5 s on the GPU thread, downloading 1.2-13 GB per 5 s of which 13-29 MB changed.
- Hottest buffer 0x12c0000000 (70 MiB, written by CS 0xeaa507075716aa3b each frame): guest writes 1743 + reads 318 per 5 s, 655 downloads (698 MiB), 149 ms.
- Next (existing switch, no build): live A/B KYTY_ASYNC_WRITE_READBACK 0/1 in one process at the same spot, stats off. It keeps exact data (the faulting thread still waits for its download); only the GPU-thread drain goes.

## 2026-10-15 14:09 — Live A/B KYTY_ASYNC_WRITE_READBACK (engine ab36e30, still camera, same spot)
| state | 5 s windows | mean frame | fps |
|---|---|---|---|
| 0 (round 1) | 9 | 69.6-72.1 ms (one 75) | 13.3-14.4 |
| 1 (round 1) | 7 | 67.8-68.95 ms | 14.5-14.8 |
| 0 (round 2) | 8 | 69.2-70.4 ms | 14.2-14.5 |
| 1 (round 2) | 7 | 67.6-68.95 ms | 14.5-14.8 |
- Result: about -1.5 ms per frame (+2 %), every ON window faster than every OFF window. Exact data (the faulting thread still waits for its own download). Small; worth keeping on.
- Note: frames are paced to 60 Hz vblanks (p99 ~84 ms = 5 vblanks, mean ~68-70 = mostly 4), so gains show up in steps.
- Earlier (14:06) an ErrorDeviceLost with no new switch on: the known pending device-lost (also 2026-10-14 19:50).

## 2026-10-15 14:57 — Live A/B KYTY_WRITE_WINDOW_KB 0/16 (engine ed3b0be, ASYNC_WRITE_READBACK=1, still camera, same spot)
| state | 5 s windows (mean frame, ms) | average |
|---|---|---|
| 0 | 66.2 67.1 66.5 67.8 / 67.1 66.9 68.3 68.2 69.2 67.6 67.8 | 67.5 |
| 16 | 66.7 64.7 64.9 65.6 65.6 67.3 65.8 / 66.2 65.6 65.2 65.4 64.9 65.8 67.6 | 65.8 |
- Result: -1.7 ms per frame (+2.6 %), both rounds. Kept on (16). With ASYNC_WRITE_READBACK: ~70 -> ~65.8 ms today (+6 %).
- Same invalidation call as a one-byte fault (pages only marked CPU-written earlier); a window with GPU data or an image is not taken.

## 2026-10-15 15:05 — Live A/B KYTY_DIRECT_READBACK and KYTY_COPY_QUEUE_INFLIGHT (ed3b0be, AWR=1, WW=16)
- DIRECT_READBACK 1 vs 0: 66.0-67.8 vs 65.6-68.0 ms. No change.
- COPY_QUEUE_INFLIGHT 1 vs 0: 65.4-67.1 vs 63.5-66.9 ms. No change.
- Frames sit at ~66.7 ms = 4 vblanks of 60 Hz: the work per frame lies between 50 and 66.7 ms, so a saving shows only when a frame drops to 3 vblanks (50 ms, 20 fps). Both left off.

## 2026-10-15 15:38-15:48 — Live bisection of the flicker (engine 603b618, KYTY_LIVE_SKIP_*)
- Compute stage first: the game closed at once (skipping compute work hangs the game). Pixel stage only after that.
- Rounds (bits/value -> flicker): 1/0 stops, 2/0 stops, 3/0 flickers, 4/4 mostly stops, 5/4 mostly stops, 6/4 mostly stops (one tank texture left), 7/4 flickers, 8/68 flickers, 9/196 stops ("poco").
- One pixel shader matches 9 bits = 0xc4: **bfacdd28a2288ac4**. Skipping it removes almost all of the flicker; a small remainder (a tank texture) comes from another shader.
- Next: dump its RDNA2 code and translation (KYTY_DECODED_DUMP_DIR / IR dump) and find the mistranslated instruction.

## 2026-10-15 15:51-15:55 — Flicker shader confirmed and dumped
- KYTY_LIVE_SKIP_BITS=0 again (explicit value; removing the key does not revert): flicker back. bfacdd28a2288ac4 confirmed.
- Frames with it skipped: ~43 ms (22-24 fps); with it back: 52-73 ms (13-19 fps), same session (camera not strictly fixed). The shader costs time as well.
- Dump (KYTY_DECODED_DUMP_DIR marker trick + --graphics-debug-dump): pixel shader, 2254 RDNA2 instructions (13.7 KiB), 5 image_sample + 3 sample_lz + 1 sample_l, 13 IMAGE_LOAD, 8 IMAGE_STORE (writes UAVs from a pixel shader), S_WQM_B64, F16 compares with SDWA high-half selects, V_CUBE* ops, s_setreg. SPIR-V 263 KiB, 13.5k instructions (1788 OpSelect, 2655 OpBitcast, 643 Phi).
- docs/shaders/bfacdd28a2288ac4.rdna.txt kept for the analysis.

## 2026-10-15 16:40 — Cause of the device losts (black screen / PC reboot reports)
- Log 20261015-140613 (ErrorDeviceLost, no new switch): VK_EXT_device_fault says the GPU read **a destroyed buffer**: ReadInvalid at host VA 0x39a055000 "in DESTROYED buffer va=0x39a000000 size=0x254000 guest=0x112444c000, destroyed 2075.5 ms ago". The guest range is now held by a larger buffer (guest=0x112444c000 size=0x35c000, table va=0x3db600000). Something still handed a shader the old buffer's device address two seconds after it was replaced (use-after-free on the GPU).
- A GPU read of freed memory loses the device; on some drivers that is a black screen, a driver reset or a reboot (users' reports, Demon's Souls too).
- Next: find which path keeps old buffer device addresses (pointer tables, cached SRT plans, indirect args) after a buffer is replaced.

## 2026-10-15 16:50 — POPS interlock is in, flicker stays; real cause found in the same shader
- Dump with b822785: SPIR-V of bfacdd28a2288ac4 has OpBeginInvocationInterlockEXT/End (1 each), ExecutionMode PixelInterlockOrderedEXT, capability 5378. User: still flickers. Interlock alone is not the fix.
- The shader read-modify-writes the same images: IMAGE_LOAD and IMAGE_STORE on the same T# (s8, s16, s36). In the SPIR-V all 13 loads are OpImageFetch on **sampled** images (OpTypeImage Sampled=1) and the 8 stores OpImageWrite on **storage** images (Sampled=2): two descriptors for one guest image. ResourceTracking::AddImage keys resources by resource_class, and ImageRead is always ImageResourceClass::Sampled (ValueOpcodes.cpp:217). So the load reads through the texture path (another view, possibly another copy, not coherent with the storage writes), and the read-modify-write sees stale or racing data: frame-to-frame differences.
- Next: a load of a T# that the shader also stores to goes to the storage resource (OpImageRead, Coherent under the interlock).

## 2026-10-17 07:00-08:10 — Ground flicker: two hypotheses ruled out
- Dumps of 10-16 18:02 detiled (64KB_R_X render-target swizzle, gpu_tiler_render_target.comp): the ground decal patches change per pixel every frame in the five RMW images of bfacdd28a2288ac4. May be intended stochastic noise resolved later by the game's TAA: not proof of a bug by itself.
- ec32e98 (loads of stored images on the storage path even when also sampled): the log shows bfacdd28's 5 loads already on the storage images ("Shaders: POPS ... (0 also sampled)", ~20 POPS shaders). Still flickers.
- 52af0e0 (KYTY_POPS_BARRIER=1: end rendering + shader-write barrier after every POPS draw): still flickers. Ordering between decal draws is not the cause.
- Next: dump the composite/final image and the pass that resolves the decal noise frame by frame, to find the first pass where the flicker appears.

## 2026-10-17 08:10 — Overall state (user report + log 20261017-065729)
- Flicker also on trees and mountains; vegetation pops in/out while moving in the forest; long black loading screen with music.
- FPS: loading 0.4-10, then 10-19 oscillating in some areas, ~23 steady in others. Render thread is the limit (82-98 %), ~24 % of it waiting on GPU for fault-driven readbacks (10-15 profile); 120k-430k write faults / 5 s on the same pages, up to 13 GB downloaded / 5 s.
- Working hypothesis (unproven): GPU-written data (culling, indirect args, visibility) read through the CPU copy with stale or mixed frames -> popping and flicker; the copies -> low, oscillating FPS. First test: turn off coherence for one hot GPU-written buffer and watch popping/flicker. Occlusion queries are still "always visible" (cost, not popping).

## 2026-10-17 08:17-08:20 — Live A/B: strict coherence (engine 7bd5a63, forest)
- live.txt: KYTY_ASYNC_READBACK, KYTY_COPY_QUEUE_READBACK, KYTY_ASYNC_WRITE_READBACK and KYTY_WRITE_WINDOW_KB to 0 (log 20261017-081640: applied at line 1194).
- User: vegetation popping, ground/trees/mountains flicker all the same. The async/windowed CPU<->GPU copy paths are not the cause. Restored to the original values afterwards.
- Remaining suspects for popping + flicker everywhere: GPU culling inputs that never go through the CPU (previous frame's depth/HiZ, history images through the image cache), and the game's TAA (motion vectors/jitter).

## 2026-10-17 08:21-08:31 — Live bisection of pixel shaders (trees/mountains flicker, engine 7bd5a63)
- KYTY_LIVE_SKIP_STAGE=1: bits1 v0 and bits2 v0 -> no flicker; bits3 v0 -> slight; bits3 v4 -> flicker. Deeper steps were inconsistent on re-test: no single PS and no 5-PS group changes it; bits4 v0 re-tested "same".
- Conclusion: no single culprit pixel shader. Skipping a quarter of the PS removes the flickering objects themselves (trees, terrain), so the flicker comes from something they all share (shadows, lighting/G-buffer inputs, the game's TAA or another full-screen compute pass), not from one draw.
- Side note: skipping bits7 value 48 (PS 272c6c9434fa2cb0, 587ddb621bb29db0, ef015ad7addc9db0) made more fog appear that looked right to the user: one of them may hide the fog.
- live.txt restored to the original values.

## 2026-10-17 08:32-08:40 — Live bisection of compute shaders (engine 7bd5a63, forest)
- KYTY_LIVE_SKIP_STAGE=2, bits4: value 0 and 1 no change; value 2 closed/hung the game (an essential CS); value 3 (7 CS) -> lighting gone and flicker gone.
- The 7: 043df42613b9b623, e0cca8c21c2df1e3, 83ceee12d11bd2a3, 7b779927d3dcd963, 021204e9e4fe4393, 105bc58e82354993, a0c0e96d0440cb53. One at a time (bits16): all still flicker except 021204e9e4fe4393, which breaks the image (the lighting itself).
- So the flicker is in the lighting result: CS 021204e9e4fe4393 or one of its inputs (shadow maps, G-buffer, light lists). Next: KYTY_DUMP_SHADER=021204e9e4fe4393 (its images frame by frame) and its RDNA dump, as done for the ground.
- live.txt restored.

## 2026-10-25 — Glitch pass (branch wolverine-fix-1025): moon in a cutscene, claw-hit sound, popping
- Moon (cutscene): the gameplay fix (KYTY_CLAMP_SAMPLER_SHADERS=59bdf02950d06cb6) is per shader hash at compile time; the cutscene draws the moon with another shader that still gets the wrapping fallback S#. New live switches act at bind time on the exact wrapping fallback words only: KYTY_FALLBACK_SAMPLER_LOG=N lists the next N shaders that bind it (set it during the cutscene); KYTY_FALLBACK_CLAMP_BITS/VALUE clamp the fallback for a hash subset (BITS>=64: all) to bisect in the running cutscene. The found hash goes into KYTY_CLAMP_SAMPLER_SHADERS.
- Claw-hit sound: AJM resample jobs (SetResampleParameters/Ex) are accepted and ignored (output always at ratio 1); PADSPK ports with no DualSense play on the main output. KYTY_AUDIO_TRACE=1 prints port types (AudioOut2 raw -> mapped), AT9 stream formats, resample requests and decode errors; KYTY_AUDIO_MUTE_TYPES (live bit mask of port types) and KYTY_PADSPK_MAIN=0 (live) tell which port plays the strange sound.
- Popping/flicker: KYTY_SKIP_LIST_MASK (live, default -1) re-enables single entries of --skip-shaders in a running game; CS 0x29be8047c4e86afe (BVH) is still skipped by default (KYTY_SKIP_BVH_DISPATCHES=0 runs it, untested in game).
