# General lessons: what worked and what did not (any game)

Source: KytyPS5 fork, Demon's Souls 01.007.000, Windows, i9-14900K (6 P + 16 E cores) + RTX 5090. Switches are that
fork's `KYTY_*` names; the ideas carry over to any HLE PS5 emulator. Title-specific items: `demons-souls.md`.

## Frame model now (compare yours before looking for work)

- Boletaria 1-1 standing, 120 cap: ~64 fps, render thread ~15 ms and ~95% busy, GPU ~65% busy. ~9.5K table draws
  (~5.5K draw calls at ~1.4 us), ~1.4K table dispatches (~1.8 us; async ones half in buffer/BDA preparation),
  ~2.4K ring copies of guest-written constants (~10 MB, ~1 ms), PM4 parsing + register observers ~9%, page-protection
  syscalls ~4%, linear copies ~4%.
- Heavy walking/turning views: 13-17K draws and every phase ~2x dearer per call (more distinct cold data) ->
  25-45 ms frames. That, not first-use work, is most of the remaining 1% lows.
- History: at ~40 fps (09-30) it was ~4K normal-path dispatches + ~1K draws at 1-5 us; the old "async thread + GPU
  overlap + less translation, all together" plan for 60 was wrong (the game's own frame label forbids the overlap);
  60 came from table mode plus coherence, upload and chain cuts.

## Wins by area (apply first if your emulator lacks them)

| Area | Change | Result | Why |
|---|---|---|---|
| Translation | Vulkan recording worker + deferred submits | 21.1 -> 23.7 fps | translation stops waiting on driver calls |
| | **Table mode**: at translation, snapshot every resource-table slot a draw's shaders read into a per-draw block (upload ring, read by device address); V#s resolved through a direct-mapped cache proven by registration/mapping/region epochs; image sets keyed by T#/S# words, validated once per frame | 1-1 44.2 -> 48.6 (steady 45 -> 52); dispatches +9.6% once programs were exact specializations with a per-buffer portable fallback | removes per-draw resource preparation; replaced per-object replay records (now ~10 draws/frame) |
| | Store a variant once per (pair, state) per frame, no per-frame budget; no record store while the variant is pending | walk 1% low +12%, +7% | budgets pushed hundreds of draws onto the slow path per frame |
| | Refusal census: a draw refused only for an optional input (a null or not-yet-streamed texture) now takes the fast path | 43.9 -> 45.4 fps; refusals 409 -> 2/frame | ~400 draws/frame paid ~5.5 us each for nothing |
| | Epoch shortcuts: set proofs re-checked only when an image/its pages changed, dynamic-state diff, targets/transits skipped while the images epoch holds, state-key cache from register-change bits, last-pair memo | each 0.5-2% | provable "nothing changed" checks instead of lookups |
| | PM4 look-ahead prefetching the next draw's indirect register tables and user-data lines | draw 1.42 -> 1.37 us | the next draw's first-level misses overlap this draw |
| | Batch boundaries wait while the GPU still runs handed work | +3.6%, 1% low +3 | fewer, larger submissions without starving the GPU |
| | Engine linear-copy compute shader on the CPU: detected from registers, writes only differing pages through the backing view keeping protection, big compares on helpers, accepted across consecutive mappings | +0.8%, +0.4%, +1.8%, -0.15 ms chain; one level 69 -> 85 fps | a refused copy made its destination GPU-owned and every table read of it a sync readback |
| | Wave64 programs proven lane-local run one lane per invocation on 32-wide GPUs | GPU 10.05 -> 9.07 ms | emulating 2 lanes per invocation doubled registers and code |
| Build | -O3 + ThinLTO; PGO, retrained after big changes | ~+4%; render -3..-5% per retrain | a stale profile loses it (a new branch drops a function's profile); ship x86-64-v3 (-march=native leaked GFNI/VNNI into a build) |
| Coherence | Readbacks on a transfer-only queue, waiting only for the last writer's timeline value | sync downloads 1.19 -> 0.29 ms/frame; +3.6% | 83% read data finished frames earlier |
| | Fault handler releases its lock before mprotect; reprotect on a worker; 64 KiB "image here" bitmaps | +1.1 fps; mprotect 608 -> 320/frame; +0.7 fps | the render thread spun ~300x/frame on that lock; cheap negative lookups |
| | Readback readiness checked only over GPU-owned pages actually written back | +6.5% | whole-window checks waited on unrelated writers |
| | A host overwrite of a whole range drops the GPU data instead of reading it back; per-mapping fences for spanning write-backs | sync downloads 0.71 -> 0; lock waits -84% | |
| | Shader writes no longer mark a whole buffer "written this command buffer" (in-order uploads) | +4%, 1% low +4 | each in-order upload broke the render pass with two full barriers |
| | BDA (pointer) reads synced only where a guest write became visible (submission, satisfied wait) | region syncs 2.5-10K -> 850/frame | |
| | Exact unmap epochs per 16 KiB granule | 1% low 17.9 -> 20.6 | "last 64 unmaps" invalidated most records |
| Uploads | Game buffers placed in shared memory blocks (not one allocation each); staging copies on helpers; an upload worker that applies write protections first and reads the guest mapping with a guarded copy | ~100 us per dedicated allocation gone; 2.3 -> 6-11 GB/s; hitch waits 27 -> 4.5 ms | |
| | Textures upload only the view's mip levels (tiled layouts put small mips first: lower levels are one suffix), partial dirty ranges also for GPU-modified images, re-dirtied staging refilled | 3.97 -> 1.43 GB per walk; level entry 16.2 -> 8.0 GB | streaming changes base level, not whole textures |
| GPU | No barrier between repeated guest storage writes in one layout/access; one barrier dedupe | barriers 7470 -> 3087/frame, 33.9 -> 40.6 fps | guest flushes already order those writes |
| | Bounds by robustBufferAccess2 instead of EXEC guards on loads | 40.6 -> 43.1 | |
| Shaders | Unoptimized pipelines first, optimized in the background (NVIDIA only) | cold stalls 22 s -> 0 | |
| | Translator: dominator tree in CFG structurization; deterministic SPIR-V ids; every game shader translated in the background at start | worst shader 862 -> 77 ms; caches hit across runs; cold-route slow translations 44 -> 5 | quadratic dominator sets; hash-order ids made caches never hit |
| | Static precompile of every runtime variant (exact, table, portable subsets) + recorded hints shipped without game code + tiered run (recorded first) | cold 1-1 sync compiles 274 -> 0; ready to play 13 min | the runtime finds pipelines by the driver's key of the full create info |
| | SPIR-V passed inline (no driver shader modules), binary store, periodic cache saves | -1.8 GB heap, -3 GB; caches survive crashes | drivers keep copies of everything handed to them |
| Threads | Render thread above the game's threads, non-exclusively on P-cores and never CPU 0; sync callers spin ~20 us before sleeping; present without the render lock | +2.5%; lows steadier; -0.3 ms | Windows interrupts land on CPU 0 (pinning there halved fps) |
| Memory | Garbage collection by last use for images (1200 flips) and buffers (600/1800); collector trigger on budget, LRU pre-filtered | small GPUs went from never collecting to stable | GPU flips and must-keep LRU heads had blocked it |
| | 16 GB RAM policy: spill idle shader plans to a scratch file, JIT arena, driver caches capped by RAM, record-only warmup | working set 17.5 -> 14.0 GB | |
| Long play | Per-area sweeps of address-keyed caches; kernel ranges in trees; blocked range sets; 256 KiB translation caches; map/unmap OS calls outside the address-space lock | revisit stalls 0.3-0.8 s gone; walk 58 -> 70; lock holds 0.53 -> 0.04 s | the game cuts memory finer every load (9K -> 51K mappings) |
| Stutter | Free-range search under a shared lock; kept file handles + positional reads + background read-through of small streamed audio | >100 ms frames 8-12 -> 0; 1% low 5.9 -> 44.6 | lock convoys and synchronous disk reads, not CPU work |
| | Streaming without re-creation: pools kept on partial unmap, dirty ranges only, pooled detile scratch, binary-searched interval queries, no GPU finish on unmap | worst frame 200 -> 113 ms | a new VMA block per detile was a 2.5 s freeze |
| | Windows partial unmaps parked in their view (no system call; the view released whole once emptied, split only if something is mapped over a parked range; pages trimmed from the working set on a thread of their own) | 1-1 walk 1% low 35 -> 49, longest frame 43 -> 22 ms | splitting cost 0.05 ms per MiB of the VIEW on the game's job threads; a log showed the game maps over <1% of freed ranges within 1 s |

## Dead ends (do not retry without a new argument)

| Idea | Result | Reason |
|---|---|---|
| Reuse full per-draw preparation; per-program stage reuse; repeat-draw fast paths | -3.5% .. flat | bookkeeping ate the ceiling |
| Render thread split; speculative parallel translation (8 workers); second thread for async queues | flat; -15%; ~0.3 ms possible | hand-off and commit costs; only 25-30% of the work is thread-private (caches are shared) |
| Guest async queues (culling chain) on their own VkQueue | 31 vs 36 fps, black patches | labels are written at translation: guest cross-queue sync orders translation only |
| Native replacements of engine passes (shadow pass, culling, geometry chains) | -10 .. -35% | ~2500 memory-reuse proofs/frame plus fallbacks |
| Memoizing per-draw/dispatch results by key (dispatch records, buffer memos, a materialization memo) | slower despite 10.7K hits/frame | 96.7% of dispatch keys are new every frame (per-frame rings); hit rate is not savings |
| Readback shortcuts: snapshot guest writes, skip the copy, zero-copy/latent feedback | -1.25 fps; fell to 9 fps; -8% + crashes | two barriers per snapshot; broken ownership |
| GPU-side resource-table evaluation (in place or through a page table) | +4 ms GPU / flicker, -12% | the emulator writes labels and CP copies at translation; data must be snapshotted then |
| Deferring or offloading ring copies; ring or blocks in VRAM; host-memory mirrors | mismatches; -3.3%; -0.85%; GPU 0.84 -> 2.56 ms | the game rewrites constants within ~100 us without a label; bus writes/reads are slower |
| Trimming ring copies to the shader's static read extent | 0.25% of bytes eligible | vector loads with computed offsets |
| Pages kept writable when re-faulted often | uploads 13 MB -> 25.8 GB/frame | without protection there is no write signal |
| Narrow readback windows to the read pages | 63 -> 33 fps | 9x readbacks, 7x faults |
| Merging protection calls; folding RO into NO_ACCESS | flat; -0.67% | cost is per changed page and per downgrade (TLB shootdown IPIs on every core) |
| Software prefetch of guest tables on the render thread; helper-thread prefetch; history prefetch | worse; -9%; hit 35% | page walks stall the issuing thread; another core cannot fill this core's L2 |
| Faster lookups for pairs/variants/states, reordered set fields, 4-way resolved table, held-entry reprove | per-call cost unchanged or worse | misses overlap (MLP); fewer instructions are not fewer misses |
| Render thread pinned alone to one core; strict pinning | 1% lows worse | it cannot escape preemption |
| Table multi-draw; written-V# obtain memo | 11% eligible; ~1% (built twice) | |
| Chunked views for partial unmaps; prefault backing on map; keep textures across full unmaps | hold time same; +21 s, +6 GB; no gain | |
| Faster uploads (parallel or overlapped write-protection calls) | causal slope of frame time to upload waits 0.05-0.27 | the waits sit on the recording worker before submit, off the critical path; VirtualProtect contends across threads (7 threads 2.6x slower per call) |
| Shader objects, pipeline libraries, DX12 | slower or <=3% | bottleneck is CPU translation |
| Changing emitted SPIR-V for a speedup (fast format paths, padding) | forbidden | every user's caches miss: cold HUD 56 -> 105-218 s |

## Platform facts worth knowing early

- **Windows memory**: every protection downgrade on any thread interrupts all cores running the process (1000
  downgrade pairs/frame cost 3.5-5%; upgrades and same-protection calls are free). A partial unmap of a file view =
  whole view unmapped + remainder remapped (~0.33 us/page, a crash window if trackers' protection is not restored
  under the lock). A view mapped read-only can never become writable; VirtualProtect cannot span views. First touch
  through a second view of a section costs ~1.2 us/page, serialized per process. VirtualProtect to no-access and
  VirtualUnlock of resident view pages both cost ~0.5 us per 4 KiB page and both drop them from the working set (a
  range left mapped stays counted in use). Exception dispatch uses ~5 KiB of the faulting thread's stack (guest
  fibers!). ReadFile into protected pages fails silently.
- **NVIDIA**: vkQueuePresentKHR holds a device-wide lock (wait for the GPU before presenting); a VkPipelineCache is
  copied whole into the heap; every VkShaderModule's SPIR-V is kept; pipeline binaries embed a per-process zstd
  dictionary (build stores from many small shards); image/view creation can stall ~5 ms in the driver; Streamline
  frame generation needs VK_KHR_present_id and ~44 GiB of free address space below 1 TiB at device creation;
  memory type 0 is counted under VRAM usage.
- **AMD RDNA2**: native wave64 and unoptimized pipeline builds crashed the driver -> emulate wave32, unoptimized-first
  on NVIDIA only.

## Shader pipelines (one AAA title)

- Scale: ~21K programs (10.7K compute, 9.5K of them particle variants), ~93K pipelines, ~8.7 thread-hours of driver
  compile. NVIDIA compiles a big compute shader nearly serially inside one process and shares no stages between
  pipelines -> shard the precompile over processes.
- What stalls is a specialization key the prefetch lacks (0.1-0.5 s re-translation on the render thread), not a
  missing SPIR-V. Measure coverage per stage at the key level; normalize key fields the module ignores (recorded
  coverage CS 92.3 -> 97.5%, PS 93.9 -> 98.6%); a full playthrough's recording covers effects met once.
- Recorded play is what gets used: with a recorded tier in the store, two 1-1 walks each used ~2,200 pipelines,
  ~2,000 of them recorded-tier and 1 (then 0) from the statically guessed tier (the rest: native variants built at
  run time). Compile the recorded tier first ("ready to play"), merge every
  recording source (shipped hints + this PC's), and skip guessed variants of shaders it covers (a fifth of compile
  time; laptop from scratch 2:36 -> 1:48 h, ready to play at 0:59).
- Compile cost multipliers: wave64 on a 32-wide GPU (every instruction twice, x2.3), runtime format decode as nested
  switches (up to 9.5 MB SPIR-V), EXEC predication.
