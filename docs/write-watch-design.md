# Write watch instead of protection-based write tracking (KYTY_WRITE_WATCH)

Branch `ds-write-watch` (from `origin/ds-port-nativo` 6317056a2). Line numbers are of this branch.

## Problem

Bug check 0x1A / 0x41202 (MEMORY_MANAGEMENT, 5 minidumps): a page of the page-file-backed guest section is paged out
while its protection changes. The trackers change protections of the section's views 50-59 thousand times a second
(field logs below). Pinning the watched pages (ds-no-0x1a 1f7d3537e) exhausts RAM on 16 GB. Changing protections of
**private** memory (VirtualAlloc) is not the dangerous kind: the 0x41202 path is the prototype-PTE path of shared pages.

## Field numbers (protect calls per second, `[native-mem]` census, 10 s windows)

| allocator (engine)              | 063159-577 | share | 050844-282 | share |
|---------------------------------|-----------:|------:|-----------:|------:|
| HeapAllocator #4 (CPU heap)     | 31267      | 53.1% | 29271      | 57.6% |
| Default TaggedArena #11 (GPU frame arena) | 20843 | 35.4% | 12842 | 25.3% |
| Default TaggedArena #10 (CPU)   | 2977       | 5.1%  | 1750       | 3.4%  |
| BucketALoading #3               | 2808       | 4.8%  | 6189       | 12.2% |
| everything else                 | 983        | 1.7%  | 791        | 1.6%  |
| **total**                       | **58878**  |       | **50843**  |       |

Three allocators are 93-95% of all protect calls. #4 and #11 are not write-only upload heaps: #4 takes
fw 4375 / fwr 3232 / **fr 7946** faults/s and 363 MiB/s of GPU write-backs; #11 takes 3444 MiB/s of write-backs
(063159). So they need write watch **and** a replacement for the read watch. BucketALoading #3 is pure CPU-write
(fr 0, wb 0): the cleanest write-only consumer, but only 5-12%.

## What the section and the alias are (evidence)

* One section for all guest memory, direct **and** flexible: `GuestBackingStore` (memoryAddressSpace.inc:135-195):
  `CreateFileMappingA(... SEC_COMMIT | SEC_RESERVE)`, plus `MapViewOfFile` of the whole of it = the emulator's alias
  (`m_backing_base`, :183).
* The game's mappings are placeholder replacements by views: `ReplacePlaceholderView` (:1727, `MapViewOfFile3`
  `MEM_REPLACE_PLACEHOLDER`). Direct memory: `KernelMapDirectMemory` -> `MapBacking` (memory.cpp:4039-4044, comment:
  "Direct mappings must remain views of the single backing object. Anonymous fallbacks break aliasing and lose
  direct-memory contents when a range is unmapped and mapped again"). Flexible memory too: `FlexibleMemory::Map`
  (memory.cpp:2651-2711) maps views at backing offsets >= `PhysicalMemory::Size()` (memory.cpp:1165, 1076-1079).
* Private memory exists already: `Commit` (memoryAddressSpace.inc:~2300, `VirtualAlloc2 MEM_REPLACE_PLACEHOLDER`,
  `MappingType::Private`) for runtime ranges (stacks etc., `AllocateGuestRuntimeMemory` memory.cpp:~4720).
* What needs the alias:
  1. The game itself, for direct memory: the same direct range can be mapped at two addresses
     (`PhysicalMemory::FindMappings` / "mapped_aliases", memory.cpp:3871-3884; `GuestBackingAliases`,
     `IsUniqueGuestBackingRange`, memory.h:175-180, used by bufferCache.cpp:227/234/878/1109), and unmapped then
     mapped again with its contents kept. Flexible memory has no physical address on PS5: the game cannot alias it.
     The census reports Demon's Souls' resident memory as "through the game's view only" (log line "GPU residency
     [by kind] ... the emulator's alias only 0 + both 0"), and the lazy-commit line reports 0-8 MiB "aliased/alias".
  2. The emulator, to read and write guest bytes regardless of the trackers' protection: `TryGetBackingPointer`,
     `TryGetBackingPieces`, `TryReadBacking*`, `WriteBacking` (memory.h:119-198; 16 call sites in bufferCache.cpp,
     8 in textureCache.cpp, 5 nativeTex.cpp, plus fresh-memory.h, writeback-verify.h, rt-local-lights.cpp...). This
     use exists **only because** the guest view may be no-access/read-only. Under write watch the view stays
     writable, so the guest address itself serves; GPU write-backs then show up as writes (see risks).
  3. Diagnostics/trim: `FlushTrim` (memory.cpp:~1975), `BackingView` (fresh-memory.h:56).
* The GPU does **not** need the alias: no `VK_EXT_external_memory_host` import anywhere in src/ (grep:
  `ImportMemoryHostPointer`, `eHostAllocationEXT`: none). Every upload is a copy into Vulkan memory.

## Can guest memory be private MEM_WRITE_WATCH memory?

* MEM_WRITE_WATCH works only on private memory, chosen at reserve time. Guest memory sits in placeholders, so it must
  come from `VirtualAlloc2(MEM_RESERVE|MEM_COMMIT|MEM_WRITE_WATCH|MEM_REPLACE_PLACEHOLDER)`. Whether this Windows build
  accepts that combination is the first field question: stage 2a probes it at start-up (`[write-watch] probe`).
* **Flexible memory: yes**, no alias needed by the game. On unmap its contents are legitimately gone; on map it is
  fresh zero memory (what `FlexibleMemory::Map` does now with `ZeroBacking`). The emulator's alias readers must take
  the guest address for such ranges (`BackingPointer` returns the vaddr when the mapping is private; it is never
  no-access under write watch).
* **Direct memory that is mapped once and never aliased: yes, with care.** Unmap must copy the bytes into the section
  at their backing offset first (no live game reference: the range is being unmapped), and a later second mapping of
  the same physical range must be refused-or-converted. Conversion while the game runs is not safe; so direct memory
  goes private only on an explicit allow-list (the engine's heap runs, found by the census) after the field log shows
  `aliased` stays 0 for them. Never moved after the game has a pointer to it.
* Direct memory the game aliases or remaps: stays on the section (protection tracking, small volume).

## Consumers

* **Write detection** (page manager write watchers, buffer/texture caches, upload heaps, `UploadHeap` kept pages):
  instead of write-protect -> fault -> unprotect, the GPU thread polls `GetWriteWatch(WRITE_WATCH_FLAG_RESET)` over the
  watched private runs at the points where guest writes become visible to GPU work - `BdaDirtyRegions::Publish`
  (submission taken, WAIT_REG_MEM satisfied; uploadHeap.h:30-35) and the flip - and marks the returned pages CPU-dirty
  exactly as a write fault does today (`UpdateRegionWatchers<track,false>` pageManager.cpp:~350). Poll cost scales
  with the range watched (probe gives us/GiB); only the runs that hold watched pages are polled.
  `ResetWriteWatch` after the emulator's own writes into the range (write-backs), under the region lock.
* **Read detection** (fault on a CPU read of GPU-written data: `NO_ACCESS_PROTECTION`, pageManager.cpp:64/164/293):
  write watch cannot see reads. Replacements: (a) eager write-back at the sync point that makes the GPU result
  visible (end of the command buffer that wrote it, a fence/label the CPU waits on) - the path `eager_feedback`
  already has; (b) for textures, NativeTexturePC-style: GPU-only, never written back, explicit read on the rare CPU
  read (level 2 already keeps 3 GiB "guest no-access" for textures without per-page churn). For #4/#11 the field
  shows 363 / 3444 MiB/s write-backs: eager write-back of all of it is too much; it needs the readback to be
  limited to ranges the CPU reads (the `fr` census per allocator shows #11 has fr 0: its write-backs can be dropped
  or made lazy-at-sync; #4 has fr 7946/s and needs eager write-back of its GPU-written pages).
  Read protections that remain land on **private** memory (no 0x41202 risk).

## Order of the stages

1. (this) Design + measurement.
2. a. Classify every page-manager protect (section direct / section flexible / private), log per 10 s, probe
      MEM_WRITE_WATCH on placeholders and GetWriteWatch cost. No behaviour change. **Committed.**
   b. Flexible memory as private MEM_WRITE_WATCH memory (alias readers redirected to the vaddr for private
      mappings), write watchers on those ranges polled instead of protected. Moves exactly the "flexible" share the
      2a log shows.
   c. BucketALoading #3 and TaggedArena #10 (write-only, fr ~0): allow-listed direct runs to private + write watch.
   d. #11 (fr 0, write-backs only): private + write watch, write-backs through the vaddr + ResetWriteWatch.
   e. #4: private + write watch + eager write-back of GPU-written pages at sync points; its read watch stays as a
      protection on private memory.
3. Section protect calls -> 0 for the hot set; pinning becomes unnecessary.

## Risks

* `MEM_WRITE_WATCH` with `MEM_REPLACE_PLACEHOLDER` may be refused on some builds (probe says; fallback: keep section).
* Write watch reports the emulator's own writes (write-backs, zeroing) as guest writes: either ResetWriteWatch after
  them (a guest write in the same instant is lost -> must be done under the same lock that orders write-backs today)
  or accept a redundant upload.
* Private memory is charged to commit like the page-file section is; no RAM growth, but unmap of direct memory
  copies its bytes back into the section (CPU cost at unmap only).
* GetWriteWatch over large ranges every publish: cost must stay well under the faults it replaces (today 5.4 s of
  commits and ~150k faults per 5 s window).
