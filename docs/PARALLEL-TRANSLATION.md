# Parallel translation of guest command buffers (KYTY_SPECULATE=4)

The graphics queue's command buffers are translated ahead on worker threads (KYTY_SPECULATE_THREADS) and committed in
order on the GPU thread, each chunk checked against what ran before it (write stamps, the state it started from); a
chunk whose assumptions no longer hold is translated the normal way. With KYTY_SPECULATE unset (mode 0) the GPU thread
runs upstream's path. The October design notes (sections 1-10 of the document db18e344 removed) are in the history.

## 1. Against upstream's scout (TOUR3, 4 Nov)

One build of this branch rebased on upstream's tip (2dd7a061), four configurations, one boot each over the tour's
stops, one 10 s untraced measure a stop (fps / render_ms; one boot's spread is up to ~6 fps). (1) upstream as shipped:
mode 0, KYTY_SCOUT=0, KYTY_WRITE_GUARD=0, this branch's mode-0 levers off; (2) the same with KYTY_SCOUT=1; (3) this
branch in mode 0; (4) this branch in mode 4 (KYTY_SPECULATE=4, 8 threads, queue 49 speculated). (3) and (4) ran with
KYTY_TABLE_TARGET_READ=1. (2) and (4) were measured twice at the five stops the first run left inside the spread.

| stop | (1) upstream mode 0 | (2) upstream + scout | (3) branch mode 0 | (4) branch mode 4 | (4) against (2) |
|---|---|---|---|---|---|
| the stone arch | 48.1 / 20.53 | 44.4 / 21.96 | 49.2 / 19.79 | 54.4 / 15.85 | +10.0: wins |
| 1-1 | 54.4 / 17.87 | 61.5, 58.8 / 15.85, 16.85 | 60.5 / 16.12 | 63.8, 64.6 / 13.13, 12.58 | +4.1: wins |
| 3-1 | 59.4 / 16.55 | 62.3 / 15.78 | 60.9 / 15.62 | 69.6 / 12.08 | +7.3: wins |
| 2-1 | 89.6 / 10.74 | 89.9, 89.8 / 10.38, 10.77 | 87.7 / 9.96 | 91.3, 88.8 / 8.92, 8.99 | +0.2: tie |
| 4-1 | 69.7 / 13.74 | 72.1, 69.9 / 13.57, 14.17 | 73.9 / 13.13 | 75.6, 75.3 / 10.19, 10.56 | +4.5: wins |
| 5-1 | 82.6 / 11.35 | 84.9, 84.5 / 11.06, 11.54 | 81.5 / 10.68 | 87.2, 86.1 / 9.12, 9.47 | +2.0: tie |
| the Nexus | 86.2 / 9.34 | 86.9, 83.3 / 10.16, 11.12 | 85.0 / 8.90 | 81.9, 82.4 / 9.31, 9.23 | -3.0: loses |

Mode 4's render thread spends less time a frame at every stop; where the frame is the render thread's (the arch,
3-1, 1-1, 4-1) that is the frame's gain. At the Nexus the frame is not the render thread's (73 % busy in mode 4 against 82 % for
the scout) and mode 4 reads lower; whose time it is there is being read.

## 2. Every switch of the branch

The switches this branch adds or whose default it changes (upstream's own switches are not listed). A live switch is
read once by its first use from the environment variable KYTY_<NAME> (live-switch.h) and can be changed while running
with the live command `set <name> <value>`. Mode 0 / mode 4: the default under KYTY_SPECULATE=0 / 4; "-" where the
switch acts only in mode 4. A verify switch N checks every Nth use against the path it replaces and counts a mismatch.

| switch | what it does | mode 0 | mode 4 | verify |
|---|---|---|---|---|
| **Speculation** | | | | |
| KYTY_SPECULATE (0 or 4) | 4: graphics command buffers translated ahead on worker threads and committed in order | 0 | 4 | |
| live `mode 0\|4` (live-control.h) | KYTY_SPECULATE's mode changed while the game runs, between submissions; every mode-defaulted switch from one table (local/mode-toggle.h), a variable set wins | - | - | |
| KYTY_SPECULATE_THREADS (1..12) | the worker threads (our runs: 8) | - | 2 | |
| KYTY_SPECULATE_ASYNC | the compute queue speculated beside the graphics queue (0: none) | - | 49 | |
| KYTY_SPECULATE_ASYNC_THREADS | that queue's threads | - | 2 | |
| KYTY_MIMALLOC (Windows) | the emulator's operator new/delete from mimalloc; fixed at process start from the environment (a boot in mode 0 switched live to 4 keeps the CRT heap, so a heap A/B is two boots) | 0 | 1 | |
| KYTY_RENDER_CPUS | CPUs the workers avoid (the render thread's) | - | unset | |
| spec_chunk_in_order | a chunk's translation starts from the state pass's state at its start | - | 1 | |
| spec_late_reads | a read of memory a pending write of an unlinked chain covered is checked at commit | - | 1 | |
| spec_start_instance_free | the start-instance SGPR left out of a chunk's start-state comparison | - | 1 | |
| spec_predict_dcc | a DCC surface's clear predicted on a worker | - | 1 | |
| spec_touch_batch | a committed segment's image LRU touches under one texture-cache lock | - | 1 | |
| spec_draw_indirect | DRAW_INDIRECT as a table draw, not a hole | - | 1 | spec_draw_indirect_verify N |
| spec_read_one_set | a worker's clear read costs one search of one journal set | - | 1 | spec_read_one_set_verify N |
| spec_walk_from_stamp | the clean-read check's log walk tries the read's latest stamp first | 1 | 1 | spec_commit_verify N |
| speculate_cancel_us | how long the render thread waits on a worker's chunk before taking the rest itself | - | 1000000 (always) | |
| speculate_copy_runs | a journaled read inside one copy is one read | - | 1 | |
| speculate_big_filter | big pending writes kept as ranges in the granule filter | - | 1 | |
| speculate_stamps | the commit's write check through the log's stamps, not a range set | - | 1 | |
| speculate_write_parts | a host write logged by the parts that change | - | 1 | |
| speculate_texel_sync | texel buffers over GPU-written images synced for a worker | - | 1 | |
| speculate_batch_submits | committed buffers submitted in one vkQueueSubmit | - | 1 | |
| speculate_flush_every | a commit's held buffers flushed every Nth commit | - | 8 | |
| speculate_keep_pass | a segment that recorded nothing keeps the open render pass | - | 1 | |
| serve_in_waits (changed) | guest commands served while the render thread waits (bits); mode 4 adds 2 and 4 | 1 | 7 | |
| **The write guard** | | | | |
| KYTY_WRITE_GUARD (changed in mode 4) | a guest write lands at once, hidden from the game until older uploads are done (off in mode 0 as upstream: the protection churn behind #5 / #50) | 0 | 1 | |
| write_guard_spec | the guard in mode 4 too | - | 1 | |
| write_guard_lockfree | readers check a counting filter before the guard's lock | 1 | 1 | |
| write_guard_fast | the render thread's own table of the pages it guarded | 0 | 0 | |
| **The buffer cache** | | | | |
| eager_feedback (upstream, #300; off in mode 0 as upstream, cfbf1148) | completed copy-feedback snapshots written back before the game faults on them | 0 | 1 | eager_feedback_verify N |
| feedback_batch (upstream, #301; off in mode 0 as upstream, cfbf1148) | copy-feedback snapshots copied together as the open buffer closes | 0 | 1 | |
| **Compute** | | | | |
| fill_buffer_clears | an image-clear dispatch over plain buffer memory made as a transfer fill | 0 | 1 | |
| **The table path** | | | | |
| KYTY_TABLE_TARGET_READ (changed) | a table image set may hold a target its shaders sample | 0 | 1 | |
| table_serials (upstream, #302; off in mode 0 as upstream, cfbf1148) | a table draw's target and set checks skipped while the images' serial sums hold | 0 | 1 | table_serials 2 |
| table_memo (upstream, #303; off in mode 0 as upstream, cfbf1148) | a table draw reuses its pair's last variant, state and set while nothing they depend on moved | 0 | 1 | table_memo_verify N |
