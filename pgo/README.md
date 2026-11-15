# Profile for the released build

`PPSA03671.profdata` is the clang profile the Windows release is built with
(`-DKYTY_PGO_USE=K:/pgo/PPSA03671.profdata`, ThinLTO on, see `.github/workflows/build.yml`).
It changes which code the compiler inlines and how it lays it out. It doesn't change what the
emulator does, so no draw changes.

## What it holds

Counts per function and branch, named by function. There are no addresses and nothing about the
CPU or the GPU, so there is one x86-64 build and no build for another CPU that could reject it.
What depends on the training machine is only which code ran: paths it never took (other Vulkan
features, the instruction emulator for CPUs without a feature) count as cold and are optimized for
size. They stay correct but may be a little slower.

Only the Windows job uses it. Linux and macOS mangle names differently, so nothing would match.

## The source must be at `K:/`

clang-cl names functions in anonymous namespaces after a hash of the main file's name as given on
the command line, and CMake's Ninja generator passes absolute paths. A build from another path
loses the profile for all of those functions, some of the hottest among them. Training and CI
therefore map the checkout to the same drive:

```
subst K: "<checkout>"
cmake -S K:/ -B K:/_Build/windows ...
```

CMake warns when `KYTY_PGO_USE` is set and the source isn't at `KYTY_PGO_SOURCE_ROOT` (`K:/`).

## When it goes stale

A function whose code changed no longer matches its hash: the compiler warns
(`-Wprofile-instr-out-of-date`, or `-Wprofile-instr-missing` for a new function), gives that
function no profile, and the build goes on. The warnings aren't errors, so a stale profile never
breaks a build; it only loses gain, function by function. A build from the same tree as the
profile still gives a few of these warnings, so a handful means nothing.

Retrain when the warnings cover much of the hot code (the GPU thread: `renderer/`, `guest_gpu/`),
after a large change there, and before a release that is meant to be measured against the one
before. An upstream merge usually counts.

A newer clang reads an older profile; an older clang can't read a newer one. Retrain after moving
to a newer compiler, or at least check that CI's still reads it.

The file is about 4.5 MB and each retrain adds that much to the repository's history, so don't
retrain for every commit.

## Retraining

On one machine, one GPU run at a time, from the commit the profile is for:

1. `subst K: "<checkout>"`.
2. Configure and build from `K:/` with `-DKYTY_PGO_GENERATE=ON -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`.
3. Run the game with `KYTY_PGO_FILE=<file>.profraw` and `KYTY_PGO_DUMP_SECONDS=15`, which writes the
   counters every 15 s, so a crash or a killed process still leaves a profile. The training set:
   - a truly cold session with its precompile, walking from the jungle save;
   - a warm session standing in Trask Outpost for five minutes.
4. `llvm-profdata merge -o pgo/PPSA03671.profdata <files>.profraw`, with the `llvm-profdata` of the
   compiler that builds the release.
5. Configure and build from `K:/` with `-DKYTY_PGO_GENERATE=OFF -DKYTY_PGO_USE=K:/pgo/PPSA03671.profdata
   -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`, and compare it with ThinLTO alone from the same tree.
   Compare only boots in the same state: the title scene a boot picks changes the load later on.

Profiles from several machines can be merged the same way, which would cover more of the other
hardware's paths.
