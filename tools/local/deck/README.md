# Steam Deck development loop

Build on the Windows PC (WSL), run and measure on the Deck over SSH. The Deck build is `KYTY_STEAM_DECK=1`
(CMake `-DKYTY_STEAM_DECK=ON`, `-march=znver2`); its banner says `Release (Steam Deck)`.

One-time setup:
- WSL (Ubuntu 26.04: glibc 2.43 and libstdc++ 6.0.35, as SteamOS 3.9) with the packages of `build-linux.yml`, and a
  clone of this repository at `~/kyty-deck` (`git clone /mnt/c/.../KytyPS5-pr500 ~/kyty-deck`).
- On the Deck: SSH on (`passwd`, `sudo systemctl start sshd`), and a key: `ssh-keygen -f ~/.ssh/kyty_deck`, its `.pub`
  appended to the Deck's `~/.ssh/authorized_keys` (the owner types the password once); the key copied into WSL too.
- The game as `~/Games/PPSA01341-app0.zar` and the release in `~/KytyPS5` (its `_PipelineCache` is shared).
- To play: `run-deck.sh` (this folder; the Linux package carries it too) starts the game with the Deck's settings; the
  players' guide is `docs/STEAM-DECK.md`.

The loop (`tools/local/deck/deck.sh`, from Git Bash):

```
deck.sh build                      # this checkout (HEAD + uncommitted + untracked) into WSL, build-deck.sh
deck.sh deploy                     # kyty_emulator, translator-id, deckbench.py, the 1-1 bench save -> ~/KytyPS5
deck.sh run --label x --settle 240 --measure 20 --spin 10 --prof 10 --tracem 3 --census 5 --set NAME=VALUE
deck-series.sh "a|options" "b|options"   # one line per run
deck.sh prof x                     # CPU per thread, module and function (process samples, libraries symbolized)
deck.sh marks x [--queue q49]      # GPU ms per frame per shader, by guest queue; barriers by the function recording them
deck.sh trace x                    # a --trace / --tracew timeline: frames, render-thread gaps and what ended them
deck.sh census x --kind 13         # render-thread census; the code that recorded each barrier
deck.sh run --label k --keep ...   # leave the game running, then:
deck.sh ab "a|+r_bloomEnable=false" "b|!switch=0"   # same-boot steps: engine convars (+) and live switches (!)
deck.sh ssh 'python3 ~/KytyPS5/decklive.py "capture /home/deck/KytyPS5/prof/cap 1"'  # any live command
```

The Deck's clocks (`deck-perf.sh`, run with sudo on the Deck by its owner): `on` holds the GPU at 1600 MHz (as Game
Mode's Manual GPU Clock), `gpu <MHz>|auto`, `tdp <W>` (3-15), `off`, `status`. With the GPU clock automatic its DVFS
turns GPU savings into a lower clock (same fps): measure GPU work with it held. `gpumetrics.py` reads the APU's SMU
limits and power by rail; every run reports them (`limits`: the share of samples each limit held the clocks).

A run boots in the Deck's desktop session (`systemd-run --user`), enters the game with the scripted pad, settles,
measures (fps, render thread, and the Deck's GPU busy/clock, power, CPU clocks, memory, swap) and stops. The bench save
goes into `~/KytyPS5/_SaveData` only. Warm up first: a new pipeline compiles for up to ~1.5 s on the Deck, and the
optimized builds queue on two threads; `--settle 240` lets one run drain them, later runs start warm.
`KYTY_PREFETCH_CACHE_BUILD` (set from `translator-id`, a hash of the shader translator's sources) keeps the translated
shaders across builds that leave the translator alone.
