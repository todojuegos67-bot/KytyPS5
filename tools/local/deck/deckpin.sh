#!/usr/bin/env bash
# deckpin.sh <render cpus> <other cpus>: CPU affinity of the running kyty_emulator's threads, live (taskset): the render
# thread (the busiest Kyty.Gpu) on <render cpus>, every other thread on <other cpus>. `deckpin.sh reset` restores the
# start-up masks (render 1-7, the rest 0-7).
set -euo pipefail
pid=$(pgrep -x kyty_emulator | head -1)
[ -n "$pid" ] || { echo "no kyty_emulator"; exit 1; }
render_cpus="${1:-}" other_cpus="${2:-}"
[ "$render_cpus" != reset ] || { render_cpus=1-7; other_cpus=0-7; }
render=$(for t in /proc/$pid/task/*; do [ "$(cat $t/comm)" = Kyty.Gpu ] && echo "$(awk '{print $14}' $t/stat) $(basename $t)"; done | sort -n | tail -1 | cut -d' ' -f2)
count=0
for t in /proc/$pid/task/*; do
	tid=$(basename $t)
	if [ "$tid" = "$render" ]; then taskset -pc "$render_cpus" "$tid" > /dev/null; else taskset -pc "$other_cpus" "$tid" > /dev/null 2>&1 || true; fi
	count=$((count + 1))
done
echo "render thread $render on $render_cpus, $((count - 1)) other threads on $other_cpus"
