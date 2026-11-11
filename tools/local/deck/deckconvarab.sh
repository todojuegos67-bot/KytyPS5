#!/bin/bash
# deckconvarab.sh WORLDS "label=convars" ...: one boot per variant (game convars take effect at boot): 1-1 boot (360p,
# no projected decals), the game's own world changes to WORLDS in turn ("3 1": Latria; "2 1,5 1": Stonefang then
# Valley), each 60 s settled, 15 s measured; one line a variant and world (fps, render ms, render busy, table draws;
# the driver's GPU busy % over 5 s at the last world). DECK_SET="NAME=VALUE ..." adds emulator variables to every
# boot. Run on the Deck in ~/KytyPS5.
world=$1
shift
for variant in "$@"; do
	label=${variant%%=*}
	convars=${variant#*=}
	python3 deckbench.py stop >/dev/null 2>&1
	sleep 3
	set=(--set KYTY_RENDER_HEIGHT=360 --set "KYTY_GAME_CONVARS=r_projectedDecalEnable=false${convars:+;$convars}")
	for extra in $DECK_SET; do set+=(--set "$extra"); done
	python3 deckbench.py run --label "cv-$label" --keep --settle 30 --measure 3 "${set[@]}" >/dev/null 2>&1
	python3 decksoak.py --cycles 1 --worlds "$world" --settle 60 --stay 15 >/dev/null 2>&1
	gpu=$(for i in $(seq 50); do head -1 /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1; sleep 0.1; done | awk '{s+=$1} END {if (NR) printf "%.0f", s/NR}')
	f=$(ls -t logs/*cv-$label.log | head -1)
	n=0
	IFS=, read -ra worlds <<< "$world"
	for w in "${worlds[@]}"; do
		m=$(grep -a "LIVE_MEASURE.*label=soak$n " "$f" | grep -oE "fps=[0-9.]+|render_ms=[0-9.]+|render_busy=[0-9.%]+|p99_ms=[0-9.]+" | paste -sd" ")
		c=$(grep -a "LIVE_COUNTERS.*label=soak$n " "$f" | grep -oE "table_draws=[0-9.]+" | paste -sd" ")
		echo "$label [$convars] world $w: $m $c${gpu:+ gpu_busy(last world)=$gpu%}"
		n=$((n + 1))
	done
done
python3 deckbench.py stop >/dev/null 2>&1
