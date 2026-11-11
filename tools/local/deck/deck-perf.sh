#!/bin/bash
# deck-perf.sh (run with sudo): the Steam Deck's GPU clock and power limit, as Game Mode's Quick Access performance
# settings set them (Manual GPU Clock, TDP Limit), for measuring the emulator. A reboot (or `off`) returns them to
# Steam's defaults (in Game Mode, Steam applies its own per-game settings again).
#   deck-perf.sh status            the settings and the APU's readings now
#   deck-perf.sh on                GPU held at 1600 MHz and the CPU's energy preference "performance"
#   deck-perf.sh off               GPU clock automatic, CPU "balance_performance", TDP limit 15 W
#   deck-perf.sh gpu <MHz>|auto    GPU held at MHz (200-1600), or automatic again
#   deck-perf.sh tdp <W>           the APU power limit in watts, both PPT limits (3-15: Steam's TDP Limit slider range)
set -euo pipefail
gpu=/sys/class/drm/card0/device
hwmon=$(ls -d $gpu/hwmon/hwmon* | head -1)

gpu_clock() {
	if [ "$1" = auto ]; then
		echo r > "$gpu/pp_od_clk_voltage" || true
		echo c > "$gpu/pp_od_clk_voltage" || true
		echo auto > "$gpu/power_dpm_force_performance_level"
		return
	fi
	if ! [ "$1" -ge 200 ] 2>/dev/null || [ "$1" -gt 1600 ]; then
		echo "GPU clock $1: 200-1600 MHz" >&2
		exit 1
	fi
	echo manual > "$gpu/power_dpm_force_performance_level"
	echo "s 0 $1" > "$gpu/pp_od_clk_voltage"
	echo "s 1 $1" > "$gpu/pp_od_clk_voltage"
	echo c > "$gpu/pp_od_clk_voltage"
}

cpu_preference() {
	for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do echo "$1" > "$f"; done
}

tdp() {
	if ! [ "$1" -ge 3 ] 2>/dev/null || [ "$1" -gt 15 ]; then
		echo "TDP limit $1: 3-15 W (the LCD model's range in Steam)" >&2
		exit 1
	fi
	echo $(($1 * 1000000)) > "$hwmon/power1_cap"
	echo $(($1 * 1000000)) > "$hwmon/power2_cap"
}

case "${1:-status}" in
on) gpu_clock 1600; cpu_preference performance ;;
off) gpu_clock auto; cpu_preference balance_performance; tdp 15 ;;
gpu) gpu_clock "${2:?MHz or auto}" ;;
tdp) tdp "${2:?watts}" ;;
status) ;;
*) sed -n '2,9p' "$0"; exit 1 ;;
esac
echo "GPU clock: $(cat "$gpu/power_dpm_force_performance_level"), range $(grep -A2 OD_SCLK "$gpu/pp_od_clk_voltage" | tail -2 | awk '{print $2}' | tr '\n' ' ')(now $(($(cat "$hwmon/freq1_input") / 1000000)) MHz)"
echo "TDP limit: $(($(cat "$hwmon/power1_cap") / 1000000)) W slow, $(($(cat "$hwmon/power2_cap") / 1000000)) W fast; drawing $(awk '{printf "%.1f", $1 / 1000000}' "$hwmon/power1_average") W"
echo "CPU energy preference: $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
