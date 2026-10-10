param([string]$Label = 'nexus', [double]$Seconds = 20, [int]$IntervalMs = 250)
# One measurement of the running game (portable package or source tree): per-thread CPU (threads.ps1),
# GPU load and video memory (nvidia-smi), video memory per process (Windows GPU counters: who else holds
# VRAM), captures of the game image every 250 ms (capture.ps1, named by unix ms like the SLOW Frame lines)
# and a copy of the current run log. Everything goes to _measure\<label>-<time>\ next to the emulator.
#   powershell -ExecutionPolicy Bypass -File tools\local\windows\nexus-measure.ps1 -Label nexus-walk -Seconds 20
$S = $PSScriptRoot
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$out = Join-Path $root ("_measure\{0}-{1}" -f $Label, (Get-Date -Format 'HHmmss'))
New-Item -ItemType Directory -Force $out | Out-Null
$p = Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 1 } | Select-Object -First 1
if (!$p) { 'emulator not running'; exit 1 }
"emulator pid $($p.Id), started $($p.StartTime), output $out"

# Video memory per process before the run (dedicated MiB).
function GpuMem {
	try {
		(Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples |
			ForEach-Object { if ($_.InstanceName -match 'pid_(\d+)_') { [pscustomobject]@{ Pid = [int]$Matches[1]; MiB = $_.CookedValue / 1MB } } } |
			Group-Object Pid | ForEach-Object {
				$id = [int]$_.Name; $name = (Get-Process -Id $id -ErrorAction SilentlyContinue).ProcessName
				[pscustomobject]@{ MiB = [math]::Round(($_.Group | Measure-Object MiB -Sum).Sum); Pid = $id; Name = $name }
			} | Where-Object MiB -ge 32 | Sort-Object MiB -Descending
	} catch { "GPU counters unavailable: $_" }
}
GpuMem | Format-Table -AutoSize | Out-String -Width 200 | Set-Content "$out\vram-before.txt"

$smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
if (!$smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = "$env:SystemRoot\System32\nvidia-smi.exe" }
$gpuJob = $null
if ($smi) {
	$gpuJob = Start-Process -FilePath $smi -PassThru -WindowStyle Hidden -RedirectStandardOutput "$out\gpu.csv" -ArgumentList @(
		'--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.gr,clocks.mem,power.draw,pstate,temperature.gpu,clocks_throttle_reasons.active',
		'--format=csv', "-lms", "$IntervalMs")
}
# GPU engine use of the emulator process (3D, copy, compute), sampled with the run.
$engineJob = Start-Job -ArgumentList $p.Id, $Seconds -ScriptBlock {
	param($id, $seconds)
	$n = [math]::Max(1, [int]$seconds)
	try {
		Get-Counter "\GPU Engine(pid_${id}_*)\Utilization Percentage" -SampleInterval 1 -MaxSamples $n -ErrorAction Stop | ForEach-Object {
			$t = $_.Timestamp.ToString('HH:mm:ss')
			$_.CounterSamples | Where-Object CookedValue -gt 0.5 | ForEach-Object {
				$engine = if ($_.InstanceName -match 'engtype_(\w+)') { $Matches[1] } else { $_.InstanceName }
				"{0} {1,-12} {2,6:N1}%" -f $t, $engine, $_.CookedValue
			}
		}
	} catch { "GPU engine counters unavailable: $_" }
}
$capJob = Start-Job -ArgumentList "$S\capture.ps1", "$out\shots", $Seconds -ScriptBlock {
	param($script, $dir, $seconds) & powershell -NoProfile -ExecutionPolicy Bypass -File $script -Out $dir -Seconds $seconds -IntervalMs 250 -Width 960
}
$startMs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
& powershell -NoProfile -ExecutionPolicy Bypass -File "$S\threads.ps1" -Seconds $Seconds -Top 40 -Out "$out\threads.txt" | Out-Null
$endMs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
Wait-Job $capJob, $engineJob -Timeout 30 | Out-Null
Receive-Job $capJob | Set-Content "$out\capture.txt"
Receive-Job $engineJob | Set-Content "$out\gpu-engines.txt"
if ($gpuJob) { Stop-Process -Id $gpuJob.Id -ErrorAction SilentlyContinue }
GpuMem | Format-Table -AutoSize | Out-String -Width 200 | Set-Content "$out\vram-after.txt"

# The run log of this process (the newest in logs\) and the window title.
$log = Get-ChildItem "$root\logs\*.out.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (!$log) { $log = Get-ChildItem "$root\_Build\run-logs\*.out.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1 }
if ($log) { Copy-Item $log.FullName "$out\run.out.log" }
$p.Refresh()
@("label=$Label", "start_unix_ms=$startMs", "end_unix_ms=$endMs", "pid=$($p.Id)", "title=$($p.MainWindowTitle)",
  "log=$($log.FullName)") | Set-Content "$out\info.txt"
Get-Content "$out\threads.txt" | Select-Object -First 12
"done: $out"
