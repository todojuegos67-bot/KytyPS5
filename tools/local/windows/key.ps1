param([string]$Key = 'j', [int]$HoldMs = 150, [int]$Times = 1, [int]$GapMs = 600)
# Focus the emulator window and press a key with SendInput (scancode, as SDL reads it).
$p = Get-Process kyty_emulator -ErrorAction SilentlyContinue | Where-Object MainWindowHandle -ne 0 | Select-Object -First 1
if (!$p) { "emulator not running"; exit 1 }
Add-Type @"
using System; using System.Runtime.InteropServices;
public class K {
  [StructLayout(LayoutKind.Sequential)] public struct KEYBDINPUT { public ushort wVk, wScan; public uint dwFlags, time; public IntPtr dwExtraInfo; }
  [StructLayout(LayoutKind.Explicit, Size=40)] public struct INPUT { [FieldOffset(0)] public uint type; [FieldOffset(8)] public KEYBDINPUT ki; }
  [DllImport("user32.dll")] public static extern uint SendInput(uint n, INPUT[] i, int size);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern uint MapVirtualKey(uint code, uint type);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
  [DllImport("imm32.dll")] public static extern IntPtr ImmGetDefaultIMEWnd(IntPtr h);
  [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr h, uint msg, IntPtr w, IntPtr l);
  [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, IntPtr extra);
  public static void Send(ushort vk, bool up, bool ext) {
    var i = new INPUT[1]; i[0].type = 1; i[0].ki.wScan = (ushort)MapVirtualKey(vk, 0);
    i[0].ki.dwFlags = 8u | (up ? 2u : 0u) | (ext ? 1u : 0u);
    SendInput(1, i, Marshal.SizeOf(typeof(INPUT)));
  }
}
"@
$vk = @{ up = 0x26; down = 0x28; left = 0x25; right = 0x27; enter = 0x0D; esc = 0x1B; space = 0x20; f8 = 0x77; f9 = 0x78; f10 = 0x79; lctrl = 0xA2; lshift = 0xA0 }
$code = if ($vk.ContainsKey($Key)) { $vk[$Key] } else { [int][char]$Key.ToUpper() }
$ext = $Key -in @('up', 'down', 'left', 'right')
# Windows only lets a process take the foreground right after input: tap Alt first.
if ([K]::IsIconic($p.MainWindowHandle)) { [K]::ShowWindow($p.MainWindowHandle, 9) | Out-Null } # SW_RESTORE
if ([K]::GetForegroundWindow() -ne $p.MainWindowHandle) {
	[K]::keybd_event(0x12, 0, 0, [IntPtr]::Zero); [K]::keybd_event(0x12, 0, 2, [IntPtr]::Zero)
	[K]::SetForegroundWindow($p.MainWindowHandle) | Out-Null
	Start-Sleep -Milliseconds 300
}
# Never type into another window.
if ([K]::GetForegroundWindow() -ne $p.MainWindowHandle) { "emulator window is not in the foreground; no key sent"; exit 1 }
# An open IME (Japanese/Chinese input) swallows letter keys: close it for this window.
$ime = [K]::ImmGetDefaultIMEWnd($p.MainWindowHandle)
if ($ime -ne [IntPtr]::Zero) { [K]::SendMessage($ime, 0x283, [IntPtr]6, [IntPtr]::Zero) | Out-Null }
for ($n = 0; $n -lt $Times; $n++) {
	[K]::Send([uint16]$code, $false, $ext); Start-Sleep -Milliseconds $HoldMs; [K]::Send([uint16]$code, $true, $ext)
	if ($n + 1 -lt $Times) { Start-Sleep -Milliseconds $GapMs }
}
"sent $Key x$Times"
