using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;
using Microsoft.Win32;

namespace KytyLauncher {

// What the launcher knows about this PC and the emulator's folder, found once at start.
sealed class LauncherContext {
	public readonly string Root;          // the folder of run-windows.ps1 (the package, or the source tree)
	public readonly string EmulatorExe;   // kyty_emulator.exe (in the package, or _Build\windows of the source tree)
	public readonly string PrecompileTool;
	public readonly Settings Settings;
	public readonly long EfficiencyMask;  // the efficiency cores of a hybrid CPU as an affinity mask, or 0
	public readonly List<GpuInfo> Gpus;
	public readonly bool RtxGpu;          // DLSS (the upscaler and frame generation) needs an NVIDIA RTX graphics card
	public readonly bool AmdGpu;          // an AMD Radeon graphics card (AMD Anti-Lag)
	public readonly ulong RamBytes;
	public readonly int DisplayHeight;    // the primary display's height in pixels (Game output: Automatic), or 0
	public readonly bool FsrAvailable;    // AMD's signed FSR runtime next to the emulator (FSR frame generation)
	public string Streamline { get; private set; } // NVIDIA's signed Streamline runtime folder, or null
	public bool FrameGenAvailable { get { return Streamline != null && RtxGpu; } }

	public LauncherContext() {
		Root = FindRoot();
		EmulatorExe = File.Exists(Path.Combine(Root, "kyty_emulator.exe")) ? Path.Combine(Root, "kyty_emulator.exe")
			: Path.Combine(Root, "_Build", "windows", "kyty_emulator.exe");
		PrecompileTool = new[] { Path.Combine(Root, "kyty_shader_precompile.exe"), Path.Combine(Root, "_Build", "windows", "kyty_shader_precompile.exe") }
			.FirstOrDefault(File.Exists);
		Settings = new Settings(Path.Combine(Root, "launcher-settings.json"));
		var saved = Settings.Str("ui_language");
		Strings.Language = saved == "zh" ? UiLanguage.Chinese : saved == "en" ? UiLanguage.English : Strings.SystemLanguage();
		EfficiencyMask = Native.EfficiencyMask();
		Gpus = GpuInfo.All();
		RtxGpu = Gpus.Any(g => g.Name.IndexOf("NVIDIA", StringComparison.OrdinalIgnoreCase) >= 0 && HasWord(g.Name, "RTX"));
		AmdGpu = Gpus.Any(g => g.Name.IndexOf("AMD", StringComparison.OrdinalIgnoreCase) >= 0 || g.Name.IndexOf("Radeon", StringComparison.OrdinalIgnoreCase) >= 0);
		RamBytes = Native.TotalRam();
		DisplayHeight = Native.DisplayHeight();
		Streamline = ResolveStreamline();
		// AMD's runtime next to the emulator, signed by AMD: FSR SDK 2's loader with its frame generation DLL (FSR 4 frame
		// generation on RX 9000 cards, FSR 3.1 on others), or FidelityFX SDK 1.1.4's amd_fidelityfx_vk.dll (FSR 3.1).
		var folder = Path.GetDirectoryName(EmulatorExe) ?? Root;
		Func<string, bool> amd = name => Signature.SignedBy(Path.Combine(folder, name), "Advanced Micro Devices");
		FsrAvailable = (amd("amd_fidelityfx_loader_vk.dll") && amd("amd_fidelityfx_framegeneration_vk.dll")) ||
		               amd("amd_fidelityfx_vk.dll");
	}

	// The launcher sits next to run-windows.ps1 in a package; in the source tree it is built to _Build\launcher.
	static string FindRoot() {
		var start = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? ".";
		for (var dir = new DirectoryInfo(start); dir != null; dir = dir.Parent) {
			if (File.Exists(Path.Combine(dir.FullName, "run-windows.ps1"))) return dir.FullName;
		}
		return start;
	}

	static bool HasWord(string text, string word) {
		var at = 0;
		while ((at = text.IndexOf(word, at, StringComparison.Ordinal)) >= 0) {
			var before = at == 0 || !char.IsLetterOrDigit(text[at - 1]);
			var after = at + word.Length >= text.Length || !char.IsLetterOrDigit(text[at + word.Length]);
			if (before && after) return true;
			at += word.Length;
		}
		return false;
	}

	// NVIDIA's Streamline runtime (no release ships it): the folder chosen in the launcher, else KYTY_STREAMLINE_DIR, else
	// a "streamline" folder next to the emulator, else the SDK beside a local build (_Build\deps\streamline\sdk). Only
	// NVIDIA's own sl.interposer.dll counts (run-windows -StreamlineDir).
	string ResolveStreamline() {
		var chosen = Settings.Str("streamline");
		if (chosen.Length != 0) return FindStreamline(chosen);
		var environment = Environment.GetEnvironmentVariable("KYTY_STREAMLINE_DIR");
		if (!string.IsNullOrEmpty(environment)) return FindStreamline(environment);
		var exeDir = Path.GetDirectoryName(EmulatorExe) ?? Root;
		var candidates = new List<string> { Path.Combine(exeDir, "streamline") };
		var parent = Path.GetDirectoryName(exeDir);
		if (parent != null) candidates.Add(Path.Combine(parent, "deps", "streamline", "sdk"));
		foreach (var dir in candidates) {
			var found = FindStreamline(dir);
			if (found != null) return found;
		}
		return null;
	}

	// The runtime's folder in `dir` (the folder itself or its bin\x64) when its sl.interposer.dll is signed by NVIDIA.
	public static string FindStreamline(string dir) {
		if (string.IsNullOrEmpty(dir)) return null;
		foreach (var folder in new[] { dir, Path.Combine(dir, "bin", "x64") }) {
			var file = Path.Combine(folder, "sl.interposer.dll");
			if (File.Exists(file)) return Signature.SignedBy(file, "NVIDIA Corporation") ? folder : null;
		}
		return null;
	}

	public bool UseStreamlineFolder(string dir) {
		var found = FindStreamline(dir);
		if (found == null) return false;
		Settings.Set("streamline", dir);
		Streamline = found;
		return true;
	}

	public string LogsFolder() {
		return Directory.Exists(Path.Combine(Root, "_Build")) ? Path.Combine(Root, "_Build", "run-logs") : Path.Combine(Root, "logs");
	}

	public string Readme() {
		return new[] { Path.Combine(Root, "README.md"), Path.Combine(Root, "docs", "PORTABLE-README.md") }.FirstOrDefault(File.Exists);
	}
}

sealed class GpuInfo {
	public string Name;
	public ulong Memory; // bytes of its own video memory, 0 when unknown

	// The display adapters Windows lists (their driver's description and dedicated memory), largest memory first.
	public static List<GpuInfo> All() {
		var list = new List<GpuInfo>();
		const string ClassKey = @"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";
		try {
			using (var display = Registry.LocalMachine.OpenSubKey(ClassKey)) {
				if (display == null) return list;
				foreach (var name in display.GetSubKeyNames()) {
					try {
						using (var adapter = display.OpenSubKey(name)) {
							var desc = adapter == null ? null : adapter.GetValue("DriverDesc") as string;
							if (string.IsNullOrEmpty(desc)) continue;
							ulong memory = 0;
							var q = adapter.GetValue("HardwareInformation.qwMemorySize");
							if (q is long) memory = (ulong)(long)q;
							else if (q is byte[] && ((byte[])q).Length >= 8) memory = BitConverter.ToUInt64((byte[])q, 0);
							if (memory == 0) {
								var d = adapter.GetValue("HardwareInformation.MemorySize");
								if (d is int) memory = (uint)(int)d;
								else if (d is byte[] && ((byte[])d).Length >= 4) memory = BitConverter.ToUInt32((byte[])d, 0);
							}
							if (list.Any(g => g.Name == desc)) continue;
							list.Add(new GpuInfo { Name = desc, Memory = memory });
						}
					} catch (Exception) {
						// (Keys without access, such as "Properties".)
					}
				}
			}
		} catch (Exception) {
		}
		return list.OrderByDescending(g => g.Memory).ToList();
	}
}

static class Signature {
	// Whether `file` has a valid Authenticode signature whose signer is organization `organization` (O=...).
	public static bool SignedBy(string file, string organization) {
		if (!File.Exists(file)) return false;
		try {
			var signer = X509Certificate.CreateFromSignedFile(file);
			if (signer.Subject.IndexOf("O=" + organization, StringComparison.OrdinalIgnoreCase) < 0) return false;
		} catch (Exception) {
			return false;
		}
		return Verify(file);
	}

	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
	struct FileInfo {
		public uint cbStruct;
		public string pcwszFilePath;
		public IntPtr hFile;
		public IntPtr pgKnownSubject;
	}

	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
	struct TrustData {
		public uint cbStruct;
		public IntPtr pPolicyCallbackData;
		public IntPtr pSIPClientData;
		public uint dwUIChoice;
		public uint fdwRevocationChecks;
		public uint dwUnionChoice;
		public IntPtr pFile;
		public uint dwStateAction;
		public IntPtr hWVTStateData;
		public IntPtr pwszURLReference;
		public uint dwProvFlags;
		public uint dwUIContext;
		public IntPtr pSignatureSettings;
	}

	[DllImport("wintrust.dll", CharSet = CharSet.Unicode)]
	static extern int WinVerifyTrust(IntPtr hwnd, ref Guid action, ref TrustData data);

	// WinVerifyTrust (what Get-AuthenticodeSignature's "Valid" means), offline: no revocation lookup.
	static bool Verify(string file) {
		var action = new Guid("00AAC56B-CD44-11d0-8CC2-00C04FC295EE"); // WINTRUST_ACTION_GENERIC_VERIFY_V2
		var info = new FileInfo { cbStruct = (uint)Marshal.SizeOf(typeof(FileInfo)), pcwszFilePath = file };
		var infoPointer = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(FileInfo)));
		try {
			Marshal.StructureToPtr(info, infoPointer, false);
			var data = new TrustData {
				cbStruct = (uint)Marshal.SizeOf(typeof(TrustData)),
				dwUIChoice = 2,             // WTD_UI_NONE
				fdwRevocationChecks = 0,    // WTD_REVOKE_NONE
				dwUnionChoice = 1,          // WTD_CHOICE_FILE
				pFile = infoPointer,
				dwStateAction = 1,          // WTD_STATEACTION_VERIFY
				dwProvFlags = 0x10 | 0x1000 // WTD_REVOCATION_CHECK_NONE | WTD_CACHE_ONLY_URL_RETRIEVAL
			};
			var result = WinVerifyTrust(IntPtr.Zero, ref action, ref data);
			data.dwStateAction = 2; // WTD_STATEACTION_CLOSE
			WinVerifyTrust(IntPtr.Zero, ref action, ref data);
			return result == 0;
		} catch (Exception) {
			return false;
		} finally {
			Marshal.DestroyStructure(infoPointer, typeof(FileInfo));
			Marshal.FreeHGlobal(infoPointer);
		}
	}
}

static class Native {
	[StructLayout(LayoutKind.Sequential)]
	struct MemoryStatus {
		public uint dwLength;
		public uint dwMemoryLoad;
		public ulong ullTotalPhys, ullAvailPhys, ullTotalPageFile, ullAvailPageFile, ullTotalVirtual, ullAvailVirtual, ullAvailExtendedVirtual;
	}

	[DllImport("kernel32.dll", SetLastError = true)]
	static extern bool GlobalMemoryStatusEx(ref MemoryStatus status);

	public static ulong TotalRam() {
		var status = new MemoryStatus { dwLength = (uint)Marshal.SizeOf(typeof(MemoryStatus)) };
		return GlobalMemoryStatusEx(ref status) ? status.ullTotalPhys : 0;
	}

	// DEVMODEW, up to the display frequency (dmSize says how much of it is here).
	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
	struct DevMode {
		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName;
		public ushort dmSpecVersion, dmDriverVersion, dmSize, dmDriverExtra;
		public uint dmFields;
		public int dmPositionX, dmPositionY;
		public uint dmDisplayOrientation, dmDisplayFixedOutput;
		public short dmColor, dmDuplex, dmYResolution, dmTTOption, dmCollate;
		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName;
		public ushort dmLogPixels;
		public uint dmBitsPerPel, dmPelsWidth, dmPelsHeight, dmDisplayFlags, dmDisplayFrequency;
	}

	[DllImport("user32.dll", CharSet = CharSet.Unicode)]
	static extern bool EnumDisplaySettingsW(string deviceName, int modeNumber, ref DevMode mode);

	// The primary display's height in pixels, its current mode's (not scaled by Windows' display scaling), or 0
	// (launcher.ps1's Get-DisplayHeight).
	public static int DisplayHeight() {
		var mode = new DevMode { dmSize = (ushort)Marshal.SizeOf(typeof(DevMode)) };
		return EnumDisplaySettingsW(null, -1, ref mode) ? (int)mode.dmPelsHeight : 0;
	}

	[DllImport("user32.dll")]
	static extern IntPtr GetForegroundWindow();

	[DllImport("user32.dll")]
	static extern bool IsWindow(IntPtr window);

	[DllImport("user32.dll")]
	static extern bool SetForegroundWindow(IntPtr window);

	[DllImport("user32.dll")]
	static extern bool IsWindowVisible(IntPtr window);

	[DllImport("user32.dll")]
	static extern bool BringWindowToTop(IntPtr window);

	[DllImport("user32.dll")]
	static extern uint GetWindowThreadProcessId(IntPtr window, out uint process);

	[DllImport("user32.dll")]
	static extern bool AttachThreadInput(uint attach, uint to, bool on);

	[DllImport("kernel32.dll")]
	static extern uint GetCurrentThreadId();

	[DllImport("user32.dll")]
	static extern IntPtr GetWindow(IntPtr window, uint command);

	[DllImport("user32.dll")]
	static extern bool GetWindowRect(IntPtr window, out Rect rect);

	delegate bool EnumWindowsProc(IntPtr window, IntPtr parameter);

	[DllImport("user32.dll")]
	static extern bool EnumWindows(EnumWindowsProc callback, IntPtr parameter);

	[StructLayout(LayoutKind.Sequential)]
	struct Rect {
		public int Left, Top, Right, Bottom;
	}

	[StructLayout(LayoutKind.Sequential)]
	struct ProcessBasicInformation {
		public IntPtr ExitStatus, PebBaseAddress, AffinityMask, BasePriority, UniqueProcessId, InheritedFromUniqueProcessId;
	}

	[DllImport("ntdll.dll")]
	static extern int NtQueryInformationProcess(IntPtr process, int informationClass, ref ProcessBasicInformation information,
		int length, out int returned);

	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
	struct ProcessEntry {
		public uint Size, Usage, ProcessId;
		public IntPtr DefaultHeapId;
		public uint ModuleId, Threads, ParentProcessId;
		public int PriorityClassBase;
		public uint Flags;
		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
		public string ExeFile;
	}

	[DllImport("kernel32.dll")]
	static extern IntPtr CreateToolhelp32Snapshot(uint flags, uint process);

	[DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
	static extern bool Process32FirstW(IntPtr snapshot, ref ProcessEntry entry);

	[DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
	static extern bool Process32NextW(IntPtr snapshot, ref ProcessEntry entry);

	[DllImport("kernel32.dll")]
	static extern bool CloseHandle(IntPtr handle);

	// The window that has the focus now, as the emulator reads it from KYTY_RETURN_FOCUS (a number).
	public static string ForegroundWindow() {
		return GetForegroundWindow().ToInt64().ToString(System.Globalization.CultureInfo.InvariantCulture);
	}

	// The process that started this one (Steam, when it starts the game), as the emulator reads it from
	// KYTY_RETURN_FOCUS_PID: its largest window takes the focus back where the one in front at the start is gone.
	public static string ParentProcess() {
		var information = new ProcessBasicInformation();
		int returned;
		var parent = NtQueryInformationProcess(System.Diagnostics.Process.GetCurrentProcess().Handle, 0, ref information,
			Marshal.SizeOf(information), out returned) == 0 ? information.InheritedFromUniqueProcessId.ToInt64() : 0;
		return parent.ToString(System.Globalization.CultureInfo.InvariantCulture);
	}

	// The largest visible top-level window of a process or of one it started (Steam's windows, Big Picture's full screen
	// one among them, belong to the steamwebhelper.exe it starts), or zero.
	static IntPtr LargestWindowOf(uint process) {
		var processes = new HashSet<uint> { process };
		var snapshot = CreateToolhelp32Snapshot(2 /* TH32CS_SNAPPROCESS */, 0);
		if (snapshot != new IntPtr(-1)) {
			var entry = new ProcessEntry { Size = (uint)Marshal.SizeOf(typeof(ProcessEntry)) };
			for (var more = Process32FirstW(snapshot, ref entry); more; more = Process32NextW(snapshot, ref entry))
				if (entry.ParentProcessId == process) processes.Add(entry.ProcessId);
			CloseHandle(snapshot);
		}
		var best = IntPtr.Zero;
		long area = 0;
		EnumWindows((window, parameter) => {
			uint owner;
			Rect rect;
			if (GetWindowThreadProcessId(window, out owner) != 0 && processes.Contains(owner) && IsWindowVisible(window) &&
			    GetWindow(window, 4 /* GW_OWNER */) == IntPtr.Zero && GetWindowRect(window, out rect)) {
				long size = (long)(rect.Right - rect.Left) * (rect.Bottom - rect.Top);
				if (size > area) {
					area = size;
					best = window;
				}
			}
			return true;
		}, IntPtr.Zero);
		return best;
	}

	// The focus to the window ForegroundWindow named, else to the largest of the process ParentProcess named. Windows
	// refuses it to a process without the focus: then with the input of the foreground window's thread attached.
	public static void Focus(string window, string parent) {
		long handle, process;
		var target = long.TryParse(window, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out handle)
			? new IntPtr(handle) : IntPtr.Zero;
		if (target == IntPtr.Zero || !IsWindow(target) || !IsWindowVisible(target))
			target = long.TryParse(parent, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out process) && process > 0
				? LargestWindowOf((uint)process) : IntPtr.Zero;
		if (target == IntPtr.Zero || GetForegroundWindow() == target) return;
		TakeForeground(target);
	}

	// `window` in front with the focus (Focus's way).
	public static void TakeForeground(IntPtr window) {
		if (SetForegroundWindow(window) && GetForegroundWindow() == window) return;
		uint ignored;
		var foreground = GetWindowThreadProcessId(GetForegroundWindow(), out ignored);
		var self = GetCurrentThreadId();
		if (foreground == 0 || foreground == self || !AttachThreadInput(self, foreground, true)) return;
		BringWindowToTop(window);
		SetForegroundWindow(window);
		AttachThreadInput(self, foreground, false);
	}

	[DllImport("user32.dll", CharSet = CharSet.Unicode)]
	static extern int GetClassNameW(IntPtr window, System.Text.StringBuilder name, int length);

	// The window in front, for the log: "window 0x..., '<class>' of <program>.exe", or "no window".
	public static string ForegroundDescription() {
		var window = GetForegroundWindow();
		if (window == IntPtr.Zero) return "no window";
		var name = new System.Text.StringBuilder(256);
		GetClassNameW(window, name, name.Capacity);
		uint process;
		GetWindowThreadProcessId(window, out process);
		string program = "?";
		try {
			using (var owner = System.Diagnostics.Process.GetProcessById((int)process)) program = owner.ProcessName + ".exe";
		} catch (Exception) {
		}
		return string.Format(System.Globalization.CultureInfo.InvariantCulture, "window 0x{0:X}, '{1}' of {2}", window.ToInt64(), name, program);
	}

	// The program name of a process (ParentProcess's number), lower case without ".exe", or "".
	public static string ProcessName(string id) {
		int process;
		if (!int.TryParse(id, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out process) || process <= 0)
			return "";
		try {
			using (var owner = System.Diagnostics.Process.GetProcessById(process)) return owner.ProcessName.ToLowerInvariant();
		} catch (Exception) {
			return "";
		}
	}

	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
	struct StartupInfo {
		public int cb;
		public string lpReserved, lpDesktop, lpTitle;
		public int dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars, dwFillAttribute, dwFlags;
		public short wShowWindow, cbReserved2;
		public IntPtr lpReserved2, hStdInput, hStdOutput, hStdError;
	}

	[StructLayout(LayoutKind.Sequential)]
	struct ProcessInformation {
		public IntPtr hProcess, hThread;
		public int dwProcessId, dwThreadId;
	}

	[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
	static extern bool CreateProcessW(string application, System.Text.StringBuilder commandLine, IntPtr processAttributes,
		IntPtr threadAttributes, bool inheritHandles, uint flags, IntPtr environment, string directory, ref StartupInfo startup,
		out ProcessInformation information);

	[DllImport("kernel32.dll")]
	static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool result);

	// `exe arguments` started outside the job this process is in: Steam counts a game as running while a process of its
	// job lasts (issue #217). Broken away from the job where it allows that, else made by WMI's provider host
	// (Win32_Process.Create), which is in no job. How it started (for the log), or null.
	public static string StartOutsideJob(string exe, string arguments) {
		const uint CreateBreakawayFromJob = 0x01000000;
		var line = "\"" + exe + "\" " + arguments;
		bool inJob;
		if (!IsProcessInJob(System.Diagnostics.Process.GetCurrentProcess().Handle, IntPtr.Zero, out inJob)) inJob = true;
		var startup = new StartupInfo { cb = Marshal.SizeOf(typeof(StartupInfo)) };
		ProcessInformation information;
		if (CreateProcessW(null, new System.Text.StringBuilder(line), IntPtr.Zero, IntPtr.Zero, false, inJob ? CreateBreakawayFromJob : 0,
		                   IntPtr.Zero, null, ref startup, out information)) {
			CloseHandle(information.hThread);
			CloseHandle(information.hProcess);
			return inJob ? "broken away from the game's job" : "started (the launcher was in no job)";
		}
		var refused = Marshal.GetLastWin32Error();
		try {
			using (var processes = new System.Management.ManagementClass("Win32_Process")) {
				var result = processes.InvokeMethod("Create", new object[] { line, null, null, null });
				if (Convert.ToInt32(result, System.Globalization.CultureInfo.InvariantCulture) == 0)
					return "started by WMI (the job refused a break-away: error " + refused + ")";
			}
		} catch (Exception) {
		}
		return null;
	}

	[DllImport("kernel32.dll", SetLastError = true)]
	static extern bool GetSystemCpuSetInformation(IntPtr information, uint length, out uint returned, IntPtr process, uint flags);

	// The efficiency cores (the lowest efficiency class of a hybrid CPU, processor group 0) as an affinity mask, or 0 on a
	// CPU of one kind of core (launcher.ps1's Get-EfficiencyMask).
	public static long EfficiencyMask() {
		try {
			uint length;
			GetSystemCpuSetInformation(IntPtr.Zero, 0, out length, IntPtr.Zero, 0);
			if (length == 0) return 0;
			var buffer = Marshal.AllocHGlobal((int)length);
			var sets = new List<KeyValuePair<int, int>>(); // (logical processor, efficiency class)
			try {
				if (!GetSystemCpuSetInformation(buffer, length, out length, IntPtr.Zero, 0)) return 0;
				for (int offset = 0; offset < length; offset += Marshal.ReadInt32(buffer, offset)) {
					if (Marshal.ReadInt32(buffer, offset + 4) != 0 || Marshal.ReadInt16(buffer, offset + 12) != 0) continue;
					sets.Add(new KeyValuePair<int, int>(Marshal.ReadByte(buffer, offset + 14), Marshal.ReadByte(buffer, offset + 18)));
					if (Marshal.ReadInt32(buffer, offset) <= 0) break;
				}
			} finally {
				Marshal.FreeHGlobal(buffer);
			}
			if (sets.Select(s => s.Value).Distinct().Count() < 2) return 0;
			var slowest = sets.Min(s => s.Value);
			long mask = 0;
			foreach (var set in sets.Where(s => s.Value == slowest && s.Key < 63)) mask |= 1L << set.Key;
			return mask;
		} catch (Exception) {
			return 0;
		}
	}
}

}
