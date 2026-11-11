using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;

namespace KytyLauncher {

// The Explorer-style folder dialog (IFileOpenDialog with FOS_PICKFOLDERS) instead of the old folder tree.
static class FolderPicker {
	public static string Pick(Window owner, string title, string initial) {
		IFileOpenDialog dialog = null;
		try {
			dialog = (IFileOpenDialog)new FileOpenDialogClass();
			uint options;
			dialog.GetOptions(out options);
			dialog.SetOptions(options | 0x20 | 0x40 | 0x800); // FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST
			dialog.SetTitle(title);
			if (!string.IsNullOrEmpty(initial) && Directory.Exists(initial)) {
				var shellItem = typeof(IShellItem).GUID;
				IShellItem folder;
				if (SHCreateItemFromParsingName(initial, IntPtr.Zero, ref shellItem, out folder) == 0 && folder != null) dialog.SetFolder(folder);
			}
			var parent = owner == null ? IntPtr.Zero : new WindowInteropHelper(owner).Handle;
			if (dialog.Show(parent) != 0) return null; // (cancelled)
			IShellItem result;
			dialog.GetResult(out result);
			string path;
			result.GetDisplayName(0x80058000, out path); // SIGDN_FILESYSPATH
			return path;
		} catch (Exception) {
			// (No COM dialog: WinForms' folder tree.)
			using (var fallback = new System.Windows.Forms.FolderBrowserDialog { Description = title, SelectedPath = initial ?? "" }) {
				return fallback.ShowDialog() == System.Windows.Forms.DialogResult.OK ? fallback.SelectedPath : null;
			}
		} finally {
			if (dialog != null) Marshal.ReleaseComObject(dialog);
		}
	}

	[DllImport("shell32.dll", CharSet = CharSet.Unicode)]
	static extern int SHCreateItemFromParsingName(string path, IntPtr bindContext, ref Guid riid, out IShellItem item);

	[ComImport, Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")]
	class FileOpenDialogClass {
	}

	[ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
	interface IFileOpenDialog {
		[PreserveSig] int Show(IntPtr parent);
		void SetFileTypes(uint count, IntPtr filters);
		void SetFileTypeIndex(uint index);
		void GetFileTypeIndex(out uint index);
		void Advise(IntPtr events, out uint cookie);
		void Unadvise(uint cookie);
		void SetOptions(uint options);
		void GetOptions(out uint options);
		void SetDefaultFolder(IShellItem item);
		void SetFolder(IShellItem item);
		void GetFolder(out IShellItem item);
		void GetCurrentSelection(out IShellItem item);
		void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
		void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
		void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
		void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
		void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
		void GetResult(out IShellItem item);
		void AddPlace(IShellItem item, int alignment);
		void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string extension);
		void Close(int result);
		void SetClientGuid(ref Guid guid);
		void ClearClientData();
		void SetFilter(IntPtr filter);
		void GetResults(out IntPtr items);
		void GetSelectedItems(out IntPtr items);
	}

	[ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
	interface IShellItem {
		void BindToHandler(IntPtr bindContext, ref Guid handler, ref Guid riid, out IntPtr result);
		void GetParent(out IShellItem parent);
		void GetDisplayName(uint form, [MarshalAs(UnmanagedType.LPWStr)] out string name);
		void GetAttributes(uint mask, out uint attributes);
		void Compare(IShellItem item, uint hint, out int order);
	}
}

}
