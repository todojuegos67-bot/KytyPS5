// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;

namespace KytyLauncher {

// Optional artwork only. Never edits launch.json, bindings, saves or shader caches.
static class KeyboardPrompts {
	const string ResourcePrefix = "KytyLauncher.KeyboardPrompts.";
	const string BackupName = ".kyty-keyboard-prompts-v1";
	sealed class Icon {
		public string Name, OriginalHash, ReplacementHash;
		public byte[] Replacement;
	}
	static string Hash(byte[] data) {
		using (var sha = SHA256.Create())
			return BitConverter.ToString(sha.ComputeHash(data)).Replace("-", "").ToLowerInvariant();
	}
	static Stream Resource(string name) {
		var stream = typeof(KeyboardPrompts).Assembly.GetManifestResourceStream(ResourcePrefix + name);
		if (stream == null)
			throw new IOException("Missing keyboard prompt resource: " + name);
		return stream;
	}
	static List<Icon> Icons() {
		var icons = new List<Icon>();
		using (var reader = new StreamReader(Resource("manifest.txt"))) {
			string line;
			while ((line = reader.ReadLine()) != null) {
				var fields = line.Split(' ');
				if (fields.Length != 3 || Path.GetFileName(fields[0]) != fields[0])
					throw new IOException("Invalid keyboard prompt manifest.");
				using (var zip = new GZipStream(Resource(fields[0] + ".gz"), CompressionMode.Decompress))
				using (var bytes = new MemoryStream()) {
					zip.CopyTo(bytes);
					var icon = new Icon { Name = fields[0], OriginalHash = fields[1], ReplacementHash = fields[2], Replacement = bytes.ToArray() };
					if (Hash(icon.Replacement) != icon.ReplacementHash)
						throw new IOException("Damaged keyboard prompt: " + icon.Name);
					icons.Add(icon);
				}
			}
		}
		if (icons.Count != 22)
			throw new IOException("Incomplete keyboard prompt pack.");
		return icons;
	}
	static void NoLinks(string path) {
		for (var current = Path.GetFullPath(path); current != null; current = Path.GetDirectoryName(current)) {
			if ((File.Exists(current) || Directory.Exists(current)) && (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
				throw new IOException("Keyboard prompts cannot modify linked files or folders: " + current);
		}
	}
	static string GeneratedPath(string backup, string name, string hash) {
		return Path.Combine(backup, "generated", name, hash + ".gz");
	}
	static bool KnownGenerated(string backup, string name, string hash, byte[] current) {
		var path = GeneratedPath(backup, name, hash);
		NoLinks(path);
		if (!File.Exists(path))
			return false;
		using (var file = File.OpenRead(path))
		using (var gzip = new GZipStream(file, CompressionMode.Decompress)) {
			var bytes = new byte[current.Length];
			int at = 0;
			while (at < bytes.Length) { int n = gzip.Read(bytes, at, bytes.Length - at); if (n == 0) return false; at += n; }
			return gzip.ReadByte() == -1 && Hash(bytes) == hash;
		}
	}
	static void WriteAtomic(string path, byte[] bytes) {
		var temp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
		try {
			File.WriteAllBytes(temp, bytes);
			if (File.Exists(path))
				File.Replace(temp, path, null);
			else
				File.Move(temp, path);
		}
		finally { if (File.Exists(temp)) File.Delete(temp); }
	}

	public static void Apply(GameInfo game, bool enabled, KeyboardLayout layout = null) {
		try { if (enabled && layout != null) layout.Validate(); ApplyCore(game, enabled, layout); }
		catch (Exception error) when (error is IOException || error is UnauthorizedAccessException) {
			throw new IOException(error.Message + "\n\nIf prompts were partly applied, close the game, turn off " +
				"Game > Keyboard and mouse prompts and press Play for the same game folder to restore originals " +
				"from .kyty-keyboard-prompts-v1. Keep that folder; missing or damaged backups must be recovered " +
				"from your own original game files first.", error);
		}
	}

	static void ApplyCore(GameInfo game, bool enabled, KeyboardLayout layout) {
		if (game == null || string.IsNullOrWhiteSpace(game.Path)) {
			if (enabled)
				throw new IOException("Select an extracted Demon's Souls game folder for keyboard prompts.");
			return;
		}
		var backup = Path.Combine(game.Path, BackupName);
		if (!enabled && !Directory.Exists(backup))
			return;
		if (!enabled && Directory.GetFiles(backup, "*.ctxr").Length == 0)
			return;
		if (!Directory.Exists(game.Path) || game.TitleId != "PPSA01341" || (enabled && game.Version != "01.007.000"))
			throw new IOException("Keyboard prompts support extracted PPSA01341 version 01.007.000 only; ZArchive and other versions are not supported.");
		foreach (var process in Process.GetProcessesByName("kyty_emulator")) {
			process.Dispose();
			throw new IOException("Close the game before changing keyboard prompts.");
		}
		var folder = Path.Combine(game.Path, "ui", "textures", "buttonicons", "_ps5");
		NoLinks(folder);
		NoLinks(backup);
		Directory.CreateDirectory(backup);
		var lockPath = Path.Combine(backup, "lock");
		NoLinks(lockPath);
		using (var fileLock = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None)) {
			var icons = Icons();
			var defaults = new KeyboardLayout();
			if (enabled && layout != null)
				foreach (var icon in icons) {
					var label = layout.IconLabel(icon.Name);
					if (label != defaults.IconLabel(icon.Name))
						icon.Replacement = PromptTexture.GenerateWithTemplate(label, icon.Replacement);
				}
			var before = new Dictionary<string, byte[]>();
			var desired = new Dictionary<string, byte[]>();
			// Validate the whole batch first. Existing mods and damaged backups are never overwritten.
			foreach (var icon in icons) {
				var target = Path.Combine(folder, icon.Name);
				var original = Path.Combine(backup, icon.Name);
				NoLinks(target);
				NoLinks(original);
				var current = File.ReadAllBytes(target);
				var hash = Hash(current);
				if (hash != icon.OriginalHash && hash != icon.ReplacementHash && !KnownGenerated(backup, icon.Name, hash, current))
					throw new IOException("Unrecognized or modified prompt: " + icon.Name + ". Restore the original game icon before changing this option.");
				byte[] saved;
				if (File.Exists(original)) {
					saved = File.ReadAllBytes(original);
					if (Hash(saved) != icon.OriginalHash)
						throw new IOException("Invalid original prompt backup: " + original);
				}
				else {
					if (hash != icon.OriginalHash)
						throw new IOException("Original prompt backup is missing: " + icon.Name + ". Restore the original icon from your own backup first.");
					saved = current;
				}
				before.Add(target, current);
				desired.Add(target, enabled ? icon.Replacement : saved);
			}
			// Persist every original before replacing anything. An interrupted batch can be
			// completed or restored on the next Play; originals are retained on uninstall too.
			foreach (var icon in icons) {
				var original = Path.Combine(backup, icon.Name);
				if (!File.Exists(original))
					WriteAtomic(original, before[Path.Combine(folder, icon.Name)]);
			}
			// Record each generated variant before any game write. This also recognizes mixed
			// old/new profiles after interruption, without accepting unrelated texture mods.
			if (enabled)
				foreach (var icon in icons) {
					if (Hash(icon.Replacement) == icon.ReplacementHash)
						continue;
					var path = GeneratedPath(backup, icon.Name, Hash(icon.Replacement));
					NoLinks(path);
					Directory.CreateDirectory(Path.GetDirectoryName(path));
					using (var bytes = new MemoryStream()) {
						using (var gzip = new GZipStream(bytes, CompressionMode.Compress, true))
							gzip.Write(icon.Replacement, 0, icon.Replacement.Length);
						WriteAtomic(path, bytes.ToArray());
					}
				}
			var changed = new List<string>();
			try {
				foreach (var pair in desired) {
					if (Hash(before[pair.Key]) == Hash(pair.Value))
						continue;
					changed.Add(pair.Key);
					WriteAtomic(pair.Key, pair.Value);
				}
			}
			catch (Exception error) when (error is IOException || error is UnauthorizedAccessException) {
				var failures = new List<Exception> { error };
				// A locked file must not stop rollback of the other files in the batch.
				foreach (var path in changed) {
					try { WriteAtomic(path, before[path]); }
					catch (Exception rollback) when (rollback is IOException || rollback is UnauthorizedAccessException) {
						failures.Add(rollback);
					}
				}
				throw new IOException("Could not finish changing keyboard prompts: " + error.Message,
					new AggregateException(failures));
			}
		}
	}
}
}
