// SPDX-License-Identifier: GPL-2.0-or-later
// Uses the user's own originals as temporary fixtures; none are redistributed.
using System;
using System.IO;
using System.Linq;

namespace KytyLauncher {
sealed class GameInfo { public string Path, TitleId, Version; }
static class KeyboardPromptsTests {
	static void Check(bool value, string message) { if (!value) throw new Exception(message); }
	static void Reject(Action action) {
		try { action(); } catch (IOException error) {
			Check(error.Message.Contains("turn off") && error.Message.Contains("press Play") &&
				error.Message.Contains(".kyty-keyboard-prompts-v1"), "Missing recovery instructions");
			return;
		}
		throw new Exception("Expected the operation to be rejected.");
	}
	static int Main(string[] args) {
		var root = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "kyty-prompts-test-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(root);
		try {
			var folder = System.IO.Path.Combine(root, "ui", "textures", "buttonicons", "_ps5");
			Directory.CreateDirectory(folder);
			var game = new GameInfo { Path = root, TitleId = "PPSA01341", Version = "01.007.000" };
			KeyboardPrompts.Apply(game, false);
			Check(!Directory.Exists(System.IO.Path.Combine(root, ".kyty-keyboard-prompts-v1")), "Default off wrote a backup");
			foreach (var path in Directory.GetFiles(args[0], "*.ctxr")) File.Copy(path, System.IO.Path.Combine(folder, System.IO.Path.GetFileName(path)));
			var originals = Directory.GetFiles(folder).ToDictionary(p => p, File.ReadAllBytes);
			game.Version = "01.005.000";
			Reject(() => KeyboardPrompts.Apply(game, true));
			game.Version = "01.007.000";
			KeyboardPrompts.Apply(game, true);
			var installed = originals.Keys.ToDictionary(p => p, File.ReadAllBytes);
			Check(installed.Count(p => !p.Value.SequenceEqual(originals[p.Key])) == 22, "Expected exactly 22 changed icons");
			KeyboardPrompts.Apply(game, true);
			foreach (var p in installed) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Second enable changed an icon");
			KeyboardPrompts.Apply(game, false);
			foreach (var p in originals) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Restore did not recover exact original bytes");
			// Simulate an interrupted install: both original and replacement states are recoverable.
			var first = installed.First(p => !p.Value.SequenceEqual(originals[p.Key]));
			File.WriteAllBytes(first.Key, first.Value);
			KeyboardPrompts.Apply(game, false);
			Check(File.ReadAllBytes(first.Key).SequenceEqual(originals[first.Key]), "Partial install did not restore");
			var changed = (byte[])originals[first.Key].Clone(); changed[changed.Length - 1] ^= 1;
			File.WriteAllBytes(first.Key, changed);
			Reject(() => KeyboardPrompts.Apply(game, true));
			Check(File.ReadAllBytes(first.Key).SequenceEqual(changed), "Existing mod was overwritten");
			File.WriteAllBytes(first.Key, originals[first.Key]);
			var backup = System.IO.Path.Combine(root, ".kyty-keyboard-prompts-v1", System.IO.Path.GetFileName(first.Key));
			File.WriteAllBytes(backup, changed);
			Reject(() => KeyboardPrompts.Apply(game, true));
			foreach (var p in originals) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Rejected operation changed a game file");
			File.WriteAllBytes(backup, originals[first.Key]);
			// Force replacement to fail after the first icon; the batch must roll back.
			var second = installed.Where(p => !p.Value.SequenceEqual(originals[p.Key])).OrderBy(p => p.Key).Skip(1).First();
			using (var held = new FileStream(second.Key, FileMode.Open, FileAccess.Read, FileShare.Read)) {
				Reject(() => KeyboardPrompts.Apply(game, true));
			}
			foreach (var p in originals) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Failed batch left a modified icon");
            // Custom key labels are generated and recognized across profiles and interrupted installs.
            var layout = new KeyboardLayout();
            Check(layout.Assign("Cross", "R") == "Square", "Duplicate input should swap actions");
            Check(layout.Bindings["Square"] == "E", "Swap lost the previous input");
            layout.Assign("LeftStickUp", "I"); layout.Assign("R3", "Mouse:X1");
            layout.Sensitivity = "2.0";
            var settingsPath = System.IO.Path.Combine(root, "launcher-settings.json");
            var settings = new Settings(settingsPath); layout.Save(settings); settings.Save(true);
            var loaded = KeyboardLayout.Read(new Settings(settingsPath));
            Check(loaded.Bindings["Cross"] == "R" && loaded.Sensitivity == "2.0", "Layout did not survive reload");
            try { loaded.Assign("Cross", "F7"); throw new Exception("Reserved key accepted"); } catch (IOException) { }
            KeyboardPrompts.Apply(game, true, loaded);
            var custom = originals.Keys.ToDictionary(p => p, File.ReadAllBytes);
            foreach(var pair in custom) {
                Check(pair.Value.Take(65).SequenceEqual(installed[pair.Key].Take(65)) && pair.Value.Skip(131137).SequenceEqual(installed[pair.Key].Skip(131137)), "Custom texture changed its metadata");
            }
            Check(!custom[System.IO.Path.Combine(folder,"ps4_button_cross.ctxr")].SequenceEqual(installed[System.IO.Path.Combine(folder,"ps4_button_cross.ctxr")]), "Custom label was not generated");
            KeyboardPrompts.Apply(game, true, loaded);
            foreach(var p in custom) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Custom reapply changed artwork");
            // A saved generated variant is accepted; an unrelated edit still is not.
            var modified = (byte[])custom[first.Key].Clone(); modified[80] ^= 1;
            File.WriteAllBytes(first.Key, modified); Reject(() => KeyboardPrompts.Apply(game, false));
            File.WriteAllBytes(first.Key, custom[first.Key]);
            KeyboardPrompts.Apply(game, true);
            // Mixed generated/default states after interruption must also restore exactly.
            File.WriteAllBytes(System.IO.Path.Combine(folder,"ps4_button_cross.ctxr"), custom[System.IO.Path.Combine(folder,"ps4_button_cross.ctxr")]);
            KeyboardPrompts.Apply(game, false);
            foreach(var p in originals) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Custom profile restore lost original bytes");
            using (var held = new FileStream(second.Key, FileMode.Open, FileAccess.Read, FileShare.Read)) {
                Reject(() => KeyboardPrompts.Apply(game, true, loaded));
            }
            foreach(var p in originals) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Failed custom batch left an icon changed");
            using(var held = new FileStream(settingsPath, FileMode.Open, FileAccess.Read, FileShare.Read)) {
                try { settings.Save(true); throw new Exception("Locked settings write was not reported"); } catch(IOException) { }
            }
            KeyboardPrompts.Apply(game, true);
            File.Delete(backup);
			Reject(() => KeyboardPrompts.Apply(game, false));
			foreach (var p in installed) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Missing backup caused a partial restore");
			Console.WriteLine("PASS: default off, version guard, apply, idempotence, exact restore, partial recovery, mod/corrupt/missing backup protection, locked-file rollback, custom profile swaps/persistence/artwork/restore, protected settings writes");
			return 0;
		} finally {
			var expectedParent = System.IO.Path.GetFullPath(System.IO.Path.GetTempPath()).TrimEnd(System.IO.Path.DirectorySeparatorChar);
			Check(System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(root)) == expectedParent &&
				System.IO.Path.GetFileName(root).StartsWith("kyty-prompts-test-"), "Unexpected cleanup path");
			Directory.Delete(root, true);
		}
	}
}
}
