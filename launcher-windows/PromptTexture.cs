// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.IO;
using System.IO.Compression;
using System.Linq;
namespace KytyLauncher {
static class PromptTexture {
	static byte[] Resource(string name) {
		using (var input = typeof(PromptTexture).Assembly.GetManifestResourceStream("KytyLauncher.KeyboardPrompts." + name)) {
			if (input == null)
				throw new IOException("Missing prompt texture resource: " + name);
			using (var gzip = new GZipStream(input, CompressionMode.Decompress))
			using (var output = new MemoryStream()) { gzip.CopyTo(output); return output.ToArray(); }
		}
	}
	static readonly byte[] Blank = Resource("blank.ctxr.gz");
	static readonly int[] Offsets = ReadOffsets();
	static int[] ReadOffsets() {
		var data = Resource("pixel-offsets.bin.gz");
		if (data.Length != 21845 * 4 || Blank.Length != 131149)
			throw new IOException("Invalid prompt texture layout");
		var result = new int[21845];
		Buffer.BlockCopy(data, 0, result, 0, data.Length);
		if (result.Distinct().Count() != result.Length || result.Any(x => x < 65 || x + 4 > 131137))
			throw new IOException("Invalid prompt pixel offset");
		return result;
	}
	// Supersample a neutral keycap; all eight mips are regenerated, including small PS5 tiles.
	public static Bitmap Draw(string label) {
		var bitmap = new Bitmap(512, 512, PixelFormat.Format32bppArgb);
		using (var g = Graphics.FromImage(bitmap)) {
			g.ScaleTransform(4, 4);
			g.Clear(Color.Transparent);
			g.SmoothingMode = SmoothingMode.AntiAlias;
			g.TextRenderingHint = TextRenderingHint.AntiAliasGridFit;
			using (var border = new Pen(Color.FromArgb(155, 155, 155), 5)) {
				g.FillRectangle(Brushes.White, 10, 23, 108, 82);
				g.DrawRectangle(border, 10, 23, 108, 82);
			}
			using (var format = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center }) {
				float size = 36;
				while (size > 7) {
					using (var f = new Font("Segoe UI", size, FontStyle.Bold, GraphicsUnit.Pixel)) {
						var measured = g.MeasureString(label, f);
						if (measured.Width <= 98 && measured.Height <= 72)
							break;
					}
					size -= 1;
				}
				using (var f = new Font("Segoe UI", size, FontStyle.Bold, GraphicsUnit.Pixel))
					g.DrawString(label, f, Brushes.Black, new RectangleF(14, 27, 100, 74), format);
			}
		}
		return bitmap;
	}
	public static byte[][] Pixels(string label) {
		var result = new byte[8][];
		using (var bitmap = Draw(label))
			for (int level = 0; level < 8; level++) {
				int width = Math.Max(1, 128 >> level);
				var pixels = new byte[width * width * 4];
				using (var mip = new Bitmap(width, width, PixelFormat.Format32bppArgb)) {
					using (var g = Graphics.FromImage(mip)) { g.CompositingMode = CompositingMode.SourceCopy; g.InterpolationMode = InterpolationMode.HighQualityBicubic; g.DrawImage(bitmap, 0, 0, width, width); }
					for (int y = 0; y < width; y++)
						for (int x = 0; x < width; x++) {
							var c = mip.GetPixel(x, y);
							int at = (y * width + x) * 4;
							pixels[at] = c.R;
							pixels[at + 1] = c.G;
							pixels[at + 2] = c.B;
							pixels[at + 3] = c.A;
						}
				}
				result[level] = pixels;
			}
		return result;
	}
	public static byte[] Generate(string label) { return GenerateWithTemplate(label, Blank); }
	public static byte[] GenerateWithTemplate(string label, byte[] template) {
		if (template.Length != Blank.Length)
			throw new IOException("Invalid prompt template");
		var result = (byte[])template.Clone();
		Array.Clear(result, 65, 131072);
		int pixel = 0;
		foreach (var mip in Pixels(label))
			for (int at = 0; at < mip.Length; at += 4) {
				Buffer.BlockCopy(mip, at, result, Offsets[pixel++], 4);
			}
		return result;
	}
}
}
