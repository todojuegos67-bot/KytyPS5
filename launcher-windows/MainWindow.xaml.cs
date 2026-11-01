using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace KytyLauncher {

// An item of a combo box: what it shows and the value it stands for.
sealed class Choice {
	public readonly string Label;
	public readonly object Value;

	public Choice(string label, object value) {
		Label = label;
		Value = value;
	}

	public override string ToString() {
		return Label;
	}
}

partial class MainWindow : Window {
	// The window's sizes for each aspect ratio of the 3D.
	static readonly Dictionary<string, string[]> ResolutionsByRatio = new Dictionary<string, string[]> {
		{ "16:9", new[] { "1280x720", "1600x900", "1920x1080", "2560x1440", "3840x2160" } },
		{ "21:9", new[] { "2560x1080", "3440x1440", "3840x1600", "5120x2160" } },
		{ "32:9", new[] { "3840x1080", "5120x1440", "7680x2160" } },
	};
	// The height Demon's Souls renders its 3D at (run-windows.ps1 -RenderHeight); its output and UI stay 3840x2160. The
	// upscalers' usual shares among them (issue #129): Ultra Quality 77% (1664 of 4K, 832 of 1080p), Quality 66.7%
	// (1440), Balanced 58% (1256), Performance 50% (1080), Ultra Performance 33% (720).
	static readonly int[] RenderHeights = { 2160, 1800, 1664, 1620, 1512, 1440, 1296, 1256, 1152, 1080, 900, 832, 720 };
	// Below 720 with the 1080p output only (issue #129): DLSS's Balanced, Performance and Ultra Performance there; the
	// game's TAA down to 540 (from 360 its picture is too soft).
	static readonly int[] LowRenderHeights = { 630, 540, 360 };
	// The 1440p output's (DLSS or FSR 3, PR #165): DLAA, Ultra Quality 1104, Quality 960, Balanced 840, Performance 720,
	// Ultra Performance 480 (its third; the emulator raises a height between the third and the half to the half).
	static readonly int[] Output1440Heights = { 1440, 1152, 1104, 1080, 960, 840, 720, 480 };

	// (--screenshot pictures: no game art, another path shown.)
	public static bool ScreenshotNoArt;
	public static string ScreenshotPath;

	readonly LauncherContext context;
	readonly Commands commands;
	readonly List<FrameworkElement> pages = new List<FrameworkElement>();
	readonly DispatcherTimer gameTimer;
	GameInfo game = new GameInfo();
	StoreState storeState = StoreState.Unknown;
	Task<StoreState> storeTask;
	DateTime storeCheckedAt = DateTime.MinValue;
	bool building;      // the controls are being made: their events change nothing
	bool fillingRender; // the render heights are being listed: a selection is not a choice
	bool renderForDlss; // the render height row is the PS5 Pro path's - DLSS's or FSR 3's (settings render_dlss) - or the game's TAA's (render)
	int renderWanted;   // the height last chosen for the upscaler shown
	int pageIndex;

	TextBox gamePath, extraArgs;
	ComboBox resolution, ratio, render, upscaler, dlssModel, sharpness, output, quality, frameGen, language, audioOutput, hud, uiLanguage, gpu;
	// The Sharpening row's value for each upscaler ("0" to "0.9"), and the upscaler whose value it shows.
	string dlssSharpness = "0", fsrSharpness = "0.6", sharpnessShown = "";
	CheckBox fullscreen, aspect, wideBars, hdr, fps120, vsync, lowLatency, fsr, cheats, redzone, safeGpu, ecores, presentDirect, vrrFlips,
		antiLag, keepResident, fps30, padSpeaker, reopenLauncher;
	TextBlock renderNote, outputNote, upscalerNote, frameGenNote, streamlineText, cheatState, bannerTitle, bannerLine, storeLine;
	Image banner;
	Border bannerFallback;
	StackPanel checkList;
	Grid currentSettings;

	Settings S { get { return context.Settings; } }

	static string T(string key) {
		return Strings.T(key);
	}

	static string F(string key, params object[] values) {
		return Strings.F(key, values);
	}

	public MainWindow(LauncherContext context) {
		InitializeComponent();
		this.context = context;
		commands = new Commands(context);
		gameTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(450) };
		gameTimer.Tick += (s, e) => {
			gameTimer.Stop();
			LoadGame(gamePath.Text.Trim());
		};
		Nav.SelectionChanged += (s, e) => ShowPage(Nav.SelectedIndex);
		PlayButton.Click += (s, e) => Play();
		PrecompileButton.Click += (s, e) => Precompile("");
		Activated += (s, e) => CheckStore(false);
		Closing += (s, e) => ReadForm();
		// (A small screen: the window fits its work area.)
		var area = SystemParameters.WorkArea;
		if (Height > area.Height - 16) Height = Math.Max(MinHeight, area.Height - 16);
		if (Width > area.Width - 16) Width = Math.Max(MinWidth, area.Width - 16);
		Build();
	}

	T2 Res<T2>(string key) where T2 : class {
		return FindResource(key) as T2;
	}

	// ------------------------------------------------------------------ building the window

	void Build() {
		building = true;
		AppSubtitle.Text = T("app.subtitle");
		NavHint.Text = T("nav.hint");
		PlayButton.Content = T("button.play");
		PrecompileButton.Content = T("button.precompile");
		pages.Clear();
		PageHost.Children.Clear();
		Nav.Items.Clear();
		renderForDlss = (context.FrameGenAvailable && S.Str("upscaler") == "dlss") || S.Str("upscaler") == "fsr";
		renderWanted = renderForDlss ? S.Int("render_dlss") : S.Int("render");
		AddPage("nav.play", BuildPlayPage());
		AddPage("nav.display", BuildDisplayPage());
		AddPage("nav.graphics", BuildGraphicsPage());
		AddPage("nav.game", BuildGamePage());
		AddPage("nav.shaders", BuildShadersPage());
		AddPage("nav.advanced", BuildAdvancedPage());
		AddPage("nav.experimental", BuildExperimentalPage());
		building = false;
		Nav.SelectedIndex = Math.Max(0, Math.Min(pageIndex, pages.Count - 1));
		ShowPage(Nav.SelectedIndex);
		LoadGame(S.Str("game"));
		UpdateSummary();
	}

	void AddPage(string navKey, FrameworkElement page) {
		// (Text, not a TextBlock: an element item takes its color from the list, not from its item.)
		Nav.Items.Add(T(navKey));
		page.Visibility = Visibility.Collapsed;
		pages.Add(page);
		PageHost.Children.Add(page);
	}

	void ShowPage(int index) {
		if (index < 0) return;
		pageIndex = index;
		for (int i = 0; i < pages.Count; i++) pages[i].Visibility = i == index ? Visibility.Visible : Visibility.Collapsed;
	}

	FrameworkElement Page(string titleKey, params UIElement[] content) {
		var stack = new StackPanel { Margin = new Thickness(44, 32, 44, 40), MaxWidth = 880, HorizontalAlignment = HorizontalAlignment.Left };
		if (titleKey != null) stack.Children.Add(new TextBlock { Text = T(titleKey), Style = Res<Style>("PageTitle") });
		foreach (var element in content) stack.Children.Add(element);
		return new ScrollViewer { Content = stack, VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
			HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled, Focusable = false };
	}

	TextBlock Section(string key) {
		return new TextBlock { Text = T(key), Style = Res<Style>("SectionTitle") };
	}

	TextBlock Desc(string text) {
		return new TextBlock { Text = text, Style = Res<Style>("Desc") };
	}

	// A setting: its name on the left, the control and what it does on the right.
	FrameworkElement Row(string labelKey, FrameworkElement control, TextBlock desc = null) {
		var grid = new Grid { Margin = new Thickness(0, 7, 0, 9) };
		grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(232) });
		grid.ColumnDefinitions.Add(new ColumnDefinition());
		grid.Children.Add(new TextBlock { Text = labelKey == null ? "" : T(labelKey), Style = Res<Style>("RowLabel") });
		var right = new StackPanel();
		right.Children.Add(control);
		if (desc != null) right.Children.Add(desc);
		Grid.SetColumn(right, 1);
		grid.Children.Add(right);
		return grid;
	}

	CheckBox Switch(bool value, bool enabled = true) {
		var box = new CheckBox { IsChecked = value, IsEnabled = enabled, Style = Res<Style>("Switch") };
		box.Checked += (s, e) => Changed();
		box.Unchecked += (s, e) => Changed();
		return box;
	}

	ComboBox Combo(IEnumerable<Choice> items, int selected, double width = 300) {
		var box = new ComboBox { Width = width };
		foreach (var item in items) box.Items.Add(item);
		box.SelectedIndex = Math.Max(0, Math.Min(selected, box.Items.Count - 1));
		box.SelectionChanged += (s, e) => Changed();
		return box;
	}

	Button FlatButton(string key, RoutedEventHandler click) {
		var button = new Button { Content = T(key), Style = Res<Style>("FlatButton"), HorizontalAlignment = HorizontalAlignment.Left };
		button.Click += click;
		return button;
	}

	// A slider with the value it stands for next to it. Every number on these pages is an int, so the caller formats
	// the text and stores the rounded value; the store happens here because ReadForm only knows the controls it has
	// always had.
	FrameworkElement SliderRow(double min, double max, double step, int value, Func<double, string> text, Action<double> changed) {
		var label = new TextBlock { Foreground = Res<Brush>("TextDimBrush"), VerticalAlignment = VerticalAlignment.Center,
			MinWidth = 76, Margin = new Thickness(14, 0, 0, 0) };
		var bar = new Slider {
			Minimum = min, Maximum = max, SmallChange = step, LargeChange = step * 4, TickFrequency = step,
			IsSnapToTickEnabled = true, Width = 300, VerticalAlignment = VerticalAlignment.Center,
			Foreground = Res<Brush>("TextBrush"),
		};
		label.Text = text(value);
		bar.ValueChanged += (s, e) => { label.Text = text(e.NewValue); changed(e.NewValue); Changed(); };
		bar.Value = value;
		return Horizontal(bar, label);
	}

	static object Value(ComboBox box) {
		var choice = box.SelectedItem as Choice;
		return choice == null ? null : choice.Value;
	}

	static StackPanel Horizontal(params UIElement[] children) {
		var panel = new StackPanel { Orientation = Orientation.Horizontal };
		foreach (var child in children) panel.Children.Add(child);
		return panel;
	}

	string Paren(string a, string b) {
		return Strings.Language == UiLanguage.Chinese ? a + "（" + b + "）" : a + " (" + b + ")";
	}

	// ------------------------------------------------------------------ Play

	FrameworkElement BuildPlayPage() {
		// The game's own art from its dump (sce_sys\pic0.png), under a fade into the window; a plain gradient without one.
		var bannerGrid = new Grid { Height = 270, ClipToBounds = true };
		bannerFallback = new Border { Background = new LinearGradientBrush(Color.FromRgb(0x2A, 0x22, 0x18), Color.FromRgb(0x10, 0x10, 0x12), 35) };
		bannerGrid.Children.Add(bannerFallback);
		banner = new Image { Stretch = Stretch.UniformToFill, VerticalAlignment = VerticalAlignment.Center, HorizontalAlignment = HorizontalAlignment.Center };
		RenderOptions.SetBitmapScalingMode(banner, BitmapScalingMode.HighQuality);
		bannerGrid.Children.Add(banner);
		var fade = new LinearGradientBrush { StartPoint = new Point(0, 0), EndPoint = new Point(0, 1) };
		fade.GradientStops.Add(new GradientStop(Color.FromArgb(0x20, 0x10, 0x10, 0x12), 0));
		fade.GradientStops.Add(new GradientStop(Color.FromArgb(0x60, 0x10, 0x10, 0x12), 0.55));
		fade.GradientStops.Add(new GradientStop(Color.FromArgb(0xFF, 0x10, 0x10, 0x12), 1));
		bannerGrid.Children.Add(new Border { Background = fade });
		var side = new LinearGradientBrush { StartPoint = new Point(0, 0), EndPoint = new Point(1, 0) };
		side.GradientStops.Add(new GradientStop(Color.FromArgb(0xB0, 0x10, 0x10, 0x12), 0));
		side.GradientStops.Add(new GradientStop(Color.FromArgb(0x00, 0x10, 0x10, 0x12), 0.6));
		bannerGrid.Children.Add(new Border { Background = side });
		var titles = new StackPanel { VerticalAlignment = VerticalAlignment.Bottom, Margin = new Thickness(44, 0, 44, 22) };
		bannerTitle = new TextBlock { FontFamily = Res<FontFamily>("SerifFont"), FontSize = 34, Foreground = Brushes.White, TextWrapping = TextWrapping.Wrap };
		bannerLine = new TextBlock { FontSize = 13, Foreground = Res<Brush>("TextDimBrush"), Margin = new Thickness(2, 6, 0, 0) };
		titles.Children.Add(bannerTitle);
		titles.Children.Add(bannerLine);
		bannerGrid.Children.Add(titles);

		// Checks on the left; the game and the current settings on the right.
		var columns = new Grid { Margin = new Thickness(44, 22, 44, 32) };
		columns.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
		columns.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(24) });
		columns.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1.05, GridUnitType.Star) });

		checkList = new StackPanel();
		var checksCard = new Border { Style = Res<Style>("Card"), VerticalAlignment = VerticalAlignment.Top };
		checksCard.Child = Stack(new TextBlock { Text = T("play.checks"), Style = Res<Style>("SectionTitle"), Margin = new Thickness(0, 0, 0, 10) }, checkList);
		columns.Children.Add(checksCard);

		gamePath = new TextBox { Text = S.Str("game"), Margin = new Thickness(0, 0, 0, 8) };
		gamePath.TextChanged += (s, e) => {
			if (building) return;
			gameTimer.Stop();
			gameTimer.Start();
		};
		var browse = FlatButton("button.folder", (s, e) => BrowseGame());
		var browseZar = FlatButton("button.zar", (s, e) => BrowseZar());
		browseZar.Margin = new Thickness(8, 0, 0, 0);
		var gameCard = new Border { Style = Res<Style>("Card") };
		gameCard.Child = Stack(new TextBlock { Text = T("play.gamebox"), Style = Res<Style>("SectionTitle"), Margin = new Thickness(0, 0, 0, 10) },
			gamePath, Horizontal(browse, browseZar));
		currentSettings = new Grid();
		currentSettings.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
		currentSettings.ColumnDefinitions.Add(new ColumnDefinition());
		var settingsCard = new Border { Style = Res<Style>("Card"), Margin = new Thickness(0, 16, 0, 0) };
		settingsCard.Child = Stack(new TextBlock { Text = T("play.summary"), Style = Res<Style>("SectionTitle"), Margin = new Thickness(0, 0, 0, 8) },
			currentSettings);
		var right = Stack(gameCard, settingsCard);
		Grid.SetColumn(right, 2);
		columns.Children.Add(right);

		var stack = new StackPanel();
		stack.Children.Add(bannerGrid);
		stack.Children.Add(columns);
		return new ScrollViewer { Content = stack, VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
			HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled, Focusable = false };
	}

	static StackPanel Stack(params UIElement[] children) {
		var panel = new StackPanel();
		foreach (var child in children) panel.Children.Add(child);
		return panel;
	}

	void BrowseGame() {
		var current = gamePath.Text.Trim();
		var chosen = FolderPicker.Pick(this, T("dialog.game"), Directory.Exists(current) ? current : null);
		if (chosen != null) gamePath.Text = chosen;
	}

	void BrowseZar() {
		var current = gamePath.Text.Trim();
		var dialog = new Microsoft.Win32.OpenFileDialog { Title = T("dialog.zar"), Filter = "ZArchive (*.zar)|*.zar" };
		if (File.Exists(current)) dialog.InitialDirectory = Path.GetDirectoryName(current);
		if (dialog.ShowDialog(this) == true) gamePath.Text = dialog.FileName;
	}

	// The game at `path`: its name, art and checks (a .zar's param.json comes from the precompile program, off this thread).
	void LoadGame(string path) {
		if (!building && path != S.Str("game")) Changed();
		var zar = path.EndsWith(".zar", StringComparison.OrdinalIgnoreCase) && File.Exists(path);
		if (!zar) {
			ApplyGame(GameInfo.Read(path, context.PrecompileTool, context.Root));
			return;
		}
		ApplyGame(new GameInfo { Path = path });
		var tool = context.PrecompileTool;
		Task.Run(() => GameInfo.Read(path, tool, context.Root)).ContinueWith(task => Dispatcher.BeginInvoke(new Action(() => {
			if (gamePath.Text.Trim() == path) ApplyGame(task.Result);
		})));
	}

	void ApplyGame(GameInfo info) {
		game = info;
		GameTitle.Text = info.Found ? (info.Name.Length != 0 ? info.Name : info.TitleId) : "";
		bannerTitle.Text = info.Found ? (info.Name.Length != 0 ? info.Name : info.TitleId) : T("play.choose");
		bannerLine.Text = info.Found ? info.TitleId + "  ·  " + info.Version : "";
		banner.Source = null;
		if (info.Picture != null && !ScreenshotNoArt) {
			try {
				var image = new BitmapImage();
				image.BeginInit();
				image.CacheOption = BitmapCacheOption.OnLoad; // (the file stays free)
				image.DecodePixelWidth = 1920;
				image.UriSource = new Uri(info.Picture);
				image.EndInit();
				image.Freeze();
				banner.Source = image;
			} catch (Exception) {
			}
		}
		UpdateCheatState();
		storeTask = null;
		CheckStore(true);
	}

	// The precompiled shaders' state, asked of the precompile program (at most every 30 s: also when the window comes back
	// to the front after a precompile).
	void CheckStore(bool force) {
		if (!game.Found) {
			storeState = StoreState.Unknown;
			UpdateChecks();
			return;
		}
		if (!force && (storeTask != null && !storeTask.IsCompleted || (DateTime.Now - storeCheckedAt).TotalSeconds < 30)) return;
		storeCheckedAt = DateTime.Now;
		if (storeTask == null) storeState = StoreState.Checking;
		UpdateChecks();
		var checkedGame = game;
		var task = ShaderStore.CheckAsync(context, checkedGame);
		storeTask = task;
		task.ContinueWith(t => Dispatcher.BeginInvoke(new Action(() => {
			if (game != checkedGame || storeTask != task) return;
			storeState = t.Result;
			UpdateChecks();
		})));
	}

	enum Mark { Good, Warn, Bad, Busy }

	void AddCheck(Mark mark, string text) {
		var symbol = mark == Mark.Good ? "✓" : mark == Mark.Warn ? "!" : mark == Mark.Bad ? "✕" : "…";
		var brush = mark == Mark.Good ? Res<Brush>("GoodBrush") : mark == Mark.Warn ? Res<Brush>("WarnBrush")
			: mark == Mark.Bad ? Res<Brush>("BadBrush") : Res<Brush>("TextDimBrush");
		var row = new DockPanel { Margin = new Thickness(0, 4, 0, 6) };
		var icon = new TextBlock { Text = symbol, Foreground = brush, FontWeight = FontWeights.Bold, Width = 22, FontSize = 14 };
		DockPanel.SetDock(icon, Dock.Left);
		row.Children.Add(icon);
		row.Children.Add(new TextBlock { Text = text, TextWrapping = TextWrapping.Wrap, Foreground = Res<Brush>("TextBrush"), LineHeight = 18 });
		checkList.Children.Add(row);
	}

	void UpdateChecks() {
		if (checkList == null) return;
		checkList.Children.Clear();
		var path = gamePath == null ? S.Str("game") : gamePath.Text.Trim();
		if (game.Found) AddCheck(Mark.Good, F("game.ok", game.Name.Length != 0 ? game.Name : game.TitleId, game.TitleId, game.Version));
		else AddCheck(Mark.Bad, T(path.Length == 0 ? "game.none" : "game.bad"));
		var gpu = context.Gpus.FirstOrDefault();
		if (gpu == null) AddCheck(Mark.Warn, T("gpu.none"));
		else if (gpu.Memory > 0) AddCheck(Mark.Good, F("gpu.ok", gpu.Name, Math.Round(gpu.Memory / 1073741824.0)));
		else AddCheck(Mark.Good, F("gpu.okNoMemory", gpu.Name));
		var ramGb = Math.Round(context.RamBytes / 1073741824.0);
		AddCheck(ramGb <= 20 ? Mark.Warn : Mark.Good, F(ramGb <= 20 ? "ram.low" : "ram.ok", ramGb));
		Mark storeMark;
		var storeText = StoreStatus(out storeMark);
		if (storeText != null) AddCheck(storeMark, storeText);
		if (context.FrameGenAvailable) AddCheck(Mark.Good, T("dlss.ok"));
		else AddCheck(Mark.Busy, T(context.RtxGpu ? "dlss.noStreamline" : "dlss.noRtx"));
		AddCheck(context.FsrAvailable ? Mark.Good : Mark.Busy, T(context.FsrAvailable ? "fsr.ok" : "fsr.none"));
		if (storeLine != null) storeLine.Text = storeText ?? T("cheats.noGame");
	}

	// The precompiled shaders' line, or null without a game.
	string StoreStatus(out Mark mark) {
		mark = Mark.Busy;
		switch (storeState) {
			case StoreState.Checking: return T("shaders.checking");
			case StoreState.Current: mark = Mark.Good; return T("shaders.current");
			case StoreState.Outdated: mark = Mark.Warn; return T("shaders.outdated");
			case StoreState.OutdatedAll: mark = Mark.Warn; return T("shaders.outdatedAll");
			case StoreState.None: mark = Mark.Warn; return T("shaders.none");
			default: return game.Found ? T("shaders.unknown") : null;
		}
	}

	// ------------------------------------------------------------------ Display

	FrameworkElement BuildDisplayPage() {
		var ratioValue = S.Str("ratio");
		var ratios = new[] { new Choice(T("ratio.169"), "16:9"), new Choice("21:9", "21:9"), new Choice("32:9", "32:9") };
		ratio = Combo(ratios, Math.Max(0, Array.FindIndex(ratios, c => (string)c.Value == ratioValue)));
		resolution = new ComboBox();
		resolution.SelectionChanged += (s, e) => {
			Changed();
			WindowChanged();
		};
		UpdateResolutions(S.Str("resolution"));
		// (With 21:9 or 32:9 only: the movies and menus at 16:9 between black bars, Commands.)
		wideBars = Switch(S.Bool("wide_bars"), Ratio() != "16:9");
		ratio.SelectionChanged += (s, e) => {
			if (building) return;
			wideBars.IsEnabled = Ratio() != "16:9";
			UpdateResolutions(Value(resolution) as string ?? S.Str("resolution"));
			SelectRenderHeight();
			UpdateRenderNote();
			UpdateOutputNote();
		};
		fullscreen = Switch(S.Bool("fullscreen"));
		aspect = Switch(S.Bool("aspect"), S.Bool("fullscreen"));
		fullscreen.Checked += (s, e) => {
			aspect.IsEnabled = true;
			WindowChanged();
		};
		fullscreen.Unchecked += (s, e) => {
			aspect.IsEnabled = false;
			WindowChanged();
		};
		hdr = Switch(S.Bool("hdr"));
		fps120 = Switch(S.Bool("fps120"));
		vsync = Switch(S.Bool("vsync"));
		lowLatency = Switch(S.Bool("lowlatency"));
		return Page("display.title",
			Section("display.window"),
			Row("resolution", resolution, Desc(T("resolution.desc"))),
			Row("ratio", ratio, Desc(T("ratio.desc"))),
			Row("widebars", wideBars, Desc(T("widebars.desc"))),
			Row("fullscreen", fullscreen),
			Row("aspect", aspect, Desc(T("aspect.desc"))),
			Row("hdr", hdr, Desc(T("hdr.desc"))),
			Section("display.rate"),
			Row("fps120", fps120, Desc(T("fps120.desc"))),
			Row("vsync", vsync, Desc(T("vsync.desc"))),
			Row("lowlatency", lowLatency, Desc(T("lowlatency.desc"))));
	}

	string Ratio() {
		return ratio == null ? S.Str("ratio") : Value(ratio) as string ?? "16:9";
	}

	double RatioAspect() {
		switch (Ratio()) {
			case "21:9": return 64.0 / 27.0;
			case "32:9": return 32.0 / 9.0;
			default: return 16.0 / 9.0;
		}
	}

	// The ratio's resolutions; `wanted` kept when it is one of them, else the one nearest its height.
	void UpdateResolutions(string wanted) {
		string[] list;
		if (!ResolutionsByRatio.TryGetValue(Ratio(), out list)) list = ResolutionsByRatio["16:9"];
		var was = building;
		building = true;
		resolution.Items.Clear();
		foreach (var item in list) resolution.Items.Add(new Choice(item.Replace("x", " × "), item));
		int height;
		var parts = (wanted ?? "").Split('x');
		if (parts.Length < 2 || !int.TryParse(parts[1], out height)) height = 1440;
		var index = Array.IndexOf(list, wanted);
		if (index < 0) {
			var best = int.MaxValue;
			for (int i = 0; i < list.Length; i++) {
				var distance = Math.Abs(int.Parse(list[i].Split('x')[1]) - height);
				if (distance < best) {
					best = distance;
					index = i;
				}
			}
		}
		resolution.SelectedIndex = index;
		building = was;
		if (!building) Changed();
	}

	// ------------------------------------------------------------------ Graphics

	FrameworkElement BuildGraphicsPage() {
		// (DLSS where it runs - an NVIDIA RTX card with Streamline -, FSR 3 on any graphics card: built into the emulator.)
		var upscalers = new List<Choice> { new Choice(T("upscaler.game"), "") };
		if (context.FrameGenAvailable) upscalers.Add(new Choice(T("upscaler.dlss"), "dlss"));
		upscalers.Add(new Choice(T("upscaler.fsr"), "fsr"));
		// (No upscaler and no anti-aliasing: run-windows.ps1 -Antialiasing off, issue #220.)
		upscalers.Add(new Choice(T("upscaler.off"), "off"));
		upscaler = Combo(upscalers, Math.Max(0, upscalers.FindIndex(c => (string)c.Value == S.Str("upscaler"))));
		upscalerNote = Desc("");
		render = new ComboBox();
		renderNote = Desc("");
		render.SelectionChanged += (s, e) => {
			if (!fillingRender && render.SelectedItem != null) renderWanted = (int)Value(render);
			UpdateRenderNote();
			Changed();
		};
		var outputs = new[] { new Choice(T("output.auto"), "auto"), new Choice(T("output.4k"), "4k"), new Choice(T("output.1080"), "1080p"),
			new Choice(T("output.1440"), "1440p") };
		output = Combo(outputs, Math.Max(0, Array.FindIndex(outputs, c => (string)c.Value == S.Str("output"))));
		outputNote = Desc("");
		output.SelectionChanged += (s, e) => {
			// (The heights below 720, and DLSS's heights and their modes, follow the output.)
			SelectRenderHeight();
			UpdateRenderNote();
			UpdateOutputNote();
		};
		// The DLSS model (KYTY_DLSS_PRESET): the DLSS version's own choice, its transformer or its CNN model.
		var models = new[] { new Choice(T("dlssmodel.auto"), ""), new Choice(T("dlssmodel.k"), "K"), new Choice(T("dlssmodel.m"), "M"),
		                     new Choice(T("dlssmodel.l"), "L"), new Choice(T("dlssmodel.e"), "E") };
		dlssModel = Combo(models, Math.Max(0, Array.FindIndex(models, c => (string)c.Value == S.Str("dlss_preset"))));
		dlssModel.IsEnabled = Value(upscaler) as string == "dlss";
		// The upscaler's sharpening (KYTY_DLSS_SHARPNESS, KYTY_FSR_SHARPNESS): one row for the chosen upscaler, each with a
		// value of its own: DLSS none by default (the game's own TAA sharpens, DLSS does not: issue #224), FSR 3 its 0.6
		// (issue #227).
		var sharpnesses = new[] { new Choice(T("sharpen.off"), "0"), new Choice(T("sharpen.low"), "0.3"),
			new Choice(T("sharpen.medium"), "0.6"), new Choice(T("sharpen.high"), "0.9") };
		Func<string, string, string> sharpnessOr = (value, fallback) => Array.Exists(sharpnesses, c => (string)c.Value == value) ? value : fallback;
		dlssSharpness = sharpnessOr(S.Str("dlss_sharpness"), "0");
		fsrSharpness  = sharpnessOr(S.Str("fsr_sharpness"), "0.6");
		sharpness     = Combo(sharpnesses, 0);
		Action showSharpness = () => {
			KeepSharpness();
			sharpnessShown = Value(upscaler) as string ?? "";
			var shown = sharpnessShown == "fsr" ? fsrSharpness : dlssSharpness;
			sharpness.SelectedIndex = Math.Max(0, Array.FindIndex(sharpnesses, c => (string)c.Value == shown));
			sharpness.IsEnabled = sharpnessShown == "dlss" || sharpnessShown == "fsr";
		};
		showSharpness();
		upscaler.SelectionChanged += (s, e) => {
			if (building) return;
			// The height chosen goes back to the upscaler it was for; the newly chosen one's is shown.
			S.Set(renderForDlss ? "render_dlss" : "render", renderWanted);
			renderForDlss = Value(upscaler) as string == "dlss" || Value(upscaler) as string == "fsr";
			renderWanted = renderForDlss ? S.Int("render_dlss") : S.Int("render");
			dlssModel.IsEnabled = Value(upscaler) as string == "dlss";
			showSharpness();
			UpdateUpscalerNote();
			SelectRenderHeight();
			UpdateRenderNote();
			UpdateOutputNote();
			Changed();
		};
		SelectRenderHeight();
		UpdateRenderNote();
		UpdateUpscalerNote();
		UpdateOutputNote();

		var qualityValue = S.Str("quality").ToLowerInvariant();
		var qualities = new[] { new Choice(T("quality.standard"), "standard"), new Choice(T("quality.low"), "low"), new Choice(T("quality.lowest"), "lowest") };
		quality = Combo(qualities, Math.Max(0, Array.FindIndex(qualities, c => (string)c.Value == qualityValue)));

		var gens = new List<Choice> { new Choice(T("framegen.off"), 0) };
		for (int x = 2; x <= 6; x++) gens.Add(new Choice(x + "x", x));
		var genValue = S.Int("framegen");
		frameGen = Combo(gens, context.FrameGenAvailable && genValue >= 2 ? genValue - 1 : 0, 160);
		frameGen.IsEnabled = context.FrameGenAvailable;
		frameGenNote = Desc("");
		UpdateFrameGenNote();
		var chooseStreamline = FlatButton("button.streamline", (s, e) => ChooseStreamline());
		streamlineText = new TextBlock { Foreground = Res<Brush>("TextDimBrush"), VerticalAlignment = VerticalAlignment.Center,
			Margin = new Thickness(12, 0, 0, 0), TextTrimming = TextTrimming.CharacterEllipsis, MaxWidth = 380 };
		streamlineText.Text = context.Streamline ?? (S.Str("streamline").Length != 0 ? S.Str("streamline") : T("streamline.none"));
		fsr = Switch(context.FsrAvailable && S.Bool("fsr"), context.FsrAvailable);
		fsr.Checked += (s, e) => UpdateUpscalerNote();
		fsr.Unchecked += (s, e) => UpdateUpscalerNote();
		return Page("graphics.title",
			Section("graphics.resolution"),
			Row("upscaler", upscaler, upscalerNote),
			Row("render", render, renderNote),
			Row("output", output, outputNote),
			Row("dlssmodel", dlssModel, Desc(T("dlssmodel.desc"))),
			Row("sharpen", sharpness, Desc(T("sharpen.desc"))),
			Section("graphics.detail"),
			Row("quality", quality, Desc(T("quality.desc"))),
			Section("graphics.fg"),
			Row("framegen", frameGen, frameGenNote),
			Row("streamline", Horizontal(chooseStreamline, streamlineText), Desc(T("streamline.desc"))),
			Row(null, Horizontal(fsr, new TextBlock { Text = T("fsr"), Margin = new Thickness(12, 4, 0, 0), TextWrapping = TextWrapping.Wrap, MaxWidth = 520 }),
				Desc(T(context.FsrAvailable ? "fsr.desc" : "fsr.missing"))));
	}

	string RenderLabel(int height) {
		if (renderForDlss) return Paren(height.ToString(), SuperResolutionMode(height));
		switch (height) {
			case 2160: return Paren("2160", T("render.native"));
			case 1512: return Paren("1512", T("render.pro"));
			case 1440: return Paren("1440", T("render.ps5"));
			default: return height.ToString();
		}
	}

	// The game output shown: 1080p chosen, or Automatic for the game window chosen on the Display page (Commands.AutoHdOutput).
	bool HdOutputChosen {
		get {
			var value = output == null ? S.Str("output") : Value(output) as string;
			return value == "1080p" || (value == "auto" && Commands.AutoHdOutput(resolution == null ? S.Str("resolution") : Value(resolution) as string,
			                                                                       fullscreen == null ? S.Bool("fullscreen") : fullscreen.IsChecked == true,
			                                                                       Ratio(), context.DisplayHeight));
		}
	}

	// The output's lines: 1080 (the 1080p output), 1440 (1440p with an upscaler) or 2160.
	int OutputLines {
		get {
			if (HdOutputChosen) return 1080;
			var value = output == null ? S.Str("output") : Value(output) as string;
			return value == "1440p" && renderForDlss ? 1440 : 2160;
		}
	}

	// The output's note, with what Automatic picks for this window (and 1440p's without an upscaler).
	void UpdateOutputNote() {
		if (outputNote == null || output == null) return;
		var value = Value(output) as string;
		outputNote.Text = T("output.desc") + (value == "auto" ? " " + F("output.autoNow", HdOutputChosen ? "1080p" : "4K")
		                                      : value == "1440p" && !renderForDlss ? " " + T("output.1440NoUpscaler") : "");
	}

	// The Display page's window changed: Automatic may pick the other output, and the render heights follow it.
	void WindowChanged() {
		if (building || output == null || render == null) return;
		SelectRenderHeight();
		UpdateRenderNote();
		UpdateOutputNote();
	}

	// The DLSS mode a render height runs in: the emulator picks it from the height's share of the game's output
	// (frame-gen.cpp: 95% DLAA, 62% Quality, 55% Balanced, 45% Performance, less Ultra Performance). From 72% it is named
	// Ultra Quality (77% the usual share), DLSS's Quality mode with more lines.
	string SuperResolutionMode(int height) {
		var share = (double)height / OutputLines;
		// (FSR 3 names the same modes; at the output's own size it is FSR's native anti-aliasing.)
		var native = upscaler != null && Value(upscaler) as string == "fsr" ? "fsr.native" : "dlss.dlaa";
		return T(share >= 0.95 ? native : share >= 0.72 ? "dlss.ultraquality" : share >= 0.62 ? "dlss.quality" : share >= 0.55 ? "dlss.balanced" : share >= 0.45 ? "dlss.performance" : "dlss.ultra");
	}

	// The highest render height whose 3D at the ratio fits the game's render target memory, made for 3840x2160: no more of
	// the GPU's 64 KiB tiles (256x128, 128x128, 128x64, 64x64 pixels) than 3840x2160 (the emulator's FitsRenderTargets).
	int MaxRenderHeight() {
		var aspectValue = RatioAspect();
		var tiles = new[] { new[] { 256, 128 }, new[] { 128, 128 }, new[] { 128, 64 }, new[] { 64, 64 } };
		for (int h = 2160; h >= 720; h -= 8) {
			var w = Math.Floor(h * aspectValue / 8.0 + 0.5) * 8;
			var fits = true;
			foreach (var tile in tiles) {
				if (Math.Ceiling(w / tile[0]) * Math.Ceiling((double)h / tile[1]) > Math.Ceiling(3840.0 / tile[0]) * Math.Ceiling(2160.0 / tile[1])) fits = false;
			}
			if (fits) return h;
		}
		return 720;
	}

	// The heights the ratio allows (with DLSS those its output takes: 1080 and up for 4K, up to 1080 for 1080p; below 720
	// with the 1080p output only); the one last chosen for this upscaler, a higher one as the highest allowed, an unknown
	// one as the upscaler's default (DLSS 1080, the game's TAA 1440).
	void SelectRenderHeight() {
		var most = MaxRenderHeight();
		var lines = OutputLines;
		var allowed = lines == 1440 ? Output1440Heights.Where(h => h <= most).ToList()
		              : RenderHeights.Where(h => h <= most && (!renderForDlss || (lines == 1080 ? h <= 1080 : h >= 1080 || h == 720))).ToList();
		if (lines == 1080) allowed.AddRange(LowRenderHeights.Where(h => renderForDlss || h >= 540));
		fillingRender = true;
		var was = building;
		building = true;
		try {
			render.Items.Clear();
			foreach (var h in allowed) render.Items.Add(new Choice(RenderLabel(h), h));
			if (renderWanted > most) {
				render.SelectedIndex = 0;
				return;
			}
			foreach (var wanted in new[] { renderWanted, renderForDlss ? 1080 : 1440 }) {
				var index = allowed.IndexOf(wanted);
				if (index >= 0) {
					render.SelectedIndex = index;
					return;
				}
			}
			render.SelectedIndex = 0;
		} finally {
			fillingRender = false;
			building = was;
		}
	}

	int SelectedRender() {
		var value = render == null ? null : Value(render);
		return value == null ? (renderForDlss ? 1080 : 1440) : (int)value;
	}

	void UpdateRenderNote() {
		if (render == null || renderNote == null) return;
		var h = SelectedRender();
		var lines = OutputLines;
		// (The 1080p and 1440p outputs' 3D is at most their height.)
		var capped = lines < 2160 && h > lines;
		if (capped) h = lines;
		// (The PS5 Pro path's 4K output takes 1080 at least, or 720 for its ultra performance: a third of the output; the
		// 1440p output 720 at least, or 480.)
		var least = lines == 1440 ? 720 : 1080;
		var raised = renderForDlss && lines != 1080 && h < least && h != lines / 3;
		if (raised) h = least;
		var w = (int)(Math.Floor(h * RatioAspect() / 8.0 + 0.5) * 8);
		renderNote.Text = F(capped ? (lines == 1440 ? "render.capped1440" : "render.capped")
		                    : raised ? (lines == 1440 ? "render.raised1440" : "render.raised") : "render.desc", w, h);
	}

	void UpdateUpscalerNote() {
		var chosen = upscaler == null ? S.Str("upscaler") : Value(upscaler) as string;
		// (DLSS with FSR 3 frame generation: DLSS upscales through Streamline, AMD's runtime generates past it.)
		var fsrFrameGen = fsr != null ? fsr.IsChecked == true : context.FsrAvailable && S.Bool("fsr");
		upscalerNote.Text = chosen == "fsr" ? T("upscaler.fsrNote")
		                    : chosen == "off" ? T("upscaler.offNote")
		                    : chosen == "dlss" && fsrFrameGen ? T("upscaler.dlssFsrFg")
		                    : T(context.FrameGenAvailable ? "upscaler.ok" : !context.RtxGpu ? "upscaler.noRtx" : "upscaler.noStreamline");
	}

	void UpdateFrameGenNote() {
		frameGenNote.Text = T(context.FrameGenAvailable ? "framegen.ok" : !context.RtxGpu ? "framegen.noRtx" : "framegen.noStreamline");
	}

	void ChooseStreamline() {
		var dir = FolderPicker.Pick(this, T("dialog.streamline"), S.Str("streamline"));
		if (dir == null) return;
		if (!context.UseStreamlineFolder(dir)) {
			MessageBox.Show(this, T("dialog.streamlineBad"), T("graphics.fg"), MessageBoxButton.OK, MessageBoxImage.Warning);
			return;
		}
		// (Without an NVIDIA RTX graphics card the folder is kept and the rows stay off.)
		ReadForm();
		Build();
	}

	// ------------------------------------------------------------------ Game

	FrameworkElement BuildGamePage() {
		var languages = Strings.ConsoleLanguages.Select((name, i) => new Choice(name, i));
		language = Combo(languages, S.Int("language"), 340);
		// What the game is told its sound goes to (KYTY_AUDIO_OUTPUT, issue #207): it mixes for it.
		var audioOutputs = new[] { new Choice(T("audiooutput.tv"), ""), new Choice(T("audiooutput.headphones"), "headphones"),
			new Choice(T("audiooutput.amplifier"), "amplifier"), new Choice(T("audiooutput.soundbar"), "soundbar") };
		audioOutput = Combo(audioOutputs, Math.Max(0, Array.FindIndex(audioOutputs, c => (string)c.Value == S.Str("audio_output"))), 340);
		cheats = Switch(S.Bool("cheats"));
		var choose = FlatButton("button.cheats", (s, e) => ChooseCheats());
		choose.Margin = new Thickness(16, 0, 0, 0);
		cheatState = new TextBlock { Foreground = Res<Brush>("TextDimBrush"), VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(14, 0, 0, 0) };
		padSpeaker = Switch(S.Bool("pad_speaker"));
		return Page("game.title",
			Section("game.console"),
			Row("language", language, Desc(T("language.desc"))),
			Row("audiooutput", audioOutput, Desc(T("audiooutput.desc"))),
			Section("game.controller"),
			Row("padspeaker", padSpeaker, Desc(T("padspeaker.desc"))),
			Row("padvolume", SliderRow(0, 100, 5, S.Int("pad_speaker_volume"),
				v => ((int)v).ToString(CultureInfo.InvariantCulture), v => S.Set("pad_speaker_volume", (int)v)), Desc(T("padvolume.desc"))),
			Row("padvibration", SliderRow(0, 400, 10, S.Int("pad_vibration"),
				v => F("padStrengthValue", (v / 100.0).ToString("0.0#", CultureInfo.InvariantCulture)), v => S.Set("pad_vibration", (int)v)), Desc(T("padvibration.desc"))),
			Row("padactuator", SliderRow(5, 400, 5, S.Int("pad_actuator"),
				v => F("padStrengthValue", (v / 100.0).ToString("0.0#", CultureInfo.InvariantCulture)), v => S.Set("pad_actuator", (int)v)), Desc(T("padactuator.desc"))),
			Row("padlength", SliderRow(5, 500, 5, S.Int("pad_vibration_ms"),
				v => F("padLengthValue", ((int)v).ToString(CultureInfo.InvariantCulture)), v => S.Set("pad_vibration_ms", (int)v)), Desc(T("padlength.desc"))),
			Section("game.cheats"),
			Row("cheats", Horizontal(cheats, choose, cheatState), Desc(T("cheats.desc"))));
	}

	void UpdateCheatState() {
		if (cheatState == null) return;
		if (!game.Found) {
			cheatState.Text = T("cheats.noGame");
			return;
		}
		var file = game.CheatFile(context.Root);
		if (file == null) {
			cheatState.Text = T("cheats.noFile");
			return;
		}
		int on, total;
		CheatsWindow.Count(file, out on, out total);
		cheatState.Text = F("cheats.count", on, total);
	}

	void ChooseCheats() {
		var file = game.CheatFile(context.Root);
		if (file == null) {
			MessageBox.Show(this, T(game.Found ? "cheats.noFile" : "cheats.noGame"), T("cheats"), MessageBoxButton.OK, MessageBoxImage.Information);
			return;
		}
		CheatsWindow.Edit(this, file);
		UpdateCheatState();
	}

	// ------------------------------------------------------------------ Shaders

	FrameworkElement BuildShadersPage() {
		storeLine = new TextBlock { Foreground = Res<Brush>("TextBrush"), TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 4, 0, 0) };
		// (run-windows.ps1's estimate: 55 minutes on 22 processors.)
		var minutes = (int)(Math.Ceiling(55.0 * 22 / Math.Max(1, Environment.ProcessorCount) / 10) * 10);
		var duration = minutes < 90 ? F("duration.minutes", minutes) : F("duration.hours", minutes / 60.0);
		var precompile = FlatButton("button.precompile", (s, e) => Precompile(""));
		ecores = Switch(S.Bool("ecores") && context.EfficiencyMask != 0, context.EfficiencyMask != 0);
		var hints = FlatButton("button.hints", (s, e) => Precompile("-ExportHints"));
		return Page("shaders.title",
			storeLine,
			Section("shaders.precompile"),
			Row(null, precompile, Desc(F("precompile.desc", duration))),
			Row(null, Horizontal(ecores, new TextBlock { Text = T("ecores"), Margin = new Thickness(12, 4, 0, 0), TextWrapping = TextWrapping.Wrap, MaxWidth = 520 }),
				context.EfficiencyMask != 0 ? null : Desc(T("ecores.none"))),
			Section("shaders.share"),
			Row(null, hints, Desc(T("hints.desc"))));
	}

	// ------------------------------------------------------------------ Advanced

	FrameworkElement BuildAdvancedPage() {
		hud = Combo(new[] { new Choice(T("hud.off"), 0), new Choice(T("hud.rate"), 1), new Choice(T("hud.detailed"), 2) }, Math.Min(2, Math.Max(0, S.Int("hud"))));
		redzone = Switch(S.Bool("redzone"));
		safeGpu = Switch(S.Bool("safegpu"));
		var saved = S.Str("ui_language");
		uiLanguage = Combo(new[] { new Choice(T("uilang.auto"), ""), new Choice("English", "en"), new Choice("中文（简体）", "zh") },
			saved == "en" ? 1 : saved == "zh" ? 2 : 0, 220);
		uiLanguage.SelectionChanged += (s, e) => {
			if (building) return;
			var value = Value(uiLanguage) as string ?? "";
			S.Set("ui_language", value);
			Strings.Language = value == "zh" ? UiLanguage.Chinese : value == "en" ? UiLanguage.English : Strings.SystemLanguage();
			ReadForm();
			Build();
		};
		reopenLauncher = Switch(S.Bool("reopen_launcher"));
		var logs = FlatButton("button.logs", (s, e) => OpenLogs());
		var readme = FlatButton("button.readme", (s, e) => OpenReadme());
		var root = FlatButton("button.root", (s, e) => OpenFolder(context.Root));
		readme.Margin = root.Margin = new Thickness(10, 0, 0, 0);
		// Options for the game's start as on a run.cmd line (Commands.ExtraOptions): saved as typed.
		extraArgs = new TextBox { Text = S.Str("extra_args"), MinWidth = 420 };
		extraArgs.TextChanged += (s, e) => Changed();
		// The graphics card, shown on a PC with more than one (Commands.AddGpu): automatic or one by its name.
		var gpus = new List<Choice> { new Choice(T("gpu.auto"), "") };
		foreach (var card in context.Gpus) gpus.Add(new Choice(card.Name, card.Name));
		gpu = Combo(gpus, Math.Max(0, gpus.FindIndex(c => (string)c.Value == S.Str("gpu"))), 360);
		var content = new List<UIElement> {
			Section("advanced.stability"),
			Row("hud", hud, Desc(T("hud.desc"))),
			Row("redzone", redzone, Desc(T("redzone.desc"))),
			Row("safegpu", safeGpu, Desc(T("safegpu.desc"))) };
		if (context.Gpus.Count > 1) content.AddRange(new UIElement[] { Section("advanced.gpu"), Row("gpu", gpu, Desc(T("gpu.desc"))) });
		content.AddRange(new UIElement[] {
			Section("advanced.extra"),
			Row("extra", extraArgs, Desc(T("extra.desc"))),
			Section("advanced.launcher"),
			Row("uilang", uiLanguage),
			Row("reopen", reopenLauncher, Desc(T("reopen.desc"))),
			Row(null, Horizontal(logs, readme, root)),
			Desc(T("credit")) });
		return Page("advanced.title", content.ToArray());
	}

	// ------------------------------------------------------------------ Experimental

	// Options still being tried out (not defaults yet): a player turns one on, and its issue collects how it went.
	FrameworkElement BuildExperimentalPage() {
		presentDirect = Switch(S.Bool("present_direct"));
		vrrFlips = Switch(S.Bool("vrr_flips"));
		antiLag = Switch(S.Bool("antilag") && context.AmdGpu, context.AmdGpu);
		keepResident = Switch(S.Bool("keep_resident"));
		fps30 = Switch(S.Bool("fps30"));
		return Page("experimental.title",
			Desc(T("experimental.intro")),
			Section("experimental.framerate"),
			Row("fps30", fps30, Desc(T("fps30.desc"))),
			Row(null, Feedback(85)),
			Section("experimental.display"),
			Row("presentdirect", presentDirect, Desc(T("presentdirect.desc"))),
			Row(null, Feedback(37)),
			Row("vrrflips", vrrFlips, Desc(T("vrrflips.desc"))),
			Row(null, Feedback(221)),
			Section("experimental.memory"),
			Row("keepresident", keepResident, Desc(T("keepresident.desc"))),
			Row(null, Feedback(50)),
			Section("experimental.amd"),
			Row("antilag", antiLag, Desc(T(context.AmdGpu ? "antilag.desc" : "antilag.notAmd"))),
			Row(null, Feedback(80)));
	}

	TextBlock Feedback(int issue) {
		var link = new Hyperlink(new Run(F("feedback", issue))) {
			NavigateUri = new Uri("https://github.com/chenxiao07/KytyPS5/issues/" + issue), Foreground = Res<Brush>("GoldBrush") };
		link.RequestNavigate += (s, e) => {
			try {
				Process.Start(new ProcessStartInfo(e.Uri.AbsoluteUri) { UseShellExecute = true });
			} catch (Exception) {
			}
			e.Handled = true;
		};
		return new TextBlock(link) { FontSize = 12, Margin = new Thickness(0, -6, 0, 0) };
	}

	// ------------------------------------------------------------------ state, summary, actions

	void Changed() {
		if (building) return;
		ReadForm();
		UpdateSummary();
	}

	// The Sharpening row's value back to the upscaler it shows (the other's stays as it was).
	void KeepSharpness() {
		if (sharpness == null) return;
		var value = Value(sharpness) as string ?? "";
		if (sharpnessShown == "dlss") dlssSharpness = value;
		else if (sharpnessShown == "fsr") fsrSharpness = value;
	}

	// The controls into the settings, saved (launcher.ps1's Read-Form).
	void ReadForm() {
		if (gamePath == null || render == null) return;
		S.Set("game", gamePath.Text.Trim());
		if (extraArgs != null) S.Set("extra_args", extraArgs.Text.Trim());
		var res = Value(resolution) as string;
		if (res != null) S.Set("resolution", res);
		S.Set("fullscreen", fullscreen.IsChecked == true);
		S.Set("aspect", aspect.IsChecked == true);
		S.Set("wide_bars", wideBars.IsChecked == true);
		S.Set("language", Math.Max(0, language.SelectedIndex));
		S.Set("redzone", redzone.IsChecked == true);
		S.Set("fps120", fps120.IsChecked == true);
		S.Set("vsync", vsync.IsChecked == true);
		S.Set("lowlatency", lowLatency.IsChecked == true);
		S.Set("hdr", hdr.IsChecked == true);
		S.Set("cheats", cheats.IsChecked == true);
		S.Set("pad_speaker", padSpeaker.IsChecked == true);
		if (reopenLauncher != null) S.Set("reopen_launcher", reopenLauncher.IsChecked == true);
		S.Set("safegpu", safeGpu.IsChecked == true);
		S.Set("hud", Math.Max(0, hud.SelectedIndex));
		S.Set(renderForDlss ? "render_dlss" : "render", SelectedRender());
		S.Set("ratio", Ratio());
		S.Set("quality", Value(quality) as string ?? "standard");
		S.Set("output", Value(output) as string ?? "auto");
		// (Kept as they were where the DLLs are missing.)
		if (context.FrameGenAvailable) S.Set("framegen", frameGen.SelectedIndex > 0 ? frameGen.SelectedIndex + 1 : 0);
		// (A DLSS chosen on this PC stays while its runtime is missing: the list then has no DLSS to show.)
		var chosenUpscaler = Value(upscaler) as string ?? "";
		if (!(S.Str("upscaler") == "dlss" && !context.FrameGenAvailable && chosenUpscaler.Length == 0)) S.Set("upscaler", chosenUpscaler);
		S.Set("dlss_preset", Value(dlssModel) as string ?? "");
		KeepSharpness();
		S.Set("dlss_sharpness", dlssSharpness == "0" ? "" : dlssSharpness);
		S.Set("fsr_sharpness", fsrSharpness == "0.6" ? "" : fsrSharpness);
		S.Set("audio_output", Value(audioOutput) as string ?? "");
		S.Set("gpu", Value(gpu) as string ?? "");
		if (context.FsrAvailable) S.Set("fsr", fsr.IsChecked == true);
		S.Set("ecores", ecores.IsChecked == true);
		S.Set("present_direct", presentDirect.IsChecked == true);
		S.Set("vrr_flips", vrrFlips.IsChecked == true);
		S.Set("antilag", antiLag.IsChecked == true && context.AmdGpu);
		S.Set("keep_resident", keepResident.IsChecked == true);
		S.Set("fps30", fps30.IsChecked == true);
		S.Save();
	}

	void UpdateSummary() {
		var res = S.Str("resolution").Replace("x", "×");
		var dlss = commands.Dlss;
		var upscaledBy = dlss ? T("summary.dlss") : commands.FsrUpscaler ? T("summary.fsr3")
		                 : S.Str("upscaler") == "off" ? T("summary.noAa") : T("summary.taa");
		var hdOutput = commands.HdOutput;
		var output1440 = !hdOutput && commands.Output1440;
		var superResolution = dlss || commands.FsrUpscaler;
		var height = superResolution ? S.Int("render_dlss") : S.Int("render");
		if (hdOutput && height > 1080) height = 1080;
		if (output1440 && height > 1440) height = 1440;
		// (The PS5 Pro path's 4K output takes 1080 at least, or 720, as its render height list; the 1440p output 720, or 480.)
		if (superResolution && !hdOutput && !output1440 && height < 1080 && height != 720) height = 1080;
		if (output1440 && height < 720 && height != 480) height = 720;
		string frameGeneration;
		if (context.FsrAvailable && S.Bool("fsr")) frameGeneration = T("summary.fsr");
		else if (context.FrameGenAvailable && S.Int("framegen") >= 2) frameGeneration = "DLSS " + S.Int("framegen") + "x";
		else frameGeneration = T("framegen.off");
		var qualityKey = "quality." + (S.Str("quality").Length == 0 ? "standard" : S.Str("quality"));
		var options = new List<string>();
		if (S.Bool("vsync")) options.Add(T("summary.vsync"));
		if (S.Bool("fps30")) options.Add(T("summary.fps30"));
		else if (S.Bool("fps120")) options.Add(T("summary.fps120"));
		if (S.Bool("lowlatency")) options.Add(T("lowlatency"));
		if (S.Bool("hdr")) options.Add(T("summary.hdr"));
		if (S.Bool("wide_bars") && S.Str("ratio") != "16:9") options.Add(T("summary.wideBars"));
		if (hdOutput) options.Add(T("summary.hd"));
		if (output1440) options.Add(T("summary.1440"));
		if (S.Bool("present_direct")) options.Add(T("summary.presentDirect"));
		if (S.Bool("vrr_flips")) options.Add(T("summary.vrrFlips"));
		if (S.Bool("antilag")) options.Add(T("summary.antilag"));
		if (S.Bool("keep_resident")) options.Add(T("summary.keepResident"));
		if (!S.Bool("pad_speaker")) options.Add(T("summary.padSpeakerOff"));
		if (S.Str("extra_args").Length != 0) options.Add(T("summary.extra"));

		var parts = new List<string> {
			F(S.Bool("fullscreen") ? "summary.fullscreen" : "summary.window", res), S.Str("ratio"), F("summary.render", height),
			upscaledBy };
		if (frameGeneration != T("framegen.off")) parts.Add(frameGeneration);
		parts.AddRange(options);
		Summary.Text = string.Join("   ·   ", parts);

		if (currentSettings == null) return;
		currentSettings.Children.Clear();
		currentSettings.RowDefinitions.Clear();
		var lines = new[] {
			new[] { T("nav.display"), F(S.Bool("fullscreen") ? "summary.fullscreen" : "summary.window", res) + "  ·  " + S.Str("ratio") },
			new[] { "3D", F("summary.render", height) + "  ·  " + upscaledBy },
			new[] { T("graphics.fg"), frameGeneration },
			new[] { T("quality"), T(qualityKey) },
			new[] { T("summary.options"), options.Count == 0 ? "—" : string.Join("  ·  ", options) },
		};
		for (int i = 0; i < lines.Length; i++) {
			currentSettings.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
			var label = new TextBlock { Text = lines[i][0], Foreground = Res<Brush>("TextDimBrush"), Margin = new Thickness(0, 4, 18, 4) };
			var value = new TextBlock { Text = lines[i][1], Foreground = Res<Brush>("TextBrush"), Margin = new Thickness(0, 4, 0, 4), TextWrapping = TextWrapping.Wrap };
			Grid.SetRow(label, i);
			Grid.SetRow(value, i);
			Grid.SetColumn(value, 1);
			currentSettings.Children.Add(label);
			currentSettings.Children.Add(value);
		}
	}

	void Play() {
		ReadForm();
		Process process;
		try {
			process = commands.Start(commands.PlayArguments(game.Found ? game : null));
		} catch (Exception e) {
			MessageBox.Show(this, F("error.start", e.Message), T("error.title"), MessageBoxButton.OK, MessageBoxImage.Error);
			return;
		}
		if (process == null || !S.Bool("reopen_launcher")) {
			Close();
			return;
		}
		// (Advanced page, "Back to the launcher after the game": hidden while the game's window - run.cmd's - is open.)
		Hide();
		process.EnableRaisingEvents = true;
		process.Exited += (s, e) => Dispatcher.BeginInvoke(new Action(() => {
			Show();
			Activate();
		}));
		if (process.HasExited) {
			Show();
			Activate();
		}
	}

	void Precompile(string option) {
		ReadForm();
		try {
			commands.Start(commands.PrecompileArguments(option));
		} catch (Exception e) {
			MessageBox.Show(this, F("error.start", e.Message), T("error.title"), MessageBoxButton.OK, MessageBoxImage.Error);
		}
	}

	void OpenLogs() {
		var folder = context.LogsFolder();
		try {
			Directory.CreateDirectory(folder);
		} catch (Exception) {
		}
		OpenFolder(folder);
	}

	static void OpenFolder(string folder) {
		try {
			Process.Start(new ProcessStartInfo("explorer.exe", "\"" + folder + "\"") { UseShellExecute = true });
		} catch (Exception) {
		}
	}

	void OpenReadme() {
		var file = context.Readme();
		if (file == null) return;
		try {
			Process.Start(new ProcessStartInfo("notepad.exe", "\"" + file + "\"") { UseShellExecute = true });
		} catch (Exception) {
		}
	}

	// ------------------------------------------------------------------ test pictures

	// Each page drawn into <folder>\<n>-<page>.png without showing the window (--screenshot).
	public void SaveScreenshots(string folder) {
		Directory.CreateDirectory(folder);
		if (ScreenshotPath != null) {
			building = true;
			gamePath.Text = ScreenshotPath;
			building = false;
		}
		if (storeTask != null) {
			try {
				if (storeTask.Wait(60000)) {
					storeState = storeTask.Result;
					UpdateChecks();
				}
			} catch (Exception) {
			}
		}
		var root = (FrameworkElement)Content;
		var width = (int)Width;
		var height = (int)Height;
		var names = new[] { "play", "display", "graphics", "game", "shaders", "advanced", "experimental" };
		for (int i = 0; i < pages.Count; i++) {
			Nav.SelectedIndex = i;
			ShowPage(i);
			root.Measure(new Size(width, height));
			root.Arrange(new Rect(0, 0, width, height));
			root.UpdateLayout();
			var bitmap = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Pbgra32);
			var backdrop = new DrawingVisual();
			using (var dc = backdrop.RenderOpen()) dc.DrawRectangle(Background, null, new Rect(0, 0, width, height));
			bitmap.Render(backdrop);
			bitmap.Render(root);
			var encoder = new PngBitmapEncoder();
			encoder.Frames.Add(BitmapFrame.Create(bitmap));
			using (var stream = File.Create(Path.Combine(folder, (i + 1) + "-" + names[Math.Min(i, names.Length - 1)] + ".png"))) encoder.Save(stream);
		}
	}
}

}
