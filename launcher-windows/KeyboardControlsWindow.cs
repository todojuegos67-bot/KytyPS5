// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.Collections.Generic;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
namespace KytyLauncher {
sealed class KeyboardControlsWindow : Window {
	readonly Settings settings;
	KeyboardLayout layout;
	readonly Dictionary<string, Button> buttons = new Dictionary<string, Button>();
	readonly Dictionary<string, Image> previews = new Dictionary<string, Image>();
	readonly TextBlock status;
	readonly TextBox sensitivity;
	string capturing;
	static string T(string en, string zh) { return Strings.Language == UiLanguage.Chinese ? zh : en; }
	static readonly string[] ChineseActions = { "互动 / 确认", "翻滚 / 返回", "使用道具", "双持武器", "防御 / 左手攻击", "招架 / 左手重击", "左摇杆按下", "攻击", "重击", "锁定", "菜单", "切换法术", "切换道具", "切换左手武器", "切换右手武器", "触摸板 / 左侧", "触摸板右侧", "向前移动", "向后移动", "向左移动", "向右移动" };
	public KeyboardControlsWindow(Settings settings) {
		this.settings = settings;
		layout = KeyboardLayout.Read(settings, false);
		Title = T("Keyboard and mouse controls", "键盘和鼠标控制");
		Width = 820;
		Height = 760;
		MinWidth = 700;
		MinHeight = 500;
		WindowStartupLocation = WindowStartupLocation.CenterOwner;
		Background = (Brush)Application.Current.FindResource("WindowBrush");
		Foreground = (Brush)Application.Current.FindResource("TextBrush");
		FontFamily = new FontFamily("Segoe UI, Microsoft YaHei UI");
		FontSize = 14;
		var dock = new DockPanel { Margin = new Thickness(24) };
		Content = dock;
		var intro = new TextBlock { Text = T("Click an action, then press a key or mouse button. Assigning an input already used by another action swaps the two. F7 captures/releases the mouse in-game; F7, F8, F11 and Pause cannot be rebound.", "点击动作后按键或鼠标按钮。若输入已用于其他动作，则交换两个动作的键位。游戏内按 F7 捕获/释放鼠标；F7、F8、F11 和 Pause 不可重新绑定。"), TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 0, 0, 12) };
		DockPanel.SetDock(intro, Dock.Top);
		dock.Children.Add(intro);
		var footer = new StackPanel();
		DockPanel.SetDock(footer, Dock.Bottom);
		dock.Children.Add(footer);
		status = new TextBlock { Text = T("Changes apply with matching prompts on the next Play when the prompt option is enabled.", "启用按键提示选项后，下次游玩时应用键位和对应图标。"), TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 8, 0, 8) };
		footer.Children.Add(status);
		var sens = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 4, 0, 10) };
		sens.Children.Add(new TextBlock { Text = T("Mouse sensitivity (0.1–5; blank keeps existing): ", "鼠标灵敏度（0.1–5；留空沿用现有设置）："), VerticalAlignment = VerticalAlignment.Center });
		sensitivity = new TextBox { Text = layout.Sensitivity, Width = 75, Padding = new Thickness(5) };
		sens.Children.Add(sensitivity);
		footer.Children.Add(sens);
		var actions = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
		footer.Children.Add(actions);
		actions.Children.Add(ActionButton(T("Stop capture", "停止捕获"), () => { capturing = null; Refresh(); }));
		actions.Children.Add(ActionButton(T("Reset defaults", "恢复默认"), () => { capturing = null; layout = new KeyboardLayout(); sensitivity.Text = ""; Refresh(); status.Text = T("Default layout restored in the editor. Save to keep it.", "已在编辑器中恢复默认键位，保存后生效。"); }));
		actions.Children.Add(ActionButton(T("Cancel", "取消"), () => Close()));
		actions.Children.Add(ActionButton(T("Save", "保存"), Save));
		var rows = new StackPanel();
		dock.Children.Add(new ScrollViewer { VerticalScrollBarVisibility = ScrollBarVisibility.Auto, Content = rows });
		for (int i = 0; i < KeyboardLayout.Controls.Length; i++) {
			var control = KeyboardLayout.Controls[i];
			var row = new Grid { Margin = new Thickness(0, 2, 12, 2) };
			row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
			row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(190) });
			row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(88) });
			row.Children.Add(new TextBlock { Text = (Strings.Language == UiLanguage.Chinese ? ChineseActions[i] : KeyboardLayout.Actions[i]) + " (" + control + ")", VerticalAlignment = VerticalAlignment.Center });
			var button = new Button { Style = (Style)Application.Current.FindResource("FlatButton"), MinHeight = 34, VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(4), Content = layout.Bindings[control] };
			button.Click += (s, e) => { capturing = control; Refresh(); status.Text = T("Press a key, mouse button or scroll wheel. Use Stop capture to cancel.", "请按键、鼠标按钮或滚轮；点击停止捕获取消。"); };
			buttons.Add(control, button);
			Grid.SetColumn(button, 1);
			row.Children.Add(button);
			var image = new Image { Width = 44, Height = 44 };
			previews.Add(control, image);
			Grid.SetColumn(image, 2);
			row.Children.Add(image);
			rows.Children.Add(row);
		}
		PreviewKeyDown += OnKey;
		PreviewMouseDown += OnMouse;
		PreviewMouseWheel += (s, e) => { if (capturing != null) { Assign(e.Delta > 0 ? "Mouse:WheelUp" : "Mouse:WheelDown"); e.Handled = true; } };
		Deactivated += (s, e) => { if (capturing != null) { capturing = null; Refresh(); } };
		Refresh();
		try { layout.Validate(); }
		catch (IOException error) { status.Text = error.Message + T(" You can repair it here or reset defaults.", " 可在此修复或恢复默认设置。"); }
	}
	Button ActionButton(string text, Action action) {
		var b = new Button { Style = (Style)Application.Current.FindResource("FlatButton"), Content = text, Tag = "editor-command", Margin = new Thickness(4), Padding = new Thickness(10, 6, 10, 6) };
		b.Click += (s, e) => { capturing = null; action(); };
		return b;
	}
	void Refresh() {
		foreach (var pair in buttons) {
			pair.Value.Content = capturing == pair.Key ? T("Press an input…", "请按键…") : layout.Bindings[pair.Key];
			using (var bitmap = PromptTexture.Draw(KeyboardLayout.Label(layout.Bindings[pair.Key])))
			using (var stream = new MemoryStream()) {
				bitmap.Save(stream, System.Drawing.Imaging.ImageFormat.Png);
				stream.Position = 0;
				var source = new BitmapImage();
				source.BeginInit();
				source.CacheOption = BitmapCacheOption.OnLoad;
				source.StreamSource = stream;
				source.EndInit();
				source.Freeze();
				previews[pair.Key].Source = source;
			}
		}
	}
	void Assign(string input) {
		try {
			var other = layout.Assign(capturing, input);
			capturing = null;
			Refresh();
			status.Text = other == null ? T("Binding updated. Save to keep it.", "键位已更新，保存后生效。") : T("Swapped with ", "已交换键位：") + other;
		}
		catch (IOException error) { status.Text = error.Message; }
	}
	internal static string InputName(Key key) {
		if (key >= Key.A && key <= Key.Z)
			return key.ToString();
		if (key >= Key.D0 && key <= Key.D9)
			return ((int)key - (int)Key.D0).ToString();
		if (key >= Key.NumPad0 && key <= Key.NumPad9)
			return "Keypad " + ((int)key - (int)Key.NumPad0);
		if (key >= Key.F1 && key <= Key.F24)
			return key.ToString();
		switch (key) {
			case Key.LeftShift:
			case Key.RightShift:
				return "Left Shift";
			case Key.LeftCtrl:
			case Key.RightCtrl:
				return "Left Ctrl";
			case Key.LeftAlt:
			case Key.RightAlt:
				return "Left Alt";
			case Key.OemMinus:
				return "-";
			case Key.OemPlus:
				return "=";
			case Key.OemOpenBrackets:
				return "[";
			case Key.OemCloseBrackets:
				return "]";
			case Key.OemPipe:
			case Key.OemBackslash:
				return "\\";
			case Key.OemSemicolon:
				return ";";
			case Key.OemQuotes:
				return "'";
			case Key.OemTilde:
				return "`";
			case Key.OemComma:
				return ",";
			case Key.OemPeriod:
				return ".";
			case Key.OemQuestion:
				return "/";
			case Key.Add:
				return "Keypad +";
			case Key.Subtract:
				return "Keypad -";
			case Key.Multiply:
				return "Keypad *";
			case Key.Divide:
				return "Keypad /";
			case Key.Decimal:
				return "Keypad .";
			case Key.Enter:
				return "Return";
			case Key.Back:
				return "Backspace";
			case Key.CapsLock:
				return "CapsLock";
			case Key.PageUp:
				return "PageUp";
			case Key.PageDown:
				return "PageDown";
			default:
				return key.ToString();
		}
	}
	void OnKey(object sender, KeyEventArgs e) {
		if (capturing == null)
			return;
		e.Handled = true;
		if (!e.IsRepeat)
			Assign(InputName(e.Key == Key.System ? e.SystemKey : e.Key));
	}
	void OnMouse(object sender, MouseButtonEventArgs e) {
		if (capturing == null)
			return;
		for (var node = e.OriginalSource as DependencyObject; node != null;) {
			var button = node as Button;
			if (button != null && (string)button.Tag == "editor-command") { capturing = null; Refresh(); return; }
			node = node is Visual ? VisualTreeHelper.GetParent(node) : LogicalTreeHelper.GetParent(node);
		}
		e.Handled = true;
		Assign("Mouse:" + (e.ChangedButton == MouseButton.XButton1 ? "X1" : e.ChangedButton == MouseButton.XButton2 ? "X2" : e.ChangedButton.ToString()));
	}
	public void SaveScreenshot(string path) {
		var content = (FrameworkElement)Content;
		content.Measure(new Size(Width, Height));
		content.Arrange(new Rect(0, 0, Width, Height));
		content.UpdateLayout();
		var bitmap = new RenderTargetBitmap((int)Width, (int)Height, 96, 96, PixelFormats.Pbgra32);
		var background = new DrawingVisual();
		using (var drawing = background.RenderOpen())
			drawing.DrawRectangle(Background, null, new Rect(0, 0, Width, Height));
		bitmap.Render(background);
		bitmap.Render(content);
		var encoder = new PngBitmapEncoder();
		encoder.Frames.Add(BitmapFrame.Create(bitmap));
		using (var stream = File.Create(path))
			encoder.Save(stream);
	}
	void Save() {
		layout.Sensitivity = sensitivity.Text.Trim();
		var previous = new Dictionary<string, string>();
		foreach (var control in KeyboardLayout.Controls)
			previous[control] = settings.Str("keyboard_binding_" + control);
		var previousSensitivity = settings.Str("keyboard_mouse_sensitivity");
		try { layout.Validate(); layout.Save(settings); settings.Save(true); DialogResult = true; }
		catch (Exception error) when (error is IOException || error is UnauthorizedAccessException) {
			foreach (var pair in previous)
				settings.Set("keyboard_binding_" + pair.Key, pair.Value);
			settings.Set("keyboard_mouse_sensitivity", previousSensitivity);
			status.Text = error.Message;
		}
	}
}
}
