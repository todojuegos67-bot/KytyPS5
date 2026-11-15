#include "wolverineLauncher.h"

#include <QApplication>
#include <QCloseEvent>
#include <QComboBox>
#include <QCoreApplication>
#include <QDesktopServices>
#include <QDir>
#include <QDirIterator>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPainter>
#include <QPainterPath>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSpinBox>
#include <QStackedWidget>
#include <QThread>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <filesystem>

#if defined(_WIN32)
#include <dxgi1_6.h>
#include <windows.h> // IWYU pragma: keep
#endif

namespace {

constexpr char TitleId[]         = "PPSA03671";
constexpr char TestedVersion[]   = "01.001.005"; // the only version this build supports (1.05)
constexpr char EmulatorExe[]     =
#if defined(_WIN32)
    "kyty_emulator.exe";
#else
    "kyty_emulator";
#endif

// Colors of the Ghost of Yotei fork's launcher: a dark sidebar, light settings pages, amber marks.
constexpr char Style[] = R"(
QWidget#root { background: #ece8df; }
QWidget#sidebar { background: #171512; }
QLabel#brand { color: #efe9de; font-family: Georgia, 'Times New Roman', serif; font-size: 34px; }
QLabel#brandSub { color: #c9c1b4; font-size: 13px; letter-spacing: 2px; }
QLabel#sideFoot { color: #b8b0a3; font-size: 12px; }
QListWidget#nav { background: transparent; border: none; color: #d9d2c6; font-size: 17px; outline: 0; }
QListWidget#nav::item { padding: 14px 30px; border-left: 3px solid transparent; }
QListWidget#nav::item:selected { background: #26221d; color: #ffffff; border-left: 3px solid #e3a73b; }
QListWidget#nav::item:hover:!selected { background: #1f1c18; }
QLabel#eyebrow { color: #5d574e; font-size: 13px; letter-spacing: 2px; }
QLabel#eyebrowDark { color: #ddd6ca; font-size: 13px; letter-spacing: 1px; }
QLabel#title { color: #1b1916; font-family: Georgia, 'Times New Roman', serif; font-size: 52px; }
QLabel#titleDark { color: #f5f0e6; font-family: Georgia, 'Times New Roman', serif; font-size: 58px; }
QLabel#section { color: #1b1916; font-family: Georgia, 'Times New Roman', serif; font-size: 30px; padding-top: 18px; }
QLabel#rowLabel { color: #1b1916; font-size: 15px; }
QLabel#desc { color: #6c665d; font-size: 13px; }
QFrame#rule { background: #cfc8bc; max-height: 1px; min-height: 1px; border: none; }
QFrame#panel { background: rgba(38, 34, 29, 225); border: 1px solid rgba(255, 255, 255, 30); border-radius: 14px; }
QLabel#panelTitle { color: #f2ece2; font-family: Georgia, 'Times New Roman', serif; font-size: 30px; background: transparent; }
QLabel#checkText { color: #efe9de; font-size: 15px; background: transparent; }
QLabel#checkDetail { color: #c3bbae; font-size: 13px; background: transparent; }
QLabel#cardName { color: #f2ece2; font-size: 16px; background: transparent; }
QLabel#cardId { color: #c3bbae; font-family: Consolas, monospace; font-size: 13px; background: transparent; }
QFrame#card { background: rgba(60, 54, 46, 160); border: 2px solid #e3a73b; border-radius: 12px; }
QWidget#bottomBar { background: #171512; }
QLabel#summary { color: #e8e1d5; font-family: Consolas, monospace; font-size: 15px; }
QPushButton { color: #1b1916; background: transparent; border: 1.5px solid #1b1916; border-radius: 10px;
              padding: 12px 22px; font-size: 15px; }
QPushButton:hover { background: rgba(0, 0, 0, 18); }
QPushButton:disabled { color: #9f988d; border-color: #b9b2a6; }
QPushButton#danger { color: #b5504a; border-color: #d9a49f; }
QPushButton#barButton { color: #f3ecdf; border: 1.5px solid #d6a24a; background: rgba(214, 162, 74, 40); font-size: 17px; }
QPushButton#barButton:disabled { color: #8c857b; border-color: #5e584f; background: transparent; }
QPushButton#playButton { color: #1b1712; border: 2px solid #f1c46a; border-radius: 10px; font-family: Georgia, serif;
                         font-size: 30px; padding: 6px 70px;
                         background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e9be6b, stop:1 #b8873a); }
QPushButton#playButton:disabled { color: #6b645a; border-color: #5e584f; background: #2b2723; }
QPushButton#folderButton { color: #f3ecdf; border-color: #d6a24a; }
QComboBox, QSpinBox, QLineEdit { color: #1b1916; background: #f6f3ed; border: 1px solid #c9c2b6; border-radius: 8px;
                                 padding: 9px 14px; font-size: 15px; min-width: 200px; }
QProgressBar { background: #d9d3c8; border: none; border-radius: 4px; max-height: 8px; text-align: center; }
QProgressBar::chunk { background: #e3a73b; border-radius: 4px; }
QProgressBar#barProgress { background: #3a352f; }
QLabel#barText { color: #e8e1d5; font-size: 14px; }
QScrollArea { background: transparent; border: none; }
QWidget#pageContent { background: transparent; }
QSpinBox::up-button, QSpinBox::down-button { width: 24px; border: none; background: transparent; }
)";

QLabel* Label(const QString& text, const char* name) {
	auto* label = new QLabel(text);
	label->setObjectName(QString::fromLatin1(name));
	label->setWordWrap(true);
	return label;
}

QFrame* Rule() {
	auto* rule = new QFrame;
	rule->setObjectName(QStringLiteral("rule"));
	rule->setFixedHeight(1);
	rule->setStyleSheet(QStringLiteral("background: #cfc8bc;"));
	return rule;
}

QString Value(const QString& line, const QString& name) {
	const QRegularExpression pattern(QStringLiteral("(?:^| )%1=(\\S+)").arg(name));
	const auto               match = pattern.match(line);
	return match.hasMatch() ? match.captured(1) : QString();
}

QString Size(quint64 bytes) {
	if (bytes >= (quint64 {1} << 30u)) return QStringLiteral("%1 GB").arg(static_cast<double>(bytes) / (1u << 30u), 0, 'f', 2);
	return QStringLiteral("%1 MB").arg(static_cast<double>(bytes) / (1u << 20u), 0, 'f', 0);
}

// A round mark before a check: good, warn or bad.
QLabel* Mark(char kind) {
	auto*       mark  = new QLabel(kind == 'g' ? QStringLiteral("✓") : kind == 'w' ? QStringLiteral("!") : QStringLiteral("✕"));
	const char* color = kind == 'g' ? "#5b9b5f" : kind == 'w' ? "#d9a12f" : "#b5504a";
	mark->setFixedSize(26, 26);
	mark->setAlignment(Qt::AlignCenter);
	mark->setStyleSheet(QStringLiteral("background: %1; color: white; border-radius: 13px; font-weight: bold; font-size: 14px;")
	                        .arg(QString::fromLatin1(color)));
	return mark;
}

// The page background: the game's art (pic2, Logan cut out, else pic0) at the right, over the page color, fading out to
// the left so the panels stay readable; on the settings pages only as a banner across the top.
class Backdrop: public QWidget {
public:
	explicit Backdrop(bool dark, QWidget* parent = nullptr): QWidget(parent), m_dark(dark) {}
	void SetPicture(const QPixmap& picture) {
		m_picture = picture;
		update();
	}

protected:
	void paintEvent(QPaintEvent* /*event*/) override {
		QPainter painter(this);
		const QColor page = m_dark ? QColor(0x1d, 0x1a, 0x16) : QColor(0xec, 0xe8, 0xdf);
		painter.fillRect(rect(), page);
		if (m_picture.isNull()) return;
		const int  height = m_dark ? this->height() : std::min(this->height(), 320);
		const auto scaled = m_picture.scaledToHeight(height, Qt::SmoothTransformation);
		painter.setOpacity(m_dark ? 0.85 : 0.45);
		painter.drawPixmap(width() - scaled.width(), 0, scaled);
		painter.setOpacity(1.0);
		auto            clear = page;
		clear.setAlpha(0);
		QLinearGradient left(0, 0, width(), 0);
		left.setColorAt(0.0, page);
		left.setColorAt(0.35, page);
		left.setColorAt(0.75, clear);
		painter.fillRect(QRect(0, 0, width(), height), left);
		if (!m_dark) {
			QLinearGradient down(0, 0, 0, height);
			down.setColorAt(0.0, clear);
			down.setColorAt(1.0, page);
			painter.fillRect(QRect(0, 0, width(), height), down);
		}
	}

private:
	bool    m_dark;
	QPixmap m_picture;
};

} // namespace

ToggleSwitch::ToggleSwitch(QWidget* parent): QAbstractButton(parent) {
	setCheckable(true);
	setCursor(Qt::PointingHandCursor);
}

void ToggleSwitch::paintEvent(QPaintEvent* /*event*/) {
	QPainter painter(this);
	painter.setRenderHint(QPainter::Antialiasing);
	const QRectF track(1, 3, 52, 26);
	painter.setPen(Qt::NoPen);
	painter.setBrush(!isEnabled() ? QColor(0xc9, 0xc2, 0xb6) : isChecked() ? QColor(0x1b, 0x19, 0x16) : QColor(0xb9, 0xb2, 0xa6));
	painter.drawRoundedRect(track, 13, 13);
	painter.setBrush(Qt::white);
	painter.drawEllipse(QPointF(isChecked() ? 40 : 14, 16), 10, 10);
}

WolverineLauncher::WolverineLauncher(QWidget* parent)
    : QWidget(parent),
      m_settings(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("wolverine_launcher.ini")),
                 QSettings::IniFormat) {
	setObjectName(QStringLiteral("root"));
	setWindowTitle(QStringLiteral("KytyPS5 · Wolverine"));
	setStyleSheet(QString::fromLatin1(Style));
	resize(1500, 960);

#if defined(_WIN32)
	// The graphics card the game runs on: the high-performance one, as Windows' graphics settings order them.
	IDXGIFactory6* factory = nullptr;
	if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory6), reinterpret_cast<void**>(&factory)))) {
		IDXGIAdapter1* adapter = nullptr;
		if (SUCCEEDED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, __uuidof(IDXGIAdapter1),
		                                                   reinterpret_cast<void**>(&adapter)))) {
			DXGI_ADAPTER_DESC1 desc {};
			adapter->GetDesc1(&desc);
			m_gpu.name   = QString::fromWCharArray(desc.Description);
			m_gpu.memory = desc.DedicatedVideoMemory;
			LARGE_INTEGER umd {};
			if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd))) {
				m_gpu.driver = QStringLiteral("%1.%2.%3.%4")
				                   .arg(HIWORD(umd.HighPart))
				                   .arg(LOWORD(umd.HighPart))
				                   .arg(HIWORD(umd.LowPart))
				                   .arg(LOWORD(umd.LowPart));
			}
			adapter->Release();
		}
		factory->Release();
	}
	MEMORYSTATUSEX memory {};
	memory.dwLength = sizeof(memory);
	if (GlobalMemoryStatusEx(&memory)) m_ram = memory.ullTotalPhys;
#endif

	auto* layout = new QHBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);
	layout->addWidget(BuildSidebar());
	m_pages = new QStackedWidget;
	m_pages->addWidget(BuildPlayPage());
	m_pages->addWidget(BuildDisplayPage());
	m_pages->addWidget(BuildGraphicsPage());
	m_pages->addWidget(BuildShadersPage());
	m_pages->addWidget(BuildAdvancedPage());
	layout->addWidget(m_pages, 1);
	connect(m_nav, &QListWidget::currentRowChanged, m_pages, &QStackedWidget::setCurrentIndex);
	m_nav->setCurrentRow(0);

	m_game_watch = new QTimer(this);
	m_game_watch->setInterval(1000);
	connect(m_game_watch, &QTimer::timeout, this, [this] {
#if defined(_WIN32)
		HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(m_game_pid));
		DWORD  code    = 0;
		const bool alive = process != nullptr && GetExitCodeProcess(process, &code) && code == STILL_ACTIVE;
		if (process != nullptr) CloseHandle(process);
#else
		const bool alive = false;
#endif
		if (!alive) {
			m_game_pid = 0;
			m_game_watch->stop();
			UpdateBusy();
			UpdateCache();
		}
	});

	LoadGame();
}

WolverineLauncher::~WolverineLauncher() = default;

void WolverineLauncher::SaveScreenshots(const QString& folder) {
	QDir().mkpath(folder);
	const char* names[] = {"play", "display", "graphics", "shaders", "advanced"};
	ensurePolished();
	for (int i = 0; i < m_pages->count(); i++) {
		m_nav->setCurrentRow(i);
		grab().save(QDir(folder).filePath(QStringLiteral("%1-%2.png").arg(i + 1).arg(QString::fromLatin1(names[i]))));
	}
}

QWidget* WolverineLauncher::BuildSidebar() {
	auto* sidebar = new QWidget;
	sidebar->setObjectName(QStringLiteral("sidebar"));
	sidebar->setAttribute(Qt::WA_StyledBackground);
	sidebar->setFixedWidth(330);
	auto* layout = new QVBoxLayout(sidebar);
	layout->setContentsMargins(0, 50, 0, 30);
	auto* brand = Label(QStringLiteral("KytyPS5"), "brand");
	auto* sub   = Label(QStringLiteral("WOLVERINE BUILD"), "brandSub");
	brand->setContentsMargins(34, 0, 0, 0);
	sub->setContentsMargins(36, 0, 0, 40);
	layout->addWidget(brand);
	layout->addWidget(sub);
	m_nav = new QListWidget;
	m_nav->setObjectName(QStringLiteral("nav"));
	for (const auto& page: {tr("Play"), tr("Display"), tr("Graphics"), tr("Shaders"), tr("Advanced")}) {
		m_nav->addItem(QStringLiteral("•   ") + page);
	}
	m_nav->setCursor(Qt::PointingHandCursor);
	layout->addWidget(m_nav, 1);
	auto* foot = Label(tr("by Senaxx\nF2 performance overlay"), "sideFoot");
	foot->setContentsMargins(34, 0, 0, 0);
	layout->addWidget(foot);
	return sidebar;
}

QWidget* WolverineLauncher::BuildPlayPage() {
	auto* page = new Backdrop(true);
	m_play_page = page;
	auto* outer = new QVBoxLayout(page);
	outer->setContentsMargins(0, 0, 0, 0);
	outer->setSpacing(0);

	auto* content = new QWidget;
	auto* layout  = new QVBoxLayout(content);
	layout->setContentsMargins(72, 70, 72, 40);
	m_eyebrow  = Label({}, "eyebrowDark");
	m_headline = Label({}, "titleDark");
	layout->addWidget(m_eyebrow);
	layout->addWidget(m_headline);
	layout->addSpacing(30);

	auto* row = new QHBoxLayout;
	// The game: its card, or the folder button.
	auto* games = new QFrame;
	games->setObjectName(QStringLiteral("panel"));
	auto* games_layout = new QVBoxLayout(games);
	games_layout->setContentsMargins(34, 30, 34, 30);
	auto* folder = new QPushButton(tr("Game folder…"));
	folder->setObjectName(QStringLiteral("folderButton"));
	connect(folder, &QPushButton::clicked, this, &WolverineLauncher::ChooseFolder);
	auto* folder_row = new QHBoxLayout;
	folder_row->addWidget(folder);
	folder_row->addStretch(1);
	games_layout->addLayout(folder_row);
	games_layout->addSpacing(18);
	auto* card = new QFrame;
	card->setObjectName(QStringLiteral("card"));
	card->setFixedWidth(260);
	auto* card_layout = new QVBoxLayout(card);
	card_layout->setContentsMargins(10, 10, 10, 12);
	m_card_art = new QLabel;
	m_card_art->setFixedSize(236, 236);
	m_card_art->setAlignment(Qt::AlignCenter);
	m_card_name = Label({}, "cardName");
	m_card_id   = Label({}, "cardId");
	card_layout->addWidget(m_card_art);
	card_layout->addWidget(m_card_name);
	card_layout->addWidget(m_card_id);
	games_layout->addWidget(card);
	games_layout->addStretch(1);
	row->addWidget(games, 3);
	row->addSpacing(36);

	auto* checks = new QFrame;
	checks->setObjectName(QStringLiteral("panel"));
	auto* checks_layout = new QVBoxLayout(checks);
	checks_layout->setContentsMargins(36, 30, 36, 30);
	checks_layout->addWidget(Label(tr("Checks"), "panelTitle"));
	checks_layout->addSpacing(10);
	m_checks = new QVBoxLayout;
	checks_layout->addLayout(m_checks);
	checks_layout->addStretch(1);
	row->addWidget(checks, 2);
	layout->addLayout(row, 1);
	outer->addWidget(content, 1);

	auto* bar = new QWidget;
	bar->setObjectName(QStringLiteral("bottomBar"));
	bar->setAttribute(Qt::WA_StyledBackground);
	auto* bar_layout = new QHBoxLayout(bar);
	bar_layout->setContentsMargins(72, 22, 72, 22);
	m_summary = Label({}, "summary");
	bar_layout->addWidget(m_summary, 1);
	auto* progress_box = new QVBoxLayout;
	m_play_progress    = new QProgressBar;
	m_play_progress->setObjectName(QStringLiteral("barProgress"));
	m_play_progress->setRange(0, 1000);
	m_play_progress->setTextVisible(false);
	m_play_progress_text = Label({}, "barText");
	progress_box->addWidget(m_play_progress_text);
	progress_box->addWidget(m_play_progress);
	bar_layout->addLayout(progress_box, 1);
	m_precompile_play = new QPushButton(tr("Precompile shaders"));
	m_precompile_play->setObjectName(QStringLiteral("barButton"));
	connect(m_precompile_play, &QPushButton::clicked, this, [this] {
		if (m_precompile != nullptr) {
			StopPrecompile();
		} else {
			Precompile({});
		}
	});
	m_play = new QPushButton(tr("PLAY"));
	m_play->setObjectName(QStringLiteral("playButton"));
	connect(m_play, &QPushButton::clicked, this, &WolverineLauncher::Play);
	bar_layout->addSpacing(24);
	bar_layout->addWidget(m_precompile_play);
	bar_layout->addSpacing(18);
	bar_layout->addWidget(m_play);
	outer->addWidget(bar);
	return page;
}

QWidget* WolverineLauncher::SettingsPage(const QString& title, QVBoxLayout*& body) {
	auto* page = new Backdrop(false);
	auto* outer = new QVBoxLayout(page);
	outer->setContentsMargins(0, 0, 0, 0);
	auto* scroll = new QScrollArea;
	scroll->setWidgetResizable(true);
	scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	scroll->setFrameShape(QFrame::NoFrame);
	scroll->viewport()->setAutoFillBackground(false);
	scroll->viewport()->setStyleSheet(QStringLiteral("background: transparent;"));
	auto* content = new QWidget;
	content->setObjectName(QStringLiteral("pageContent"));
	content->setAutoFillBackground(false);
	body = new QVBoxLayout(content);
	body->setContentsMargins(72, 70, 72, 50);
	body->setSpacing(10);
	body->addWidget(Label(tr("SETTINGS"), "eyebrow"));
	body->addWidget(Label(title, "title"));
	body->addSpacing(50);
	scroll->setWidget(content);
	outer->addWidget(scroll);
	return page;
}

void WolverineLauncher::Section(QVBoxLayout* body, const QString& title) {
	body->addWidget(Label(title, "section"));
}

void WolverineLauncher::Row(QVBoxLayout* body, const QString& label, QWidget* control, const QString& description,
                            bool separator) {
	auto* row = new QHBoxLayout;
	auto* name = Label(label, "rowLabel");
	name->setFixedWidth(420);
	name->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	row->addWidget(name, 0, Qt::AlignTop);
	auto* right = new QVBoxLayout;
	if (control != nullptr) {
		auto* line = new QHBoxLayout;
		line->addWidget(control);
		line->addStretch(1);
		right->addLayout(line);
	}
	if (!description.isEmpty()) right->addWidget(Label(description, "desc"));
	row->addLayout(right, 1);
	body->addSpacing(8);
	body->addLayout(row);
	body->addSpacing(8);
	if (separator) body->addWidget(Rule());
}

QWidget* WolverineLauncher::BuildDisplayPage() {
	QVBoxLayout* body = nullptr;
	auto*        page = SettingsPage(tr("Display"), body);
	Section(body, tr("Window"));
	m_resolution = new QComboBox;
	for (const auto* resolution: {"1280x720", "1600x900", "1920x1080", "2560x1440", "3840x2160"}) {
		m_resolution->addItem(QString::fromLatin1(resolution).replace(QLatin1Char('x'), QStringLiteral("×")),
		                      QString::fromLatin1(resolution));
	}
	m_resolution->setCurrentIndex(std::max(0, m_resolution->findData(m_settings.value("resolution", "1920x1080").toString())));
	Row(body, tr("Resolution"), m_resolution, tr("The window's size; in fullscreen the screen mode."));
	m_fullscreen = new ToggleSwitch;
	m_fullscreen->setChecked(m_settings.value("fullscreen", false).toBool());
	Row(body, tr("Fullscreen"), m_fullscreen, {});
	m_stretch = new ToggleSwitch;
	m_stretch->setChecked(m_settings.value("stretch", false).toBool());
	Row(body, tr("Stretch to the window"), m_stretch, tr("Off keeps the game's 16:9 with black bars."));
	m_hdr = new ToggleSwitch;
	m_hdr->setChecked(m_settings.value("hdr", false).toBool());
	Row(body, tr("HDR"), m_hdr, tr("Turn on \"Use HDR\" in Windows display settings first; SDR otherwise."));
	m_hide_cursor = new ToggleSwitch;
	m_hide_cursor->setChecked(m_settings.value("hide_cursor", true).toBool());
	Row(body, tr("Hide the mouse cursor"), m_hide_cursor, {}, false);
	body->addStretch(1);
	for (auto* toggle: {m_fullscreen, m_stretch, m_hdr, m_hide_cursor}) connect(toggle, &ToggleSwitch::toggled, this, &WolverineLauncher::Save);
	connect(m_resolution, &QComboBox::currentIndexChanged, this, &WolverineLauncher::Save);
	return page;
}

QWidget* WolverineLauncher::BuildGraphicsPage() {
	QVBoxLayout* body = nullptr;
	auto*        page = SettingsPage(tr("Graphics"), body);
	Section(body, tr("Frame rate"));
	m_present = new QComboBox;
	m_present->addItem(tr("Immediate (no VSync)"), QStringLiteral("Immediate"));
	m_present->addItem(tr("Mailbox"), QStringLiteral("Mailbox"));
	m_present->addItem(tr("VSync (FIFO)"), QStringLiteral("Fifo"));
	m_present->setCurrentIndex(std::max(0, m_present->findData(m_settings.value("present_mode", "Immediate").toString())));
	Row(body, tr("Presentation"), m_present, tr("VSync shows no tearing; Immediate shows each frame as soon as it is done."));
	m_overlay = new ToggleSwitch;
	m_overlay->setChecked(m_settings.value("perf_overlay", false).toBool());
	Row(body, tr("Performance overlay"), m_overlay, tr("The frame rate and frame times; F2 in the game switches it."));
	Section(body, tr("Upscaling and frame generation"));
	m_upscaler = new QComboBox;
	m_upscaler->addItem(tr("FSR 3.1 (built in)"), QStringLiteral("FSR"));
	m_upscaler->addItem(tr("NVIDIA DLSS"), QStringLiteral("DLSS"));
	m_upscaler->setCurrentIndex(std::max(0, m_upscaler->findData(m_settings.value("upscaler", "FSR").toString())));
	Row(body, tr("Upscaler"), m_upscaler,
	    tr("What upscales the game's PS5 Pro image. DLSS needs an NVIDIA RTX card and your own nvngx_dlss.dll: nothing "
	       "from NVIDIA comes with this build."));
	m_dlss_row         = new QWidget;
	auto* dlss_layout  = new QHBoxLayout(m_dlss_row);
	dlss_layout->setContentsMargins(0, 0, 0, 0);
	m_dlss_folder = new QLineEdit(m_settings.value("dlss_folder", QString()).toString());
	m_dlss_folder->setMinimumWidth(520);
	auto* dlss_browse = new QPushButton(tr("Browse"));
	dlss_layout->addWidget(m_dlss_folder);
	dlss_layout->addWidget(dlss_browse);
	Row(body, tr("DLSS folder"), m_dlss_row, tr("The folder that holds nvngx_dlss.dll, for example from a PC game that has DLSS."));
	m_frame_gen = Label({}, "rowLabel");
	Row(body, tr("Frame generation"), m_frame_gen,
	    tr("One generated frame between each two of the game's, so twice the frames shown. It comes with FSR 3.1 "
	       "upscaling; DLSS frame generation comes later."));
	Section(body, tr("Render resolution"));
	m_render = new QComboBox;
	m_render->addItem(tr("Dynamic (the game's)"), QStringLiteral("auto"));
	for (const auto* step: {"1664x936", "1792x1008", "1920x1080", "2048x1152", "2304x1296", "2560x1440", "2816x1584",
	                        "3072x1728", "3328x1872"}) {
		m_render->addItem(QString::fromLatin1(step).replace(QLatin1Char('x'), QStringLiteral("×")), QString::fromLatin1(step));
	}
	m_render->setCurrentIndex(std::max(0, m_render->findData(m_settings.value("render_resolution", "auto").toString())));
	Row(body, tr("3D render size"), m_render,
	    tr("The size the game draws its 3D at before its own upscale to 2560×1440. Dynamic sits at 1664×936 below 60 fps; "
	       "a larger size is sharper (hair, foliage) and costs GPU time and video memory."));
	Section(body, tr("Graphics card"));
	m_gpu_index = new QSpinBox;
	m_gpu_index->setRange(-1, 8);
	m_gpu_index->setSpecialValueText(tr("Automatic"));
	m_gpu_index->setValue(m_settings.value("gpu", -1).toInt());
	Row(body, tr("Graphics card to use"), m_gpu_index,
	    tr("For PCs with more than one graphics card: the game and the shader precompile both use the one chosen "
	       "(its Vulkan index). After changing it, precompile the shaders again."),
	    false);
	body->addStretch(1);
	connect(m_present, &QComboBox::currentIndexChanged, this, &WolverineLauncher::Save);
	connect(m_render, &QComboBox::currentIndexChanged, this, &WolverineLauncher::Save);
	connect(m_overlay, &ToggleSwitch::toggled, this, &WolverineLauncher::Save);
	connect(m_upscaler, &QComboBox::currentIndexChanged, this, &WolverineLauncher::Save);
	connect(m_dlss_folder, &QLineEdit::editingFinished, this, &WolverineLauncher::Save);
	connect(dlss_browse, &QPushButton::clicked, this, [this] {
		const auto dir = QFileDialog::getExistingDirectory(this, tr("The folder with nvngx_dlss.dll"), m_dlss_folder->text());
		if (dir.isEmpty()) return;
		m_dlss_folder->setText(QDir::toNativeSeparators(dir));
		Save();
	});
	connect(m_gpu_index, &QSpinBox::valueChanged, this, &WolverineLauncher::Save);
	return page;
}

QWidget* WolverineLauncher::BuildShadersPage() {
	QVBoxLayout* body = nullptr;
	auto*        page = SettingsPage(tr("Shaders"), body);
	Section(body, tr("Precompile"));
	auto* buttons       = new QWidget;
	auto* buttons_layout = new QHBoxLayout(buttons);
	buttons_layout->setContentsMargins(0, 0, 0, 0);
	m_precompile_button = new QPushButton(tr("Precompile shaders"));
	m_stop_button       = new QPushButton(tr("Pause"));
	buttons_layout->addWidget(m_precompile_button);
	buttons_layout->addWidget(m_stop_button);
	connect(m_precompile_button, &QPushButton::clicked, this, [this] { Precompile({}); });
	connect(m_stop_button, &QPushButton::clicked, this, &WolverineLauncher::StopPrecompile);
	Row(body, tr("Precompile all shaders"), buttons,
	    tr("Translates every shader of the game for this graphics card once, so new effects do not stutter. Closing the "
	       "launcher pauses it. Needed again after a driver update."),
	    false);
	auto* progress = new QWidget;
	auto* progress_layout = new QVBoxLayout(progress);
	progress_layout->setContentsMargins(0, 0, 0, 0);
	m_progress = new QProgressBar;
	m_progress->setRange(0, 1000);
	m_progress->setTextVisible(false);
	m_progress->setMinimumWidth(520);
	m_progress_text = Label({}, "desc");
	progress_layout->addWidget(m_progress);
	progress_layout->addWidget(m_progress_text);
	Row(body, {}, progress, {});
	m_on_start = new ToggleSwitch;
	m_on_start->setChecked(m_settings.value("precompile_on_start", true).toBool());
	Row(body, tr("Precompile on start"), m_on_start,
	    tr("Shaders seen in earlier sessions are prepared in the background while the game loads. Off also stops "
	       "recording new shaders for precompiling."));
	m_threads = new QSpinBox;
	m_threads->setRange(0, 64);
	m_threads->setSpecialValueText(tr("Default"));
	m_threads->setValue(m_settings.value("precompile_threads", 0).toInt());
	Row(body, tr("Precompile threads"), m_threads,
	    tr("More threads finish sooner; fewer keep the PC responsive while you play."), false);

	Section(body, tr("Cache"));
	auto* cache        = new QWidget;
	auto* cache_layout = new QVBoxLayout(cache);
	cache_layout->setContentsMargins(0, 0, 0, 0);
	auto* size_row = new QHBoxLayout;
	size_row->addWidget(Label(tr("Size on disk"), "desc"));
	m_cache_size = Label({}, "cardId");
	m_cache_size->setStyleSheet(QStringLiteral("color: #1b1916; font-family: Consolas, monospace; font-size: 15px;"));
	size_row->addWidget(m_cache_size);
	size_row->addStretch(1);
	cache_layout->addLayout(size_row);
	auto* cache_buttons = new QHBoxLayout;
	auto* open          = new QPushButton(tr("Open folder"));
	m_clear_button      = new QPushButton(tr("Clear cache"));
	m_clear_button->setObjectName(QStringLiteral("danger"));
	cache_buttons->addWidget(open);
	cache_buttons->addWidget(m_clear_button);
	cache_buttons->addStretch(1);
	cache_layout->addLayout(cache_buttons);
	Row(body, tr("Shader cache"), cache, tr("The compiled shaders, kept next to the emulator in _PipelineCache."));
	connect(open, &QPushButton::clicked, this, [this] {
		const auto folder = QFileInfo(Emulator()).dir().filePath(QStringLiteral("_PipelineCache"));
		QDir().mkpath(folder);
		QDesktopServices::openUrl(QUrl::fromLocalFile(folder));
	});
	connect(m_clear_button, &QPushButton::clicked, this, [this] {
		if (QMessageBox::question(this, tr("Clear cache"),
		                          tr("Delete the compiled shaders and the shaders recorded for precompiling? The next start "
		                             "compiles them again.")) != QMessageBox::Yes) {
			return;
		}
		QDir folder(QFileInfo(Emulator()).dir().filePath(QStringLiteral("_PipelineCache")));
		for (const auto& file: folder.entryList({QString::fromLatin1(TitleId) + QStringLiteral(".*")}, QDir::Files)) {
			folder.remove(file);
		}
		m_settings.remove(QStringLiteral("precompiled"));
		UpdateCache();
		UpdateChecks();
	});

	Section(body, tr("Share"));
	m_export_button = new QPushButton(tr("Export shader hints"));
	connect(m_export_button, &QPushButton::clicked, this, [this] {
		const auto file = QFileDialog::getSaveFileName(this, tr("Export shader hints"),
		                                               QDir::home().filePath(QStringLiteral("my-hints-%1.kyhints").arg(QString::fromLatin1(TitleId))),
		                                               tr("Shader hints (*.kyhints)"));
		if (!file.isEmpty()) Precompile({QStringLiteral("--export-hints"), QDir::toNativeSeparators(file)});
	});
	Row(body, tr("Shader hints"), m_export_button,
	    tr("The shaders your play recorded, without the game's code: the next packages can precompile them for "
	       "everyone."),
	    false);
	body->addStretch(1);
	connect(m_on_start, &ToggleSwitch::toggled, this, &WolverineLauncher::Save);
	connect(m_threads, &QSpinBox::valueChanged, this, &WolverineLauncher::Save);
	return page;
}

QWidget* WolverineLauncher::BuildAdvancedPage() {
	QVBoxLayout* body = nullptr;
	auto*        page = SettingsPage(tr("Advanced"), body);
	Section(body, tr("Game"));
	m_skip_logos = new ToggleSwitch;
	m_skip_logos->setChecked(m_settings.value("skip_logos", true).toBool());
	Row(body, tr("Skip the startup logos"), m_skip_logos, tr("The title shows after about 20 seconds instead of 60."));
	m_language = new QComboBox;
	const std::pair<int, const char*> languages[] = {{1, "English (US)"}, {18, "English (UK)"}, {2, "French"}, {3, "Spanish"},
	                                                 {4, "German"},       {5, "Italian"},       {6, "Dutch"},  {7, "Portuguese"},
	                                                 {0, "Japanese"}};
	for (const auto& [code, name]: languages) m_language->addItem(QString::fromLatin1(name), code);
	m_language->setCurrentIndex(std::max(0, m_language->findData(m_settings.value("language", 1).toInt())));
	Row(body, tr("Console language"), m_language, tr("The language the game is told; it decides its text and voices."));
	m_user_name = new QLineEdit(m_settings.value("user_name", QStringLiteral("Kyty")).toString());
	m_user_name->setMaxLength(16);
	Row(body, tr("User name"), m_user_name, {});
	Section(body, tr("Troubleshooting"));
	m_log_file = new ToggleSwitch;
	m_log_file->setChecked(m_settings.value("log_file", false).toBool());
	Row(body, tr("Log file"), m_log_file, tr("The emulator's log, _kyty.txt next to it: attach it to a bug report."));
	m_extra = new QLineEdit(m_settings.value("extra", QString()).toString());
	m_extra->setMinimumWidth(520);
	Row(body, tr("Command-line options"), m_extra,
	    tr("Added to the game's start, e.g. --vblank-frequency 120. If the game no longer starts, clear this box."), false);
	body->addWidget(Label(tr("Always on for Marvel's Wolverine: readback of linear images, tessellation and red-zone "
	                         "protection."),
	                      "desc"));
	body->addStretch(1);
	connect(m_skip_logos, &ToggleSwitch::toggled, this, &WolverineLauncher::Save);
	connect(m_log_file, &ToggleSwitch::toggled, this, &WolverineLauncher::Save);
	connect(m_language, &QComboBox::currentIndexChanged, this, &WolverineLauncher::Save);
	connect(m_user_name, &QLineEdit::editingFinished, this, &WolverineLauncher::Save);
	connect(m_extra, &QLineEdit::editingFinished, this, &WolverineLauncher::Save);
	return page;
}

void WolverineLauncher::Save() {
	m_settings.setValue("resolution", m_resolution->currentData());
	m_settings.setValue("fullscreen", m_fullscreen->isChecked());
	m_settings.setValue("stretch", m_stretch->isChecked());
	m_settings.setValue("hdr", m_hdr->isChecked());
	m_settings.setValue("hide_cursor", m_hide_cursor->isChecked());
	m_settings.setValue("present_mode", m_present->currentData());
	m_settings.setValue("render_resolution", m_render->currentData());
	m_settings.setValue("perf_overlay", m_overlay->isChecked());
	m_settings.setValue("gpu", m_gpu_index->value());
	m_settings.setValue("upscaler", m_upscaler->currentData());
	m_settings.setValue("dlss_folder", m_dlss_folder->text().trimmed());
	m_settings.setValue("precompile_on_start", m_on_start->isChecked());
	m_settings.setValue("precompile_threads", m_threads->value());
	m_settings.setValue("skip_logos", m_skip_logos->isChecked());
	m_settings.setValue("log_file", m_log_file->isChecked());
	m_settings.setValue("language", m_language->currentData());
	// The field counts characters, the emulator refuses more than 16 UTF-8 bytes (Config::MAX_USER_NAME_LENGTH).
	auto user_name = m_user_name->text();
	while (user_name.toUtf8().size() > 16) user_name.chop(user_name.back().isLowSurrogate() ? 2 : 1);
	if (user_name != m_user_name->text()) m_user_name->setText(user_name);
	m_settings.setValue("user_name", user_name.isEmpty() ? QStringLiteral("Kyty") : user_name);
	m_settings.setValue("extra", m_extra->text());
	UpdateSummary();
	UpdateChecks();
}

QString WolverineLauncher::Emulator() const {
	return QDir(QCoreApplication::applicationDirPath()).filePath(QString::fromLatin1(EmulatorExe));
}

QString WolverineLauncher::HintsFile() const {
	const auto file = QFileInfo(Emulator()).dir().filePath(QStringLiteral("hints/%1.kyhints").arg(QString::fromLatin1(TitleId)));
	return QFileInfo::exists(file) ? file : QString();
}

// What a complete precompile was made for: the graphics card, its driver, the shipped hints and the emulator build. A
// change of any needs another precompile.
QString WolverineLauncher::PrecompileStamp() const {
	const QFileInfo hints(HintsFile());
	const QFileInfo emulator(Emulator());
	return QStringLiteral("%1|%2|%3|%4:%5|%6:%7")
	    .arg(m_gpu.name, m_gpu.driver)
	    .arg(m_settings.value("gpu", -1).toInt())
	    .arg(hints.exists() ? hints.size() : 0)
	    .arg(hints.exists() ? hints.lastModified().toSecsSinceEpoch() : 0)
	    .arg(emulator.size())
	    .arg(emulator.lastModified().toSecsSinceEpoch());
}

// A package that lost its hints never looks precompiled, whatever an older stamp says.
bool WolverineLauncher::Precompiled() const {
	return !HintsFile().isEmpty() && m_settings.value("precompiled").toString() == PrecompileStamp();
}

// Whether the game will generate frames, as the emulator decides it (Fsr3::FrameGen::Requested): Pro mode, the built-in
// FSR 3.1 (FSR chosen, no --fsr-folder), and neither KYTY_FSR3 nor KYTY_FSR3_FG at 0. Command-line options come last,
// so theirs win. Off: `why` says what turns it off.
bool WolverineLauncher::FrameGeneration(QString* why) const {
	const auto extra  = QProcess::splitCommand(m_settings.value("extra", QString()).toString());
	const auto option = [&](const QString& name) {
		const auto at = extra.lastIndexOf(name);
		return at < 0 ? QString() : extra.value(at + 1);
	};
	const auto environment = QProcessEnvironment::systemEnvironment();
	auto       upscaler    = option(QStringLiteral("--pssr-upscaler"));
	if (upscaler.isEmpty()) upscaler = m_settings.value("upscaler", "FSR").toString();
	if (option(QStringLiteral("--trinity")) == QStringLiteral("false")) {
		*why = tr("base PS5 mode (--trinity false)");
	} else if (upscaler == QStringLiteral("DLSS")) {
		*why = tr("with DLSS (DLSS frame generation comes later)");
	} else if (upscaler != QStringLiteral("FSR")) {
		*why = tr("with %1 upscaling").arg(upscaler);
	} else if (!option(QStringLiteral("--fsr-folder")).isEmpty()) {
		*why = tr("with AMD's FidelityFX DLLs (--fsr-folder)");
	} else if (environment.value(QStringLiteral("KYTY_FSR3")) == QStringLiteral("0") ||
	           environment.value(QStringLiteral("KYTY_FSR3_FG")) == QStringLiteral("0")) {
		*why = tr("KYTY_FSR3 or KYTY_FSR3_FG is 0");
	} else {
		return true;
	}
	return false;
}

QStringList WolverineLauncher::GameArguments() const {
	QStringList args;
	const auto  size = m_settings.value("resolution", "1920x1080").toString().split(QLatin1Char('x'));
	args << QStringLiteral("--game") << QDir::toNativeSeparators(QDir(m_game.dir).filePath(QStringLiteral("eboot.bin")));
	args << QStringLiteral("--readback-linear-images") << QStringLiteral("--tessellation") << QStringLiteral("--redzone");
	args << QStringLiteral("--screen-width") << size.value(0, QStringLiteral("1920")) << QStringLiteral("--screen-height")
	     << size.value(1, QStringLiteral("1080"));
	args << QStringLiteral("--present-mode") << m_settings.value("present_mode", "Immediate").toString();
	if (const auto render = m_settings.value("render_resolution", "auto").toString(); render != QStringLiteral("auto")) {
		args << QStringLiteral("--render-resolution") << render;
	}
	args << QStringLiteral("--console-language") << QString::number(m_settings.value("language", 1).toInt());
	args << QStringLiteral("--user-name") << m_settings.value("user_name", QStringLiteral("Kyty")).toString();
	const std::pair<const char*, const char*> flags[] = {{"fullscreen", "--fullscreen"},   {"stretch", "--stretch"},
	                                                     {"hdr", "--hdr"},                 {"hide_cursor", "--hide-cursor"},
	                                                     {"perf_overlay", "--perf-overlay"}, {"skip_logos", "--skip-notice-screen"}};
	for (const auto& [key, flag]: flags) {
		if (m_settings.value(key, QString::fromLatin1(key) == QStringLiteral("skip_logos") ||
		                              QString::fromLatin1(key) == QStringLiteral("hide_cursor"))
		        .toBool()) {
			args << QString::fromLatin1(flag);
		}
	}
	if (const auto gpu = m_settings.value("gpu", -1).toInt(); gpu >= 0) args << QStringLiteral("--gpu") << QString::number(gpu);
	const auto upscaler = m_settings.value("upscaler", "FSR").toString();
	args << QStringLiteral("--pssr-upscaler") << upscaler;
	// Only with DLSS: the folder alone also pins the DLSS render step (wolverineRenderStep.cpp).
	if (const auto folder = m_settings.value("dlss_folder", QString()).toString(); upscaler == QStringLiteral("DLSS") && !folder.isEmpty()) {
		args << QStringLiteral("--dlss-folder") << QDir::toNativeSeparators(folder);
	}
	// 67: DLSS frame generation (--dlssg-folder, --dlssg-multiplier 2-4) goes here, with its case in FrameGeneration().
	// It runs on the built-in FSR 3.1 upscale in Pro mode and replaces FSR's frame generation; never with DLSS upscaling.
	if (m_settings.value("log_file", false).toBool()) {
		args << QStringLiteral("--printf-direction") << QStringLiteral("File") << QStringLiteral("--printf-output-file")
		     << QStringLiteral("_kyty.txt");
	}
	args << QProcess::splitCommand(m_settings.value("extra", QString()).toString());
	return args;
}

void WolverineLauncher::LoadGame() {
	m_game     = {};
	m_game.dir = m_settings.value("game_dir").toString();
	QFile param(QDir(m_game.dir).filePath(QStringLiteral("sce_sys/param.json")));
	if (!m_game.dir.isEmpty() && QFileInfo::exists(QDir(m_game.dir).filePath(QStringLiteral("eboot.bin"))) &&
	    param.open(QIODevice::ReadOnly)) {
		const auto json      = QJsonDocument::fromJson(param.readAll()).object();
		m_game.found         = true;
		m_game.title_id      = json.value(QStringLiteral("titleId")).toString();
		m_game.version       = json.value(QStringLiteral("contentVersion")).toString();
		const auto localized = json.value(QStringLiteral("localizedParameters")).toObject();
		m_game.name = localized.value(localized.value(QStringLiteral("defaultLanguage")).toString()).toObject().value(QStringLiteral("titleName")).toString();
		// The art comes from the user's own game files.
		m_game.icon.load(QDir(m_game.dir).filePath(QStringLiteral("sce_sys/icon0.png")));
		if (!m_game.background.load(QDir(m_game.dir).filePath(QStringLiteral("sce_sys/pic2.png")))) {
			m_game.background.load(QDir(m_game.dir).filePath(QStringLiteral("sce_sys/pic0.png")));
		}
	}
	if (!m_game.icon.isNull()) {
		setWindowIcon(QIcon(m_game.icon));
		m_card_art->setPixmap(m_game.icon.scaled(m_card_art->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	} else {
		m_card_art->setText(tr("No game"));
		m_card_art->setStyleSheet(QStringLiteral("color: #c3bbae;"));
	}
	for (int i = 0; i < m_pages->count(); i++) static_cast<Backdrop*>(m_pages->widget(i))->SetPicture(m_game.background);
	m_card_name->setText(m_game.found ? m_game.name : tr("Choose the game folder"));
	m_card_id->setText(m_game.found ? QStringLiteral("%1 · %2").arg(m_game.title_id, m_game.version) : QString());
	UpdateChecks();
	UpdateSummary();
	UpdateBusy();
	UpdateCache();
}

void WolverineLauncher::ChooseFolder() {
	const auto dir = QFileDialog::getExistingDirectory(this, tr("The game folder (eboot.bin, sce_sys)"),
	                                                   m_settings.value("game_dir").toString());
	if (dir.isEmpty()) return;
	m_settings.setValue("game_dir", dir);
	LoadGame();
}

// Empties a layout and every layout nested in it. Widgets are hidden now and deleted later: the one that asked for the
// update (the copy button) may still be in its signal. A label in a nested layout that is only taken out stays visible
// where it was, under the next rows.
static void ClearLayout(QLayout* layout) {
	while (auto* item = layout->takeAt(0)) {
		if (auto* widget = item->widget()) {
			widget->hide();
			widget->deleteLater();
		}
		if (auto* child = item->layout()) ClearLayout(child);
		delete item;
	}
}

void WolverineLauncher::UpdateChecks() {
	ClearLayout(m_checks);
	const auto add = [this](char kind, const QString& text, const QString& detail) {
		auto* row = new QHBoxLayout;
		row->setSpacing(16);
		row->addWidget(Mark(kind), 0, Qt::AlignTop);
		auto* lines = new QVBoxLayout;
		lines->setSpacing(4);
		lines->addWidget(Label(text, "checkText"));
		if (!detail.isEmpty()) lines->addWidget(Label(detail, "checkDetail"));
		row->addLayout(lines, 1);
		m_checks->addLayout(row);
		m_checks->addSpacing(14);
	};
	const bool right_game = m_game.found && m_game.title_id == QString::fromLatin1(TitleId);
	if (!m_game.found) {
		add('b', tr("No game folder"), tr("Choose the folder of your Marvel's Wolverine dump: eboot.bin and sce_sys."));
	} else if (!right_game) {
		add('b', tr("Not Marvel's Wolverine"), tr("%1 is %2: this build plays %3 only.").arg(m_game.dir, m_game.title_id, QString::fromLatin1(TitleId)));
	} else {
		add('g', tr("Game folder found"), QStringLiteral("%1 · eboot.bin, sce_sys").arg(QDir::toNativeSeparators(m_game.dir)));
		if (m_game.version == QString::fromLatin1(TestedVersion)) {
			add('g', tr("Game version %1 · tested").arg(m_game.version), {});
		} else {
			add('b', tr("Game version %1 · not supported").arg(m_game.version), tr("This build plays version 1.05 (01.001.005) only."));
		}
	}
	if (m_gpu.name.isEmpty()) {
		add('w', tr("Graphics card not identified"), {});
	} else {
		const auto gb = static_cast<double>(m_gpu.memory) / (1u << 30u);
		add(gb + 0.5 < 8.0 ? 'b' : 'g', QStringLiteral("%1 · %2 GB").arg(m_gpu.name).arg(gb, 0, 'f', 0),
		    gb + 0.5 < 8.0 ? tr("8 GB of video memory at least.") : QString());
	}
	if (m_settings.value("upscaler", "FSR").toString() == QStringLiteral("DLSS")) {
		const auto folder = m_settings.value("dlss_folder", QString()).toString();
		if (!m_gpu.name.isEmpty() && !m_gpu.name.contains(QStringLiteral("NVIDIA"), Qt::CaseInsensitive)) {
			add('w', tr("DLSS needs an NVIDIA RTX card"), tr("Choose FSR 3.1 on the Graphics page."));
		} else if (folder.isEmpty() || !QFileInfo::exists(QDir(folder).filePath(QStringLiteral("nvngx_dlss.dll")))) {
			add('w', tr("DLSS: no nvngx_dlss.dll"), tr("Choose its folder on the Graphics page."));
		} else {
			add('g', tr("DLSS · nvngx_dlss.dll found"), tr("No frame generation with DLSS yet."));
		}
	}
	if (m_ram != 0) {
		const auto gb = static_cast<double>(m_ram) / (1u << 30u);
		add(gb <= 20.0 ? 'w' : 'g', tr("%1 GB of RAM").arg(gb, 0, 'f', 0),
		    gb <= 20.0 ? tr("Close other programs while playing, and keep the page file 'System managed'.") : QString());
	}
	if (right_game) {
		if (HintsFile().isEmpty()) {
			add('w', tr("Shader hints missing"),
			    tr("hints\\%1.kyhints is not beside the emulator, so there is nothing to precompile. Unpack the whole release zip.")
			        .arg(QString::fromLatin1(TitleId)));
		} else if (Precompiled()) {
			add('g', tr("Shaders precompiled for this graphics card"), {});
		} else if (m_settings.contains("precompiled")) {
			add('w', tr("Shaders: precompile again"),
			    tr("The graphics card, its driver or the shipped shader hints changed since the last precompile. Play precompiles them first."));
		} else {
			add('w', tr("Shaders not precompiled yet"),
			    tr("Play precompiles them first, once for this graphics card and driver, then starts the game."));
		}
		// The working folder of a game run is the emulator's: each release's zip unpacks to a new one.
		if (!QDir(QFileInfo(Emulator()).dir().filePath(QStringLiteral("_SaveData"))).exists()) {
			add('w', tr("No saves in this folder"),
			    tr("Saves stay beside the release that wrote them, in _SaveData. New here? Ignore this. Otherwise bring them over."));
			auto* copy = new QPushButton(tr("Copy saves from another folder"));
			copy->setObjectName(QStringLiteral("barButton"));
			connect(copy, &QPushButton::clicked, this, &WolverineLauncher::CopySaves);
			m_checks->addWidget(copy, 0, Qt::AlignLeft);
			m_checks->addSpacing(14);
		}
	}
}

void WolverineLauncher::CopySaves() {
	const auto chosen = QFileDialog::getExistingDirectory(this, tr("The old release's folder, or its _SaveData"));
	if (chosen.isEmpty()) return;
	QDir from(chosen);
	if (from.dirName() != QStringLiteral("_SaveData")) from.setPath(from.filePath(QStringLiteral("_SaveData")));
	if (!from.exists()) {
		QMessageBox::warning(this, tr("Saves"), tr("No _SaveData in %1.").arg(QDir::toNativeSeparators(chosen)));
		return;
	}
	const auto      target = QFileInfo(Emulator()).dir().filePath(QStringLiteral("_SaveData"));
	std::error_code error;
	std::filesystem::copy(std::filesystem::path(from.absolutePath().toStdU16String()),
	                      std::filesystem::path(target.toStdU16String()), std::filesystem::copy_options::recursive, error);
	if (error) {
		QMessageBox::warning(this, tr("Saves"), tr("Could not copy every save:\n%1").arg(QString::fromStdString(error.message())));
	}
	UpdateChecks();
}

void WolverineLauncher::UpdateSummary() {
	const auto resolution = m_settings.value("resolution", "1920x1080").toString().replace(QLatin1Char('x'), QStringLiteral("×"));
	QStringList parts {resolution, m_settings.value("fullscreen", false).toBool() ? tr("fullscreen") : tr("window"),
	                   m_settings.value("present_mode", "Immediate").toString()};
	if (m_settings.value("hdr", false).toBool()) parts << QStringLiteral("HDR");
	const bool dlss = m_settings.value("upscaler", "FSR").toString() == QStringLiteral("DLSS");
	QString    why;
	const bool generated = FrameGeneration(&why);
	parts << (dlss ? QStringLiteral("DLSS") : QStringLiteral("FSR 3.1"))
	      << (generated ? tr("frame generation 2x") : tr("frame generation off"));
	m_frame_gen->setText(generated ? tr("2x · FSR 3.1") : tr("Off %1").arg(why));
	m_dlss_row->setEnabled(dlss);
	m_summary->setText(parts.join(QStringLiteral(" · ")));
	const bool right_game = m_game.found && m_game.title_id == QString::fromLatin1(TitleId);
	m_eyebrow->setText(right_game ? QStringLiteral("%1 · FOUND").arg(m_game.title_id) : tr("NO GAME"));
	m_headline->setText(right_game ? tr("Ready to play") : tr("Choose your game"));
}

void WolverineLauncher::UpdateBusy() {
	const bool right_game = m_game.found && m_game.title_id == QString::fromLatin1(TitleId);
	const bool precompiling = m_precompile != nullptr;
	const bool playing      = m_game_pid != 0;
	m_play->setEnabled(right_game && !precompiling && !playing);
	m_play->setText(playing ? tr("PLAYING") : tr("PLAY"));
	m_precompile_play->setEnabled(right_game && !playing);
	m_precompile_play->setText(precompiling ? tr("Pause") : tr("Precompile shaders"));
	m_precompile_button->setEnabled(right_game && !precompiling && !playing);
	m_export_button->setEnabled(right_game && !precompiling && !playing);
	m_clear_button->setEnabled(!precompiling && !playing);
	m_stop_button->setVisible(precompiling);
	m_progress->setVisible(precompiling);
	m_play_progress->setVisible(precompiling);
	m_play_progress_text->setVisible(precompiling || !m_play_progress_text->text().isEmpty());
	m_summary->setVisible(!precompiling);
}

void WolverineLauncher::UpdateCache() {
	quint64 bytes = 0;
	QDir    folder(QFileInfo(Emulator()).dir().filePath(QStringLiteral("_PipelineCache")));
	for (const auto& file: folder.entryInfoList({QString::fromLatin1(TitleId) + QStringLiteral(".*")}, QDir::Files)) {
		bytes += static_cast<quint64>(file.size());
	}
	m_cache_size->setText(bytes == 0 ? tr("No cache yet") : Size(bytes));
}

void WolverineLauncher::Play() {
	if (m_precompile != nullptr || m_game_pid != 0) return;
	// Senaxx, 2026-11-13: "Precompile first, always": a player's first start never compiles shaders while playing.
	if (!Precompiled() && !HintsFile().isEmpty()) {
		m_play_after = true;
		Precompile({});
		m_play_after = m_precompile != nullptr;
		return;
	}
	StartGame();
}

void WolverineLauncher::StartGame() {
	QProcess process;
	process.setProgram(Emulator());
	process.setArguments(GameArguments());
	process.setWorkingDirectory(QFileInfo(Emulator()).absolutePath());
	auto environment = QProcessEnvironment::systemEnvironment();
	// Off also stops the recording (pipelineCache.cpp, InitializePrecompile).
	if (!m_settings.value("precompile_on_start", true).toBool()) environment.insert(QStringLiteral("KYTY_PRECOMPILE"), QStringLiteral("0"));
	if (const auto threads = m_settings.value("precompile_threads", 0).toInt(); threads > 0) {
		environment.insert(QStringLiteral("KYTY_PRECOMPILE_THREADS"), QString::number(threads));
	}
	process.setProcessEnvironment(environment);
	if (!process.startDetached(&m_game_pid)) {
		QMessageBox::critical(this, tr("Error"), tr("Could not start:\n%1").arg(Emulator()));
		m_game_pid = 0;
		return;
	}
	m_game_watch->start();
	UpdateBusy();
}

void WolverineLauncher::Precompile(const QStringList& extra) {
	if (m_precompile != nullptr || m_game_pid != 0 || !m_game.found) return;
	QStringList options = extra;
	if (options.isEmpty()) {
		const auto hints = HintsFile();
		options = hints.isEmpty() ? QStringList {QStringLiteral("--precompile-only")}
		                          : QStringList {QStringLiteral("--import-hints"), QDir::toNativeSeparators(hints)};
		if (const auto threads = m_settings.value("precompile_threads", 0).toInt(); threads > 0) {
			options << QStringLiteral("--precompile-threads") << QString::number(threads);
		}
	}
	m_precompile_stamp = PrecompileStamp(); // the settings it starts with; the card can be changed while it runs
	m_precompile = new QProcess(this);
	m_precompile->setProgram(Emulator());
	// The game's own arguments: the same device and options, so the game's pipelines are the ones compiled.
	m_precompile->setArguments(GameArguments() + options);
	m_precompile->setWorkingDirectory(QFileInfo(Emulator()).absolutePath());
	m_precompile->setStandardErrorFile(QFileInfo(Emulator()).dir().filePath(QStringLiteral("_precompile_log.txt")));
	m_pending.clear();
	m_precompile_status.clear();
	m_progress->setValue(0);
	m_play_progress->setValue(0);
	for (auto* text: {m_progress_text, m_play_progress_text}) text->setText(tr("Starting…"));
	connect(m_precompile, &QProcess::readyReadStandardOutput, this, [this] {
		m_pending += m_precompile->readAllStandardOutput();
		for (qsizetype end = m_pending.indexOf('\n'); end >= 0; end = m_pending.indexOf('\n')) {
			PrecompileLine(QString::fromUtf8(m_pending.left(end)).trimmed());
			m_pending.remove(0, end + 1);
		}
	});
	// Qt emits no finished for this one.
	connect(m_precompile, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
		if (error != QProcess::FailedToStart) return;
		for (auto* text: {m_progress_text, m_play_progress_text}) text->setText(tr("Could not start:\n%1").arg(Emulator()));
		m_play_after = false;
		m_precompile->deleteLater();
		m_precompile = nullptr;
		UpdateBusy();
	});
	connect(m_precompile, &QProcess::finished, this, [this](int code, QProcess::ExitStatus status) {
		const bool complete = status == QProcess::NormalExit && code == 0 && m_precompile_status == QStringLiteral("complete");
		if (complete) m_settings.setValue("precompiled", m_precompile_stamp);
		if (status != QProcess::NormalExit || (code != 0 && code != 3)) {
			for (auto* text: {m_progress_text, m_play_progress_text}) {
				text->setText(tr("The precompile ended (code %1); see _precompile_log.txt.").arg(code));
			}
		}
		m_precompile->deleteLater();
		m_precompile = nullptr;
		UpdateBusy();
		UpdateChecks();
		UpdateCache();
		// Play's precompile (Stop clears m_play_after): the game starts when it ends, even after a failure, which then
		// compiles while playing rather than never starting. Directly, not through Play: a stamp that still didn't match
		// would precompile again, in a loop.
		if (std::exchange(m_play_after, false)) StartGame();
	});
	m_precompile->start();
	UpdateBusy();
}

void WolverineLauncher::StopPrecompile() {
	if (m_precompile == nullptr) return;
	// Never killed: a killed job becomes a suspect. "quit" lets the running compiles finish and saves.
	m_play_after = false;
	m_precompile->write("quit\n");
	for (auto* text: {m_progress_text, m_play_progress_text}) {
		text->setText(tr("Pausing: the running compiles finish, then the work done is saved…"));
	}
}

// The emulator's PRECOMPILE_* lines (pipelineCache.cpp, RunPrecompileOnly).
void WolverineLauncher::PrecompileLine(const QString& line) {
	QString text;
	int     value = -1;
	if (line.startsWith(QStringLiteral("PRECOMPILE_HINTS_PROGRESS "))) {
		const auto read  = Value(line, QStringLiteral("read_mb")).toDouble();
		const auto total = std::max(1.0, Value(line, QStringLiteral("total_mb")).toDouble());
		text  = tr("Finding the shaders in the game files… %1 MB").arg(read, 0, 'f', 0);
		value = static_cast<int>(std::min(1.0, read / total) * 1000.0);
	} else if (line.startsWith(QStringLiteral("PRECOMPILE_HINTS "))) {
		if (Value(line, QStringLiteral("status")) != QStringLiteral("ok")) {
			text = tr("The shader hints could not be used (%1).").arg(Value(line, QStringLiteral("reason")));
		} else if (Value(line, QStringLiteral("kind")) == QStringLiteral("export")) {
			text = tr("Exported %1 shaders and %2 pipelines.").arg(Value(line, QStringLiteral("stages")), Value(line, QStringLiteral("pipelines")));
		}
	} else if (line.startsWith(QStringLiteral("PRECOMPILE_START "))) {
		text = tr("%1 shaders and pipelines to compile").arg(Value(line, QStringLiteral("total")));
	} else if (line.startsWith(QStringLiteral("PRECOMPILE_PROGRESS "))) {
		const auto done  = Value(line, QStringLiteral("done")).toDouble();
		const auto total = Value(line, QStringLiteral("total")).toDouble();
		const auto eta   = Value(line, QStringLiteral("eta_s")).toInt();
		value            = total > 0 ? static_cast<int>(done / total * 1000.0) : 1000;
		const auto phase = Value(line, QStringLiteral("phase"));
		if (phase == QStringLiteral("save")) {
			text = tr("Saving…");
		} else if (phase == QStringLiteral("compile")) {
			text = tr("Precompiling %1 of %2 · about %3 left")
			           .arg(done, 0, 'f', 0)
			           .arg(total, 0, 'f', 0)
			           .arg(eta >= 90 ? tr("%1 minutes").arg((eta + 30) / 60) : tr("%1 seconds").arg(eta));
		}
	} else if (line.startsWith(QStringLiteral("PRECOMPILE_DONE "))) {
		m_precompile_status = Value(line, QStringLiteral("status"));
		if (m_precompile_status == QStringLiteral("complete") && Value(line, QStringLiteral("total")).toInt() == 0 && HintsFile().isEmpty()) {
			m_precompile_status = QStringLiteral("nothing"); // stores no stamp
			text = tr("No shader hints found (hints\\%1.kyhints); nothing to precompile.").arg(QString::fromLatin1(TitleId));
		} else if (m_precompile_status == QStringLiteral("complete")) {
			text = tr("Precompiled: %1 in %2 s.").arg(Value(line, QStringLiteral("total")), Value(line, QStringLiteral("seconds")));
		} else if (m_precompile_status == QStringLiteral("stopped")) {
			text = tr("Paused after %1 of %2: the next precompile continues.")
			           .arg(Value(line, QStringLiteral("done")), Value(line, QStringLiteral("total")));
		}
	}
	if (!text.isEmpty()) {
		for (auto* label: {m_progress_text, m_play_progress_text}) label->setText(text);
	}
	if (value >= 0) {
		m_progress->setValue(value);
		m_play_progress->setValue(value);
	}
}

void WolverineLauncher::closeEvent(QCloseEvent* event) {
	// Closing the launcher pauses a precompile: "quit", then its save.
	if (m_precompile != nullptr) {
		m_precompile->disconnect(this);
		m_precompile->write("quit\n");
		m_precompile->waitForFinished(120000);
	}
	Save();
	event->accept();
}

