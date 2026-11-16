#ifndef WOLVERINE_LAUNCHER_H
#define WOLVERINE_LAUNCHER_H

#include <QAbstractButton>
#include <QPixmap>
#include <QSettings>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <memory>
#include <vector>

class QComboBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QProcess;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QStackedWidget;
class QTimer;
class QVBoxLayout;

// An on/off switch drawn as a pill, as the Ghost of Yotei fork's launcher has them.
class ToggleSwitch: public QAbstractButton {
	Q_OBJECT

public:
	explicit ToggleSwitch(QWidget* parent = nullptr);
	[[nodiscard]] QSize sizeHint() const override { return {54, 30}; }

protected:
	void paintEvent(QPaintEvent* event) override;
};

// Marvel's Wolverine's start window (session 12): Play with the checks, the settings pages, and the shader precompile
// (kyty_emulator --precompile-only / --import-hints). Laid out after the Ghost of Yotei fork's launcher; the art comes
// from the user's own game folder (sce_sys), nothing of the game ships with it.
class WolverineLauncher: public QWidget {
	Q_OBJECT

public:
	explicit WolverineLauncher(QWidget* parent = nullptr);
	~WolverineLauncher() override;
	// Each page drawn into <folder>/<n>-<page>.png without showing the window (--screenshot).
	void SaveScreenshots(const QString& folder);

protected:
	void closeEvent(QCloseEvent* event) override;
	void CopySaves();

private:
	struct Game {
		bool    found = false;
		QString dir, title_id, name, version;
		QPixmap icon, background, banner;
	};
	struct Gpu {
		QString name;
		quint64 memory = 0;
		QString driver;
	};

	QWidget* BuildSidebar();
	QWidget* BuildPlayPage();
	QWidget* BuildDisplayPage();
	QWidget* BuildGraphicsPage();
	QWidget* BuildShadersPage();
	QWidget* BuildAdvancedPage();
	QWidget* SettingsPage(const QString& title, QVBoxLayout*& body);
	void     Section(QVBoxLayout* body, const QString& title);
	void     Row(QVBoxLayout* body, const QString& label, QWidget* control, const QString& description, bool separator = true);

	void        LoadGame();
	void        UpdateChecks();
	void        UpdateSummary();
	void        UpdateBusy();
	void        UpdateCache();
	void        ChooseFolder();
	void        ChooseArchive();
	void        Play();
	void        StartGame();
	void        Precompile(const QStringList& extra);
	void        StopPrecompile();
	void        PrecompileLine(const QString& line);
	void        Save();
	[[nodiscard]] QString     Emulator() const;
	[[nodiscard]] QStringList GameArguments() const;
	[[nodiscard]] QString     HintsFile() const;
	[[nodiscard]] QString     PrecompileStamp() const;
	[[nodiscard]] bool        Precompiled() const;
	[[nodiscard]] bool        FrameGeneration(QString* why) const;

	QSettings m_settings;
	Game      m_game;
	Gpu       m_gpu;
	quint64   m_ram = 0;

	QListWidget*    m_nav   = nullptr;
	QStackedWidget* m_pages = nullptr;
	// Play
	QWidget*     m_play_page = nullptr;
	QLabel*      m_eyebrow   = nullptr;
	QLabel*      m_headline  = nullptr;
	QLabel*      m_card_art  = nullptr;
	QLabel*      m_card_name = nullptr;
	QLabel*      m_card_id   = nullptr;
	QVBoxLayout* m_checks    = nullptr;
	QLabel*      m_summary   = nullptr;
	QPushButton* m_play      = nullptr;
	QPushButton* m_precompile_play = nullptr;
	QProgressBar* m_play_progress  = nullptr;
	QLabel*       m_play_progress_text = nullptr;
	// Settings
	QComboBox*    m_resolution  = nullptr;
	ToggleSwitch* m_fullscreen  = nullptr;
	ToggleSwitch* m_stretch     = nullptr;
	ToggleSwitch* m_hdr         = nullptr;
	ToggleSwitch* m_hide_cursor = nullptr;
	QComboBox*    m_present     = nullptr;
	QComboBox*    m_render      = nullptr;
	ToggleSwitch* m_overlay     = nullptr;
	QSpinBox*     m_gpu_index   = nullptr;
	QComboBox*    m_upscaler    = nullptr;
	QWidget*      m_dlss_row    = nullptr;
	QLineEdit*    m_dlss_folder = nullptr;
	QLabel*       m_frame_gen   = nullptr;
	ToggleSwitch* m_on_start    = nullptr;
	QSpinBox*     m_threads     = nullptr;
	QLabel*       m_cache_size  = nullptr;
	QPushButton*  m_precompile_button = nullptr;
	QPushButton*  m_export_button     = nullptr;
	QPushButton*  m_stop_button       = nullptr;
	QPushButton*  m_clear_button      = nullptr;
	QProgressBar* m_progress          = nullptr;
	QLabel*       m_progress_text     = nullptr;
	ToggleSwitch* m_skip_logos  = nullptr;
	ToggleSwitch* m_log_file    = nullptr;
	QComboBox*    m_language    = nullptr;
	QLineEdit*    m_user_name   = nullptr;
	QLineEdit*    m_extra       = nullptr;

	QProcess*  m_precompile = nullptr;
	QByteArray m_pending;
	QString    m_precompile_status;
	QString    m_precompile_stamp;
	bool       m_play_after = false;
	qint64     m_game_pid   = 0;
	QTimer*    m_game_watch = nullptr;
};

#endif // WOLVERINE_LAUNCHER_H
