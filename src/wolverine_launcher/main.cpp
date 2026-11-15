#include "wolverineLauncher.h"

#include <QApplication>
#include <QStyleHints>

int main(int argc, char* argv[]) {
	QApplication app(argc, argv);
	QApplication::setApplicationName(QStringLiteral("Wolverine launcher"));
	// Its own colors on every PC: Windows' dark mode would paint the light settings pages dark.
	QApplication::setStyle(QStringLiteral("Fusion"));
	QApplication::styleHints()->setColorScheme(Qt::ColorScheme::Light);
	WolverineLauncher window;
	if (const auto at = QApplication::arguments().indexOf(QStringLiteral("--screenshot")); at > 0 && at + 1 < argc) {
		window.SaveScreenshots(QApplication::arguments().at(at + 1));
		return 0;
	}
	window.show();
	return QApplication::exec();
}
