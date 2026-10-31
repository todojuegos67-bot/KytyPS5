#ifndef EMULATOR_SRC_LIBS_TROPHY_PAGE_H_
#define EMULATOR_SRC_LIBS_TROPHY_PAGE_H_

// (ds-troph) The emulator's in-game F1 menu (Volver al juego, Ver trofeos, Cerrar el juego) and its trophy page. The
// window thread feeds it SDL events (HandleEvent: true when the event is the menu's, the game never gets it); the
// system overlay draws it between ImGui::NewFrame and Render (Draw).

union SDL_Event;

namespace Libs::TrophyPage {

bool               HandleEvent(const SDL_Event& event);
bool               IsOpen();
unsigned long long Revision(); // changes whenever its picture does
void               Draw(float width, float height);

} // namespace Libs::TrophyPage

#endif
