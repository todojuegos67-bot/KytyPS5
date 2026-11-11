#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_ADDCONT_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_ADDCONT_H_

#include "common/common.h"

#include <filesystem>
#include <string>
#include <vector>

namespace Libs {

// Add-on content - the console's addcont - is kept beside the emulator, one directory per add-on:
//
//     Content\<title id>\<folder>\
//         sce_sys\param.json
//
// A folder answers to one entitlement label: the last dash-separated part of the contentId in its
// param.json, which is the sixteen characters the guest asks about, or its own name when there is no
// readable param.json. The mount in libAppContent and the entitlement list in libNet answer from
// this one pairing, so a folder cannot mount under a label the list calls unowned, and a label that
// matches nothing names the directory to rename instead of taking another add-on's files.
//
// A folder's contentId says which add-on it is, so it is read as JSON and its shape is checked (the
// console's own: UP0001-PPSA01341_00-XXXXXXXXXXXXXXXX) before any of it is used; a folder whose
// param.json is not that is treated as having no param.json at all, which leaves the folder-name
// fallback above in charge.
//
// Content the game should be told it owns although no folder holds its files - a deluxe edition's
// extras, say - is declared in dlc_emu.ini beside the emulator (the same place as Content\):
//
//     [PSAL]
//     content_id=UP0001-PPSA01341_00-XXXXXXXXXXXXXXXX
//     download_status=NO_EXTRA_DATA
//     service_label=0
//
// Only complete, title-matched declarations are read, unknown keys make a section invalid, and a
// label that a folder already answers to is not declared twice.
namespace Addcont {

// The guest's entitlement label field holds sixteen characters, so a folder's label is cut to those
// below: everything that compares a label - the mount, the list, the single-label question - works
// with the same sixteen.
constexpr size_t LABEL_SIZE = 16;

struct Folder {
	std::string           label;
	std::filesystem::path path;
};

// An add-on the game owns without its files being present, declared in dlc_emu.ini.
struct LocalEntitlement {
	std::string content_id;
	std::string label;
};

// Content\<title id>\, where the add-ons of a title are looked for.
std::filesystem::path Directory(const std::string& title_id);

// The add-ons of that directory, each with the label it answers to, in name order, one per label.
std::vector<Folder> Folders(const std::string& title_id);

// The license-only add-ons dlc_emu.ini declares for this title, one per label.
std::vector<LocalEntitlement> LocalEntitlements(const std::string& title_id);

} // namespace Addcont

} // namespace Libs

#endif /* EMULATOR_INCLUDE_EMULATOR_LIBS_ADDCONT_H_ */
