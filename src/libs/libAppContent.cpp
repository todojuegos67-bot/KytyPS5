#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "loader/symbolDatabase.h"
#include "loader/systemContent.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "kernel/fileSystem.h"
#include "libs/addcont.h"

namespace Libs {

LIB_VERSION("AppContent", 1, "AppContentUtil", 1, 1);

namespace AppContent {

constexpr uint32_t APP_CONTENT_APPPARAM_SKU_FLAG_FULL   = 3;
constexpr int      APP_CONTENT_ERROR_PARAMETER          = -2133262334; /* 0x80D90002 */
constexpr int      APP_CONTENT_ERROR_NOT_MOUNTED        = -2133262332; /* 0x80D90004 */
constexpr int      APP_CONTENT_ERROR_MOUNT_FULL         = -2133262330; /* 0x80D90006 */
constexpr int      APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT = -2133262329; /* 0x80D90007 */

static std::string g_title_id;

struct AppContentInitParam {
	char reserved[32];
};

struct AppContentBootParam {
	char     reserved1[4];
	uint32_t attr;
	char     reserved2[32];
};

struct AppContentMountPoint {
	char data[16];
};

struct NpUnifiedEntitlementLabel {
	char data[17];
	char padding[3];
};

static bool TryGetTitleIdFromContentId(std::string* title_id) {
	std::string content_id;
	if (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &content_id) || content_id.empty()) {
		return false;
	}

	const uint32_t dash = Common::FindIndex(content_id, '-');
	if (!Common::IndexValid(content_id, dash)) {
		return false;
	}

	const uint32_t underscore = Common::FindIndex(content_id, '_', dash + 1);
	if (!Common::IndexValid(content_id, underscore) || underscore <= dash + 1) {
		return false;
	}

	*title_id = Common::Mid(content_id, dash + 1, underscore - dash - 1);
	return !title_id->empty();
}

static bool ResolveTitleId(std::string* title_id) {
	if (Loader::SystemContentParamSfoGetString("TITLE_ID", title_id) && !title_id->empty()) {
		return true;
	}

	if (TryGetTitleIdFromContentId(title_id)) {
		LOGF("\t PS5 mode: TITLE_ID missing, using CONTENT_ID title segment\n");
		return true;
	}

	return false;
}

int KYTY_SYSV_ABI AppContentInitialize(const AppContentInitParam* init_param,
                                       AppContentBootParam*       boot_param) {
	PRINT_NAME();

	LOGF("\t init_param = 0x%016" PRIx64 "\n"
	     "\t boot_param = 0x%016" PRIx64 "\n",
	     reinterpret_cast<uint64_t>(init_param), reinterpret_cast<uint64_t>(boot_param));

	if (boot_param != nullptr) {
		std::memset(boot_param, 0, sizeof(*boot_param));
	}

	if (ResolveTitleId(&g_title_id)) {
		LOGF("\t title_id   = %s\n", g_title_id.c_str());
	} else {
		LOGF("\t TITLE_ID missing\n");
	}

	// printf, not LOGF: this line is the canary that says a game uses AppContent at all. It comes
	// after the title id is known, because that is what the add-on directory is named.
	std::printf("AppContent: initialized; add-on content is looked for in %s\n",
	            Addcont::Directory(g_title_id).string().c_str());

	return OK;
}

int KYTY_SYSV_ABI AppContentAppParamGetInt(uint32_t param_id, int32_t* value) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(value == nullptr);

	*value     = 0;
	bool found = false;

	LOGF("\t param_id = %u\n", param_id);

	switch (param_id) {
		case 0:
			*value = APP_CONTENT_APPPARAM_SKU_FLAG_FULL;
			found  = true;
			break;
		case 1: found = Loader::SystemContentParamSfoGetInt("USER_DEFINED_PARAM_1", value); break;
		case 2: found = Loader::SystemContentParamSfoGetInt("USER_DEFINED_PARAM_2", value); break;
		case 3: found = Loader::SystemContentParamSfoGetInt("USER_DEFINED_PARAM_3", value); break;
		case 4: found = Loader::SystemContentParamSfoGetInt("USER_DEFINED_PARAM_4", value); break;
		default: EXIT("unknown param_id: %u\n", param_id);
	}

	LOGF("\t value    = %d [%s]\n", *value, found ? "found" : "not found");

	return OK;
}

static int GetAvailableSpaceKb(const AppContentMountPoint* mount_point,
                               size_t*                     available_space_kb) {
	LOGF("\t mount_point        = 0x%016" PRIx64 "\n"
	     "\t available_space_kb = 0x%016" PRIx64 "\n",
	     reinterpret_cast<uint64_t>(mount_point), reinterpret_cast<uint64_t>(available_space_kb));

	if (available_space_kb == nullptr) {
		return APP_CONTENT_ERROR_PARAMETER;
	}

	std::error_code ec;
	const auto      space = std::filesystem::space(std::filesystem::current_path(), ec);
	if (ec) {
		LOGF("\t filesystem::space failed: %s\n", ec.message().c_str());
		*available_space_kb = 1024u * 1024u;
	} else {
		*available_space_kb = static_cast<size_t>(space.available / 1024u);
	}

	if (mount_point != nullptr) {
		char mount_name[sizeof(mount_point->data) + 1] {};
		std::memcpy(mount_name, mount_point->data, sizeof(mount_point->data));
		LOGF("\t mount             = %s\n", mount_name);
	}
	LOGF("\t available_space_kb = %" PRIu64 "\n", static_cast<uint64_t>(*available_space_kb));

	return OK;
}

int KYTY_SYSV_ABI AppContentDownloadDataGetAvailableSpaceKb(const AppContentMountPoint* mount_point,
                                                            size_t* available_space_kb) {
	PRINT_NAME();
	return GetAvailableSpaceKb(mount_point, available_space_kb);
}

} // namespace AppContent

// The add-on directory itself is shared with the entitlement side in libNet, so both answer from one
// pairing: see libs/addcont.h. This is where it is read and implemented, in Libs::Addcont.
namespace Addcont {

std::filesystem::path Directory(const std::string& title_id) {
	return std::filesystem::path("Content") / title_id;
}

// The console's own content id: UP0001-PPSA01341_00-XXXXXXXXXXXXXXXX.
static bool IsContentId(const std::string& content_id, const std::string& title_id) {
	static const std::regex pattern("^[A-Z]{2}[0-9]{4}-[A-Z]{4}[0-9]{5}_[0-9]{2}-[A-Z0-9]{16}$");
	if (!std::regex_match(content_id, pattern)) {
		return false;
	}
	return title_id.empty() || content_id.compare(7, title_id.size(), title_id) == 0;
}

static std::string ContentId(const std::filesystem::path& folder, const std::string& title_id) {
	std::ifstream param(folder / "sce_sys" / "param.json");
	if (!param) {
		return {};
	}
	const auto param_json = nlohmann::json::parse(param, nullptr, false);
	if (!param_json.is_object()) {
		return {};
	}
	// applicationCategoryType marks an application's own param.json, never an add-on's.
	if (param_json.contains("applicationCategoryType")) {
		return {};
	}
	const auto content_id = param_json.value("contentId", std::string {});
	return IsContentId(content_id, title_id) ? content_id : std::string {};
}

// The entitlement label a folder answers to: the last dash-separated part of its contentId, which is
// what the guest asks about, or the folder's own name when there is no readable param.json, so a
// dump can simply be renamed to the label it carries. Cut to the guest's label field, so the mount
// and the entitlement list always speak of the same sixteen characters.
static std::string FolderLabel(const std::filesystem::path& folder, const std::string& title_id) {
	const auto content_id = ContentId(folder, title_id);
	const auto dash       = content_id.rfind('-');

	auto label = (dash != std::string::npos && dash + 1 < content_id.size())
	                 ? content_id.substr(dash + 1)
	                 : folder.filename().string();

	const auto begin = label.find_first_not_of(" \t\r\n");
	const auto end   = label.find_last_not_of(" \t\r\n");
	label            = begin == std::string::npos ? std::string() : label.substr(begin, end - begin + 1);

	if (label.size() > LABEL_SIZE) {
		label.resize(LABEL_SIZE);
	}
	return label;
}

std::vector<Folder> Folders(const std::string& title_id) {
	std::vector<Folder> addons;

	std::error_code error;
	const auto      root = Directory(title_id);
	if (!std::filesystem::is_directory(root, error)) {
		return addons;
	}
	std::vector<std::filesystem::path> paths;
	for (const auto& entry: std::filesystem::directory_iterator(root, error)) {
		std::error_code entry_error;
		// A link could stand for another add-on's folder, or lead back to this one.
		if (entry.is_directory(entry_error) && !entry.is_symlink(entry_error) && !entry_error) {
			paths.push_back(entry.path());
		}
	}
	std::sort(paths.begin(), paths.end());

	// One folder per label, the first in name order, so the mount and the list pick the same one.
	std::set<std::string> labels;
	for (const auto& path: paths) {
		const auto label = FolderLabel(path, title_id);
		if (!label.empty() && labels.insert(label).second) {
			addons.push_back({label, path});
		}
	}
	return addons;
}

std::vector<LocalEntitlement> LocalEntitlements(const std::string& title_id) {
	std::vector<LocalEntitlement> result;
	if (title_id.size() != 9) {
		return result;
	}
	std::ifstream file("dlc_emu.ini", std::ios::binary);
	if (!file) {
		return result;
	}
	// A handful of keys at most; anything larger is not a file of ours to read.
	std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (contents.size() > 64 * 1024) {
		return result;
	}

	const auto trim = [](std::string value) {
		const auto first = value.find_first_not_of(" \t\r\n");
		if (first == std::string::npos) {
			return std::string {};
		}
		const auto last = value.find_last_not_of(" \t\r\n");
		return value.substr(first, last - first + 1);
	};

	std::set<std::string>              labels;
	std::map<std::string, std::string> fields;
	bool                               psal    = false;
	bool                               invalid = false;
	const auto                         flush   = [&] {
        const auto id      = fields.find("content_id");
        const auto status  = fields.find("download_status");
        const auto service = fields.find("service_label");
        if (!psal || invalid || id == fields.end() || status == fields.end() ||
            status->second != "NO_EXTRA_DATA") {
            return;
        }
        // Only the platform service carries add-ons here.
        if (service != fields.end() && service->second != "0") {
            return;
        }
        if (!IsContentId(id->second, title_id)) {
            return;
        }
        const auto entitlement_label = id->second.substr(20);
        if (labels.insert(entitlement_label).second) {
            result.push_back({id->second, entitlement_label});
        }
	};

	std::istringstream stream(contents);
	std::string        line;
	while (std::getline(stream, line)) {
		line = trim(line);
		if (line.empty() || line[0] == '#' || line[0] == ';') {
			continue;
		}
		if (line[0] == '[') {
			flush();
			fields.clear();
			invalid = false;
			psal    = line == "[PSAL]";
			continue;
		}
		if (!psal) {
			continue;
		}
		const auto equals = line.find('=');
		if (equals == std::string::npos) {
			invalid = true;
			continue;
		}
		const auto key   = trim(line.substr(0, equals));
		const auto value = trim(line.substr(equals + 1));
		// Unknown keys could change what ownership means here; they are not ignored silently.
		if (key != "content_id" && key != "download_status" && key != "service_label" &&
		    key != "entitlement_key") {
			invalid = true;
			continue;
		}
		if (!fields.emplace(key, value).second) {
			invalid = true;
		}
	}
	flush();
	return result;
}

} // namespace Addcont

namespace AppContent {

static std::string EntitlementText(const NpUnifiedEntitlementLabel* label) {
	std::string text;
	for (size_t i = 0; i < sizeof(label->data) && label->data[i] != '\0'; i++) {
		text.push_back(label->data[i]);
	}
	return text;
}

// Mounts are remembered: the guest hands back the point it was given, and a mount of the same label
// again is handed the same one, so the console's own limit is not spent by unmounting and mounting.
static constexpr size_t ADDCONT_MOUNT_MAX = 64;

struct AddcontMount {
	std::string label;
	std::string point;
};

static std::mutex                g_addcont_mutex;
static std::vector<AddcontMount> g_addcont_mounts;

int KYTY_SYSV_ABI AppContentAddcontMount(uint32_t                         service_label,
                                         const NpUnifiedEntitlementLabel* entitlement_label,
                                         AppContentMountPoint*            mount_point) {
	PRINT_NAME();

	if (entitlement_label == nullptr || mount_point == nullptr) {
		return APP_CONTENT_ERROR_PARAMETER;
	}

	std::memset(mount_point->data, 0, sizeof(mount_point->data));

	const auto label = EntitlementText(entitlement_label);

	// Only a folder that answers to this label mounts, which is the same pairing the entitlement list
	// in libNet answers from: a mount never carries another add-on's files, and a label that matches
	// nothing says which directory to rename instead.
	const auto folders = Addcont::Folders(g_title_id);

	std::filesystem::path chosen;
	for (const auto& folder: folders) {
		if (folder.label == label) {
			chosen = folder.path;
			break;
		}
	}

	if (chosen.empty()) {
		// printf, not LOGF: the console's LOGF is silent in the release configuration.
		std::printf("AppContent: no add-on for '%s' under %s\n", label.c_str(),
		            Addcont::Directory(g_title_id).string().c_str());
		return APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT;
	}

	std::string point;
	{
		std::lock_guard lock(g_addcont_mutex);
		for (const auto& mount: g_addcont_mounts) {
			if (mount.label == label) {
				point = mount.point;
				break;
			}
		}
		if (point.empty()) {
			if (g_addcont_mounts.size() >= ADDCONT_MOUNT_MAX) {
				std::printf("AppContent: %zu add-ons are mounted already\n", g_addcont_mounts.size());
				return APP_CONTENT_ERROR_MOUNT_FULL;
			}
			// The lowest number no mount holds (after an unmount the count would name a point still in use).
			for (uint32_t index = 0; point.empty(); index++) {
				const auto name = "/addcont" + std::to_string(index);
				if (std::none_of(g_addcont_mounts.begin(), g_addcont_mounts.end(),
				                 [&](const AddcontMount& mount) { return mount.point == name; })) {
					point = name;
				}
			}
			LibKernel::FileSystem::Mount(chosen, point);
			g_addcont_mounts.push_back({label, point});
		}
	}

	std::printf("AppContent: add-on '%s' -> %s (mounted at %s)\n", label.c_str(),
	            chosen.string().c_str(), point.c_str());

	std::snprintf(mount_point->data, sizeof(mount_point->data), "%s", point.c_str());
	return OK;
}

int KYTY_SYSV_ABI AppContentAddcontUnmount(const AppContentMountPoint* mount_point) {
	PRINT_NAME();

	LOGF("\t mount_point = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(mount_point));

	if (mount_point == nullptr) {
		return APP_CONTENT_ERROR_PARAMETER;
	}

	if (mount_point->data[0] == '\0') {
		return APP_CONTENT_ERROR_NOT_MOUNTED;
	}

	{
		std::lock_guard lock(g_addcont_mutex);
		std::erase_if(g_addcont_mounts,
		              [mount_point](const auto& mount) { return mount.point == mount_point->data; });
	}
	LibKernel::FileSystem::Umount(mount_point->data);

	std::printf("AppContent: add-on unmounted (%s)\n", mount_point->data);

	return OK;
}

} // namespace AppContent

namespace LibAppContentTemporary {

namespace AppContentTemporary {

using AppContentMountPoint = AppContent::AppContentMountPoint;

static int KYTY_SYSV_ABI AppContentTemporaryDataMount2(uint32_t              option,
                                                       AppContentMountPoint* mount_point) {
	PRINT_NAME();

	LOGF("\t option      = 0x%08" PRIx32 "\n"
	     "\t mount_point = 0x%016" PRIx64 "\n",
	     option, reinterpret_cast<uint64_t>(mount_point));

	if (mount_point == nullptr) {
		return AppContent::APP_CONTENT_ERROR_PARAMETER;
	}

	std::memset(mount_point->data, 0, sizeof(mount_point->data));
	std::memcpy(mount_point->data, "/temp0", sizeof("/temp0") - 1);

	LOGF("\t mount      = %s\n", mount_point->data);

	return OK;
}

static int KYTY_SYSV_ABI AppContentTemporaryDataFormat(const AppContentMountPoint* mount_point) {
	PRINT_NAME();

	LOGF("\t mount_point = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(mount_point));

	if (mount_point == nullptr) {
		return AppContent::APP_CONTENT_ERROR_PARAMETER;
	}

	return OK;
}

static int KYTY_SYSV_ABI AppContentTemporaryDataGetAvailableSpaceKb(
    const AppContentMountPoint* mount_point, size_t* available_space_kb) {
	PRINT_NAME();
	return AppContent::GetAvailableSpaceKb(mount_point, available_space_kb);
}

} // namespace AppContentTemporary

LIB_DEFINE(InitAppContent_1_Temporary) {
	LIB_FUNC("buYbeLOGWmA", AppContentTemporary::AppContentTemporaryDataMount2);
	LIB_FUNC("a5N7lAG0y2Q", AppContentTemporary::AppContentTemporaryDataFormat);
	LIB_FUNC("SaKib2Ug0yI", AppContentTemporary::AppContentTemporaryDataGetAvailableSpaceKb);
}

} // namespace LibAppContentTemporary

LIB_DEFINE(InitAppContent_1) {
	LIB_FUNC("R9lA82OraNs", AppContent::AppContentInitialize);
	LIB_FUNC("99b82IKXpH4", AppContent::AppContentAppParamGetInt);
	LIB_FUNC("VANhIWcqYak", AppContent::AppContentAddcontMount);
	LIB_FUNC("3rHWaV-1KC4", AppContent::AppContentAddcontUnmount);
	LIB_FUNC("Gl6w5i0JokY", AppContent::AppContentDownloadDataGetAvailableSpaceKb);
	LibAppContentTemporary::InitAppContent_1_Temporary(s);
}

} // namespace Libs
