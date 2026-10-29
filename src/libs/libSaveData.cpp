#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "kernel/fileSystem.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/saveDataMountSlots.h"
#include "libs/saveDataBackup.h"
#include "libs/saveDataCapacity.h"
#include "libs/saveDataInfo.h"
#include "loader/symbolDatabase.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <fstream>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <vector>

namespace Libs {

LIB_VERSION("SaveData", 1, "SaveData", 1, 1);

namespace SaveData {

// TODO(): specify dir at launcher
static constexpr char     SAVE_DATA_DIR[]      = "_SaveData";

struct SceSaveDataDirName {
	char data[32];
};

struct SceSaveDataTitleId {
	char data[10];
	char padding[6];
};

struct SceSaveDataFingerprint {
	char data[65];
	char padding[15];
};

struct SaveDataParam;

enum class SaveDataSortKey : uint32_t {
	DirName    = 0,
	UserParam  = 1,
	Blocks     = 2,
	Mtime      = 3,
	FreeBlocks = 5,
};

enum class SaveDataSortOrder : uint32_t {
	Ascent  = 0,
	Descent = 1,
};

struct SaveDataDirNameSearchCond {
	int32_t                   user_id;
	int32_t                   pad;
	const SceSaveDataTitleId* title_id;
	const SceSaveDataDirName* dir_name;
	SaveDataSortKey           key;
	SaveDataSortOrder         order;
	uint8_t                   reserved[32];
};

struct SaveDataSearchInfo {
	uint64_t blocks;
	uint64_t free_blocks;
	uint8_t  reserved[32];
};

struct SaveDataDirNameSearchResult {
	uint32_t            hit_num;
	int32_t             pad;
	SceSaveDataDirName* dir_names;
	uint32_t            dir_names_num;
	uint32_t            set_num;
	SaveDataParam*      params;
	SaveDataSearchInfo* infos;
	uint8_t             reserved[12];
	int32_t             pad2;
};

struct SaveDataMountPoint {
	char data[16];
};

struct SaveDataMount3 {
	int                       user_id;
	int                       pad;
	const SceSaveDataDirName* dir_name;
	uint64_t                  blocks;
	uint64_t                  system_blocks;
	uint32_t                  mount_mode;
	int                       pad2;
	int32_t                   resource;
	uint8_t                   reserved[32];
};

struct SaveDataMountResult {
	SaveDataMountPoint mount_point;
	uint64_t           required_blocks;
	uint32_t           unused;
	uint32_t           mount_status;
	uint8_t            reserved[28];
	int                pad;
};

struct SaveDataParam {
	char     title[128];
	char     sub_title[128];
	char     detail[1024];
	uint32_t user_param;
	int      pad;
	int64_t  mtime;
	uint8_t  reserved[32];
};

struct SaveDataMountInfo {
	uint64_t blocks;
	uint64_t free_blocks;
	uint8_t  reserved[32];
};

struct SaveDataIcon {
	void*   buf;
	size_t  buf_size;
	size_t  data_size;
	uint8_t reserved[32];
};

struct SaveDataMemoryData {
	void*   buf;
	size_t  buf_size;
	int64_t offset;
	uint8_t reserved[40];
};

struct SaveDataMemoryGet2 {
	int32_t             user_id;
	uint8_t             padding[4];
	SaveDataMemoryData* data;
	SaveDataParam*      param;
	SaveDataIcon*       icon;
	uint32_t            slot_id;
	uint8_t             reserved[28];
};

struct SaveDataMemorySetup2 {
	uint32_t             option;
	int32_t              user_id;
	size_t               memory_size;
	size_t               icon_memory_size;
	const SaveDataParam* init_param;
	const SaveDataIcon*  init_icon;
	uint32_t             slot_id;
	uint8_t              reserved[20];
};

struct SaveDataMemorySetupResult {
	size_t  existed_memory_size;
	uint8_t reserved[16];
};

struct SaveDataMemorySet2 {
	int32_t                   user_id;
	uint8_t                   padding[4];
	const SaveDataMemoryData* data;
	const SaveDataParam*      param;
	const SaveDataIcon*       icon;
	uint32_t                  data_num;
	uint32_t                  slot_id;
	uint8_t                   reserved[24];
};

struct SaveDataTransferringMount {
	int32_t                   user_id;
	const SceSaveDataTitleId* title_id;
	const SceSaveDataDirName* dir_name;
	const void*               fingerprint;
	uint8_t                   reserved[32];
};

struct SaveDataPrepareParam {
	int32_t  resource;
	uint32_t prepare_mode;
	uint8_t  reserved[32];
};

struct SaveDataCommitParam {
	int32_t  resource;
	uint32_t commit_mode;
	uint8_t  reserved[32];
};

struct SaveDataDelete {
	int32_t                   user_id;
	int32_t                   pad;
	const SceSaveDataTitleId* title_id;
	const SceSaveDataDirName* dir_name;
	uint32_t                  unused;
	uint8_t                   reserved[32];
	int32_t                   pad2;
};

struct SaveDataEvent {
	uint32_t           type;
	int32_t            error_code;
	int32_t            user_id;
	uint8_t            padding[4];
	SceSaveDataTitleId title_id;
	SceSaveDataDirName dir_name;
	uint8_t            reserved[40];
};

struct SaveDataBackup {
	int32_t                   user_id;
	int32_t                   pad;
	const SceSaveDataTitleId* title_id;
	const SceSaveDataDirName* dir_name;
	const void*               fingerprint;
	uint8_t                   reserved[32];
};

static constexpr uint32_t SAVE_DATA_UMOUNT_MODE_BACKUP_ASYNC = (1u << 16u);
static constexpr uint32_t SAVE_DATA_COMMIT_MODE_BACKUP_ASYNC = 1u;

static constexpr uint32_t SAVE_DATA_EVENT_TYPE_UMOUNT_BACKUP_END = 1u;
static constexpr uint32_t SAVE_DATA_EVENT_TYPE_BACKUP_END        = 2u;
static constexpr uint32_t SAVE_DATA_EVENT_TYPE_COMMIT_BACKUP_END = 4u;

static std::vector<uint8_t>      g_save_data_memory(0x10000);
static int32_t                   g_next_transaction_resource = 1;
static std::deque<SaveDataEvent> g_save_data_events;
static SaveDataMountSlots        g_mount_slots;
static Common::Mutex             g_mount_mutex;
// (ds-pc-port8) The PC port's save-and-quit (F1): once set, no new save can start (BlockForExit).
static std::atomic<bool>         g_exit_block {false};
static std::atomic<uint64_t>     g_player_mounts {0};

static std::string get_title_id();
// The save data memory (SaveDataSetSaveDataMemory2) kept on disk: it was only in RAM, and lost at every exit.
static std::filesystem::path save_memory_file() {
	return std::filesystem::path(SAVE_DATA_DIR) / get_title_id() / "_SaveDataMemory.bin";
}
static void load_save_memory() {
	static bool loaded = false;
	if (loaded) return;
	loaded = true;
	std::ifstream file(save_memory_file(), std::ios::binary);
	if (!file) return;
	std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (data.size() > g_save_data_memory.size()) g_save_data_memory.resize(data.size());
	std::memcpy(g_save_data_memory.data(), data.data(), data.size());
	std::printf("[save] save data memory loaded: %zu bytes\n", data.size());
}
static void store_save_memory() {
	std::error_code error;
	std::filesystem::create_directories(save_memory_file().parent_path(), error);
	const auto temporary = save_memory_file().string() + ".tmp";
	{
		std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
		if (!file) return;
		file.write(reinterpret_cast<const char*>(g_save_data_memory.data()), static_cast<std::streamsize>(g_save_data_memory.size()));
		if (!file) return;
	}
	std::filesystem::rename(temporary, save_memory_file(), error);
}

static std::string get_title_id() {
	std::string title_id;
	if (!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) {
		title_id = "UNKNOWN";
	}

	return title_id;
}

static void queue_save_data_event(uint32_t type, int32_t user_id,
                                  const SceSaveDataTitleId* title_id,
                                  const SceSaveDataDirName* dir_name, int32_t error_code = OK) {
	SaveDataEvent event = {};
	event.type          = type;
	event.error_code    = error_code;
	event.user_id       = user_id;
	if (title_id != nullptr) {
		std::memcpy(&event.title_id, title_id, sizeof(event.title_id));
	}
	if (dir_name != nullptr) {
		std::memcpy(&event.dir_name, dir_name, sizeof(event.dir_name));
	}

	g_save_data_events.push_back(event);
	if (g_save_data_events.size() > 20) {
		g_save_data_events.pop_front();
	}
}

static bool dir_name_match(const char* str, const char* pattern) {
	while (*str != '\0' && *pattern != '\0') {
		if (*pattern == '%') {
			for (const char* s = str;; s++) {
				if (dir_name_match(s, pattern + 1)) {
					return true;
				}
				if (*s == '\0') {
					break;
				}
			}
			return false;
		}
		if (*pattern == '_') {
			str++;
			pattern++;
			continue;
		}
		if (*pattern != *str) {
			return false;
		}
		str++;
		pattern++;
	}
	return *str == '\0' && *pattern == '\0';
}

static std::filesystem::path capacity_metadata(std::string_view name) {
	// Keep emulator bookkeeping outside guest-visible save directories.
	return std::filesystem::path(SAVE_DATA_DIR) / ".kyty-capacity" / get_title_id() /
	       (std::string(name) + ".blocks");
}

// The texts and user parameter the game gave a slot (SaveDataSetParam), kept beside the capacity bookkeeping: the
// system's save data list shows them (SaveDataDialog), and SaveDataGetParam and searches report them as the PS5 does.
struct StoredParam {
	char     title[128];
	char     sub_title[128];
	char     detail[1024];
	uint32_t user_param;
};

static std::filesystem::path param_metadata(std::string_view name) {
	return std::filesystem::path(SAVE_DATA_DIR) / ".kyty-param" / get_title_id() / (std::string(name) + ".param");
}

static StoredParam read_param(std::string_view name) {
	StoredParam   param {};
	std::ifstream file(param_metadata(name), std::ios::binary);
	if (!file.read(reinterpret_cast<char*>(&param), sizeof(param))) param = {};
	param.title[sizeof(param.title) - 1]         = '\0';
	param.sub_title[sizeof(param.sub_title) - 1] = '\0';
	param.detail[sizeof(param.detail) - 1]       = '\0';
	return param;
}

// (Written beside it, flushed, then renamed over it: a crash leaves the old or the new one, not zeros. Issue #73.)
static void write_param(std::string_view name, const StoredParam& param) {
	std::error_code ec;
	const auto      path = param_metadata(name);
	std::filesystem::create_directories(path.parent_path(), ec);
	auto temporary = path;
	temporary += ".tmp";
	{
		std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
		file.write(reinterpret_cast<const char*>(&param), sizeof(param));
		if (!file) return;
	}
	Backup::FlushFile(temporary);
	std::filesystem::rename(temporary, path, ec);
}

// Issue #73 (libs/saveDataBackup.h): a slot's files, and the folder of its backups.
static Backup::SlotFiles slot_files(std::string_view name) {
	return {std::filesystem::path(SAVE_DATA_DIR) / get_title_id() / std::string(name), param_metadata(name),
	        capacity_metadata(name)};
}

static std::filesystem::path backup_root(std::string_view name) {
	return std::filesystem::path(SAVE_DATA_DIR) / ".kyty-backup" / get_title_id() / std::string(name);
}

// The slot's files flushed to disk, and a backup of them made when they changed since the last one.
static void keep_safe(std::string_view name) {
	static bool noted = false;
	const auto  slot  = slot_files(name);
	Backup::FlushSlot(slot);
	if (const auto made = Backup::Make(slot, backup_root(name), std::time(nullptr)); made && !noted) {
		noted = true;
		std::printf("Save data: each save is also kept in %s (the last %zu, at least %d minutes apart)\n",
		            backup_root(name).parent_path().string().c_str(), Backup::Generations,
		            static_cast<int>(Backup::ReplaceSeconds / 60));
	}
}

// A slot left broken by a crash (a file of zeros) gets its newest good backup back before the game reads it.
static void repair(std::string_view name) {
	const auto slot = slot_files(name);
	if (!Backup::Broken(slot)) return;
	const auto root = backup_root(name);
	if (const auto restored = Backup::Restore(slot, root, std::time(nullptr))) {
		std::printf("Save data: %.*s had a file of zeros (as a crash while saving leaves it): its backup %s is put back; "
		            "the broken files are kept in %s\n",
		            static_cast<int>(name.size()), name.data(), restored->c_str(), root.string().c_str());
	} else {
		std::printf("Save data: %.*s has a file of zeros (as a crash while saving leaves it) and no good backup in %s\n",
		            static_cast<int>(name.size()), name.data(), root.string().c_str());
	}
	std::fflush(stdout);
}

// A slot's modification time (time_t) as the PS5 reports it: the time of its last write, here its newest file.
static int64_t save_data_mtime(const std::filesystem::path& directory) {
	int64_t newest = 0;
	for (const auto& entry: Common::File::GetDirEntries(directory)) {
		if (entry.is_file) {
			newest = std::max(newest, static_cast<int64_t>(
			                              Common::File::GetLastWriteTimeUTC(directory / entry.name).ToUnix()));
		}
	}
	return newest;
}

SaveSlotInfo GetSaveSlotInfo(std::string_view dir_name) {
	const auto param = read_param(dir_name);
	return {param.title, param.sub_title, param.detail,
	        save_data_mtime(std::filesystem::path(SAVE_DATA_DIR) / get_title_id() / std::string(dir_name))};
}

static int MountedSlotsLocked() {
	int mounted = 0;
	for (size_t slot = 0; slot < SaveDataMountSlots::COUNT; slot++) {
		if (g_mount_slots.IsMounted(slot)) mounted++;
	}
	return mounted;
}

int MountedSlots() {
	Common::LockGuard lock(g_mount_mutex);
	return MountedSlotsLocked();
}

uint64_t PlayerProfileMounts() {
	return g_player_mounts.load(std::memory_order_acquire);
}

std::string TitleId() {
	return get_title_id();
}

bool BlockForExit(int timeout_ms, int settle_ms) {
	using Clock      = std::chrono::steady_clock;
	const auto start = Clock::now();
	auto       idle  = Clock::time_point {};
	bool       quiet = false;
	for (;;) {
		const bool busy = MountedSlots() != 0 || LibKernel::FileSystem::OpenSaveWrites() != 0;
		const auto now  = Clock::now();
		if (busy) {
			quiet = false;
		} else if (!quiet) {
			quiet = true;
			idle  = now;
		}
		if (quiet && now - idle >= std::chrono::milliseconds(settle_ms)) {
			Common::LockGuard lock(g_mount_mutex);
			if (MountedSlotsLocked() == 0 && LibKernel::FileSystem::OpenSaveWrites() == 0) {
				g_exit_block.store(true, std::memory_order_release);
				std::printf("[save] closing (F1): no save in progress; new saves refused from now on\n");
				std::fflush(stdout);
				return true;
			}
			quiet = false;
		}
		if (now - start >= std::chrono::milliseconds(timeout_ms)) {
			Common::LockGuard lock(g_mount_mutex);
			g_exit_block.store(true, std::memory_order_release);
			std::printf("[save] closing (F1): a save still busy after %d ms (%d slot(s) mounted, %d file(s) open); new saves refused\n",
			            timeout_ms, MountedSlotsLocked(), LibKernel::FileSystem::OpenSaveWrites());
			std::fflush(stdout);
			return false;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
}

void FlushAll() {
	const auto title = std::filesystem::path(SAVE_DATA_DIR) / get_title_id();
	std::error_code error;
	for (const auto& entry: std::filesystem::directory_iterator(title, error)) {
		if (entry.is_directory(error)) Backup::FlushSlot(slot_files(entry.path().filename().string()));
	}
}

static int mount_save_data(int slot, std::string_view dir_name, const std::string& directory,
                           uint32_t status, SaveDataMountResult* result) {
	const std::string mount_point = SaveDataMountSlots::MountPoint(static_cast<size_t>(slot));
	LibKernel::FileSystem::Mount(directory, mount_point);
	g_mount_slots.Mount(static_cast<size_t>(slot), dir_name);
	std::snprintf(result->mount_point.data, sizeof(result->mount_point.data), "%s",
	              mount_point.c_str());
	result->required_blocks = 0;
	result->mount_status    = status;
	return OK;
}

int KYTY_SYSV_ABI SaveDataInitialize3(const void* /*init*/) {
	PRINT_NAME();

	// EXIT_NOT_IMPLEMENTED(init != nullptr);

	return OK;
}

int KYTY_SYSV_ABI SaveDataTerminate() {
	PRINT_NAME();

	Common::LockGuard lock(g_mount_mutex);
	if (!g_mount_slots.Empty()) {
		return SAVE_DATA_ERROR_BUSY;
	}
	g_save_data_events.clear();

	return OK;
}

int KYTY_SYSV_ABI SaveDataCreateTransactionResource(uint32_t size) {
	PRINT_NAME();

	LOGF("\t size = %" PRIu32 "\n", size);

	return g_next_transaction_resource++;
}

int KYTY_SYSV_ABI SaveDataDeleteTransactionResource(int32_t resource) {
	PRINT_NAME();

	LOGF("\t resource = %" PRId32 "\n", resource);

	return OK;
}

int KYTY_SYSV_ABI SaveDataDirNameSearch(const SaveDataDirNameSearchCond* cond,
                                        SaveDataDirNameSearchResult*     result) {
	PRINT_NAME();

	if (cond == nullptr || result == nullptr ||
	    static_cast<uint32_t>(cond->key) > static_cast<uint32_t>(SaveDataSortKey::FreeBlocks) ||
	    static_cast<uint32_t>(cond->order) > static_cast<uint32_t>(SaveDataSortOrder::Descent)) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t user_id       = %d\n"
	     "\t title_id      = %s\n"
	     "\t dir_name      = %s\n"
	     "\t key           = %" PRIu32 "\n"
	     "\t order         = %" PRIu32 "\n"
	     "\t dir_names_num = %" PRIu32 "\n",
	     cond->user_id, cond->title_id != nullptr ? cond->title_id->data : "<default>",
	     cond->dir_name != nullptr ? cond->dir_name->data : "<all>",
	     static_cast<uint32_t>(cond->key), static_cast<uint32_t>(cond->order),
	     result->dir_names_num);

	result->hit_num = 0;
	result->pad     = 0;
	result->set_num = 0;
	std::memset(result->reserved, 0, sizeof(result->reserved));
	result->pad2 = 0;

	std::vector<std::string> dir_list;
	std::string              root =
	    Common::FixDirectorySlash((std::string(SAVE_DATA_DIR) + "/" + get_title_id()));

	if (Common::File::IsDirectoryExisting(root)) {
		for (const auto& entry: Common::File::GetDirEntries(root)) {
			if (!entry.is_file && entry.name != "." && entry.name != ".." &&
			    !Common::StartsWith(entry.name, "sce_")) {
				if (cond->dir_name == nullptr || cond->dir_name->data[0] == '\0' ||
				    dir_name_match(Common::ToLower(entry.name).c_str(),
				                   Common::ToLower(std::string(cond->dir_name->data)).c_str())) {
					dir_list.push_back(entry.name);
				}
			}
		}
	}

	// Sorted by the key the game asks for, ties by name. Games pick the slot to continue from this order or from
	// the reported times (Demon's Souls' Continue: the newest PlayerProfile); sorting by name only made it load
	// PlayerProfile0 whichever character was played last (issue #26).
	struct Slot {
		std::string name;
		int64_t     mtime;
		uint64_t    blocks;
		uint64_t    free_blocks;
		bool        capacity;
		StoredParam param;
	};
	std::vector<Slot> slots;
	for (const auto& name: dir_list) {
		const auto directory = std::filesystem::path(root) / name;
		const auto capacity  = ReadSaveDataCapacity(directory, capacity_metadata(name));
		slots.push_back({name, save_data_mtime(directory), capacity ? capacity->blocks : 0,
		                 capacity ? capacity->free_blocks : 0, capacity.has_value(), read_param(name)});
	}
	std::sort(slots.begin(), slots.end(), [key = cond->key](const Slot& a, const Slot& b) {
		uint64_t va = 0;
		uint64_t vb = 0;
		switch (key) {
			case SaveDataSortKey::Mtime:
				va = static_cast<uint64_t>(a.mtime);
				vb = static_cast<uint64_t>(b.mtime);
				break;
			case SaveDataSortKey::Blocks:
				va = a.blocks;
				vb = b.blocks;
				break;
			case SaveDataSortKey::FreeBlocks:
				va = a.free_blocks;
				vb = b.free_blocks;
				break;
			case SaveDataSortKey::UserParam:
				va = a.param.user_param;
				vb = b.param.user_param;
				break;
			default: break; // DirName
		}
		return va != vb ? va < vb : std::strcmp(a.name.c_str(), b.name.c_str()) < 0;
	});
	if (cond->order == SaveDataSortOrder::Descent) {
		std::reverse(slots.begin(), slots.end());
	}

	auto max_count = (result->dir_names_num < slots.size() ? result->dir_names_num : slots.size());

	result->hit_num = static_cast<uint32_t>(slots.size());
	result->set_num = static_cast<uint32_t>(max_count);

	std::printf("SaveData: search %s by key %" PRIu32 "%s: %zu slots, first %s\n",
	            cond->dir_name != nullptr && cond->dir_name->data[0] != '\0' ? cond->dir_name->data : "<all>",
	            static_cast<uint32_t>(cond->key), cond->order == SaveDataSortOrder::Descent ? " descending" : "",
	            slots.size(), slots.empty() ? "-" : slots.front().name.c_str());

	for (size_t i = 0; i < max_count; i++) {
		const auto& slot = slots[i];
		std::snprintf(result->dir_names[i].data, sizeof(result->dir_names[i].data), "%s",
		              slot.name.c_str());
		if (result->params != nullptr) {
			auto& param = result->params[i];
			param       = {};
			std::memcpy(param.title, slot.param.title, sizeof(param.title));
			std::memcpy(param.sub_title, slot.param.sub_title, sizeof(param.sub_title));
			std::memcpy(param.detail, slot.param.detail, sizeof(param.detail));
			param.user_param = slot.param.user_param;
			param.mtime      = slot.mtime;
		}
		if (result->infos != nullptr) {
			result->infos[i] = {};
			if (!slot.capacity) { return SAVE_DATA_ERROR_INTERNAL; }
			result->infos[i].blocks      = slot.blocks;
			result->infos[i].free_blocks = slot.free_blocks;
		}
	}

	return OK;
}

int KYTY_SYSV_ABI SaveDataMount3(const SaveDataMount3* mount, SaveDataMountResult* mount_result) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(mount == nullptr);
	EXIT_NOT_IMPLEMENTED(mount_result == nullptr);
	EXIT_NOT_IMPLEMENTED(mount->dir_name == nullptr);

	LOGF("\t user_id       = %d\n"
	     "\t dir_name      = %s\n"
	     "\t blocks        = %" PRIu64 "\n"
	     "\t system_blocks = %" PRIu64 "\n"
	     "\t mount_mode    = %" PRIu32 "\n"
	     "\t resource      = %" PRId32 "\n",
	     mount->user_id, mount->dir_name->data, mount->blocks, mount->system_blocks,
	     mount->mount_mode, mount->resource);

	*mount_result = {};

	Common::LockGuard lock(g_mount_mutex);
	const std::string dir_name = mount->dir_name->data;
	const std::string mount_dir =
	    std::string(SAVE_DATA_DIR) + "/" + get_title_id() + "/" + dir_name;
	if (g_exit_block.load(std::memory_order_acquire)) {
		std::printf("[save] mount %s mode 0x%x: refused, the game is closing (F1)\n", dir_name.c_str(), mount->mount_mode);
		return SAVE_DATA_ERROR_BUSY;
	}
	const bool create  = ((mount->mount_mode & 4u) != 0);
	const bool create2 = ((mount->mount_mode & 32u) != 0);
	const bool open    = (!create && !create2 && ((mount->mount_mode & 3u) != 0));

	const int slot = g_mount_slots.FindAvailable(dir_name);
	if (slot == SaveDataMountSlots::BUSY) {
		{ std::printf("[save] mount %s mode 0x%x: SAVE_DATA_ERROR_BUSY\n", dir_name.c_str(), mount->mount_mode); return SAVE_DATA_ERROR_BUSY; }
	}
	if (slot == SaveDataMountSlots::FULL) {
		{ std::printf("[save] mount %s mode 0x%x: SAVE_DATA_ERROR_MOUNT_FULL\n", dir_name.c_str(), mount->mount_mode); return SAVE_DATA_ERROR_MOUNT_FULL; }
	}

	if (!create && !create2 && !open) {
		EXIT("unknown mount mode: %u", mount->mount_mode);
	}

	if (open && !Common::File::IsDirectoryExisting(mount_dir)) {
		{ std::printf("[save] mount %s mode 0x%x: SAVE_DATA_ERROR_NOT_FOUND\n", dir_name.c_str(), mount->mount_mode); return SAVE_DATA_ERROR_NOT_FOUND; }
	}

	if (create && Common::File::IsDirectoryExisting(mount_dir)) {
		{ std::printf("[save] mount %s mode 0x%x: SAVE_DATA_ERROR_EXISTS\n", dir_name.c_str(), mount->mount_mode); return SAVE_DATA_ERROR_EXISTS; }
	}

	bool created = false;
	if ((create || create2) && !Common::File::IsDirectoryExisting(mount_dir)) {
		Common::File::CreateDirectories(mount_dir);
		created = true;

		EXIT_NOT_IMPLEMENTED((!Common::File::IsDirectoryExisting(mount_dir)));
	}

	if (!created) {
		repair(dir_name);
	}

	const auto metadata = capacity_metadata(dir_name);
	const auto capacity = ReadSaveDataCapacity(mount_dir, metadata);
	if (!capacity || !WriteSaveDataCapacity(metadata, std::max(capacity->blocks, mount->blocks))) {
		{ std::printf("[save] mount %s mode 0x%x: SAVE_DATA_ERROR_INTERNAL\n", dir_name.c_str(), mount->mount_mode); return SAVE_DATA_ERROR_INTERNAL; }
	}

	const int result = mount_save_data(slot, dir_name, mount_dir, created ? 1u : 0u, mount_result);
	std::printf("[save] mount %s mode 0x%x%s: %d\n", dir_name.c_str(), mount->mount_mode, created ? " (created)" : "", result);
	if (result == OK && dir_name.find("PlayerProfile") != std::string::npos) g_player_mounts.fetch_add(1, std::memory_order_acq_rel);
	return result;
}

int KYTY_SYSV_ABI SaveDataSetupSaveDataMemory2(const SaveDataMemorySetup2* setup_param,
                                               SaveDataMemorySetupResult*  result) {
	PRINT_NAME();

	if (setup_param == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t option           = 0x%08" PRIx32 "\n"
	     "\t user_id          = %" PRId32 "\n"
	     "\t memory_size      = %" PRIu64 "\n"
	     "\t icon_memory_size = %" PRIu64 "\n"
	     "\t slot_id          = %" PRIu32 "\n",
	     setup_param->option, setup_param->user_id, static_cast<uint64_t>(setup_param->memory_size),
	     static_cast<uint64_t>(setup_param->icon_memory_size), setup_param->slot_id);

	load_save_memory();
	if (setup_param->memory_size > g_save_data_memory.size()) {
		g_save_data_memory.resize(setup_param->memory_size);
	}
	std::printf("[save] save data memory setup: %llu bytes, slot %u\n",
	            static_cast<unsigned long long>(setup_param->memory_size), setup_param->slot_id);

	if (result != nullptr) {
		*result                     = {};
		result->existed_memory_size = g_save_data_memory.size();
	}

	return OK;
}

int KYTY_SYSV_ABI SaveDataGetSaveDataMemory2(SaveDataMemoryGet2* get_param) {
	PRINT_NAME();

	if (get_param == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t user_id  = %" PRId32 "\n"
	     "\t data     = 0x%016" PRIx64 "\n"
	     "\t param    = 0x%016" PRIx64 "\n"
	     "\t icon     = 0x%016" PRIx64 "\n"
	     "\t slot_id  = %" PRIu32 "\n",
	     get_param->user_id, reinterpret_cast<uint64_t>(get_param->data),
	     reinterpret_cast<uint64_t>(get_param->param), reinterpret_cast<uint64_t>(get_param->icon),
	     get_param->slot_id);

	load_save_memory();
	if (get_param->data != nullptr) {
		auto* data = get_param->data;
		if (data->buf == nullptr || data->offset < 0) {
			return SAVE_DATA_ERROR_PARAMETER;
		}

		const auto offset = static_cast<size_t>(data->offset);
		if (offset + data->buf_size > g_save_data_memory.size()) {
			g_save_data_memory.resize(offset + data->buf_size);
		}
		std::memcpy(data->buf, g_save_data_memory.data() + offset, data->buf_size);
	}
	if (get_param->param != nullptr) {
		std::memset(get_param->param, 0, sizeof(*get_param->param));
	}
	if (get_param->icon != nullptr) {
		get_param->icon->data_size = 0;
	}

	return OK;
}

int KYTY_SYSV_ABI SaveDataSetSaveDataMemory2(const SaveDataMemorySet2* set_param) {
	PRINT_NAME();

	if (set_param == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t user_id  = %" PRId32 "\n"
	     "\t data     = 0x%016" PRIx64 "\n"
	     "\t param    = 0x%016" PRIx64 "\n"
	     "\t icon     = 0x%016" PRIx64 "\n"
	     "\t data_num = %" PRIu32 "\n"
	     "\t slot_id  = %" PRIu32 "\n",
	     set_param->user_id, reinterpret_cast<uint64_t>(set_param->data),
	     reinterpret_cast<uint64_t>(set_param->param), reinterpret_cast<uint64_t>(set_param->icon),
	     set_param->data_num, set_param->slot_id);

	if (g_exit_block.load(std::memory_order_acquire)) {
		return SAVE_DATA_ERROR_BUSY; // (ds-pc-port8) closing (F1)
	}
	const uint32_t data_num = (set_param->data_num == 0 ? 1 : set_param->data_num);
	if (set_param->data != nullptr) {
		for (uint32_t i = 0; i < data_num; i++) {
			const auto& data = set_param->data[i];
			if (data.buf == nullptr || data.offset < 0) {
				return SAVE_DATA_ERROR_PARAMETER;
			}

			const auto offset = static_cast<size_t>(data.offset);
			if (offset + data.buf_size > g_save_data_memory.size()) {
				g_save_data_memory.resize(offset + data.buf_size);
			}
			std::memcpy(g_save_data_memory.data() + offset, data.buf, data.buf_size);
		}
		store_save_memory();
		std::printf("[save] save data memory written (%u blocks)\n", data_num);
	}

	return OK;
}

int KYTY_SYSV_ABI SaveDataTransferringMount(const SaveDataTransferringMount* mount,
                                            SaveDataMountResult*             mount_result) {
	PRINT_NAME();

	if (mount == nullptr || mount_result == nullptr || mount->dir_name == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t user_id  = %" PRId32 "\n"
	     "\t title_id = %s\n"
	     "\t dir_name = %s\n",
	     mount->user_id, mount->title_id != nullptr ? mount->title_id->data : "<null>",
	     mount->dir_name->data);

	*mount_result = {};

	Common::LockGuard lock(g_mount_mutex);
	const std::string dir_name = mount->dir_name->data;
	const std::string mount_dir =
	    std::string(SAVE_DATA_DIR) + "/" + get_title_id() + "/" + dir_name;
	const int slot = g_mount_slots.FindAvailable(dir_name);
	if (slot == SaveDataMountSlots::BUSY) {
		return SAVE_DATA_ERROR_BUSY;
	}
	if (slot == SaveDataMountSlots::FULL) {
		return SAVE_DATA_ERROR_MOUNT_FULL;
	}

	if (!Common::File::IsDirectoryExisting(mount_dir)) {
		Common::File::CreateDirectories(mount_dir);
	}

	return mount_save_data(slot, dir_name, mount_dir, 1, mount_result);
}

int KYTY_SYSV_ABI SaveDataUmount2(uint32_t mode, const SaveDataMountPoint* mount_point) {
	PRINT_NAME();

	if (mount_point == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t mode        = %" PRIu32 "\n"
	     "\t mount_point = %s\n",
	     mode, mount_point->data);

	Common::LockGuard lock(g_mount_mutex);
	const std::string point = mount_point->data;
	const int         slot  = g_mount_slots.Find(point);
	if (slot == SaveDataMountSlots::FULL) {
		return SAVE_DATA_ERROR_NOT_FOUND;
	}

	const std::string dir_name = g_mount_slots.Directory(static_cast<size_t>(slot));
	LibKernel::FileSystem::Umount(point);
	g_mount_slots.Release(static_cast<size_t>(slot));
	keep_safe(dir_name);
	if ((mode & SAVE_DATA_UMOUNT_MODE_BACKUP_ASYNC) != 0) {
		queue_save_data_event(SAVE_DATA_EVENT_TYPE_UMOUNT_BACKUP_END, 0, nullptr, nullptr);
	}

	return OK;
}

int KYTY_SYSV_ABI SaveDataPrepare(const SaveDataMountPoint*   mount_point,
                                  const SaveDataPrepareParam* param) {
	PRINT_NAME();

	if (mount_point == nullptr || param == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t mount_point  = %s\n"
	     "\t resource     = %" PRId32 "\n"
	     "\t prepare_mode = %" PRIu32 "\n",
	     mount_point->data, param->resource, param->prepare_mode);

	return OK;
}

int KYTY_SYSV_ABI SaveDataCommit(const SaveDataCommitParam* param) {
	PRINT_NAME();

	if (param == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t resource    = %" PRId32 "\n"
	     "\t commit_mode = %" PRIu32 "\n",
	     param->resource, param->commit_mode);

	{
		Common::LockGuard lock(g_mount_mutex);
		for (size_t slot = 0; slot < SaveDataMountSlots::COUNT; slot++) {
			if (g_mount_slots.IsMounted(slot)) keep_safe(g_mount_slots.Directory(slot));
		}
	}
	if ((param->commit_mode & SAVE_DATA_COMMIT_MODE_BACKUP_ASYNC) != 0) {
		queue_save_data_event(SAVE_DATA_EVENT_TYPE_COMMIT_BACKUP_END, 0, nullptr, nullptr);
	}

	return OK;
}

int KYTY_SYSV_ABI SaveDataDelete(const SaveDataDelete* del) {
	PRINT_NAME();

	if (del == nullptr || del->dir_name == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t user_id  = %" PRId32 "\n"
	     "\t title_id = %s\n"
	     "\t dir_name = %s\n",
	     del->user_id, del->title_id != nullptr ? del->title_id->data : "<default>",
	     del->dir_name->data);

	std::string dir =
	    std::string(SAVE_DATA_DIR) + "/" + get_title_id() + "/" + std::string(del->dir_name->data);
	if (Common::File::IsDirectoryExisting(dir)) {
		Common::File::DeleteDirectory(dir);
		std::error_code ec;
		std::filesystem::remove(capacity_metadata(del->dir_name->data), ec);
		if (ec) { return SAVE_DATA_ERROR_INTERNAL; }
	}

	return OK;
}

int KYTY_SYSV_ABI SaveDataGetParam(const SaveDataMountPoint* mount_point, uint32_t param_type,
                                   void* param_buf, size_t param_buf_size, size_t* got_size) {
	PRINT_NAME();

	if (mount_point == nullptr || param_buf == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t mount_point    = %s\n"
	     "\t param_type     = %u\n"
	     "\t param_buf_size = %" PRIu64 "\n",
	     mount_point->data, param_type, param_buf_size);

	// The mounted slot's modification time and the texts and user parameter the game last set (SetParam).
	int64_t     mtime = 0;
	StoredParam stored {};
	{
		Common::LockGuard lock(g_mount_mutex);
		const int         slot = g_mount_slots.Find(mount_point->data);
		if (slot != SaveDataMountSlots::FULL) {
			const std::string name = g_mount_slots.Directory(static_cast<size_t>(slot));
			mtime                  = save_data_mtime(std::filesystem::path(SAVE_DATA_DIR) / get_title_id() / name);
			stored                 = read_param(name);
		}
	}

	if (param_type == 0) {
		if (param_buf_size < sizeof(SaveDataParam)) {
			return SAVE_DATA_ERROR_PARAMETER;
		}
		auto* param = static_cast<SaveDataParam*>(param_buf);
		std::memset(param, 0, sizeof(SaveDataParam));
		std::memcpy(param->title, stored.title, sizeof(param->title));
		std::memcpy(param->sub_title, stored.sub_title, sizeof(param->sub_title));
		std::memcpy(param->detail, stored.detail, sizeof(param->detail));
		param->user_param = stored.user_param;
		param->mtime      = mtime;
		if (got_size != nullptr) {
			*got_size = sizeof(SaveDataParam);
		}
		return OK;
	}

	std::memset(param_buf, 0, param_buf_size);
	const auto copy = [&](const void* field, size_t size) { std::memcpy(param_buf, field, std::min(size, param_buf_size)); };
	switch (param_type) {
		case 1: copy(stored.title, sizeof(stored.title)); break;
		case 2: copy(stored.sub_title, sizeof(stored.sub_title)); break;
		case 3: copy(stored.detail, sizeof(stored.detail)); break;
		case 4: copy(&stored.user_param, sizeof(stored.user_param)); break;
		case 5: copy(&mtime, sizeof(mtime)); break;
		default: break;
	}
	if (got_size != nullptr) {
		*got_size = param_buf_size;
	}

	return OK;
}

int KYTY_SYSV_ABI SaveDataLoadIcon(const SaveDataMountPoint* mount_point, SaveDataIcon* icon) {
	PRINT_NAME();

	if (mount_point == nullptr || icon == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t mount_point = %s\n"
	     "\t buf         = %016" PRIx64 "\n"
	     "\t buf_size    = %" PRIu64 "\n",
	     mount_point->data, reinterpret_cast<uint64_t>(icon->buf), icon->buf_size);

	icon->data_size = 0;

	return OK;
}

int KYTY_SYSV_ABI SaveDataSaveIconByPath(const SaveDataMountPoint* mount_point, const char* path) {
	PRINT_NAME();

	if (mount_point == nullptr || path == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t mount_point = %s\n"
	     "\t path        = %s\n",
	     mount_point->data, path);

	return OK;
}

int KYTY_SYSV_ABI SaveDataSyncSaveDataMemory(const void* sync_param) {
	PRINT_NAME();

	LOGF("\t sync_param = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(sync_param));

	return OK;
}

int KYTY_SYSV_ABI SaveDataGetEventResult(const void* event_param, SaveDataEvent* event) {
	PRINT_NAME();

	LOGF("\t event_param = 0x%016" PRIx64 "\n"
	     "\t event       = 0x%016" PRIx64 "\n",
	     reinterpret_cast<uint64_t>(event_param), reinterpret_cast<uint64_t>(event));

	if (event == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	if (g_save_data_events.empty()) {
		return SAVE_DATA_ERROR_NOT_FOUND;
	}

	*event = g_save_data_events.front();
	g_save_data_events.pop_front();

	LOGF("\t type       = %" PRIu32 "\n"
	     "\t error_code = %" PRId32 "\n"
	     "\t user_id    = %" PRId32 "\n"
	     "\t title_id   = %.10s\n"
	     "\t dir_name   = %.32s\n",
	     event->type, event->error_code, event->user_id, event->title_id.data,
	     event->dir_name.data);

	return OK;
}

int KYTY_SYSV_ABI SaveDataBackup(const SaveDataBackup* backup) {
	PRINT_NAME();

	LOGF("\t backup = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(backup));

	if (backup == nullptr || backup->dir_name == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}

	LOGF("\t user_id = %" PRId32 "\n"
	     "\t title_id = %.10s\n"
	     "\t dir_name = %.32s\n",
	     backup->user_id, backup->title_id != nullptr ? backup->title_id->data : "<default>",
	     backup->dir_name->data);

	{
		Common::LockGuard lock(g_mount_mutex);
		keep_safe(std::string(backup->dir_name->data, strnlen(backup->dir_name->data, sizeof(backup->dir_name->data))));
	}
	queue_save_data_event(SAVE_DATA_EVENT_TYPE_BACKUP_END, backup->user_id, backup->title_id,
	                      backup->dir_name);

	return OK;
}

int KYTY_SYSV_ABI SaveDataSetParam(const SaveDataMountPoint* mount_point, uint32_t param_type,
                                   const void* param_buf, size_t param_buf_size) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(mount_point == nullptr);

	LOGF("\t mount_point    = %s\n"
	     "\t param_type     = %u\n"
	     "\t param_buf_size = %" PRIu64 "\n",
	     mount_point->data, param_type, param_buf_size);

	if (param_buf == nullptr) {
		return SAVE_DATA_ERROR_PARAMETER;
	}
	std::string name;
	{
		Common::LockGuard lock(g_mount_mutex);
		const int         slot = g_mount_slots.Find(mount_point->data);
		if (slot == SaveDataMountSlots::FULL) {
			return SAVE_DATA_ERROR_PARAMETER;
		}
		name = g_mount_slots.Directory(static_cast<size_t>(slot));
	}
	// Kept for the slot (read_param): the system's list shows them, GetParam and searches report them.
	auto       stored = read_param(name);
	const auto text   = [&](char* field, size_t size, const char* source, size_t source_size) {
		std::memset(field, 0, size);
		std::memcpy(field, source, std::min(size - 1, strnlen(source, source_size)));
	};
	switch (param_type) {
		case 0: {
			if (param_buf_size < sizeof(SaveDataParam)) {
				return SAVE_DATA_ERROR_PARAMETER;
			}
			const auto* p = static_cast<const SaveDataParam*>(param_buf);
			LOGF("\t title      = %s\n"
			     "\t sub_title  = %s\n"
			     "\t detail     = %s\n"
			     "\t user_param = %u\n",
			     p->title, p->sub_title, p->detail, p->user_param);
			text(stored.title, sizeof(stored.title), p->title, sizeof(p->title));
			text(stored.sub_title, sizeof(stored.sub_title), p->sub_title, sizeof(p->sub_title));
			text(stored.detail, sizeof(stored.detail), p->detail, sizeof(p->detail));
			stored.user_param = p->user_param;
			break;
		}
		case 1: text(stored.title, sizeof(stored.title), static_cast<const char*>(param_buf), param_buf_size); break;
		case 2:
			text(stored.sub_title, sizeof(stored.sub_title), static_cast<const char*>(param_buf), param_buf_size);
			break;
		case 3: text(stored.detail, sizeof(stored.detail), static_cast<const char*>(param_buf), param_buf_size); break;
		case 4:
			if (param_buf_size < sizeof(stored.user_param)) {
				return SAVE_DATA_ERROR_PARAMETER;
			}
			std::memcpy(&stored.user_param, param_buf, sizeof(stored.user_param));
			break;
		default: LOGF("\t unsupported param_type, accepting as no-op\n"); return OK;
	}
	write_param(name, stored);

	return OK;
}

int KYTY_SYSV_ABI SaveDataGetMountInfo(const SaveDataMountPoint* mount_point,
                                       SaveDataMountInfo*        info) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(mount_point == nullptr);
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	*info = {};

	Common::LockGuard lock(g_mount_mutex);
	const int slot = g_mount_slots.Find(mount_point->data);
	if (slot == SaveDataMountSlots::FULL) { return SAVE_DATA_ERROR_NOT_MOUNTED; }
	const auto& name = g_mount_slots.Directory(static_cast<size_t>(slot));
	const auto capacity = ReadSaveDataCapacity(
	    std::filesystem::path(SAVE_DATA_DIR) / get_title_id() / name, capacity_metadata(name));
	if (!capacity) { return SAVE_DATA_ERROR_INTERNAL; }
	info->blocks = capacity->blocks;
	info->free_blocks = capacity->free_blocks;

	return OK;
}

int KYTY_SYSV_ABI SaveDataSaveIcon(const SaveDataMountPoint* mount_point,
                                   const SaveDataIcon*       icon) {
	EXIT_NOT_IMPLEMENTED(mount_point == nullptr);
	EXIT_NOT_IMPLEMENTED(icon == nullptr);

	LOGF("\t buf       = %016" PRIx64 "\n"
	     "\t buf_size  = %" PRIu64 "\n"
	     "\t data_size = %" PRIu64 "\n",
	     reinterpret_cast<uint64_t>(icon->buf), icon->buf_size, icon->data_size);

	return OK;
}

} // namespace SaveData

namespace LibSaveDataNative {

LIB_VERSION("SaveData_native", 1, "SaveData_native", 1, 1);

LIB_DEFINE(InitSaveDataNative_1) {
	LIB_FUNC("TywrFKCoLGY", ::Libs::SaveData::SaveDataInitialize3);
	LIB_FUNC("dyIhnXq-0SM", ::Libs::SaveData::SaveDataDirNameSearch);
	LIB_FUNC("PHnuI4LhuRk", ::Libs::SaveData::SaveDataDirNameSearch);
	LIB_FUNC("ZP4e7rlzOUk", ::Libs::SaveData::SaveDataMount3);
	LIB_FUNC("gjRZNnw0JPE", ::Libs::SaveData::SaveDataCreateTransactionResource);
	LIB_FUNC("lJUQuaKqoKY", ::Libs::SaveData::SaveDataDeleteTransactionResource);
	LIB_FUNC("sDCBrmc61XU", ::Libs::SaveData::SaveDataPrepare);
	LIB_FUNC("ie7qhZ4X0Cc", ::Libs::SaveData::SaveDataCommit);
	LIB_FUNC("oQySEUfgXRA", ::Libs::SaveData::SaveDataSetupSaveDataMemory2);
	LIB_FUNC("QwOO7vegnV8", ::Libs::SaveData::SaveDataGetSaveDataMemory2);
	LIB_FUNC("cduy9v4YmT4", ::Libs::SaveData::SaveDataSetSaveDataMemory2);
	LIB_FUNC("wiT9jeC7xPw", ::Libs::SaveData::SaveDataSyncSaveDataMemory);
	LIB_FUNC("j8xKtiFj0SY", ::Libs::SaveData::SaveDataGetEventResult);
	LIB_FUNC("WAzWTZm1H+I", ::Libs::SaveData::SaveDataTransferringMount);
	LIB_FUNC("RjMlsR8EXrw", ::Libs::SaveData::SaveDataTransferringMount);
	LIB_FUNC("z1JA8-iJt3k", ::Libs::SaveData::SaveDataBackup);
	LIB_FUNC("uW4vfTwMQVo", ::Libs::SaveData::SaveDataUmount2);
	LIB_FUNC("S1GkePI17zQ", ::Libs::SaveData::SaveDataDelete);
	LIB_FUNC("85zul--eGXs", ::Libs::SaveData::SaveDataSetParam);
	LIB_FUNC("XgvSuIdnMlw", ::Libs::SaveData::SaveDataGetParam);
	LIB_FUNC("65VH0Qaaz6s", ::Libs::SaveData::SaveDataGetMountInfo);
	LIB_FUNC("c88Yy54Mx0w", ::Libs::SaveData::SaveDataSaveIcon);
	LIB_FUNC("Z7z6HXWORJY", ::Libs::SaveData::SaveDataSaveIconByPath);
	LIB_FUNC("cGjO3wM3V28", ::Libs::SaveData::SaveDataLoadIcon);
	LIB_FUNC("X4MYzukPc3g", ::Libs::SaveData::SaveDataDirNameSearch);
	LIB_FUNC("yKDy8S5yLA0", ::Libs::SaveData::SaveDataTerminate);
}

} // namespace LibSaveDataNative

LIB_DEFINE(InitSaveData_1) {
	LIB_FUNC("TywrFKCoLGY", SaveData::SaveDataInitialize3);
	LIB_FUNC("dyIhnXq-0SM", SaveData::SaveDataDirNameSearch);
	LIB_FUNC("ZP4e7rlzOUk", SaveData::SaveDataMount3);
	LIB_FUNC("gjRZNnw0JPE", SaveData::SaveDataCreateTransactionResource);
	LIB_FUNC("lJUQuaKqoKY", SaveData::SaveDataDeleteTransactionResource);
	LIB_FUNC("sDCBrmc61XU", SaveData::SaveDataPrepare);
	LIB_FUNC("ie7qhZ4X0Cc", SaveData::SaveDataCommit);
	LIB_FUNC("oQySEUfgXRA", SaveData::SaveDataSetupSaveDataMemory2);
	LIB_FUNC("QwOO7vegnV8", SaveData::SaveDataGetSaveDataMemory2);
	LIB_FUNC("cduy9v4YmT4", SaveData::SaveDataSetSaveDataMemory2);
	LIB_FUNC("wiT9jeC7xPw", SaveData::SaveDataSyncSaveDataMemory);
	LIB_FUNC("j8xKtiFj0SY", SaveData::SaveDataGetEventResult);
	LIB_FUNC("WAzWTZm1H+I", SaveData::SaveDataTransferringMount);
	LIB_FUNC("RjMlsR8EXrw", SaveData::SaveDataTransferringMount);
	LIB_FUNC("z1JA8-iJt3k", SaveData::SaveDataBackup);
	LIB_FUNC("uW4vfTwMQVo", SaveData::SaveDataUmount2);
	LIB_FUNC("S1GkePI17zQ", SaveData::SaveDataDelete);
	LIB_FUNC("85zul--eGXs", SaveData::SaveDataSetParam);
	LIB_FUNC("XgvSuIdnMlw", SaveData::SaveDataGetParam);
	LIB_FUNC("65VH0Qaaz6s", SaveData::SaveDataGetMountInfo);
	LIB_FUNC("c88Yy54Mx0w", SaveData::SaveDataSaveIcon);
	LIB_FUNC("Z7z6HXWORJY", SaveData::SaveDataSaveIconByPath);
	LIB_FUNC("cGjO3wM3V28", SaveData::SaveDataLoadIcon);
	LIB_FUNC("X4MYzukPc3g", SaveData::SaveDataDirNameSearch);
	LIB_FUNC("yKDy8S5yLA0", SaveData::SaveDataTerminate);

	LibSaveDataNative::InitSaveDataNative_1(s);
}

} // namespace Libs
