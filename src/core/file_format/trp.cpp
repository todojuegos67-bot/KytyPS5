// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <system_error>
#include <fmt/format.h>

#include "common/aes.h"
#include "common/key_manager.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/file_format/trp.h"

static void DecryptEFSM(std::span<const u8, 16> trophyKey, std::span<const u8, 16> NPcommID,
                        std::span<const u8, 16> efsmIv, std::span<const u8> ciphertext,
                        std::span<u8> decrypted) {
    // Step 1: Encrypt NPcommID
    std::array<u8, 16> trophyIv{};
    std::array<u8, 16> trpKey;
    // Convert spans to pointers for the aes functions
    aes::encrypt_cbc(NPcommID.data(), NPcommID.size(), trophyKey.data(), trophyKey.size(),
                     trophyIv.data(), trpKey.data(), trpKey.size(), false);

    // Step 2: Decrypt EFSM
    aes::decrypt_cbc(ciphertext.data(), ciphertext.size(), trpKey.data(), trpKey.size(),
                     const_cast<u8*>(efsmIv.data()), decrypted.data(), decrypted.size(), nullptr);
}

std::string TRP::FindNpCommId(std::span<const u8> trp) {
    const auto& key = KeyManager::GetInstance()->GetAllKeys().TrophyKeySet.ReleaseTrophyKey;
    if (key.size() != 16) {
        LOG_ERROR(Common_Filesystem,
                  "[trophy] no trophy key (TrophyKeySet.ReleaseTrophyKey in keys.json): the "
                  "NP communication ID of a game without npbind.dat cannot be found");
        return {};
    }
    if (trp.size() < sizeof(TrpHeader)) {
        return {};
    }
    TrpHeader header;
    std::memcpy(&header, trp.data(), sizeof(header));
    if (header.magic != TRP_MAGIC) {
        return {};
    }
    for (u32 i = 0; i < header.entry_num; i++) {
        const size_t at = sizeof(TrpHeader) + size_t(i) * header.entry_size;
        if (at + sizeof(TrpEntry) > trp.size()) {
            break;
        }
        TrpEntry entry;
        std::memcpy(&entry, trp.data() + at, sizeof(entry));
        if (entry.flag != ENTRY_FLAG_ENCRYPTED_XML || entry.entry_len < 48 ||
            entry.entry_pos + 32 > trp.size()) {
            continue;
        }
        const u8* iv = trp.data() + entry.entry_pos;
        const u8* block = iv + 16;
        for (int n = 0; n <= 99999; n++) {
            std::array<u8, 16> id{};
            const auto text = fmt::format("NPWR{:05d}_00", n);
            std::memcpy(id.data(), text.data(), text.size());
            std::array<u8, 16> zero_iv{};
            std::array<u8, 16> trp_key;
            aes::encrypt_cbc(id.data(), id.size(), key.data(), key.size(), zero_iv.data(),
                             trp_key.data(), trp_key.size(), false);
            std::array<u8, 16> out;
            std::array<u8, 16> iv_copy;
            std::memcpy(iv_copy.data(), iv, 16);
            aes::decrypt_cbc(block, 16, trp_key.data(), trp_key.size(), iv_copy.data(),
                             out.data(), out.size(), nullptr);
            if (out[0] == '<' && (out[1] == '!' || out[1] == '?') &&
                std::all_of(out.begin(), out.end(), [](u8 c) { return c >= 9 && c < 127; })) {
                LOG_INFO(Common_Filesystem, "[trophy] TRP decrypts with NP communication ID {}",
                         text);
                return text;
            }
        }
        LOG_ERROR(Common_Filesystem,
                  "[trophy] no NPWRxxxxx_00 ID decrypts the TRP (wrong trophy key?)");
        return {};
    }
    return {};
}

TRP::TRP() = default;
TRP::~TRP() = default;

static void removePadding(std::vector<u8>& vec) {
    for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
        if (*it == '>') {
            size_t pos = std::distance(vec.begin(), it.base());
            vec.resize(pos);
            break;
        }
    }
}

bool TRP::Extract(const std::filesystem::path& trophyPath, std::string npCommId,
                  const std::filesystem::path& outputPath) {
    // Retrieve trophy key
    const auto& user_key_vec =
        KeyManager::GetInstance()->GetAllKeys().TrophyKeySet.ReleaseTrophyKey;

    if (user_key_vec.size() != 16) {
        LOG_INFO(Common_Filesystem, "Trophy decryption key is not specified");
        return false;
    }

    std::array<u8, 16> user_key{};
    std::copy(user_key_vec.begin(), user_key_vec.end(), user_key.begin());

    s32 trpFileIndex = 0;
    bool success = true;
    const auto cleanupOnFailure = [&outputPath] {
        std::error_code ec;
        std::filesystem::remove_all(outputPath, ec);
    };
    try {
        if (trophyPath.extension() != ".trp") {
            return false;
        }
        Common::FS::IOFile file(trophyPath, Common::FS::FileAccessMode::Read);
        if (!file.IsOpen()) {
            LOG_ERROR(Common_Filesystem, "Unable to open trophy file: {}", trophyPath.string());
            return false;
        }

        TrpHeader header;
        if (!file.Read(header)) {
            LOG_ERROR(Common_Filesystem, "Failed to read TRP header from {}", trophyPath.string());
            return false;
        }

        if (header.magic != TRP_MAGIC) {
            LOG_ERROR(Common_Filesystem,
                      "Wrong trophy magic number in {}: got {:#010x}, expected {:#010x}",
                      trophyPath.string(), static_cast<u32>(header.magic), TRP_MAGIC);
            return false;
        }

        s64 seekPos = sizeof(TrpHeader);
        std::error_code dir_ec;
        std::filesystem::create_directories(outputPath / "Icons", dir_ec);
        if (!dir_ec) {
            std::filesystem::create_directories(outputPath / "Xml", dir_ec);
        }
        if (dir_ec) {
            LOG_ERROR(Common_Filesystem, "Failed to create output directories for {}: {}", npCommId,
                      dir_ec.message());
            cleanupOnFailure();
            return false;
        }

        // Process each entry in the TRP file
        for (u32 i = 0; i < header.entry_num; i++) {
            if (!file.Seek(seekPos)) {
                LOG_ERROR(Common_Filesystem, "Failed to seek to TRP entry offset");
                success = false;
                break;
            }
            seekPos += static_cast<s64>(header.entry_size);

            TrpEntry entry;
            if (!file.Read(entry)) {
                LOG_ERROR(Common_Filesystem, "Failed to read TRP entry");
                success = false;
                break;
            }

            std::string_view name(entry.entry_name);

            if (entry.flag == ENTRY_FLAG_PNG) {
                if (!ProcessPngEntry(file, entry, outputPath, name)) {
                    success = false;
                    // Continue with next entry
                }
            } else if (entry.flag == ENTRY_FLAG_ENCRYPTED_XML) {
                // Check if we have a valid NPCommID for decryption
                if (npCommId.size() >= 12 && npCommId[0] == 'N' && npCommId[1] == 'P') {
                    if (!ProcessEncryptedXmlEntry(file, entry, outputPath, name, user_key,
                                                  npCommId)) {
                        success = false;
                        // Continue with next entry
                    }
                } else {
                    LOG_WARNING(Common_Filesystem,
                                "Skipping encrypted XML entry - invalid NPCommID");
                    // Skip this entry but continue
                }
            } else {
                LOG_DEBUG(Common_Filesystem, "Unknown entry flag: {} for {}",
                          static_cast<u32>(entry.flag), name);
            }
            trpFileIndex++;
        }

    } catch (const std::filesystem::filesystem_error& e) {
        LOG_CRITICAL(Common_Filesystem, "Filesystem error during trophy extraction: {}", e.what());
        cleanupOnFailure();
        return false;
    } catch (const std::exception& e) {
        LOG_CRITICAL(Common_Filesystem, "Error during trophy extraction: {}", e.what());
        cleanupOnFailure();
        return false;
    }

    if (!success) {
        cleanupOnFailure();
        return false;
    }

    LOG_INFO(Common_Filesystem, "Successfully extracted {} trophy files for {}", trpFileIndex,
             npCommId);
    return true;
}

bool TRP::ProcessPngEntry(Common::FS::IOFile& file, const TrpEntry& entry,
                          const std::filesystem::path& outputPath, std::string_view name) {
    if (!file.Seek(entry.entry_pos)) {
        LOG_ERROR(Common_Filesystem, "Failed to seek to PNG entry offset");
        return false;
    }

    std::vector<u8> icon(entry.entry_len);
    if (!file.Read(icon)) {
        LOG_ERROR(Common_Filesystem, "Failed to read PNG data");
        return false;
    }

    auto outputFile = outputPath / "Icons" / name;
    size_t written = Common::FS::IOFile::WriteBytes(outputFile, icon);
    if (written != icon.size()) {
        LOG_ERROR(Common_Filesystem, "PNG write failed: wanted {} bytes, wrote {}", icon.size(),
                  written);
        return false;
    }

    return true;
}

bool TRP::ProcessEncryptedXmlEntry(Common::FS::IOFile& file, const TrpEntry& entry,
                                   const std::filesystem::path& outputPath, std::string_view name,
                                   const std::array<u8, 16>& user_key,
                                   const std::string& npCommId) {
    constexpr size_t IV_LEN = 16;

    if (!file.Seek(entry.entry_pos)) {
        LOG_ERROR(Common_Filesystem, "Failed to seek to encrypted XML entry offset");
        return false;
    }

    std::array<u8, IV_LEN> esfmIv;
    if (!file.Read(esfmIv)) {
        LOG_ERROR(Common_Filesystem, "Failed to read IV for encrypted XML");
        return false;
    }

    if (entry.entry_len <= IV_LEN) {
        LOG_ERROR(Common_Filesystem, "Encrypted XML entry too small");
        return false;
    }

    // Skip to the encrypted data (after IV)
    if (!file.Seek(entry.entry_pos + IV_LEN)) {
        LOG_ERROR(Common_Filesystem, "Failed to seek to encrypted data");
        return false;
    }

    std::vector<u8> ESFM(entry.entry_len - IV_LEN);
    std::vector<u8> XML(entry.entry_len - IV_LEN);

    if (!file.Read(ESFM)) {
        LOG_ERROR(Common_Filesystem, "Failed to read encrypted XML data");
        return false;
    }

    // Decrypt the data
    std::span<const u8, 16> key_span(user_key);

    // Convert npCommId string to span (pad or truncate to 16 bytes)
    std::array<u8, 16> npcommid_array{};
    size_t copy_len = std::min(npCommId.size(), npcommid_array.size());
    std::memcpy(npcommid_array.data(), npCommId.data(), copy_len);
    std::span<const u8, 16> npcommid_span(npcommid_array);

    DecryptEFSM(key_span, npcommid_span, esfmIv, ESFM, XML);

    // Remove padding
    removePadding(XML);

    // Create output filename
    std::string xml_name(entry.entry_name);
    size_t pos = xml_name.find("ESFM");
    if (pos != std::string::npos) {
        xml_name.replace(pos, 4, "XML");
    }

    auto outputFile = outputPath / "Xml" / xml_name;
    size_t written = Common::FS::IOFile::WriteBytes(outputFile, XML);
    if (written != XML.size()) {
        LOG_ERROR(Common_Filesystem, "XML write failed: wanted {} bytes, wrote {}", XML.size(),
                  written);
        return false;
    }

    return true;
}