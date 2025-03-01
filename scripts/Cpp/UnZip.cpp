#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <future>        // for std::async, std::future
#include <filesystem>    // C++17
#include <cstring>       // for std::memset

extern "C" {
    #include "../Chead/miniz-3.0.2/miniz.h"     // Please place it in the same folder
}

namespace fs = std::filesystem;

/**
 * @brief Extract a single file from an in-memory ZIP to the output directory.
 *
 * @param zip_data    The entire ZIP file loaded into memory.
 * @param file_index  Which file entry to extract (0-based).
 * @param output_dir  Where to place the extracted file.
 * @return true on success, false on failure.
 */
bool extract_file_from_zip(const std::vector<char>& zip_data, size_t file_index, const std::string& output_dir)
{
    mz_zip_archive zip_archive;
    std::memset(&zip_archive, 0, sizeof(zip_archive));

    // Initialize a read-only zip archive from memory
    if (!mz_zip_reader_init_mem(&zip_archive, zip_data.data(), zip_data.size(), 0)) {
        std::cerr << "[Error] Could not init zip reader from memory.\n";
        return false;
    }

    // Get the name of the file within the archive
    mz_zip_archive_file_stat file_stat;
    if (!mz_zip_reader_file_stat(&zip_archive, file_index, &file_stat)) {
        std::cerr << "[Error] Could not get file stat for entry " << file_index << ".\n";
        mz_zip_reader_end(&zip_archive);
        return false;
    }
    std::string internal_name = file_stat.m_filename; // e.g. "folder/test.txt"

    // Build a path in the output directory
    fs::path output_path = fs::path(output_dir) / internal_name;

    // If the entry is a directory (ends with '/'), create the directory and finish
    if (mz_zip_reader_is_file_a_directory(&zip_archive, file_index)) {
        std::error_code ec;
        fs::create_directories(output_path, ec); // create_directories won't fail if already exists
        // It's just a directory, no extraction needed
        mz_zip_reader_end(&zip_archive);
        return true;
    }

    // Ensure parent directories exist
    fs::create_directories(output_path.parent_path());

    // Extract the file to disk
    bool result = mz_zip_reader_extract_to_file(&zip_archive, file_index, output_path.string().c_str(), 0);
    if (!result) {
        std::cerr << "[Error] Failed to extract file: " << internal_name << "\n";
        mz_zip_reader_end(&zip_archive);
        return false;
    }

    // End the zip reader
    mz_zip_reader_end(&zip_archive);

    std::cout << "Extracted: " << internal_name << " -> " << output_path.string() << "\n";
    return true;
}

/**
 * @brief Multi-threaded unzip: read a ZIP file, launch threads to extract each entry in parallel.
 *
 * @param zip_path    Path to the .zip file to extract
 * @param output_dir  Directory to place the unzipped files
 */
bool unzip_in_parallel(const std::string& zip_path, const std::string& output_dir)
{
    // Read the entire ZIP file into memory
    std::ifstream ifs(zip_path, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "[Error] Cannot open ZIP file: " << zip_path << "\n";
        return false;
    }
    ifs.seekg(0, std::ios::end);
    std::streamoff zip_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<char> zip_data(zip_size);
    if (!ifs.read(zip_data.data(), zip_size)) {
        std::cerr << "[Error] Failed to read ZIP data.\n";
        return false;
    }
    ifs.close();

    // Create a temporary zip_archive to query the number of entries
    mz_zip_archive tmp_archive;
    std::memset(&tmp_archive, 0, sizeof(tmp_archive));
    if (!mz_zip_reader_init_mem(&tmp_archive, zip_data.data(), zip_data.size(), 0)) {
        std::cerr << "[Error] Failed to init ZIP from memory for scanning.\n";
        return false;
    }
    mz_uint num_files = mz_zip_reader_get_num_files(&tmp_archive);
    std::cout << "ZIP contains " << num_files << " entries.\n";

    // We can now close this temporary handle
    mz_zip_reader_end(&tmp_archive);

    if (num_files == 0) {
        std::cout << "[Info] No entries to extract.\n";
        return true;
    }

    // Create the output directory if needed
    fs::create_directories(output_dir);

    // Launch threads to extract each entry
    std::vector<std::future<bool>> futures;
    futures.reserve(num_files);
    for (size_t i = 0; i < num_files; i++) {
        futures.push_back(std::async(std::launch::async, [&, i]() {
            return extract_file_from_zip(zip_data, i, output_dir);
        }));
    }

    // Wait for all threads
    bool all_ok = true;
    for (size_t i = 0; i < num_files; i++) {
        bool ok = futures[i].get();
        if (!ok) {
            std::cerr << "[Fail] Could not extract index " << i << "\n";
            all_ok = false;
        }
    }
    if (all_ok) {
        std::cout << "All files successfully extracted.\n";
    } else {
        std::cerr << "Some files failed to extract.\n";
    }
    return all_ok;
}

int main(int argc, char* argv[])
{
    // Usage: multi_thread_unzip <zipfile> [output_dir]
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <zipfile> [output_dir]\n";
        return 1;
    }
    std::string zip_path = argv[1];
    std::string output_dir = (argc >= 3) ? argv[2] : "./unzipped";

    std::cout << "Multi-threaded Unzip\n";
    std::cout << "ZIP file: " << zip_path << "\n";
    std::cout << "Output dir: " << output_dir << "\n";

    bool result = unzip_in_parallel(zip_path, output_dir);
    if (!result) {
        std::cerr << "[Error] Extraction encountered issues.\n";
        return 1;
    }

    std::cout << "Done.\n";
    return 0;
}
