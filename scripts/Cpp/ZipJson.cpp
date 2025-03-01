#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>  // C++17
#include <cstring>     // for memset

extern "C" {
    #include "../Chead/miniz-3.0.2/miniz.h"     // Please place it in the same folder
}

namespace fs = std::filesystem;

/**
 * Create a ZIP file containing a single file using miniz.
 * 
 * @param file_to_zip   - Path to the file to be compressed (e.g. "mydata.json")
 * @param out_zip_name  - Name of the output ZIP file (e.g. "mydata.zip")
 * @return true if success
 */
bool create_zip_for_single_file(const std::string& file_to_zip, const std::string& out_zip_name) {
    // Read the file to be compressed
    std::ifstream ifs(file_to_zip, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "[Error] Unable to open file: " << file_to_zip << "\n";
        return false;
    }
    ifs.seekg(0, std::ios::end);
    auto file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<char> buffer(file_size);
    if (!ifs.read(buffer.data(), file_size)) {
        std::cerr << "[Error] Failed to read file: " << file_to_zip << "\n";
        return false;
    }
    ifs.close();

    // Initialize miniz writer
    mz_zip_archive zip_archive;
    std::memset(&zip_archive, 0, sizeof(zip_archive));

    // If out_zip_name already exists, it will be overwritten
    if (!mz_zip_writer_init_file(&zip_archive, out_zip_name.c_str(), 0)) {
        std::cerr << "[miniz] Unable to initialize ZIP: " << out_zip_name << "\n";
        return false;
    }

    // File name in the ZIP => only the original file name (without path)
    fs::path p(file_to_zip);
    std::string filename_in_zip = p.filename().string();

    // Add to ZIP
    if (!mz_zip_writer_add_mem(
            &zip_archive,
            filename_in_zip.c_str(),
            buffer.data(),
            buffer.size(),
            MZ_BEST_COMPRESSION))
    {
        std::cerr << "[miniz] Failed to add file: " << file_to_zip << " => " << out_zip_name << "\n";
        mz_zip_writer_end(&zip_archive);
        return false;
    }

    // Finalize & end
    bool success = true;
    if (!mz_zip_writer_finalize_archive(&zip_archive)) {
        std::cerr << "[miniz] Finalize failed\n";
        success = false;
    }
    if (!mz_zip_writer_end(&zip_archive)) {
        std::cerr << "[miniz] Writer end failed\n";
        success = false;
    }

    return success;
}

/**
 * Compress multiple ZIP files (or any files) into a final large ZIP.
 * 
 * @param files        - List of file paths (these are the .zip files)
 * @param output_zip   - Name of the final output ZIP file (e.g. "all_zips.zip")
 */
bool create_final_zip(const std::vector<std::string>& files, const std::string& output_zip) {
    mz_zip_archive zip_archive;
    std::memset(&zip_archive, 0, sizeof(zip_archive));

    if (!mz_zip_writer_init_file(&zip_archive, output_zip.c_str(), 0)) {
        std::cerr << "[miniz] Unable to initialize final ZIP: " << output_zip << "\n";
        return false;
    }

    bool success = true;
    for (auto& file_path : files) {
        // Read file into buffer
        std::ifstream ifs(file_path, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "[Error] Unable to open file: " << file_path << "\n";
            success = false;
            continue;
        }
        ifs.seekg(0, std::ios::end);
        auto size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!ifs.read(buffer.data(), size)) {
            std::cerr << "[Error] Failed to read file: " << file_path << "\n";
            success = false;
            continue;
        }
        ifs.close();

        // Add to final ZIP
        // File name = the pure filename()
        fs::path p(file_path);
        std::string name_in_zip = p.filename().string(); // e.g. "xxx.zip"

        if (!mz_zip_writer_add_mem(&zip_archive,
                                   name_in_zip.c_str(),
                                   buffer.data(),
                                   buffer.size(),
                                   MZ_BEST_COMPRESSION))
        {
            std::cerr << "[miniz] Failed to add file: " << file_path << "\n";
            success = false;
        } else {
            std::cout << "Added " << file_path << " -> " << output_zip << "\n";
        }
    }

    if (!mz_zip_writer_finalize_archive(&zip_archive)) {
        std::cerr << "[miniz] Finalize failed\n";
        success = false;
    }
    if (!mz_zip_writer_end(&zip_archive)) {
        std::cerr << "[miniz] Writer end failed\n";
        success = false;
    }

    if (success) {
        std::cout << "=== Successfully created final package: " << output_zip << " ===\n";
    }
    return success;
}

int main(int argc, char* argv[]) {
    // argv[1]: Directory to traverse, default ./data
    // argv[2]: Final output file, default all_zips.zip
    std::string input_dir = "./data";
    std::string final_zip = "all_zips.zip";
    if (argc > 1) input_dir = argv[1];
    if (argc > 2) final_zip = argv[2];

    std::cout << "Step 1: Creating individual ZIP files for each .json...\n";

    // 1) Collect all json files
    std::vector<std::string> json_files;
    for (auto& entry : fs::recursive_directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".json") {
            json_files.push_back(entry.path().string());
        }
    }
    if (json_files.empty()) {
        std::cout << "No .json files found, nothing to do\n";
        return 0;
    }

    // 2) Create a single .zip for each JSON
    std::vector<std::string> created_zips;
    for (auto& jf : json_files) {
        fs::path p(jf);
        // For example "abc.json" -> "abc.zip"
        std::string zip_name = p.stem().string() + ".zip";
        // Create
        bool ok = create_zip_for_single_file(jf, zip_name);
        if (ok) {
            std::cout << "[OK] Created " << zip_name << "\n";
            created_zips.push_back(zip_name);
        } else {
            std::cerr << "[Fail] Failed to create " << zip_name << "\n";
        }
    }

    if (created_zips.empty()) {
        std::cout << "No successful .zip files\n";
        return 0;
    }

    std::cout << "\nStep 2: Packing all .zip files into " << final_zip << "\n";
    create_final_zip(created_zips, final_zip);

    std::cout << "\n=== All done ===\n";
    return 0;
}
