/* 
Leaksmy Heng
CS5330
March 4, 2025
Project3
This is some other helper functions used across this project.
*/


#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>


bool copyFile(const std::string& source, const std::string& destination) {
    /**
     * Function to copy file from one source to another.
     */
    try {
        std::filesystem::copy(source, destination, std::filesystem::copy_options::overwrite_existing);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error copying file: " << e.what() << std::endl;
        return false;
    }
}


bool deleteFile(const std::string& source) {
    /**
     * Function to delete file.
     */
    int status = remove("myfile.txt");
    // Check if the file has been successfully removed
    if (status != 0) {
        return false;
    }
    return true;
}


int count_png_file(const std::string& directory) {
    /**
     * Function to count total number of png file in a specified directory.
     */
    int total = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            total += 1;
        }
    }
    return total;
}
