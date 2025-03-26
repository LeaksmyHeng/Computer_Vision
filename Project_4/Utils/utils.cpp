/* 
Leaksmy Heng
CS5330
March 4, 2025
Project3
This is some other helper functions used across this project.
*/

#include <opencv2/opencv.hpp>
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
    int status = std::filesystem::remove(source);
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


void write_calibration_to_csv(const cv::Mat& camera_matrix, const std::vector<double>& distortion_coefficients, const std::string& filename) {
    /**
     * Function to save camera instrinsic value to file.
     */
    // Open the file for writing
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file for writing!" << std::endl;
        return;
    }

    // Write the camera matrix
    file << "Camera Matrix\n";
    for (int i = 0; i < camera_matrix.rows; ++i) {
        for (int j = 0; j < camera_matrix.cols; ++j) {
            file << camera_matrix.at<double>(i, j);
            if (j < camera_matrix.cols - 1) file << ","; // Add a comma between values
        }
        file << "\n"; // New line after each row
    }

    // Write the distortion coefficients
    file << "Distortion Coefficients\n";
    for (size_t i = 0; i < distortion_coefficients.size(); ++i) {
        file << distortion_coefficients[i];
        if (i < distortion_coefficients.size() - 1) file << ","; // Add a comma between values
    }
    file << "\n";
    // Close the file
    file.close();
}


void read_calibration_from_csv(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    /**
     * Function to read camera instrinsic value from csv file.
     */
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Unable to open file for reading!" << std::endl;
        return;
    }

    std::string line;
    bool readingCameraMatrix = false;
    bool readingDistCoeffs = false;

    int row = 0;
    while (getline(file, line)) {
        if (line.find("Camera Matrix") != std::string::npos) {
            readingCameraMatrix = true;
            readingDistCoeffs = false;
            row = 0;
            continue;
        } else if (line.find("Distortion Coefficients") != std::string::npos) {
            readingCameraMatrix = false;
            readingDistCoeffs = true;
            row = 0;
            continue;
        }

        std::stringstream ss(line);
        std::string value;

        if (readingCameraMatrix) {
            int col = 0;
            while (getline(ss, value, ',')) {
                cameraMatrix.at<double>(row, col++) = std::stod(value);
            }
            row++;
        } else if (readingDistCoeffs) {
            int col = 0;
            while (getline(ss, value, ',')) {
                distCoeffs.at<double>(row++, col) = std::stod(value);
            }
        }
    }
    file.close();
}
