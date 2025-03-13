/**
 * Leaksmy Heng
 * CS5330
 * March 4, 2025
 * Header file for chess board corner detection
 */

#ifndef UTILS_H
#define UTILS_H

/**
 * Copy file from source to destination.
 */
bool copyFile(const std::string& source, const std::string& destination);

/**
 * Delete file from specifying source.
 */
bool deleteFile(const std::string& source);

/**
 * Count total number of .png file in a specified directory.
 */
int count_png_file(const std::string& directory);

/**
 * Write camera matrix and distortion coefficient to a csv file.
 */
void write_calibration_to_csv(const cv::Mat& camera_matrix, const std::vector<double>& distortion_coefficients, const std::string& filename);

/**
 * Read camera matrix and distortion coefficient to a csv file.
 */
void read_calibration_from_csv(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs);

#endif // UTILS_H
