/**
 * Leaksmy Heng
 * CS5330
 * March 4, 2025
 * Header file for chess board corner detection
 */

#ifndef UTILS_H
#define UTILS_H

bool copyFile(const std::string& source, const std::string& destination);
bool deleteFile(const std::string& source);
int count_png_file(const std::string& directory);
void saveCalibrationResults(const cv::Mat& camera_matrix, const std::vector<double>& distortion_coefficients, const std::string& filename);

#endif // UTILS_H
