/**
 * Leaksmy Heng
 * CS5330
 * March 1, 2025
 * Header file for chess board corner detection
 */

#ifndef CHESSBOARD_CORNER_DETECTION_H
#define CHESSBOARD_CORNER_DETECTION_H
 
#include <opencv2/opencv.hpp>
 
using namespace cv;


/**
 * Function to detect the checkboard corner and save the vector corners.
 * By default, the save vector corner is false.
 */
void checkboard_corner_detection(cv::Mat &frame, cv::Size patternsize);

/**
 * When users press "S" or "s", the save_vector_corners will flip to true
 * When it flip to true, accessing the grayscale_latest_output_image in Image directory
 * and create a 3D world coordinate system of that image
 */
void calibration_image_selection(cv::Mat &latest_image, 
    cv::Size patternsize,
    bool save_vector_corners,
    std::vector<cv::Vec3f> point_set,
    std::vector<std::vector<cv::Vec3f>> point_list,
    std::vector<std::vector<cv::Point2f>> corner_list
);


/**
 * When users press "C" or "c", the images in the calibrated_images folder will get calibrated.
 * This is only executed if there are at least 5 images used to calibrate.
 */
double camera_calibration(int number_of_calibrated_images,
    int count_png_images, 
    const std::string& directory, 
    cv::Size patternsize, 
    std::vector<cv::Vec3f> point_set, 
    std::vector<std::vector<cv::Vec3f>> point_list, 
    std::vector<std::vector<cv::Point2f>> corner_list
);


#endif // CHESSBOARD_CORNER_DETECTION_H
