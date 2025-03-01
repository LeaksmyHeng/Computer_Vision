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

void checkboard_corner_detection(cv::Mat &frame);

#endif // CHESSBOARD_CORNER_DETECTION_H
