/**
 * Leaksmy Heng
 * CS5330
 * Feb-13-2025
 * Header file to implement openning to the video after using k-mean for segnmentation.
 */


#ifndef MORPHOLOGICAL_H
#define MORPHOLOGICAL_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * Applying openning which is  erosion follow by dilation.
 */
void applying_opening(cv::Mat &src, cv::Mat &dst);

#endif // MORPHOLOGICAL_H