/**
 * Leaksmy Heng
 * Jan 16 2025
 * This is the header file for filter.cpp
 */

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>


void greyScale( cv::Mat &src, cv::Mat &dst );
int AlternativeGrayscale( cv::Mat &src, cv::Mat &dst );

#endif // FILTERS_H
