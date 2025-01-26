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
int SepiaFilter( cv::Mat &src, cv::Mat &dst );
int vignetting( cv::Mat &src, cv::Mat &dst );
int blur5x5_1( cv::Mat &src, cv::Mat &dst );
int blur5x5_2( cv::Mat &src, cv::Mat &dst );
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

#endif // FILTER_H
