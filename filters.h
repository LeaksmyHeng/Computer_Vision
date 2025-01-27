/**
 * Leaksmy Heng
 * Jan 16 2025
 * This is the header file for filters.cpp
 */

#ifndef FILTERS_H
#define FILTERS_H

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
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );
int coolTone( cv::Mat &src, cv::Mat &dst );
int lowPassFilter(cv::Mat &src, cv::Mat &dst);
int highPassFilter(cv::Mat &src, cv::Mat &dst);
int highPassFaceDetection(cv::Mat &src, cv::Mat &dst);

#endif // FILTERS_H
