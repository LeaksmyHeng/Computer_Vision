/**
 * Leaksmy Heng
 * CS5330
 * Feb-13-2025
 * Header file to implement thresholding to segregate images into foreground and background
 */


#ifndef THRESHOLDING_H
#define THRESHOLDING_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


/*
Implemented k-mean to seperated foreground and background in the image.
Using k-mean as it is dynamic to lighting.
k is the clustering. By default, it is set to 2. 
*/
int kMeanImplementation(cv::Mat &src, cv::Mat &dst , int k=2, int max_iteration=10, double epsilon=1.0);


#endif // FEATURES_H