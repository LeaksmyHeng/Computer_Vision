#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

cv::Mat baselineMatching(const cv::Mat& image);

#endif // FEATURES_H