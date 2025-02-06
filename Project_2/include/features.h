#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

cv::Mat baselineMatching(const cv::Mat& image);
cv::Mat histogram(const cv::Mat& image, int numberOfBins = 8);

#endif // FEATURES_H