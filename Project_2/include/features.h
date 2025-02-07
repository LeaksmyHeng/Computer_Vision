#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

cv::Mat baselineMatching(const cv::Mat& image);
cv::Mat histogram(const cv::Mat& image, int numberOfBins = 8, bool is1D = false);
cv::Mat multiHistograms(const cv::Mat &image, int numberOfBins = 8);
cv::Mat texture(const cv::Mat& image);
cv::Mat colorTexture(const cv::Mat &image, int numberOfBins);

#endif // FEATURES_H