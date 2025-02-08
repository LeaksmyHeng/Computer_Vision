/*
  Leaksmy Heng
  CS5330
  Feb 07, 2025
  Project2
  Header file for feature extraction file.
  The docstring of each method is in features.cpp
*/

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
vector<float> extractColorFeatures(const string& image);

#endif // FEATURES_H