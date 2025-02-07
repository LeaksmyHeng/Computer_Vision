#ifndef DISTANCEMETRIC_H
#define DISTANCEMETRIC_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

double sumOfSquaredDifference(const cv::Mat &targetImage, const cv::Mat &image);
double histogramIntersection(const cv::Mat &targetImage, const cv::Mat &image);
double chiSquareDistance(const cv::Mat& targetImage, const cv::Mat& image);
double weightedDistance(const cv::Mat& colorHistogram1, const cv::Mat& textureHistogram1, const cv::Mat& colorHistogram2, const cv::Mat& textureHistogram2);
double sumOfSquaredDifferenceVector(const vector<float>& targetImage, const vector<float>& image);

#endif // DISTANCEMETRIC_H