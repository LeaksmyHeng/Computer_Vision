#ifndef DISTANCEMETRIC_H
#define DISTANCEMETRIC_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

double sumOfSquaredDifference(cv::Mat &targetImage, cv::Mat &image);

#endif // DISTANCEMETRIC_H