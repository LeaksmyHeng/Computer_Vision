/**
 * Leaksmy Heng
 * CS5330
 * Feb 12, 2017
 * Header file for image segmentation
 */

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/opencv.hpp>

using namespace cv;

/**
 * function use to find connected component. I use opencv function for this.
 */
void applying_connectedComponents(cv::Mat &src, cv::Mat &dst, cv::Mat &stats, cv::Mat &centroids, bool is_save_to_file=false);

#endif // SEGMENTATION_H
