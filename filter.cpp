/**
 * Leaksmy Heng
 * Jan 16 2025
 * This is a helper file that is used to put all of your image manipulation functions.
 */

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "filter.h"

using namespace cv;
using namespace std;

void greyScale( cv::Mat &src, cv::Mat &dst ) {
    /**
     * This function is used to convert color to greyscale
     */
    // convert color to greyscale through (COLOR_BGR2GRAY)
    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
}

int AlternativeGrayscale(cv::Mat &src, cv::Mat &dst) {
    /**
     * Since opencv already used luminosity method, therefore,
     * I'll use the Average Method which is (R+G+B) / 3
     * Source: https://www.baeldung.com/cs/convert-rgb-to-grayscale
     */
    dst = src.clone();  // Make a copy of the source image for the destination

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(y, x);

            // Average Method to convert to greyscale
            uchar grayscale = static_cast<uchar>((pixel[2] + pixel[1] +pixel[0]) / 3);

            // Set all channels (R, G, B) to the same grayscale value
            pixel[0] = grayscale; // Blue
            pixel[1] = grayscale; // Green
            pixel[2] = grayscale; // Red
        }
    }
    return 0;
}
