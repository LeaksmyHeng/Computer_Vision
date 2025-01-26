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

int SepiaFilter( cv::Mat &src, cv::Mat &dst ) {
    /**
     * Implement a Sepia tone filter
     * Formular: 
     * 0.272, 0.534, 0.131    // Blue coefficients for R, G, B  (pay attention to the channel order)
     * 0.349, 0.686, 0.168    // Green coefficients
     * 0.393, 0.769, 0.189    // Red coefficients
     */
    dst = src.clone();

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(y, x);

            // store source red, green, blue pixel
            uchar blue = pixel[0];
            uchar green = pixel[1];
            uchar red = pixel[2];

            // store new red, green, and blue pixel
            int newBlue = static_cast<int>(red * 0.272 + green * 0.534 + blue * 0.131);
            int newGreen = static_cast<int>(red * 0.349 + green * 0.686 + blue * 0.168);
            int newRed = static_cast<int>(red * 0.393 + green * 0.769 + blue * 0.189);
            
            pixel[2] = cv::saturate_cast<uchar>(newRed);
            pixel[1] = cv::saturate_cast<uchar>(newGreen);
            pixel[0] = cv::saturate_cast<uchar>(newBlue);
        }
    }
    return 0;
}

int vignetting( cv::Mat &src, cv::Mat &dst ) {
    /**
     * Implement a Vignetting edge
     */

    // get the center of the image (x,y)
    cv::Point center(src.cols / 2, src.rows / 2);

    // clone greyframe or src as we are going to alter its value
    dst = src.clone();

    // get radius from center point
    double radius = std::sqrt(center.x * center.x + center.y * center.y);
    printf("radius is %f\n", radius);

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(y, x);

            // calculate distance from center
            double distance = std::sqrt(std::pow(x - center.x, 2) + std::pow(y - center.y, 2));
            // printf("distancce is %f\n", distance);

            // caculate vignette strength based on distance
            double vignetteWeight = 1.0 - (distance / radius);

            // Clamp the vignette weight to [0, 1]
            vignetteWeight = std::max(0.0, std::min(1.0, vignetteWeight));
            
            // Apply the vignette effect by multiplying the pixel values by the vignette weight
            // if (distance >= radius) {
            // apply vignette effect across all pixel not just the one that are distance >= radius
            // as the vignetteWeight will take care of it
            for (int c = 0; c < 3; c++) {
                pixel[c] = cv::saturate_cast<uchar>(src.at<cv::Vec3b>(y, x)[c] * vignetteWeight);
            }
            // }
        }
    }
    return 0;
}
