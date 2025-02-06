/*
  Leaksmy Heng
  Feb-04-2024
  CS5330
  This fill contain information related to distance metrics.
*/

#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

double sumOfSquaredDifference(cv::Mat &targetImage, cv::Mat &image) {
    /**
     * Function to calculate sum of squared different between two images.
     * Formula SSD = sum of ( (R1 - R2)^2 + (G1 - G2)^2 + (B1 - B2)^2)
     */

    double result = 0.0;

    // we know targetImage and image will always have the same size as
    // it goes through the feature extraction (7*7)
    // but just in case, we will do a check here
    if (targetImage.size() != image.size()) {
        printf("Invalid size for targetimage and image");
        return -1;
    }

    // loop through each row and cols
    for (int y = 0; y < targetImage.rows; ++y) {
        for (int x = 0; x < targetImage.cols; ++x) {
            // Get the color values (BGR)
            cv::Vec3b targetImageColor = targetImage.at<cv::Vec3b>(y, x);
            cv::Vec3b imageColor = image.at<cv::Vec3b>(y, x);

            // Compute the squared differences for each channel (Blue, Green, Red)
            double diffB = (targetImageColor[0] - imageColor[0]) * (targetImageColor[0] - imageColor[0]);
            double diffG = (targetImageColor[1] - imageColor[1]) * (targetImageColor[1] - imageColor[1]);
            double diffR = (targetImageColor[2] - imageColor[2]) * (targetImageColor[2] - imageColor[2]);

            // Add up the squared differences for this pixel
            result += diffB + diffG + diffR;
        }
    }

    // printf("Sum of square result is %f\n", result);
    return result;
}

double histogramIntersection(const cv::Mat &targetImage, const cv::Mat &image) {
    /*
     * Function to calculate histogramIntersection between 2 images
     * Histogram intersection = 1 - sum of min(xi, yi)
     */
    
    double result = 0.0;
    // printf("Before looping in histogram intersection\n");
    for (int b = 0; b < targetImage.size[0]; ++b) {
        for (int g = 0; g < targetImage.size[1]; ++g) {
            for (int r = 0; r < targetImage.size[2]; ++r) {
                // Add the minimum value between corresponding bins from both histograms
                result += std::min(targetImage.at<float>(b, g, r), image.at<float>(b, g, r));
            }
        }
    }

    // printf("Histogram result is %f\n", result);
    return result;
}
