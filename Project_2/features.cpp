/*
  Leaksmy Heng
  Utility functions for storing image features.
*/

#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


cv::Mat baselineMatching(const cv::Mat &image) {
    /**
     * Use the 7x7 square in the middle of the image as a feature vector.
     * 
     * param image: used cv::Mat to store the image and have it as const so we are not going to modify it
     * 
     * return: img of the 7x7
     */
    
    // check if image size is smaller than 7.
    // if it is throw error
    if (image.cols < 7 || image.rows < 7) {
        throw std::out_of_range("Image is too small for a 7x7 region.");
    }

    // get the top left corner of the image with 7*7
    int x_coordinate = (image.cols / 2) - 3;
    int y_coordinate = (image.rows / 2) - 3;
    
    // Extracting a 7x7 patch from all three channels
    // using cv::rec(x, y, width, height)
    // which we found the coordinate x and y and width and height would be 7
    // because we are using 7x7 square img as feature
    cv::Rect rect(x_coordinate, y_coordinate, 7, 7);
    // cv::imshow("Extracted Image 7x7", image(rect).clone());
    // cv::waitKey(0);
    return image(rect).clone();
}


cv::Mat histogram(const cv::Mat &image, int numberOfBins) {
    /**
    * Convert image to histogram.
    */

    // initalized a 3D histogram with a specify number of bins for each color
    int histogramSize[] = {numberOfBins, numberOfBins, numberOfBins};

    // since know number of bins, now we calculate the range of pixel in 
    // each of the bins, which in this case is 32 (256 is max so divide by 8 bins)
    // therefore, we know bins 1: 0-36, bin2: 36-68, ... for each R,G,B
    int range = 256 / numberOfBins;

    // we need to count the pixel in each bins
    // first create the mat (matrix) to store the histogram
    // initiaze all value to 0
    cv::Mat feature = Mat::zeros(3, histogramSize, CV_32F);

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b imageColor = image.at<cv::Vec3b>(y, x);
            int blue = imageColor[0] / range;
            int green = imageColor[1] / range;
            int red = imageColor[2] / range;

            // increment the count of how many pixels fall into the bin corresponding to (b, g, r)
            feature.at<float>(blue, green, red)++;
        }
    }

    // Normalize the histogram using L1 normalization so that the sum of squared values equals 1
    // achieving this by dividing it with total pixel
    normalize(feature, feature, 1, 0, cv::NORM_L1, -1, cv::Mat());

    // Return the histogram
    return feature;
  }
