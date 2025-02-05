/*
  Leaksmy Heng
  Utility functions for storing image features.
*/

#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


cv::Mat baselineMatching(const cv::Mat& image) {
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
