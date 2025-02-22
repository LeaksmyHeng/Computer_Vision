/**
 * Leaksmy Heng
 * CS5330
 * Feb-13-2025
 * Implementing openning by first applying erosion then dilation.
 * 
 * reference:
 * https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
 */


#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void applying_opening(cv::Mat &src, cv::Mat &dst) {
    /**
     * Function to apply opening which is erosion followed by dilation. The intension here is to remove noise or open interior holes.
     * https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f
     */

    // uses 4 connected kernel for erosion process and 8 in dilation process
    // https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    // first try is 4 connected kernel, second try 25x25
    cv::Mat fourByfourKernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3));
    cv::Mat eightByeightKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));

    // since i want to apply 2 different kernel to erosion and dilation, I won't use the cv::MORPH_OPEN
    // cv::morphologyEx(frame, frame, cv.MORPH_OPEN, );
    cv::Mat erodedFrame;
    cv::erode(src, erodedFrame, fourByfourKernel);

    cv::Mat dilationFrame;
    cv::dilate(erodedFrame, dst, eightByeightKernel);
}
