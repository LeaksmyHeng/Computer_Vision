/**
 * Leaksmy Heng
 * CS5330
 * Feb-13-2025
 * Implementing segmentation after applying thresholding and morphological operation on the video frame.
 * 
 * reference:
 * https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gae57b028a2b2ca327227c2399a9d53241
 * https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connectedcomponentswithstats-in-python
 * https://gist.github.com/JinpengLI/2cf814fe25222c645dd04e04be4de5a6
 * 
 *
 */

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace cv;
using namespace std;


cv::Scalar generateRandomColor() {
    /**
     * Add random color generator for cv::scalar
     */
    int r, g, b;
    int threshold = 200; // Any value above this will be considered "too light"

    do {
        r = rand() % 256; // Random number between 0 and 255
        g = rand() % 256;
        b = rand() % 256;
    } while (r > threshold && g > threshold && b > threshold); 
    return cv::Scalar(b, g, r);
}

void applying_connectedComponents(cv::Mat &src, cv::Mat &dst, cv::Mat &stats, cv::Mat &centroids){
    /**
     * Applying connected components function.
     * 
     * :param src: source image or in our case video frame
     * :param dst: destination image or video frame
     * :param states: matrix used to store stats data
     * :param centroid: matrix to stored centroid data
     */
    // threshold the image to make sure it is black and white
    int result = cv::connectedComponentsWithStats(src, dst, stats, centroids, 8, CV_32S);

    // draw boundary on the dst image
    for (int i = 1; i < result; i++) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        
        // draw rectangle with white color
        // todo: i specified red but still get white
        // i even use the random color generator but still get white
        cv::rectangle(dst, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 2);

        // print out the stats
        std::cout << "Region " << i << ":" << std::endl;
        std::cout << "Area: " << stats.at<int>(i, cv::CC_STAT_AREA) << std::endl;
        std::cout << "Bounding Box: (" 
                  << stats.at<int>(i, cv::CC_STAT_LEFT) << ", "
                  << stats.at<int>(i, cv::CC_STAT_TOP) << ") -> (" 
                  << stats.at<int>(i, cv::CC_STAT_WIDTH) << " x " 
                  << stats.at<int>(i, cv::CC_STAT_HEIGHT) << ")" << std::endl;
        std::cout << "  Centroid: (" 
                  << centroids.at<double>(i, 0) << ", "
                  << centroids.at<double>(i, 1) << ")" << std::endl;
    }
}
