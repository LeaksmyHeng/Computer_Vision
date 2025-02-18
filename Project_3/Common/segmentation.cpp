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

using namespace cv;
using namespace std;


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
        cv::rectangle(dst, cv::Rect(x, y, w, h), cv::Scalar(255), 2);

        // print out the stats
        std::cout << "Component " << i << ":" << std::endl;
        std::cout << "  Area: " << stats.at<int>(i, cv::CC_STAT_AREA) << std::endl;
        std::cout << "  Bounding Box: (" 
                  << stats.at<int>(i, cv::CC_STAT_LEFT) << ", "
                  << stats.at<int>(i, cv::CC_STAT_TOP) << ") -> (" 
                  << stats.at<int>(i, cv::CC_STAT_WIDTH) << " x " 
                  << stats.at<int>(i, cv::CC_STAT_HEIGHT) << ")" << std::endl;
        std::cout << "  Centroid: (" 
                  << centroids.at<double>(i, 0) << ", "
                  << centroids.at<double>(i, 1) << ")" << std::endl;
    }
}
