/* 
Leaksmy Heng
CS5330
March 1, 2025
Project4
This is for the first task where we have to detect and extract target corners
*/

#include <opencv2/opencv.hpp>

using namespace cv;


void checkboard_corner_detection(cv::Mat &frame) {
    // I am using the bool cv::findChessboardCorners
    // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a

    // convert frame to grayscale because the input image in findCheckboardCorners must be an 8-bit grayscale or color image.
    cv::Mat grayscale;
    cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);

    // pattern size based on the assignment is 9x6
    cv::Size patternsize(9,6);
    std::vector<cv::Point2f> corners;

    // sample flag CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK found in findChessboardcorner
    // but for right now, just have it has None
    bool patternfound = findChessboardCorners(grayscale, patternsize, corners);
    if (patternfound) {
        // std::cout << "Found chess board corner" << std::endl;
        cv::Size winSize(11,11);
        cv::Size zeroZone(-1,-1);
        cv::TermCriteria criterial(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1);

        // https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
        cv::cornerSubPix(grayscale, corners, Size(11, 11), Size(-1, -1), criterial);
        std::cout << "Number of corners found: " << corners.size() << std::endl;

        // draw the target corner and show it
        // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga6a10b0bb120c4907e5eabbcd22319022
        cv::drawChessboardCorners(grayscale, patternsize, corners, patternfound);
        // cv::imshow("grayscale", grayscale);

        if (corners.size() >= 1) {
            // get the corner_set[i].x and corner_set[i].y
            // coordinate of the first corner.
            std::cout << "0" << " Corner x and y coordinate is " << corners[0].x << ", " << corners[0].y << std::endl;
        }
    }
}
