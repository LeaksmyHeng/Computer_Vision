/* 
Leaksmy Heng
CS5330
March 11, 2025
Project3
This is the pipeline of the program.
This is for task 7
*/

#include <opencv2/opencv.hpp>

using namespace cv;

// variables for trackbar parameters
int blockSize = 2;
int apertureSize = 3;
int k = 4;
int thresh = 150;
int blurAmount = 1;

int main() {
    /**
     * Script for task7. This implement Harris Corner.
     */
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }

    // Create window and trackbars
    cv::namedWindow("Harris Corner Detection", WINDOW_AUTOSIZE);
    cv::createTrackbar("Block Size", "Harris Corner Detection", &blockSize, 10);
    cv::createTrackbar("Aperture", "Harris Corner Detection", &apertureSize, 7);
    cv::createTrackbar("k (x100)", "Harris Corner Detection", &k, 100);
    cv::createTrackbar("Threshold", "Harris Corner Detection", &thresh, 255);
    cv::createTrackbar("Blur", "Harris Corner Detection", &blurAmount, 5);

    Mat frame, grayscale, dst;
    while(true) {
        cap >> frame;
        if(frame.empty()) break;

        // Convert to grayscale
        cv::cvtColor(frame, grayscale, COLOR_BGR2GRAY);

        // Apply Gaussian blur for noise reduction
        // because harris corner detection uses image gradient which amplify high-frequency noise
        // so use gaussian here to smooth it out too
        cv::GaussianBlur(grayscale, grayscale, Size(blurAmount*2+1, blurAmount*2+1), 0);

        // Harris corner detection parameters
        int actualBlockSize = blockSize > 0 ? blockSize : 1;
        int actualAperture = apertureSize > 0 ? apertureSize*2+1 : 1;
        double kValue = k / 1000.0;

        // Detect corners
        cv::cornerHarris(grayscale, dst, actualBlockSize, actualAperture, kValue);

        // Normalize and threshold
        cv::Mat dst_norm, dst_norm_scaled;
        cv::normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1);
        cv::convertScaleAbs(dst_norm, dst_norm_scaled);

        // Draw corners on original image
        for(int i = 0; i < dst_norm.rows; i++) {
            for(int j = 0; j < dst_norm.cols; j++) {
                if((int)dst_norm.at<float>(i,j) > thresh) {
                    cv::circle(frame, Point(j,i), 4, cv::Scalar(0,0,255), 2);
                }
            }
        }

        cv::imshow("Harris Corner Detection", frame);

        if(waitKey(10) == 27) {
            break;
        };
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
