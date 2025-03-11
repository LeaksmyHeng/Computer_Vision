#include <opencv2/opencv.hpp>

using namespace cv;

// Global variables for trackbar parameters
int blockSize = 2;
int apertureSize = 3;
int k = 4;
int thresh = 150;
int blurAmount = 1;

int main() {
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
        cvtColor(frame, grayscale, COLOR_BGR2GRAY);

        // Apply Gaussian blur
        GaussianBlur(grayscale, grayscale, Size(blurAmount*2+1, blurAmount*2+1), 0);

        // Harris corner detection parameters
        int actualBlockSize = blockSize > 0 ? blockSize : 1;
        int actualAperture = apertureSize > 0 ? apertureSize*2+1 : 1;
        double kValue = k / 1000.0;

        // Detect corners
        cornerHarris(grayscale, dst, actualBlockSize, actualAperture, kValue);

        // Normalize and threshold
        Mat dst_norm, dst_norm_scaled;
        normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1);
        convertScaleAbs(dst_norm, dst_norm_scaled);

        // Draw corners on original image
        for(int i = 0; i < dst_norm.rows; i++) {
            for(int j = 0; j < dst_norm.cols; j++) {
                if((int)dst_norm.at<float>(i,j) > thresh) {
                    circle(frame, Point(j,i), 4, Scalar(0,0,255), 2);
                }
            }
        }

        imshow("Harris Corner Detection", frame);

        // Exit on ESC
        if(waitKey(10) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
