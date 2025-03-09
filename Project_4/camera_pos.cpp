/* 
Leaksmy Heng
CS5330
March 08, 2025
Project3
This is the pipeline of the program.
This is for task 4
*/


#include <opencv2/opencv.hpp>

#include "Utils_header/utils.h"


using namespace cv;


int main(int argc, char *argv[]) {
    /**
     * Main function. This is for task 4.
     */
    // open video from webcam
    cv::VideoCapture *capdev;

    // open the video device
    // 1 is my external webcam. use 0 for laptop camera
    capdev = new cv::VideoCapture(0);
    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                    (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;

    // pattern size of the chessboard specified in the instruction
    cv::Size patternsize(9,6);

    // hold the 3D coordinates of each corner in world space
    // I measure the square and it is about 2.5cm
    std::vector<cv::Vec3f> point_set;
    float square_size = 0.025f; // Size of a square in meters
    for (int i = 0; i < patternsize.height; ++i) {
        for (int j = 0; j < patternsize.width; ++j) {
            point_set.push_back(cv::Vec3f(j * square_size, i * square_size, 0));
        }
    }

    while (true) {
        *capdev >> frame;
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }

        char key = cv::waitKey(10);
        if (key == 'q' || key == 'Q') {
            printf("Quitting the program\n");
            break;
        }

        cv::Mat camera_mat = Mat::eye(3, 3, CV_64F);
        cv::Mat dist_coeffs = Mat::zeros(5, 1, CV_64F);
        
        std::string file_name = "C:/Users/Leaksmy Heng/Documents/GitHub/CS5330/Computer_Vision/Project_4/build/Debug/output_file.csv";
        read_calibration_from_csv(file_name, camera_mat, dist_coeffs);

        // std::cout << "Camera Matrix:\n" << camera_mat << std::endl;
        // std::cout << "Distortion Coefficients:\n" << dist_coeffs << std::endl;

        // check if camera_mat and dis_coeffs are empty
        if (camera_mat.empty() || dist_coeffs.empty()) {
            std::cout << "camera mat and dist coeff are empty" << std::endl;
            return -1;
        }
 
        // convert current frame to grayscale
        cv::Mat grayscale;
        cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);

        // these part of code are dup from task1
        std::vector<cv::Point2f> corners;
        bool patternfound = findChessboardCorners(grayscale, patternsize, corners);

        if (patternfound) {
            cv::TermCriteria criterial(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1);
            cv::cornerSubPix(grayscale, corners, Size(11, 11), Size(-1, -1), criterial);
            // std::cout << "Number of corners found: " << corners.size() << std::endl;
            
            // solvePnP
            // std::cout << "Proceed to Calculate Current Position of the Camera." << std::endl;
            cv::Mat rvec, tvec;
            bool result = cv::solvePnP(point_set, corners, camera_mat, dist_coeffs, rvec, tvec);
            if (!result) {
                std::cout << "solvePnP failed!" << std::endl;
            }
            else {
                std::cout << "Rotation: " << rvec.t() << std::endl;
                std::cout << "Translation: " << tvec.t() << std::endl;
            }
        }
        
        cv::imshow("Video", frame);
    }
    delete capdev;
    // Closes all the frames
    destroyAllWindows();
    return 0;
}
