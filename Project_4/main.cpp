/* 
Leaksmy Heng
CS5330
Feb 22, 2025
Project3
This is the pipeline of the program.
Press i will turn the program to inferencing mode
Press n will turn the program to training mode. Users will be prompted to label the image.
*/


#include <opencv2/opencv.hpp>

#include "Utils_header/image_detection.h"
#include "Utils_header/utils.h"


using namespace cv;


int main(int argc, char *argv[]) {
    /**
     * Main function. I use step one where I ran everything in here.
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
    // boolean to store the vector of corners
    bool vector_corners_save = false;

    // hold the 3D coordinates of each corner in world space
    std::vector<cv::Vec3f> point_set;
    std::vector<std::vector<cv::Vec3f> > point_list;
    std::vector<std::vector<cv::Point2f> > corner_list;

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
        else if (key == 's' || key == 'S') {
            printf("Store the vector of corners found by the last successful target detection into a corner_list.\n");
            vector_corners_save = true;
            // get image from Image directory and test that it was capture and display correctly.
            String path = "C:/Users/Leaksmy Heng/Documents/GitHub/CS5330/Computer_Vision/Project_4/Image/grayscale_latest_output_image.png";
            cv::Mat latest_image = cv::imread(path, IMREAD_GRAYSCALE);
            cv::imshow("image", latest_image);

            calibration_image_selection(latest_image, patternsize, vector_corners_save, point_set, point_list, corner_list);

            // delete the image and stored it in the Image/calibration folder
            // we can then check in the folder to see how many images we have for part 3 later
            std::time_t epoch_time = std::time(nullptr);
            std::time_t currentTime = std::time(0); // Get current epoch time
            std::string epochString = std::to_string(currentTime); // Convert epoch time to string
            std::string directory = "C:/Users/Leaksmy Heng/Documents/GitHub/CS5330/Computer_Vision/Project_4/Image/Calibrated_images/";
            std::string img = "grayscale_latest_output_image.png";
            std::string dest = directory + epochString + img;
            bool copy_image = copyFile(path, dest);
            if (copy_image) {
                std::cout << "Copy file successfully" << std::endl;
            }
        }

        // call the chess board corner detection
        checkboard_corner_detection(frame, patternsize);
        cv::imshow("Video", frame);
    }
    delete capdev;
    // Closes all the frames
    destroyAllWindows();
    return 0;
}
