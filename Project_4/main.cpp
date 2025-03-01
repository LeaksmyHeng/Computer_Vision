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
        
        // call the chess board corner detection
        checkboard_corner_detection(frame);
        cv::imshow("Video", frame);
    }
    delete capdev;
    // Closes all the frames
    destroyAllWindows();
    return 0;
}
