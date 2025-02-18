/* 
Leaksmy Heng
CS5330
Feb 07, 2025
Project2
This is the pipeline of the program
I implemented step 1 where i call the feature function and distance method function
all in one for loop (for loop that is used to go through the images in the database file)
*/


#include <opencv2/opencv.hpp>

#include "Common_Header/thresholding.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    /**
     * Main function. I use step one where I ran everything in here.
     * reference:
     * code from project_1 to open and close the video frame
     */
    // open video from webcam
    cv::VideoCapture *capdev;

    // open the video device
    // 1 is my external webcam. use 0 for laptop camera
    capdev = new cv::VideoCapture(1);
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
    cv::Mat grayFrame;

    while (true) {
        *capdev >> frame;
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }

        // task1: creating thresholding
        // int kMeanImplementation(cv::Mat &src, cv::Mat &dst , int k=2, int max_iteration=10, double epsilon=1.0)
        kMeanImplementation(frame, grayFrame, 2, 3, 1.0);
        cv::imshow("Video", grayFrame);


        char key = cv::waitKey(10);
        // std::cout << "Key pressed: \n" << static_cast<char>(key) << std::endl;
        if( key == 'q') {
            printf("Quiting the program\n");
            break;
        }
    }
    delete capdev;
    // Closes all the frames
    destroyAllWindows();
    return 0;
}
