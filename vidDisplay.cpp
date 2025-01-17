/**
 * Leaksmy Heng
 * Jan 16 2025
 * Read an image from a file and display it
 */

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    /**
     * This function open up the video camera and display it.
     * If users press 'q' => quite the program
     * If users press 's' => save an image to a file
     * Reference code:
     * https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
     * 
     */
    cv::VideoCapture *capdev;

    // open the video device
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
        // get a new frame from the camera, treat as a stream
        *capdev >> frame;
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }                
        cv::imshow("Video", frame);

        // see if there is a waiting keystroke
        char key = cv::waitKey(10);
        std::cout << "Key pressed: \n" << static_cast<char>(key) << std::endl;
        if( key == 'q') {
            printf("Quiting the program\n");
            break;
        }
        else if (key == 's') {
            printf("Saving the image frame to saved_image.jpg\n");
            cv::imwrite("saved_image.jpg", frame);
        }
        else if (key == 'g') {
            printf("Updating the video to grey scale instead of color.\n");
        }
    }

    delete capdev;
    // Closes all the frames
    destroyAllWindows();
    return(0);
}
