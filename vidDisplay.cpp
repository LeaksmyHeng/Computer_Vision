/**
 * Leaksmy Heng
 * Jan 16 2025
 * Read an image from a file and display it
 */

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "filter.h"

using namespace cv;
using namespace std;


int main(int argc, char *argv[]) {
    /**
     * This function open up the video camera and display it.
     * If users press 'q' => quite the program
     * If users press 's' => save an image to a file
     * If users press 'g' => change the video to greyscale instead of color
     * If users press 'c' => change the video to color again
     * If users press 'h' => change to the alternative greyscale instead of opencv greyscale
     * Reference code:
     * https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
     * https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
     * https://www.baeldung.com/cs/convert-rgb-to-grayscale
     */

    // open video from webcamp
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
    cv::Mat grayFrame;

    // 0 indicate that this is RGB video
    // 1 indicate that this is a greyscale video from opencv
    // -1 indicate that this is an alternative greyscale video i have generated
    int isGreyScaleVideo = 0;

    while (true) {
        // get a new frame from the camera, treat as a stream
        *capdev >> frame;
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }

        // use cvtColor to convert to greyscale and display it
        if (isGreyScaleVideo == 1) {
            greyScale( frame, grayFrame );
            cv::imshow("Video", grayFrame);
        }
        else if (isGreyScaleVideo == -1)
        {
            AlternativeGrayscale(frame, grayFrame);
            cv::imshow("Video", grayFrame);
        }
        // default function to display the color video
        else {
            cv::imshow("Video", frame);
        }

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
            isGreyScaleVideo = 1;
        }
        else if (key == 'c') {
            printf("Updating the video to back to color.\n");
            isGreyScaleVideo = 0;
        }
        else if (key == 'h') {
            printf("Updating the video to my greyscale image.\n");
            isGreyScaleVideo = -1;
        }
    }

    delete capdev;
    // Closes all the frames
    destroyAllWindows();
    return(0);
}
