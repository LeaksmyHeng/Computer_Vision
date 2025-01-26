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
     * If users press 'p' => change to the sepia filter
     * If users press 'v' => apply sepia filter with vignetting
     * if users press 'f' => apply 5*5 blurred filter
     * if users press 'b' => apply 5*5 blurred filter but using seperable filter instead
     * if users press 'x' => apply sobel 3x3 filter horizontal edge
     * if users press 'y' => apply sobel 3x3 filter vertial edge
     * 
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
    cv::Mat vignettingFrame;
    cv::Mat blurredFilter1;
    cv::Mat blurredFilter2;
    // all for sobel filter
    cv::Mat sobelX, sobelY, sobelAbsX, sobelAbsY, sobelVisualX, sobelVisualY;
    // gradient magnitude image
    cv::Mat gradientMagnitude;

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
        // apply sepia filter if user press p
        else if (isGreyScaleVideo == -2)
        {
            SepiaFilter(frame, grayFrame);
            cv::imshow("Video", grayFrame);
        }
        // apply sepia filter with vignetting if user press v
        else if (isGreyScaleVideo == -3)
        {
            SepiaFilter(frame, grayFrame);
            vignetting(grayFrame, vignettingFrame);
            cv::imshow("Video", vignettingFrame);
        }
        // apply 5x5 gaussian blurred filter
        else if (isGreyScaleVideo == -4)
        {
            blur5x5_1( frame, blurredFilter1 );
            cv::imshow("Video", blurredFilter1);
        }
        // apply 5x5 gaussian blurred filter using separable methods
        else if (isGreyScaleVideo == -5)
        {
            blur5x5_2( frame, blurredFilter2 );
            cv::imshow("Video", blurredFilter2);
        }
        else if (isGreyScaleVideo == -6)
        {
            sobelX3x3( frame, sobelX );
            // display image from (0-255)
            cv::convertScaleAbs(sobelX, sobelVisualX);
            cv::imshow("Video", sobelVisualX);
        }
        else if (isGreyScaleVideo == -7)
        {
            sobelY3x3( frame, sobelY );
            // display image from (0-255)
            cv::convertScaleAbs(sobelY, sobelVisualY);
            cv::imshow("Video", sobelVisualY);
        }
        else if (isGreyScaleVideo == -8)
        {
            sobelX3x3(frame, sobelX);
            sobelY3x3(frame, sobelY);
            magnitude(sobelX, sobelY, gradientMagnitude);
            cv::imshow("Video", gradientMagnitude);
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
            printf("Updating the video to my greyscale.\n");
            isGreyScaleVideo = -1;
        }
        else if (key == 'p') {
            printf("Updating the video to my Sepia filter.\n");
            isGreyScaleVideo = -2;
        }
        else if (key == 'v') {
            printf("Updating the video to Sepia filter with vignetting edge.\n");
            isGreyScaleVideo = -3;
        }
        else if (key == 'f') {
            printf("Updating the video to Gaussian 5x5 blurred filter.\n");
            isGreyScaleVideo = -4;
        }
        else if (key == 'b') {
            printf("Updating the video to Gaussian 5x5 blurred filter - separable functions.\n");
            isGreyScaleVideo = -5;
        }
        else if (key == 'x') {
            printf("Updating the video to sobelX3x3.\n");
            isGreyScaleVideo = -6;
        }
        else if (key == 'y') {
            printf("Updating the video to sobelY3x3.\n");
            isGreyScaleVideo = -7;
        }
        else if (key == 'm') {
            printf("Updating the video to gradient magnitude image.\n");
            isGreyScaleVideo = -8;
        }
    }

    delete capdev;
    // Closes all the frames
    destroyAllWindows();
    return(0);
}
