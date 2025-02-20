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
#include "Common_Header/morphological.h"
#include "Common_Header/segmentation.h"

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
    cv::Mat morphologicalFrame;
    cv::Mat stats, centroids, connectedComponent;

    bool save_to_file = false;

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
        // If 'N' or 'n' is pressed, save the feature vectors to file
        else if (key == 'n' || key == 'N') {
            save_to_file = true;  // Enable saving feature vectors to file
            printf("Collecting features and saving them...");
        }
        else {
            save_to_file = false; // Normal mode, no feature saving
        }

        // task1: creating thresholding
        // int kMeanImplementation(cv::Mat &src, cv::Mat &dst , int k=2, int max_iteration=10, double epsilon=1.0)
        kMeanImplementation(frame, grayFrame, 2, 3, 1.0);
        // task 2 applying morphology which I did using openning
        applying_opening(grayFrame, morphologicalFrame);
        applying_connectedComponents(morphologicalFrame, connectedComponent, stats, centroids, save_to_file);
        
        // cv::imshow("Video", frame);
        // cv::imshow("Thresholding", grayFrame);
        // cv::imshow("Morphological", morphologicalFrame);

        
        // i received an error on Error: Assertion failed (src_depth != CV_16F && src_depth != CV_32S) in convertToShow
        // which mean the connectedComponent can't be display so i normalized it here.
        // Normalize to range [0, 255] and convert to 8-bit unsigned image for display
        if (connectedComponent.type() == CV_16F) {
            cv::normalize(connectedComponent, connectedComponent, 0, 255, cv::NORM_MINMAX);  // Normalize the image to the range 0-255
            connectedComponent.convertTo(connectedComponent, CV_8U);
        }
        // normalize and convert a CV_32S image similarly if needed
        else if (connectedComponent.type() == CV_32S) {
            cv::normalize(connectedComponent, connectedComponent, 0, 255, cv::NORM_MINMAX);  // Normalize the image to the range 0-255
            connectedComponent.convertTo(connectedComponent, CV_8U);  // Convert to 8-bit unsigned image
        }
        cv::imshow("ConnectedComponent", connectedComponent * 50);

    }
    delete capdev;
    // Closes all the frames
    destroyAllWindows();
    return 0;
}
