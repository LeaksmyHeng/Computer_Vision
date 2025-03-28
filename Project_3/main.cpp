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
#include <iostream>
#include <filesystem>
#include <fstream>

#include "Common_Header/thresholding.h"
#include "Common_Header/morphological.h"
#include "Common_Header/segmentation.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;


#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>

#include "Common_Header/thresholding.h"
#include "Common_Header/morphological.h"
#include "Common_Header/segmentation.h"
#include "Common_Header/classifying_image.h"

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
    string currentLabel;
    vector<ObjectFeature> featureList;
    const std::string outputFilename = "object_features.csv";

    // // Calculate feature statistics
    // FeatureStats featureStats = calculateFeatureStats(featureList);

    bool showInference = false; // Add a flag to control Inference display

    while (true) {
        *capdev >> frame;
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }
        cv::Mat original_frame = frame.clone();
        //cv::imshow("frame", frame);
        char key = cv::waitKey(10);
        if (key == 'q' || key == 'Q') {
            save_features_to_csv(featureList, outputFilename);
            printf("Quitting the program\n");
            break;
        }
        // If 'N' or 'n' is pressed, save the feature vectors to file
        // when i have the file open csv and write it one by one, my program become in efficient
        // as of now, I have it store to a vector.
        // when users press 's' or 'S', then write it to a file and clear that vector.
        else if (key == 'n' || key == 'N') {
            save_to_file = true;  // Enable saving feature vectors to file
            std::cout << "Collecting features and saving them...";
            std::cout << "Enter a label for the current object: ";
            std::getline(std::cin, currentLabel);
            cv::imshow("ConnectedComponent", connectedComponent * 50);
        }
        else if (key == 's' || key == 'S') {
            cv::imshow("ConnectedComponent", connectedComponent * 50);
            std::cout << "Save to file";
            save_features_to_csv(featureList, outputFilename);      // save to csv
            save_to_file = false;                                   // Disable saving to memory
            featureList.clear();                                    // clear the struct
            
        }
        else if (key == 'i' || key == 'I') {
            // apply_classification_to_video(original_frame, featureList, stats, centroids);
            showInference = !showInference;  // Toggle the flag
        }
        else {
            save_to_file = false; // Normal mode, no feature saving
        }

        // task1: creating thresholding
        // int kMeanImplementation(cv::Mat &src, cv::Mat &dst , int k=2, int max_iteration=10, double epsilon=1.0)
        // kMeanImplementation(frame, grayFrame, 2, 3, 1.0);
        GaussianBlur(frame, frame, Size(5, 5), 0);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        // cv::threshold(frame, grayFrame, 125, 255, cv::THRESH_BINARY);
        cv::threshold(frame, grayFrame, 127, 255, cv::THRESH_BINARY_INV);
        
        // task 2 applying morphology which I did using openning
        applying_opening(grayFrame, morphologicalFrame);
        // srand(static_cast<unsigned int>(time(0)));
        applying_connectedComponents(morphologicalFrame, connectedComponent, stats, centroids, save_to_file, currentLabel, featureList);
        
         if (showInference) {
            // Load features from CSV file
            vector<ObjectFeature> features_from_csv = load_feature_from_csv(outputFilename);
            vector<double> standard_deviation = calculateStandardDeviations(features_from_csv);
            double threshold = setInitialThreshold(features_from_csv, standard_deviation);
            
            // // print out the standard deviation of the training features and threshold
            // for (size_t i = 0; i < standard_deviation.size(); ++i) {
            //     std::cout << "Standard deviation of feature " << i + 1 << ": " << standard_deviation[i] << std::endl;
            // }
            // std::cout << "Threshold is: " << threshold << std::endl;
            vector<vector<double>> features_from_frame = extractFeaturesFromFrame(original_frame);
            for (const auto& feature_vector : features_from_frame) {
                // // classifying using euclidean
                string label = classifyObjectWithUnknownDetection(feature_vector, features_from_csv, standard_deviation, threshold);
                // classifying using knn
                // string label = classifyObjectUsingKNN(feature_vector, features_from_csv, standard_deviation);

                // std::cout << "Label is " << label << std::endl;
                // Display the label on the frame (you might want to adjust the position for each object)
                putText(original_frame, label, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
            }
            
            // // Display the label on the frame
            // putText(original_frame, label, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
            cv::imshow("Inference", original_frame);
        }
        else {
            cv::imshow("Video", original_frame);
        }

        
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
        // cv::imshow("ConnectedComponent", connectedComponent * 50);

    }
    delete capdev;
    // Closes all the frames
    destroyAllWindows();
    return 0;
}
