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
    cv::Size imageSize = frame.size();

    // pattern size of the chessboard specified in the instruction
    cv::Size patternsize(9,6);
    // boolean to store the vector of corners
    bool vector_corners_save = false;

    // hold the 3D coordinates of each corner in world space
    std::vector<cv::Vec3f> point_set;
    std::vector<std::vector<cv::Vec3f> > point_list;
    std::vector<std::vector<cv::Point2f> > corner_list;

    // store how many calibrated image we have
    int number_of_calibrated_images = 0;

    // as per task3 instruction
    // make the camera matrix 3x3 cv::Mat of type CV_64FC1
    // initialized all elements to 0 and alter it later
    cv::Mat camera_matrix = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
    camera_matrix.at<double>(0, 0) = 1.0;
    camera_matrix.at<double>(1, 1) = 1.0;
    camera_matrix.at<double>(2, 2) = 1.0;
    camera_matrix.at<double>(0, 2) = frame.cols/2.0;
    camera_matrix.at<double>(1, 2) = frame.rows/2.0;

    std::vector<double> distortion_coefficients;

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
            std::string img = "_grayscale_latest_output_image.png";
            std::string dest = directory + epochString + img;
            bool copy_image = copyFile(path, dest);
            if (copy_image) {
                std::cout << "Copy file successfully" << std::endl;
                bool delete_file = deleteFile(path);
                if (delete_file) {
                    std::cout << "delete file successfully" << std::endl;
                }
            }

            // increment the number of calibrated image
            number_of_calibrated_images += 1;
        }
        
        // if users press c or C, calibrate the image if possible
        else if (key == 'c' || key == 'C') {
            // if this is less than 5, check the folders to see if there are at least 5 png file in there, if it is,
            // we can call calibration_image_selection
            std::string directory = "C:/Users/Leaksmy Heng/Documents/GitHub/CS5330/Computer_Vision/Project_4/Image/Calibrated_Images/";
            int count_png_images = count_png_file(directory); 

            std::vector<cv::Mat> rotations, translations;
            
            double camera_cal = camera_calibration(
                number_of_calibrated_images, 
                count_png_images, 
                directory,
                patternsize,
                point_set, 
                point_list, 
                corner_list,
                camera_matrix,
                distortion_coefficients,
                rotations,
                translations);
            
            // enable users to save rotations and translations
            // the prompt will pop up to ask if users want to write out the intrinsic parameters to a file
            // both the camera_matrix and the distortion_ceofficients
            char decision;
            std::cout << "Do you want to save the intrinsic parameters to file? Type Y for yes otherwise type whatever. ";
            std::cin >> decision;
            if (decision == 'Y' || decision == 'y') {
                std::cout << "Saving to file" << std::endl;
                // call save to file function
                std::string fileName = "output_file.csv";
                write_calibration_to_csv(camera_matrix, distortion_coefficients, fileName);
            }
            else {
                std::cout << "Okay not saving then." << std::endl;
            }
        }

        // call the chess board corner detection
        // uncomment this if you just want to do task 1
        checkboard_corner_detection(frame, patternsize);
        cv::imshow("Video", frame);
    }
    delete capdev;
    // Closes all the frames
    destroyAllWindows();
    return 0;
}
