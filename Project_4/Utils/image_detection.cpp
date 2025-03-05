/* 
Leaksmy Heng
CS5330
March 1, 2025
Project4
This is for the first task where we have to detect and extract target corners
*/

#include <opencv2/opencv.hpp>

using namespace cv;


void checkboard_corner_detection(cv::Mat &frame, cv::Size patternsize) {
    /**
     * Task1:
     * This task is to Detect and Extract Target Corners.
     * First I convert the frame from RBG to grayscale
     * Then apply the findChessboardCorners function from opencv. This function return bool.
     * If it returns True, that means it was able to find the checkboard.
     * If chessboard is found, I then use cornerSubPix fun to get the number of corners in the chess board
     * I print out the number of corner found and draw the chess board using drawChessboardCorners
     */
    
    // I am using the bool cv::findChessboardCorners
    // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a

    // convert frame to grayscale because the input image in findCheckboardCorners must be an 8-bit grayscale or color image.
    cv::Mat grayscale;
    cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);

    // pattern size based on the assignment is 9x6
    std::vector<cv::Point2f> corners;

    // sample flag CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK found in findChessboardcorner
    // but for right now, just have it has None
    bool patternfound = findChessboardCorners(grayscale, patternsize, corners);
    if (patternfound) {

        // Task2: if check board found, capture the image (latest image)
        // https://stackoverflow.com/questions/33503138/how-to-extract-video-frames-and-save-them-as-images-using-c
        String grayscale_name = "grayscale_latest_output_image.png";
        String frame_name = "latest_output_image.png";
        cv::imwrite("C:/Users/Leaksmy Heng/Documents/GitHub/CS5330/Computer_Vision/Project_4/Image/" + grayscale_name, grayscale);
        cv::imwrite("C:/Users/Leaksmy Heng/Documents/GitHub/CS5330/Computer_Vision/Project_4/Image/" + frame_name, frame);

        // std::cout << "Found chess board corner" << std::endl;
        cv::Size winSize(11,11);
        cv::Size zeroZone(-1,-1);
        cv::TermCriteria criterial(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1);

        // https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
        cv::cornerSubPix(grayscale, corners, Size(11, 11), Size(-1, -1), criterial);
        std::cout << "Number of corners found: " << corners.size() << std::endl;

        // draw the target corner and show it
        // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga6a10b0bb120c4907e5eabbcd22319022
        cv::drawChessboardCorners(grayscale, patternsize, corners, patternfound);
        // cv::imshow("grayscale", grayscale);

        if (corners.size() >= 1) {
            // get the corner_set[i].x and corner_set[i].y
            // coordinate of the first corner.
            std::cout << "0" << " Corner x and y coordinate is " << corners[0].x << ", " << corners[0].y << std::endl;
        }
    }
}


void calibration_image_selection(cv::Mat &latest_image, cv::Size patternsize, bool save_vector_corners, std::vector<cv::Vec3f> point_set, std::vector<std::vector<cv::Vec3f>> point_list, std::vector<std::vector<cv::Point2f>> corner_list) {
/**
 * Function to select calibration image.
 */
    // creating 3D work cooridnates of each corners
    // therefore loop through the patternsize
    // I'll just measure  the world in units of target squares
    // therefore, I won't multiple j and i with the size of the square cause just assume each one is 1 unit
    for (int i = 0; i < patternsize.height; i++) {
        for (int j = 0; j < patternsize.width; j++) {
            // std::cout << "i, j, z coordinate is " << j << ", " << i << ", " << 0 << std::endl;
            point_set.push_back(cv::Vec3f(j, i, 0));
        }
    }

    // the latest image here is already grayscale, therefore not converting it to grayscale
    // all the images here are all images with pattern found because we capture it in task 1 calibration_image_selection
    // therefore, do not have to check for if found
    std::vector<cv::Point2f> corners;
    bool patternfound = findChessboardCorners(latest_image, patternsize, corners);
    cv::Size winSize(11,11);
    cv::Size zeroZone(-1,-1);
    cv::TermCriteria criterial(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1);

    cv::cornerSubPix(latest_image, corners, Size(11, 11), Size(-1, -1), criterial);
    std::cout << "Number of corners found: " << corners.size() << std::endl;
    
    cv::drawChessboardCorners(latest_image, patternsize, corners, patternfound);
    // if (corners.size() >= 1) {
    //     std::cout << "0" << " Corner x and y coordinate is " << corners[0].x << ", " << corners[0].y << std::endl;
    // }
    corner_list.push_back(corners);
    point_list.push_back(point_set);
}
