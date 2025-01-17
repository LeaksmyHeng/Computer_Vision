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

int main(int argc, char** argv ) {
    /**
    * Function to read image and display it.
    **/

    std::string image_path = samples::findFile("C:/Users/Leaksmy Heng/Documents/GitHub/CS5330/Module_1/photos/arch.jpeg");
    Mat image = imread(image_path, IMREAD_COLOR); // Read an image from file
    
    if (image.empty()) {
        std::cout << "No image found: " << image_path << std::endl;
        return 1;
    }

    imshow("Display window", image);    // Display an image in an OpenCV window
    
    while (true) {
        int k = waitKey(0);
        std::cout << "Key pressed: " << static_cast<char>(k) << std::endl;
        // 'q', the program should quit
        if (k == 'q') {
            printf("quite the program");
            // Close the window
            cv::destroyAllWindows(); 
            return 0;
        }
    }
}
