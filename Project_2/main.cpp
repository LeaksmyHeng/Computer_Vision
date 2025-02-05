#include <opencv2/opencv.hpp>

#include "include/features.h"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {

    // todo: change this to dynamic path
    cv::Mat target = cv::imread("C:/Users/Leaksmy Heng/Documents/GitHub/CS5330/Computer_Vision/Project_2/olympus/test/pic.0009.jpg");
    if (target.empty()) {
        printf("No image found: %s\n", argv[1]);
        return 1;
    }

    printf("Found target image: %s\n", argv[1]);

    // display original image
    cv::imshow("Target Image", target);
    cv::waitKey(0);

    // display after going through baseline feature extraction
    cv::Mat baselineImage = baselineMatching(target);
    cv::imshow("Extracted baseline image", baselineImage);
    cv::waitKey(0);  

    return 0;
}
