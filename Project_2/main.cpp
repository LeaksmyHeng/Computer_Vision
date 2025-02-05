#include <iostream>
#include <filesystem>
#include <map>
#include <opencv2/opencv.hpp>

#include "include/features.h"
#include "include/distance_metric.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main(int argc, char *argv[]) {

    // // there are 5 arguments
    // if (argc < 5) {
    //     printf("Error: At least 5 arguments are required.");
    //     return -1;
    // }

    std::string targetImage = argv[1];
    fs::path imageDatabase = argv[2];
    std::string featureComputingMethod = argv[3];
    std::string distanceMetric = argv[4];

    // std::int outPutImage[256] = argv[5];

    // checking if target image exist (for argv[1])
    cv::Mat target_image = cv::imread(targetImage);
    if (target_image.empty()) {
        printf("Image file path: %s\n", argv[1]);
        return 1;
    }
    cv::Mat targetImageFeature;
    if (featureComputingMethod == "baseline") {
        targetImageFeature = baselineMatching(target_image);
    }

    // checking if imageDatabase exist (for argv[2])
    if (!fs::exists(imageDatabase) || !fs::is_directory(imageDatabase)) {
        printf("Cannot open directory image database directory");
        return -1;
    }

    // initialized the dictonary result that store file name and the distance metric result
    // this will be sorted and display the image with the closest match
    // https://www.geeksforgeeks.org/how-to-create-a-dictionary-in-cpp/
    map<std::string, double> resultDict;
    
    // This part of the code is from readfiles-1.cpp, but i converted it to window friendly
    // loop over all the files in the image file listing
    for (const auto& entry : fs::directory_iterator(imageDatabase)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().string();
            if (filename.ends_with(".jpg") || filename.ends_with(".png") ||
                filename.ends_with(".ppm") || filename.ends_with(".tif")) {

                // Read the current image in the database
                // if can't be read, just continue
                cv::Mat image = cv::imread(filename);
                if (image.empty()) {
                    continue;
                }

                // Compute features for the current image in the database
                cv::Mat computingFeaturesImage;
                if (featureComputingMethod == "baseline") {
                    computingFeaturesImage = baselineMatching(image);
                }

                // Compute the SSD distance with the target image features
                double result = 0.0;
                if (distanceMetric == "SSD") {
                    result = sumOfSquaredDifference(computingFeaturesImage, targetImageFeature);
                    resultDict[filename.c_str()] = result;
                }
            }
        }
    }

    
    for (auto it : resultDict)
        cout << it.first << ": " << it.second << endl;

    printf("Done looping through file");


    return 0;
}
