#include <iostream>
#include <filesystem>
#include <map>
#include <opencv2/opencv.hpp>

#include "include/features.h"
#include "include/distance_metric.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

bool compareFileNameAndItsDistanceValue(const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
    /***
     * Comparing the distance value by their key value pair.
     * 
     * :param a: the first map used for compmaring to b
     * :param b: the second map used for comparing to a
     */
    return a.second < b.second;
}

int main(int argc, char *argv[]) {

    // there are 5 arguments
    // if (argc < 5) {
    //     printf("Error: At least 5 arguments are required. %d were found\n", argc);
    //     return -1;
    // }

    std::string targetImage = argv[1];
    fs::path imageDatabase = argv[2];
    std::string featureComputingMethod = argv[3];
    std::string distanceMetric = argv[4];
    int outPutImage = std::stoi(argv[5]);

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
    else if (featureComputingMethod == "histogram") {
        targetImageFeature = histogram(target_image);
    }

    // checking if imageDatabase exist (for argv[2])
    if (!fs::exists(imageDatabase) || !fs::is_directory(imageDatabase)) {
        printf("Cannot open directory image database directory");
        return -1;
    }

    // initialized the dictonary result that store file name and the distance metric result
    // use map as I want to store it as key value pair
    // this will be sorted and display the image with the closest match
    // https://www.geeksforgeeks.org/how-to-create-a-dictionary-in-cpp/
    map<std::string, double> resultDict;
    
    // This part of the code is from readfiles-1.cpp, but i converted it to window friendly
    // loop over all the files in the image file listing
    for (const auto& entry : fs::directory_iterator(imageDatabase)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().string();

            // check if filename is the same as target then continue
            if (filename == targetImage) {
                std::cout << "Found target image in database: " << filename << std::endl;
                continue;
            }

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
                else if (featureComputingMethod == "histogram") {
                    computingFeaturesImage = histogram(target_image);
                }

                // Compute the SSD distance with the target image features
                double result = 0.0;
                if (distanceMetric == "SSD") {
                    result = sumOfSquaredDifference(computingFeaturesImage, targetImageFeature);
                    // if result is 0 that means we are computing the same image. so skip
                    if (result == 0.0) {
                        std::cout << "Found target image in database: " << filename << std::endl;
                        continue;
                    }
                    resultDict[filename.c_str()] = result;
                }
            }
        }
    }

    
    // sorted the resultDict in ascending order of its value (distance metric result)
    // by: convert map to a vector pair
    std::vector<std::pair<std::string, double>> resultVector(resultDict.begin(), resultDict.end());
    std::sort(resultVector.begin(), resultVector.end(), compareFileNameAndItsDistanceValue);

    std::cout << "Top "<< outPutImage <<" images sorted by "<< distanceMetric <<" values (ascending):" << std::endl;
    for (int i = 0; i < outPutImage && i < resultVector.size(); i++) {
        // .first is key and .second is value
        string imagePath = resultVector[i].first;
        Mat img = imread(imagePath);

        if (!img.empty()) {
            cout << "Displaying image: " << imagePath << " with SSD: " << resultVector[i].second << endl;
            // Show the image using OpenCV
            imshow("Top " + to_string(i + 1) + " Image", img);
            waitKey(0);
        }
    }

    printf("Done looping through file");


    return 0;
}
