/* 
Leaksmy Heng
CS5330
Feb 07, 2025
Project2
This is the pipeline of the program
I implemented step 1 where i call the feature function and distance method function
all in one for loop (for loop that is used to go through the images in the database file)
*/

#include <iostream>
#include <filesystem>
#include <map>
#include <opencv2/opencv.hpp>

#include "include/features.h"
#include "include/distance_metric.h"
#include "include/csv_util.h"

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

string getFileName(const char* fileName) {
    /**
     * Function to extract file name. Currenctly, my filename is like a/b/c/photo.jpg
     * which if we used the entirely path, the img won't be found in the excel
     * so just extract the last part out to get the photos name
     */
    return std::filesystem::path(fileName).filename().string();
}

int main(int argc, char *argv[]) {
    /**
     * Main function. I use step one where I ran everything in here.
     */

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

    /*******************This is for task 5 *****************/
    char csvFilename[] = "C:/Users/Leaksmy Heng/Documents/GitHub/CS5330/Computer_Vision/Project_2/ResNet18_olym.csv";
    vector<char *> filenames;
    vector<vector<float>> data;
    if (read_image_data_csv(csvFilename, filenames, data, 0) != 0) {
        printf("Error while trying to read image data csv.\n");
        return -1;
    }
    // convert the filenames into the dictionary so we could easily look up when calculating distance between image
    unordered_map<string, vector<float>> dataMap;
    for (size_t i = 0; i < filenames.size(); i++) {
        dataMap[string(filenames[i])] = data[i];  // Store as <filename, feature_vector>
    }

    /****************** Feature *******************/
    cv::Mat targetImageFeature;
    cv::Mat targetColorFeature;
    cv::Mat targetTextureFeature;
    std::string targetImageName;
    vector<float> targetColorExtraction;

    if (featureComputingMethod == "baseline") {
        targetImageFeature = baselineMatching(target_image);
    }
    else if (featureComputingMethod == "histogram") {
        targetImageFeature = histogram(target_image);
    }
    else if (featureComputingMethod == "multiHistogram") {
        targetImageFeature = histogram(target_image);
    }
    else if (featureComputingMethod == "colorTexture") {
        targetColorFeature = histogram(target_image, 8, true);
        targetTextureFeature = texture(target_image);
    }
    else if (featureComputingMethod == "DNN") {
        targetImageName = getFileName(targetImage.c_str());
    }
    else if (featureComputingMethod == "CBIR") {
        targetImageName = getFileName(targetImage.c_str());
        targetColorExtraction = extractColorFeatures(targetImage);
    }

    // checking if imageDatabase exist (for argv[2])
    if (!fs::exists(imageDatabase) || !fs::is_directory(imageDatabase)) {
        printf("Cannot open directory image database directory\n");
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

                /****************** Features *******************/
                // Compute features for the current image in the database
                cv::Mat computingFeaturesImage;
                cv::Mat colorFeatureImage;
                cv::Mat textureFeatureImage;
                if (featureComputingMethod == "baseline") {
                    computingFeaturesImage = baselineMatching(image);
                }
                else if (featureComputingMethod == "histogram") {
                    computingFeaturesImage = histogram(image);
                }
                else if (featureComputingMethod == "multiHistogram") {
                    computingFeaturesImage = histogram(image);
                }
                else if (featureComputingMethod == "colorTexture") {
                    // computingFeaturesImage = colorTexture(image, 8);
                    colorFeatureImage = histogram(image, 8, true);
                    textureFeatureImage = texture(image);
                }
                // else if (featureComputingMethod == "DNN") {
                //     printf("Deep network embedding feature.\n");
                // }

                /****************** Distance Metric *******************/
                // Compute the SSD distance with the target image features
                double result = 0.0;
                if (distanceMetric == "SSD") {
                    result = sumOfSquaredDifference(computingFeaturesImage, targetImageFeature);
                }
                if (distanceMetric == "histogramIntersection") {
                    result = histogramIntersection(targetImageFeature, computingFeaturesImage);
                }
                if (distanceMetric == "weightedDistance") {
                    result = weightedDistance(colorFeatureImage, targetImageFeature, targetColorFeature, targetTextureFeature);
                }
                if (distanceMetric == "SSD_V") {
                    std::string imageName = getFileName(filename.c_str());
                    // std::cout << targetImageName << " and feature image: " << imageName << "\n" << std::endl;
                    auto targetImageFound = dataMap.find(targetImageName);
                    auto featureImageFound = dataMap.find(imageName);
                    if (targetImageFound == dataMap.end() || featureImageFound == dataMap.end()) {
                        printf("Error while trying to find targetimage or featureimage\n");
                        return -1;
                    }
                    result = sumOfSquaredDifferenceVector(targetImageFound->second, featureImageFound->second);
                }
                if (distanceMetric == "CBIR") {
                    vector<float> imageColorFeature = extractColorFeatures(filename);
                    std::string imageName = getFileName(filename.c_str());
                    // calculating distance using cosineDistance and SSD
                    auto targetImageFound = dataMap.find(targetImageName);
                    auto featureImageFound = dataMap.find(imageName);
                    if (targetImageFound == dataMap.end() || featureImageFound == dataMap.end()) {
                        printf("Error while trying to find targetimage or featureimage\n");
                        return -1;
                    }
                    double cosine = cosineDistance(targetImageFound->second, featureImageFound->second);
                    double ssd = sumOfSquaredDifferenceVector(targetColorExtraction, imageColorFeature);
                    // result = 0.5 * cosine + 0.5 * ssd;
                    // result = 0.75 * cosine + 0.25 * ssd;
                    result = 0.25 * cosine + 0.75 * ssd;
                }

                // if result is 0 that means we are computing the same image. so skip
                if (result == 0.0) {
                    std::cout << "Found target image in database: " << filename << std::endl;
                    continue;
                }
                // write the filename and its distance to the map
                resultDict[filename.c_str()] = result;
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
        cv::Mat img = imread(imagePath);

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
