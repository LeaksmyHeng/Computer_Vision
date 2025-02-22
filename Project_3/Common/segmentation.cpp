/**
 * Leaksmy Heng
 * CS5330
 * Feb-13-2025
 * Implementing segmentation after applying thresholding and morphological operation on the video frame.
 * 
 * reference:
 * https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gae57b028a2b2ca327227c2399a9d53241
 * https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connectedcomponentswithstats-in-python
 * https://gist.github.com/JinpengLI/2cf814fe25222c645dd04e04be4de5a6
 * 
 *
 */

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <filesystem>
#include <fstream>

using namespace cv;
using namespace std;


struct ObjectFeature {
    /**
     * Struct to store the feature.
     */
    int regionId;
    string label;
    vector<double> featureVector;
};


std::vector<cv::Scalar> fixedColors = {
    cv::Scalar(255, 100, 100), // Light Red
    cv::Scalar(255, 0, 0),     // Blue
    cv::Scalar(255, 0, 255),   // Pink
    cv::Scalar(0, 255, 255),   // Yellow
    cv::Scalar(0, 255, 0)      // Green
};


cv::Scalar generateFixedRandomColor(const std::vector<cv::Scalar>& colors) {
    /**
     * picked random color out of the color index
     */
    int index = rand() % colors.size();
    return colors[index];
}


cv::Scalar generateRandomColor() {
    /**
     * Add random color generator for cv::scalar
     */
    static unsigned int seed = time(0); // Initialize once on first call
    seed += 1;                          // Increment the seed for each call
    return cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
}


cv::RotatedRect bounding_box(cv::Mat& region, double x, double y, double theta) {
    /**
     * This part of code is borrowed from https://github.com/RobuRishabh/Real_time_2D_Object_Recognition/blob/main/src/features.cpp
     */
    int X_max = INT_MIN, X_min = INT_MAX, Y_max = INT_MIN, Y_min = INT_MAX;
    int a = 0;
    while (a < region.rows) {
        int b = 0;
        while (b < region.cols) {
            if (region.at<uchar>(a, b) == 255) {
                int projectedX = (a - x) * cos(theta) + (b - y) * sin(theta);
                int projectedY = -(a - x) * sin(theta) + (b - y) * cos(theta);
                X_max = std::max(X_max, projectedX);
                X_min = std::min(X_min, projectedX);
                Y_max = std::max(Y_max, projectedY);
                Y_min = std::min(Y_min, projectedY);
            }
            b++;
        }
        a++;
    }

    int X_len = X_max - X_min;
    int Y_len = Y_max - Y_min;

    cv::Point centroid = cv::Point(x, y);
    cv::Size size = cv::Size(X_len, Y_len);

    return cv::RotatedRect(centroid, size, theta * 180.0 / CV_PI);
}


bool checkIfFileExists(const std::string& filePath) {
    // https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exists-using-standard-c-c11-14-17-c
    return std::filesystem::exists(filePath);
}


void save_features_to_csv(const vector<ObjectFeature> &featureList, const std::string &filename) {
    /**
     * Function to save information in the vector object to csv file.
     */
    // Open the file in append mode to save the features
    std::ofstream file(filename, std::ios::app);

    if (!file.is_open()) {
        std::cerr << "Failed to open file for saving features!" << std::endl;
        return;
    }

    // Write the feature vectors to the file
    for (const auto &objFeature : featureList) {
        file << objFeature.regionId << "," << objFeature.label;
        for (const auto &feature : objFeature.featureVector) {
            file << "," << feature;
        }
        file << "\n";
    }

    file.close();
    std::cout << "Features saved to CSV file." << std::endl;
}


void applying_feature_region(cv::Mat &src, cv::Mat &dst, cv::Mat &stats, cv::Mat &centroids, int regionId, string label, vector<ObjectFeature> &feature_list, bool save_file) {
    /**
     * Function to apply feature region to the specified region id.
     */
    
     // Get the stats and centroid for the given region
    int x = stats.at<int>(regionId, cv::CC_STAT_LEFT);
    int y = stats.at<int>(regionId, cv::CC_STAT_TOP);
    int w = stats.at<int>(regionId, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(regionId, cv::CC_STAT_HEIGHT);
    // Extract the region of interest (ROI)
    cv::Mat region = src(cv::Rect(x, y, w, h));
    // genting centroid of the region
    cv::Point2f centroid = cv::Point2f(centroids.at<double>(regionId, 0), centroids.at<double>(regionId, 1));
    // Calculate moments for the region (second-order moments)
    cv::Moments moments = cv::moments(region, true);
    // calculating axis of least central moment
    double angle = 0.5 * std::atan2(2 * moments.mu11, moments.mu02 - moments.mu20);
    
    // Calculate percent filled
    double area = moments.m00;  // Central moment m00 is the area of the region
    double boundingBoxArea = w * h;
    double percentFilled = (area / boundingBoxArea) * 100;

    // Bounding box height/width ratio
    double bboxRatio = static_cast<double>(h) / static_cast<double>(w);

    // getting the oob https://docs.opencv.org/3.4/de/d62/tutorial_bounding_rotated_ellipses.html
    cv::RotatedRect obb = bounding_box(region, centroid.x, centroid.y, angle);
    // cv::RotatedRect obb = cv::minAreaRect(region);
    cv::Point2f vertices[4];
    obb.points(vertices);

    // Draw the region and OBB on the image
    for (int i = 0; i < 4; i++) {
        // srand(static_cast<unsigned int>(time(0)));
        // cv::Scalar color = generateRandomColor();
        cv::Scalar color = generateFixedRandomColor(fixedColors);
        cv::line(dst, vertices[i], vertices[(i + 1) % 4], color, 5);
    }

    // add all feature to a feature
    std::vector<double> feature_vector = { percentFilled, bboxRatio, angle };

    if (save_file) {
        ObjectFeature objFeature = {regionId, label, feature_vector};
        feature_list.push_back(objFeature);
        std::cout << "Feature vector saved for region ID: " << regionId << std::endl;
    }

    // Display the calculated features
    std::cout << "Region " << regionId << " Features:" << std::endl;
    std::cout << "  Percent Filled: " << percentFilled << "%" << std::endl;
    std::cout << "  Bounding Box Ratio (Height/Width): " << bboxRatio << std::endl;
    std::cout << "  Orientation Angle (Axis of Least Central Moment): " << angle << " radians" << std::endl;
}


void applying_connectedComponents(cv::Mat &src, cv::Mat &dst, cv::Mat &stats, cv::Mat &centroids, bool is_save_to_file, string label, vector<ObjectFeature> &feature_list){
    /**
     * Applying connected components function.
     * 
     * :param src: source image or in our case video frame
     * :param dst: destination image or video frame
     * :param states: matrix used to store stats data
     * :param centroid: matrix to stored centroid data
     */
    // this CV_32S output 4 channel RGBA
    cv::Mat labels;
    int result = cv::connectedComponentsWithStats(src, labels, stats, centroids);
    cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
    
    
    // draw boundary on the dst image
    for (int i = 1; i < result; i++) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // ignore small region
        if (x > 50 && y > 50) {
            // Generate random color for each region
            // draw rectangle with white color
            // todo: i specified red but still get white, so play with the luminous cause your image has 4 channel
            // cv::Scalar color = generateRandomColor();
            cv::Scalar color = generateFixedRandomColor(fixedColors);
            cv::rectangle(dst, cv::Rect(x, y, w, h), color, 3);
            // // print out the stats
            // std::cout << "Region " << i << ":" << std::endl;
            // std::cout << "Area: " << stats.at<int>(i, cv::CC_STAT_AREA) << std::endl;
            // std::cout << "Bounding Box: (" 
            //           << x << ", "
            //           << y << ") -> (" 
            //           << w << " x " 
            //           << h << ")" << std::endl;
            // std::cout << "  Centroid: (" 
            //           << centroids.at<double>(i, 0) << ", "
            //           << centroids.at<double>(i, 1) << ")" << std::endl;
            
            // applying feature region function
            applying_feature_region(src, dst, stats, centroids, i, label, feature_list, is_save_to_file);
        }
    }
    
}
