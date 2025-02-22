/**
 * Leaksmy Heng
 * CS5330
 * Feb-13-2025
 * This file contains image classification.
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

using namespace cv;
using namespace std;


struct ObjectFeature {
    int regionId;
    std::string label;
    std::vector<double> featureVector;
};

// Structure to hold feature statistics
struct FeatureStats {
    std::vector<double> mean;
    std::vector<double> stdev;
};

void print_object_feature(const ObjectFeature& obj) {
    /**
     * Print out object feature to make sure we read from csv correctly.
     */
    // Print regionId and label
    std::cout << "Region ID: " << obj.regionId << std::endl;
    std::cout << "Label: " << obj.label << std::endl;

    // Print feature vector
    std::cout << "Feature Vector: ";
    for (double feature : obj.featureVector) {
        std::cout << feature << " ";
    }
    std::cout << std::endl;
}


std::vector<ObjectFeature> load_feature_from_csv(const std::string& filename) {
    /**
     * Function to load feature from csv file.
     */
    std::vector<ObjectFeature> featureList;
    std::ifstream file(filename);
    std::string line;
  
    // Skip header row
    std::getline(file, line);
  
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        ObjectFeature feature;
        std::string token;
    
        // Read RegionID
        std::getline(ss, token, ',');
        feature.regionId = std::stoi(token);
    
        // Read Label
        std::getline(ss, feature.label, ',');
    
        // Important: Clear the vector before adding new features
        feature.featureVector.clear();
    
        while (std::getline(ss, token, ',')) {
            try {
            feature.featureVector.push_back(std::stod(token));
            } catch (const std::invalid_argument& e) {
            // Handle the case where a token cannot be converted to a double
            std::cerr << "Warning: Invalid feature value encountered: " << token << ". Skipping." << std::endl;
            // You might want to set a default value or take other appropriate action
            }
        }
    
        featureList.push_back(feature);
    }

    // for (const ObjectFeature& obj : featureList) {
    //     print_object_feature(obj);
    // }
    return featureList;
}


vector<double> extractFeaturesFromFrame(const Mat& frame) {
    vector<double> features;
    
    // Convert to grayscale
    Mat grayFrame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
    
    // Threshold the image
    Mat binaryFrame;
    threshold(grayFrame, binaryFrame, 128, 255, THRESH_BINARY);
    
    // Find contours
    vector<vector<Point>> contours;
    findContours(binaryFrame, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        std::cout << "Feature not found";
        return features;
    }
    
    // Find the largest contour (assuming it's the object of interest)
    int largestContourIdx = 0;
    double largestArea = 0;
    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > largestArea) {
            largestArea = area;
            largestContourIdx = i;
        }
    }
    
    // 1. Percent Filled
    Rect boundingRect = cv::boundingRect(contours[largestContourIdx]);
    double totalArea = boundingRect.width * boundingRect.height;
    double percentFilled = largestArea / totalArea;
    features.push_back(percentFilled);
    
    // 2. Bounding Box Ratio
    double bboxRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
    features.push_back(bboxRatio);
    
    // 3. Axis of Least Central Moment
    Moments moments = cv::moments(contours[largestContourIdx]);
    double mu20 = moments.mu20 / moments.m00;
    double mu02 = moments.mu02 / moments.m00;
    double mu11 = moments.mu11 / moments.m00;
    double common = sqrt(4 * mu11 * mu11 + (mu20 - mu02) * (mu20 - mu02));
    double axisOfLeastCentralMoment = atan2(2 * mu11, mu20 - mu02 + common);
    features.push_back(axisOfLeastCentralMoment);
    
    return features;
}


double compute_scaled_distance(const vector<double> &feature1, const vector<double> &feature2, const vector<double> &stdev) {
    /**
     * Function to compute euclidian distance between two feature.
     */
    double distance = 0.0;
    for (size_t i = 0; i < feature1.size(); ++i) {
        double diff = (feature1[i] - feature2[i]) / stdev[i];
        distance += diff * diff;
    }
    
    return sqrt(distance);
}

string classifyObjectWithUnknownDetection(const vector<double>& newFeatureVector, const vector<ObjectFeature>& featureList, const vector<double>& stdevs, double threshold) {
    double minDistance = numeric_limits<double>::max();
    string label = "Unknown";
    for (const ObjectFeature& obj : featureList) {
        print_object_feature(obj);
    }

    for (const auto& knownObject : featureList) {
        double distance = compute_scaled_distance(newFeatureVector, knownObject.featureVector, stdevs);
        std::cout << "distance: " << distance << "minDsitance: " << minDistance << std::endl;
        if (distance < minDistance) {
            minDistance = distance;
            label = knownObject.label;
        }
    }

    // If the minimum distance is greater than the threshold, classify as "Unknown"
    if (minDistance > threshold) {
        label = "Unknown";
    }

    return label;
}


vector<double> calculateStandardDeviations(const vector<ObjectFeature>& featureList) {
    size_t featureCount = featureList[0].featureVector.size();
    vector<double> means(featureCount, 0.0);
    vector<double> variances(featureCount, 0.0);
    vector<double> stdevs(featureCount, 0.0);
    
    // Calculate means
    for (const auto& object : featureList) {
        for (size_t i = 0; i < featureCount; ++i) {
            means[i] += object.featureVector[i];
        }
    }
    for (auto& mean : means) {
        mean /= featureList.size();
    }
    
    // Calculate variances
    for (const auto& object : featureList) {
        for (size_t i = 0; i < featureCount; ++i) {
            double diff = object.featureVector[i] - means[i];
            variances[i] += diff * diff;
        }
    }
    
    // Calculate standard deviations
    for (size_t i = 0; i < featureCount; ++i) {
        stdevs[i] = sqrt(variances[i] / (featureList.size() - 1));
    }
    
    return stdevs;
}

double setInitialThreshold(const vector<ObjectFeature>& featureList, const vector<double>& stdevs) {
    // Use 3 standard deviations as an initial threshold
    double threshold = 0.0;
    for (const auto& stddev : stdevs) {
        threshold += 3 * stddev;
    }
    threshold /= stdevs.size();
    return threshold;
}


void processVideoStream(VideoCapture& cap, const vector<ObjectFeature>& featureList, const vector<double>& stdevs, double threshold) {
    Mat frame;
    while (cap.read(frame)) {
        // Extract features from the current frame
        vector<double> currentFeatures = extractFeaturesFromFrame(frame);
        
        if (!currentFeatures.empty()) {
            // Classify the object
            string label = classifyObjectWithUnknownDetection(currentFeatures, featureList, stdevs, threshold);
            
            // Display the label on the frame
            putText(frame, label, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        }
        
        imshow("Object Recognition", frame);
        // if (waitKey(1) == 27) break; // Exit on ESC key
    }
}
