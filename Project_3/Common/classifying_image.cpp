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

    for (const ObjectFeature& obj : featureList) {
        print_object_feature(obj);
    }
    return featureList;
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