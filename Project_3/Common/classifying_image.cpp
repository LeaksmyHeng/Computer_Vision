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
        feature.featureVector.clear();
    
        while (std::getline(ss, token, ',')) {
            try {
            feature.featureVector.push_back(std::stod(token));
            } catch (const std::invalid_argument& e) {
            // Handle the case where a token cannot be converted to a double
            std::cerr << "Invalid feature value encountered: " << token << ". Skipping." << std::endl;
            }
        }
    
        featureList.push_back(feature);
    }

    // for (const ObjectFeature& obj : featureList) {
    //     print_object_feature(obj);
    // }
    return featureList;
}


vector<vector<double>> extractFeaturesFromFrame(const Mat& frame) {
    /**
     * Function to extract the feature from frame. This is used in inferencing.
     */
    
    vector<vector<double>> allFeatures;
    
    cv::Mat grayFrame;
    cv::Mat binaryFrame;

    GaussianBlur(frame, grayFrame, Size(5, 5), 0);
    cv::cvtColor(grayFrame, grayFrame, cv::COLOR_BGR2GRAY);
    // cv::threshold(frame, grayFrame, 125, 255, cv::THRESH_BINARY);
    cv::threshold(grayFrame, binaryFrame, 127, 255, cv::THRESH_BINARY_INV);
    
    cv::Mat labels, stats, centroids;
    int result = cv::connectedComponentsWithStats(binaryFrame, labels, stats, centroids);
    cv::cvtColor(binaryFrame, binaryFrame, cv::COLOR_GRAY2BGR);

    for (int i = 1; i < result; i++) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // ignore small region
        if (x > 50 && y > 50) {
            cv::Mat region = grayFrame(cv::Rect(x, y, w, h));
            cv::Point2f centroid = cv::Point2f(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
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
            // features.push_back(percentFilled);
            // features.push_back(bboxRatio);
            // features.push_back(angle);
            std::cout << "Feature vector: " << percentFilled << ", " << bboxRatio << ", "<< angle << std::endl;
            vector<double> feature_vector = { percentFilled, bboxRatio, angle };
            allFeatures.push_back(feature_vector);

            cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 255, 0), 2);
        }
    }
    return allFeatures;
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


double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
    /**
     * Regular euclidean distance.
     */
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}


vector<double> calculateStandardDeviations(const vector<ObjectFeature>& featureList) {
    /**
     * Calculate the standard deviation of all the training feature vectors.
     */
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
    /**
     * Find the appropriate threshold of the features using 3 sigma rules by multiplying each stdev by 3,
     * sum it all up then divide it with the total number of stdevs
     */
    // Use 3 standard deviations as an initial threshold
    double threshold = 0.0;
    for (const auto& stddev : stdevs) {
        threshold += 3 * stddev;
    }
    threshold /= stdevs.size();
    return threshold;
}


string classifyObjectWithUnknownDetection(const vector<double>& newFeatureVector, const vector<ObjectFeature>& featureList, const vector<double>& stdevs, double threshold) {
    /**
     * Function to classifying the object in the frame.
     */
    double minDistance = numeric_limits<double>::max();
    string label = "Unknown";
    // // to verify that we do have feature list
    // for (const ObjectFeature& obj : featureList) {
    //     print_object_feature(obj);
    // }

    for (const auto& knownObject : featureList) {
        double distance = compute_scaled_distance(newFeatureVector, knownObject.featureVector, stdevs);
        // std::cout << "distance: " << distance << "minDsitance: " << minDistance << std::endl;
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


string classifyObjectUsingKNN(const vector<double>& newFeatureVector, const vector<ObjectFeature>& featureList, const vector<double>& stdevs) {
    /**
     * Function to use knn to classify the object.
     */
    string label = "Unknown";
    int k = 3;  // Number of nearest neighbors to consider

    // calculate distances to all objects in the training set
    vector<pair<double, ObjectFeature>> distances;
    for (const auto& knownObject : featureList) {
        double dist = compute_scaled_distance(newFeatureVector, knownObject.featureVector, stdevs);
        distances.push_back(make_pair(dist, knownObject));  // Store distance along with ObjectFeature
    }

    // sort distances in ascending order (smallest distance comes first)
    sort(distances.begin(), distances.end(), [](const pair<double, ObjectFeature>& a, const pair<double, ObjectFeature>& b) {
        return a.first < b.first;
    });

    // compute the average feature vector of the K nearest neighbors
    vector<double> avgFeatureVector(newFeatureVector.size(), 0.0);
    for (int i = 0; i < k; ++i) {
        const auto& neighbor = distances[i].second;
        for (size_t j = 0; j < neighbor.featureVector.size(); ++j) {
            avgFeatureVector[j] += neighbor.featureVector[j];  // Accumulate feature values
        }
    }

    // average the features
    for (size_t i = 0; i < avgFeatureVector.size(); ++i) {
        avgFeatureVector[i] /= k;  // Divide by K to get the average
    }

    // find the label of the nearest neighbor whose feature vector that is closest to the average feature vector
    double minDistance = DBL_MAX;
    for (const auto& knownObject : featureList) {
        double dist = compute_scaled_distance(avgFeatureVector, knownObject.featureVector, stdevs);
        if (dist < minDistance) {
            minDistance = dist;
            label = knownObject.label;
        }
    }

    return label;
}
