/**
 * Leaksmy Heng
 * CS5330
 * Feb 12, 2017
 * Header file for image classification
 */

#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H
 
#include <opencv2/opencv.hpp>
 
using namespace cv;


struct FeatureStats {
    std::vector<double> mean;
    std::vector<double> stdev;
};
std::vector<ObjectFeature> load_feature_from_csv(const std::string& filename);

#endif // CLASSIFICATION_H