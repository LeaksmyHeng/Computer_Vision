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
vector<vector<double>> extractFeaturesFromFrame(const Mat& frame);
vector<double> calculateStandardDeviations(const vector<ObjectFeature>& featureList);
double setInitialThreshold(const vector<ObjectFeature>& featureList, const vector<double>& stdevs);
string classifyObjectWithUnknownDetection(const vector<double>& newFeatureVector, const vector<ObjectFeature>& featureList, const vector<double>& stdevs, double threshold);
string classifyObjectUsingKNN(const vector<double>& newFeatureVector, const vector<ObjectFeature>& featureList, const vector<double>& stdevs);

#endif // CLASSIFICATION_H