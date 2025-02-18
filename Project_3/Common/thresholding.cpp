/**
 * Leaksmy Heng
 * CS5330
 * Feb-13-2025
 * Implementing thresholding to seperate foreground and background using k-mean cluster.
 * Default k to 2
 * 
 * reference:
 * https://medium.com/imagescv/7-smart-techniques-for-background-removal-using-python-bb8a60fdd504#:~:text=Thresholding,the%20foreground%20from%20the%20background.
 * 
 */


#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int kMeanImplementation(cv::Mat &src, cv::Mat &dst , int k=2, int max_iteration=10, double epsilon=1.0) {
    /***
     * Implement k-mean clustering for thresholding.
     * 
     * param src: source video frame
     * param dst: destination image
     * param k: number of cluster. Default to 2
     * param max_iteration: number of iteration of running k-mean cluster. default to 10
     * param epsilon: the threshold value determine when the algorithms should stop iterating. Default to 1.0
     */

    // https://docs.opencv.org/4.x/d9/dde/samples_2cpp_2kmeans_8cpp-example.html
    // https://softwareengineering.stackexchange.com/questions/314222/how-to-implement-k-means-algorithm-on-rgb-images

    // first convert image from RGB to CIE-Lab as according the softwareengineering.stackexchange, CIE-lab is better suited 
    // for k-mean cluster
    // https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
    dst = src.clone();
    cv::cvtColor(src, dst, cv::COLOR_BGR2Lab);

    // get height, width, and channels of the image
    int height = src.rows;
    int width = src.cols;
    int channels = src.channels();
    // 480 x 640 x 3
    cout << "Dimensions of the video frame: " << height << "height" << width << "width" << channels << "channels";

    // convert this to a 2D array
    // convert it to CV_32F because the k-mean cluster implemented in opencv uses that
    cv::Mat reshaped = dst.reshape(1, height*width);
    reshaped.convertTo(reshaped, CV_32F);

    // kmean
    // https://docs.opencv.org/3.4/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88
    // https://docs.opencv.org/3.4/d9/dde/samples_2cpp_2kmeans_8cpp-example.html
    cv::Mat labels, centers;
    // preparing the criterial that is used to stop the iteration in kmean
    cv::TermCriteria criterial(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, max_iteration, epsilon);
    cv::kmeans(reshaped, k, labels, criterial, 3, cv::KMEANS_PP_CENTERS, centers);

    // label is the output after running kmean
    // now convert label back to image
    labels = labels.reshape(1, src.rows);
    labels.convertTo(labels, CV_32S);

    // convert centers
    centers = centers.reshape(1, k);
    centers.convertTo(centers, CV_8U);

    // map pixel to the corresponding cluster center
    dst = cv::Mat(src.size(), src.type());
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            int clusterId = labels.at<int>(i * width + j);
            dst.at<cv::Vec3b>(i,j) = centers.row(clusterId);
        }
    }

    return 0;
}