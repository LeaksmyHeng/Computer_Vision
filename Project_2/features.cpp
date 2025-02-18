/*
  Leaksmy Heng
  CS5330
  Feb 07, 2025
  Project2
  Utility functions for storing image features.
*/

#include <math.h>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


bool checkingEmptyImage(const cv::Mat &image) {
	/**
	 * Function to check if image is empty.
	 * If it is empty, return true otherwise false
	 */
	if (image.empty()) {
        printf("Empty image\n");
        return true;
    }
	return false;
}

cv::Mat baselineMatching(const cv::Mat &image) {
    /**
     * Use the 7x7 square in the middle of the image as a feature vector.
     * 
     * param image: used cv::Mat to store the image and have it as const so we are not going to modify it
     * 
     * return: img of the 7x7
     */

	if (checkingEmptyImage(image)) {
		return cv::Mat();
	}
    
    // check if image size is smaller than 7.
    // return empty Mat as we are trying to get 7x7 metrics
    if (image.cols < 7 || image.rows < 7) {
		printf("Image too small for 7x7 region.\n");
		return cv::Mat();
    }

    // get the top left corner of the image with 7*7
    int x_coordinate = (image.cols / 2) - 3;
    int y_coordinate = (image.rows / 2) - 3;
    
    // Extracting a 7x7 patch from all three channels
    // using cv::rec(x, y, width, height)
    // which we found the coordinate x and y and width and height would be 7
    // because we are using 7x7 square img as feature
    cv::Rect rect(x_coordinate, y_coordinate, 7, 7);
    // cv::imshow("Extracted Image 7x7", image(rect).clone());
    // cv::waitKey(0);
    return image(rect).clone();
}

cv::Mat histogram(const cv::Mat &image, int numberOfBins, bool is1D) {
    /**
    * Convert image to histogram.
    */
    if (checkingEmptyImage(image)) {
		return cv::Mat();
	}

    // initalized a 3D histogram with a specify number of bins for each color
    int histogramSize[] = {numberOfBins, numberOfBins, numberOfBins};

    // since know number of bins, now we calculate the range of pixel in 
    // each of the bins, which in this case is 32 (256 is max so divide by 8 bins)
    // therefore, we know bins 1: 0-36, bin2: 36-68, ... for each R,G,B
    int range = 256 / numberOfBins;

    // we need to count the pixel in each bins
    // first create the mat (matrix) to store the histogram
    // initiaze all value to 0
    cv::Mat feature = Mat::zeros(3, histogramSize, CV_32F);

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b imageColor = image.at<cv::Vec3b>(y, x);
            int blue = imageColor[0] / range;
            int green = imageColor[1] / range;
            int red = imageColor[2] / range;

            // increment the count of how many pixels fall into the bin corresponding to (b, g, r)
            feature.at<float>(blue, green, red)++;
        }
    }

    // Normalize the histogram using L1 normalization so that the sum of squared values equals 1
    // achieving this by dividing it with total pixel
    normalize(feature, feature, 1, 0, cv::NORM_L1, -1, cv::Mat());

    // if 1D is true, convert this to 1D
    if (is1D) {
        return cv::Mat(feature).reshape(1, 1);
    }

    // Return the histogram
    return feature;
  }

cv::Mat multiHistogram(const cv::Mat &image, int numberOfBins = 8) {
	/**
	 * Function to create multi histogram.
	 */

    if (checkingEmptyImage(image)) {
		return cv::Mat();
	}

    // Get the dimensions of the image
    int height = image.rows;
    int width = image.cols;

    // Get the top half and bottom half of the region
	// rect is x, y, width and height.
    cv::Rect topRegion(0, 0, width, height / 2);  				// Top
    cv::Rect bottomRegion(0, height / 2, width, height / 2); 	// Bottom

    // Extract the top and bottom halves of the image
    cv::Mat topHalf = image(topRegion);
    cv::Mat bottomHalf = image(bottomRegion);

    // Compute the histograms for the top and bottom halves
	// using the RGB histogram from above function
    cv::Mat histTop = histogram(topHalf, numberOfBins, false);
    cv::Mat histBottom = histogram(bottomHalf, numberOfBins, false);

    // Check if histograms are empty
    if (histTop.empty() || histBottom.empty()) {
        printf("Either top histogram or bottom histogram is empty.\n");
        return cv::Mat();
    }

    // Concatenate the histograms (top and bottom halves) horizontally
	// printf("trying to concatenated hist.\n");
    cv::Mat concatenatedHist;
    cv::hconcat(histTop, histBottom, concatenatedHist);

    // Normalize the concatenated histogram (to range [0, 1])
	// using L1 because i want the pixel to be divided by overall pixel
    cv::normalize(concatenatedHist, concatenatedHist, 0, 1, cv::NORM_L1, -1, cv::Mat());

    // Return the concatenated histogram as a cv::Mat
    return concatenatedHist;
}

cv::Mat texture(const cv::Mat& image) {
	/**
	 * Compute texture histogram as feature vector.
	 * In this case, we use Sobel magnitude image and use a histogram of gradient magnitudes as the texture feature. 
	 */

	// to get to sobel, we need to convert image to grayscale
	cv::Mat grayScaleImage;
	cv::cvtColor(image, grayScaleImage, COLOR_BGR2GRAY);

	// in the first project, i created sobel filter not on full image, but start at 1 and end with row-1 or col-1
	// therefore, I will just call opencv sobel filter here
	// https://stackoverflow.com/questions/8377091/what-are-the-differences-between-cv-8u-and-cv-32f-and-what-should-i-worry-about
	cv::Mat sobelX, sobelY;
	cv::Sobel(grayScaleImage, sobelX, CV_32F, 1, 0, 3);
	cv::Sobel(grayScaleImage, sobelY, CV_32F, 0, 1, 3);

	// compute gradient magnitude now that we have sobelX and sobelY
	cv::Mat gradientMagnitude;
	cv::magnitude(sobelX, sobelY, gradientMagnitude);

	// normalized the gradient magnitude to ensure it is between [0,255]
	// norm_minmax = https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#normalize
	cv::normalize(gradientMagnitude, gradientMagnitude, 0, 255, cv::NORM_MINMAX);

	// printf("Showing images");
	// imshow("Original Image", image);
    // imshow("Gradient Magnitude", gradientMagnitude);
	// waitKey(0);

	// turn gradientMagnitude to a histogram
    // this is a 1D histogram
	int histSize = 8;
	float rangeStart = 0.0;
	float rangeEnd = 400.0;

	// initialized histogram to all 0 (1D cause of grayscale)
	cv::Mat textureHist = cv::Mat::zeros(1, histSize, CV_32F);
	// calculate binsize which is 50
	float binSize = (rangeEnd - rangeStart) / histSize;

    // loop through each row and cols of the gradient magnitude
	for (int i=0; i<gradientMagnitude.rows; i++) {
		for (int j=0; j<gradientMagnitude.cols; j++) {
			// get pixel at possition i & j
			float gradientValue = gradientMagnitude.at<float>(i, j);
			// check if gradient is within range
			if (gradientValue >= rangeStart && gradientValue < rangeEnd) {
                // Calculate the corresponding bin index
                int binIndex = static_cast<int>((gradientValue - rangeStart) / binSize);
                
                // Increment the histogram bin
                if (binIndex >= 0 && binIndex < histSize) {
                    textureHist.at<float>(0, binIndex)++;
                }
            }
		}
	}

	// normalized histogram using L1
	cv::normalize(textureHist, textureHist, 1, 0, cv::NORM_L1);
	return textureHist;
}

vector<float> extractColorFeatures(const string& image) {
    /**
     * Extracting both RBG and hue and then average it and store it in a vector.
     * 
     * :param image: image path you want to extract its color
     */
    cv::Mat img = cv::imread(image);
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::Scalar meanRGB = cv::mean(img);
    cv::Scalar meanHSV = cv::mean(hsv);
    return { static_cast<float>(meanRGB[2]), static_cast<float>(meanRGB[1]), static_cast<float>(meanRGB[0]), static_cast<float>(meanHSV[0]) };
}

cv::Mat reshapeTo2D(cv::Mat& mat) {
    if (mat.dims == 3) {
        // Reshape 3D mat into 2D (1 row, total cols = channels * bins)
        int newCols = mat.size[1] * mat.size[2]; // Multiply size[1] and size[2] to flatten channels
        mat = mat.reshape(1, 1);  // Reshape to 1 row, multiple columns
    }
    return mat;
}

std::vector<float> matToVector(cv::Mat& mat) {
    /**
     * Convert cv::mat to vector 1D vector so we could concatenate it with the
     * colorTexture histogram which was in  1D
     */
    // Check if matrix is 3D and reshape it to 2D
    if (mat.dims == 3) {
        mat = reshapeTo2D(mat);
    }

    // Mat should be 2D, flatten it to 1D vector
    std::vector<float> vec;
    mat.reshape(1, 1).copyTo(vec);
    return vec;
}

cv::Mat colorTexture(const cv::Mat& image, int numberOfBins = 8) {
    /***
     * Function to merge histogram and texture to ColorText.
     * ColorHistogram is in 3D; therefore, need to convert it to 1D
     */
    // Compute histograms
    cv::Mat colorHistogram = histogram(image, numberOfBins, false);
    cv::Mat textureHistogram = texture(image);

    // Ensure matrices are CV_32F for consistency
    colorHistogram.convertTo(colorHistogram, CV_32F);
    textureHistogram.convertTo(textureHistogram, CV_32F);

    // Convert both histograms to vectors if they are 2D (or reshaped to 2D)
    std::vector<float> colorVec;
    std::vector<float> textureVec;

    colorVec = matToVector(colorHistogram);
    textureVec = matToVector(textureHistogram);

    // Concatenate vectors
    std::vector<float> combinedVec;
    combinedVec.reserve(colorVec.size() + textureVec.size());
    combinedVec.insert(combinedVec.end(), colorVec.begin(), colorVec.end());
    combinedVec.insert(combinedVec.end(), textureVec.begin(), textureVec.end());

    // Convert back to cv::Mat (1 row, N columns)
    return cv::Mat(combinedVec).reshape(1, 1);
}
