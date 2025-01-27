/**
 * Leaksmy Heng
 * Jan 16 2025
 * This is a helper file that is used to put all of your image manipulation functions.
 */

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "filter.h"

using namespace cv;
using namespace std;

void greyScale( cv::Mat &src, cv::Mat &dst ) {
    /**
     * This function is used to convert color to greyscale
     */
    // convert color to greyscale through (COLOR_BGR2GRAY)
    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
}

int AlternativeGrayscale(cv::Mat &src, cv::Mat &dst) {
    /**
     * Since opencv already used luminosity method, therefore,
     * I'll use the Average Method which is (R+G+B) / 3
     * Source: https://www.baeldung.com/cs/convert-rgb-to-grayscale
     */
    dst = src.clone();  // Make a copy of the source image for the destination

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(y, x);

            // Average Method to convert to greyscale
            uchar grayscale = static_cast<uchar>((pixel[2] + pixel[1] +pixel[0]) / 3);

            // Set all channels (R, G, B) to the same grayscale value
            pixel[0] = grayscale; // Blue
            pixel[1] = grayscale; // Green
            pixel[2] = grayscale; // Red
        }
    }
    return 0;
}

int SepiaFilter( cv::Mat &src, cv::Mat &dst ) {
    /**
     * Implement a Sepia tone filter
     * Formular: 
     * 0.272, 0.534, 0.131    // Blue coefficients for R, G, B  (pay attention to the channel order)
     * 0.349, 0.686, 0.168    // Green coefficients
     * 0.393, 0.769, 0.189    // Red coefficients
     */
    dst = src.clone();

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(y, x);

            // store source red, green, blue pixel
            uchar blue = pixel[0];
            uchar green = pixel[1];
            uchar red = pixel[2];

            // store new red, green, and blue pixel
            int newBlue = static_cast<int>(red * 0.272 + green * 0.534 + blue * 0.131);
            int newGreen = static_cast<int>(red * 0.349 + green * 0.686 + blue * 0.168);
            int newRed = static_cast<int>(red * 0.393 + green * 0.769 + blue * 0.189);
            
            pixel[0] = cv::saturate_cast<uchar>(newBlue);
            pixel[1] = cv::saturate_cast<uchar>(newGreen);
            pixel[2] = cv::saturate_cast<uchar>(newRed);
        }
    }
    return 0;
}

int vignetting( cv::Mat &src, cv::Mat &dst ) {
    /**
     * Implement a Vignetting edge
     */

    // get the center of the image (x,y)
    cv::Point center(src.cols / 2, src.rows / 2);

    // clone greyframe or src as we are going to alter its value
    dst = src.clone();

    // get radius from center point
    double radius = std::sqrt(center.x * center.x + center.y * center.y);
    printf("radius is %f\n", radius);

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(y, x);

            // calculate distance from center
            double distance = std::sqrt(std::pow(x - center.x, 2) + std::pow(y - center.y, 2));
            // printf("distancce is %f\n", distance);

            // caculate vignette strength based on distance
            double vignetteWeight = 1.0 - (distance / radius);

            // Clamp the vignette weight to [0, 1]
            vignetteWeight = std::max(0.0, std::min(1.0, vignetteWeight));
            
            // Apply the vignette effect by multiplying the pixel values by the vignette weight
            // if (distance >= radius) {
            // apply vignette effect across all pixel not just the one that are distance >= radius
            // as the vignetteWeight will take care of it
            for (int c = 0; c < 3; c++) {
                pixel[c] = cv::saturate_cast<uchar>(src.at<cv::Vec3b>(y, x)[c] * vignetteWeight);
            }
            // }
        }
    }
    return 0;
}

int blur5x5_1( cv::Mat &src, cv::Mat &dst ) {
    /**
     * Implement Gaussian 5x5 filter
     * Source: https://stackoverflow.com/questions/1696113/how-do-i-gaussian-blur-an-image-without-using-any-in-built-gaussian-functions
     * 
     */
    // clone the source image
    dst = src.clone();

    // integer approximation of a 5x5 Gaussian
    int guassianFilter[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    // loop through each rows and cols
    // but don't have to modify the outer two rows and columns
    for (int y = 2; y < src.rows - 2; ++y) {
        for (int x = 2; x < src.cols - 2; ++x) {
            
            // temp variable for color channels
            int sumR = 0;
            int sumG = 0;
            int sumB = 0;

            // total sum of the guassian filter
            // (1+2+4+2+1)*2+(2+4+8+4+2)*2+4+8+16+8+4 = 100
            int totalSum = 100;

            // apply filter
            // https://www.geeksforgeeks.org/gaussian-filter-generation-c/
            for (int fy = -2; fy <= 2; ++fy) {
                for (int fx = -2; fx <= 2; ++fx) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(y + fy, x + fx);
                    // get the filter weight
                    int filterWeight = guassianFilter[fy + 2][fx + 2];
                    sumB += pixel[0] * filterWeight;
                    sumG += pixel[1] * filterWeight;
                    sumR += pixel[2] * filterWeight;
                }
            }

            // normalized the pixel at destination image
            dst.at<cv::Vec3b>(y, x)[0] = sumB / totalSum;
            dst.at<cv::Vec3b>(y, x)[1] = sumG / totalSum;
            dst.at<cv::Vec3b>(y, x)[2] = sumR / totalSum;
        }
    }
    return 0;
}

int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    /**
     * Implement Gaussian 5x5 filter using separable methods.
     */
    dst = src.clone(); // Clone the source image to the destination image

    // Define the 1x5 separable Gaussian kernel
    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 10; // Sum of kernel values (1 + 2 + 4 + 2 + 1)

    // Vertical Pass (1x5 filter)
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 2; x < src.cols - 2; ++x) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply the 1x5 kernel horizontally
            for (int kx = -2; kx <= 2; ++kx) {
                // access pixel using pointer ptr
                cv::Vec3b pixel = src.ptr<cv::Vec3b>(y)[x + kx];
                int weight = kernel[kx + 2];

                // Accumulate weighted values for each channel
                sumB += pixel[0] * weight;
                sumG += pixel[1] * weight;
                sumR += pixel[2] * weight;
            }

            // Normalize and set the pixel value in the destination image
            dst.ptr<cv::Vec3b>(y)[x][0] = sumB / kernelSum;
            dst.ptr<cv::Vec3b>(y)[x][1] = sumG / kernelSum;
            dst.ptr<cv::Vec3b>(y)[x][2] = sumR / kernelSum;
        }
    }


    // Vertical Pass (1x5 filter)
    for (int x = 0; x < src.cols; ++x) {
        for (int y = 2; y < src.rows - 2; ++y) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply the 1x5 kernel vertically
            for (int ky = -2; ky <= 2; ++ky) {
                cv::Vec3b pixel = src.ptr<cv::Vec3b>(y + ky)[x];
                int weight = kernel[ky + 2];

                // Accumulate weighted values for each channel
                sumB += pixel[0] * weight;
                sumG += pixel[1] * weight;
                sumR += pixel[2] * weight;
            }

            // Normalize and set the pixel value in the destination image
            dst.ptr<cv::Vec3b>(y)[x][0] = sumB / kernelSum;
            dst.ptr<cv::Vec3b>(y)[x][1] = sumG / kernelSum;
            dst.ptr<cv::Vec3b>(y)[x][2] = sumR / kernelSum;
        }
    }
    return 0;
}

int sobelX3x3( cv::Mat &src, cv::Mat &dst ) {
    /**
     * Implement Sobel 3x3 Sobel Filter horizontally
     */

    // Create the destination image as CV_16SC3 because if we only do clone,
    // it is normally cv_8UC3 wich is not right for sobel edge
    dst = cv::Mat(src.size(), CV_16SC3);

    int sobelFilterX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply the Sobel X kernel (horizontal edges)
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x + kx);
                    int weight = sobelFilterX[ky + 1][kx + 1];

                    // Accumulate the weighted sums for each channel
                    sumB += pixel[0] * weight;
                    sumG += pixel[1] * weight;
                    sumR += pixel[2] * weight;
                }
            }

            // Set the result in the destination image
            dst.at<cv::Vec3s>(y, x)[0] = sumB;
            dst.at<cv::Vec3s>(y, x)[1] = sumG;
            dst.at<cv::Vec3s>(y, x)[2] = sumR;
        }
    }

    return 0;
}

int sobelY3x3( cv::Mat &src, cv::Mat &dst ) {
    /**
     * Implement Sobel 3x3 Sobel Filter vertical edge
     */

    dst = cv::Mat(src.size(), CV_16SC3);

    int sobelFilterY[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };
    
    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x + kx);
                    int weight = sobelFilterY[ky + 1][kx + 1];

                    sumB += pixel[0] * weight;
                    sumG += pixel[1] * weight;
                    sumR += pixel[2] * weight;
                }
            }

            dst.at<cv::Vec3s>(y, x)[0] = sumB;
            dst.at<cv::Vec3s>(y, x)[1] = sumG;
            dst.at<cv::Vec3s>(y, x)[2] = sumR;
        }
    }
    
    return 0;
}

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ) {
    /***
     * Implementing gradient magnitude.
     */
    // check to make sure sx and sy are all CV_16SC3
    // as we are trying to generate a gradient magnitude image from the X and Y Sobel images
    if (sx.type() != CV_16SC3 || sy.type() != CV_16SC3) {
        printf("Type of sx and sy should be CV_16SC3 as this is sobel images/frame");
        return -1;
    }

    // convert back to regular cv_8UC3 (normal colored frame)
    dst = cv::Mat(sx.size(), CV_8UC3);

    for (int y = 0; y < sx.rows; ++y) {
        for (int x = 0; x < sx.cols; ++x) {
            // Access the pixel values in sx and sy
            cv::Vec3s pixelSx = sx.at<cv::Vec3s>(y, x);
            cv::Vec3s pixelSy = sy.at<cv::Vec3s>(y, x);

            // Compute the gradient magnitude for each channel
            cv::Vec3b pixelDst; // 8-bit destination pixel
            for (int c = 0; c < 3; ++c) {
                int gx = pixelSx[c];
                int gy = pixelSy[c];
                int mag = static_cast<int>(std::sqrt(gx * gx + gy * gy));

                // Clamp the value to the range [0, 255] and assign to the destination pixel
                pixelDst[c] = static_cast<uchar>(std::min(mag, 255));
            }

            // Set the result in the destination image
            dst.at<cv::Vec3b>(y, x) = pixelDst;
        }
    }
    return 0;
}

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ) {
    /**
     * function that takes in a color image, blurs the image, 
     * and then quantizes the image into a fixed number of levels as specified by a parameter.
     * default levels here is 10.
     */
    // check type here in case we were just using the sobel filter with CV_16
    if (src.type() != CV_8UC3 ) {
        printf("Type of source frame should be CV_8UC3 as this is sobel images/frame");
        return -1;
    }

    cv::Mat blurredFrame;
    blur5x5_2(src, blurredFrame);

    // create dst image
    dst = cv::Mat(blurredFrame.size(), CV_8UC3);
    // bucket size
    float b = 255.0f / levels;

    // Then you can take a color channel value x and execute xt = x / b, 
    // followed by the statement xf = xt * b
    for (int y = 0; y < blurredFrame.rows; ++y) {
        for (int x = 0; x < blurredFrame.cols; ++x) {
            cv::Vec3b pixel = blurredFrame.at<cv::Vec3b>(y, x); // Access the blurred pixel
            cv::Vec3b quantizedPixel;

            for (int c = 0; c < 3; ++c) {
                // take a color channel value x and execute xt = x / b,
                int xt = static_cast<int>(pixel[c] / b);
                // followed by the statement xf = xt * b
                quantizedPixel[c] = static_cast<uchar>(xt * b);
            }
            dst.at<cv::Vec3b>(y, x) = quantizedPixel;
        }
    }
    return 0;
}

int coolTone( cv::Mat &src, cv::Mat &dst ) {
    /**
     * This is part 11 and 12 of the project.
     * I choosed to implement cool toon filter if user press t
     * Cool Tone is implemented by increasing blue channel, and decrease red channel but leave green as is.
     */
    dst = src.clone();

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(y, x);
            
            pixel[0] = cv::saturate_cast<uchar>(pixel[0] * 1.5);
            pixel[1] = cv::saturate_cast<uchar>(pixel[1]);
            pixel[2] = cv::saturate_cast<uchar>(pixel[2] * 0.6);
        }
    }
    return 0;
}

