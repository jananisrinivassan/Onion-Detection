#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

string detect_onion_color(Mat image) {
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // Define thresholds for each color
    Scalar lower_red = Scalar(0, 50, 50);
    Scalar upper_red = Scalar(10, 255, 255);
    Scalar lower_pink = Scalar(150, 50, 50);
    Scalar upper_pink = Scalar(170, 255, 255);
    Scalar lower_white = Scalar(0, 0, 200);
    Scalar upper_white = Scalar(180, 30, 255);

    // Create masks for each color
    Mat mask_red, mask_pink, mask_white;
    inRange(hsv, lower_red, upper_red, mask_red);
    inRange(hsv, lower_pink, upper_pink, mask_pink);
    inRange(hsv, lower_white, upper_white, mask_white);

    // Count the number of non-zero pixels in each mask
    int red_pixels = countNonZero(mask_red);
    int pink_pixels = countNonZero(mask_pink);
    int white_pixels = countNonZero(mask_white);

    // Determine the predominant color based on the number of pixels
    int max_pixels = max({ red_pixels, pink_pixels, white_pixels });
    if (max_pixels == red_pixels)
        return "Red";
    else if (max_pixels == pink_pixels)
        return "Pink";
    else if (max_pixels == white_pixels)
        return "White";
    else
        return "Unknown";
}

vector<int> count_onion_colors(Mat image) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Apply Gaussian blur to reduce noise
    GaussianBlur(gray, gray, Size(5, 5), 0);

    // Threshold the image to obtain binary mask
    Mat thresh;
    threshold(gray, thresh, 220, 255, THRESH_BINARY);

    // Find contours in the binary m
