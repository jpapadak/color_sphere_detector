/* 
 * File:   main.cpp
 * Author: John Papadakis
 *
 * Created on January 25, 2018, 2:41 PM
 */

#include <SphereDectector.hpp>

/*
 * 
 */
int main(int argc, char** argv) {
    cv::Mat input = cv::imread("spheres.jpg");
    cv::Mat rgb;
    cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);
    cv::imshow("Input Image", rgb);
    cv::waitKey(0);
    SphereDetector detector;
    detector.detect(rgb);
    return 0;
}

