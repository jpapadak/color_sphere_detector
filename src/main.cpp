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
    cv::imshow("Input Image", input);
    cv::waitKey(0);
    SphereDetector detector;
    detector.detect(input);
    return 0;
}

