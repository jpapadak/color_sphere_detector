/* 
 * File:   testing.cpp
 * Author: John Papadakis
 *
 * Created on February 6, 2018, 7:15 PM
*/

#include <SphereDectector.hpp>


int main(int argc, char** argv) {
    
    cv::Mat input = cv::imread(std::string(std::getenv("HOME")) + "/spheres.png");
    cv::cvtColor(input, input, CV_BGR2RGB);
    SphereDetector detector;
    detector.detect(input);
    cv::waitKey();
    
    return 0;
}