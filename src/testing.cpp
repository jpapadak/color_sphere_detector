/* 
 * File:   testing.cpp
 * Author: John Papadakis
 *
 * Created on February 6, 2018, 7:15 PM
*/

#include <colored_sphere_detector/SphereDetector.hpp>


int main(int argc, char** argv) {
    
    std::string path = std::string(std::getenv("HOME")) + "/spheres.png";
    cv::Mat input = cv::imread(path, cv::ImreadModes::IMREAD_COLOR);
    cv::cvtColor(input, input, CV_BGR2RGB);
    SphereDetector detector;
    detector.config.visualize = true;   
    detector.detectCircles(input);
    cv::waitKey();
    
    return 0;
}