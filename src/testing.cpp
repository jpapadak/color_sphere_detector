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
    detector.config.visualize = true;   
    detector.config.margin_x = 100; 
    detector.config.margin_y = 75;
    detector.config.colorful_threshold = .10;
    detector.config.color_likelihood_threshold = .98;
    detector.config.bounding_box_ratio_threshold = .93;
    detector.config.min_radius_threshold = 10;
    detector.config.max_radius_threshold = 50;
    detector.config.circular_area_ratio_threshold = .75;
    detector.detect(input);
    cv::waitKey();
    
    return 0;
}