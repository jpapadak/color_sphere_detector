/* 
 * File:   main.cpp
 * Author: John Papadakis
 *
 * Created on January 25, 2018, 2:41 PM
 */

#include <SphereDectector.hpp>
#include <rgbd_drivers_uncc/rgbd_driver.hpp>

/*
 * 
 */

extern volatile bool run;

int main(int argc, char** argv) {
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
    
    RGBD_OpenCV_Driver rgbd_driver;
    
    boost::function<void (cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&)> callback(
            boost::bind(&SphereDetector::rgbd_callback, &detector, _1, _2, _3, _4));    
    rgbd_driver.setCallback(callback);

    rgbd_driver.initialize();
    run = true;
    rgbd_driver.camera_thread(nullptr);
    
    
    rgbd_driver.shutdown();
    return 0;
}

