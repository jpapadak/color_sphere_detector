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
    
    std::cout << "Opening bag file: " << argv[1] << " for reading...\n";
    RGBD_BagFile_Driver bagfile_reader(argv[1]);
    
    boost::function<void (cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&)> callback(
            boost::bind(&SphereDetector::rgbd_callback, &detector, _1, _2, _3, _4));    
    bagfile_reader.setCallback(callback);

    std::vector<std::string> topics;
    topics.push_back(std::string("/camera/rgb/image_color"));
    //topics.push_back(std::string("/camera/rgb/input_image"));
    topics.push_back(std::string("/camera/rgb/camera_info"));
    //topics.push_back(std::string("/camera/depth_registered/input_image"));
    topics.push_back(std::string("/camera/depth/image"));
    bagfile_reader.setTopics(topics);
    
    while(bagfile_reader.readNextRGBDMessage()) {
        detector.getDetections();
    }
    
    return 0;
}

