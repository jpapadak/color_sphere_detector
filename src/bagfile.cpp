/* 
 * File:   main.cpp
 * Author: John Papadakis
 *
 * Created on January 25, 2018, 2:41 PM
 */

#include <colored_sphere_detector/ColoredSphereDetector.hpp>
#include <rgbd_drivers_uncc/rgbd_driver.hpp>

int main(int argc, char** argv) {
    ColoredSphereDetector detector;
    detector.visualize = true;
    
    std::cout << "Opening bag file: " << argv[1] << " for reading...\n";
    RGBD_BagFile_Driver bagfile_reader(argv[1]);
    
    boost::function<void (cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&)> callback(
            boost::bind(&ColoredSphereDetector::rgbd_callback, &detector, _1, _2, _4));    
    bagfile_reader.setCallback(callback);

    std::vector<std::string> topics;
//    topics.push_back(std::string("/camera/rgb/image_color"));
    topics.push_back(std::string("camera/rgb/input_image"));
    topics.push_back(std::string("camera/rgb/camera_info"));
    topics.push_back(std::string("camera/depth_registered/input_image"));
//    topics.push_back(std::string("/camera/depth/image"));
    bagfile_reader.setTopics(topics);
    
    while(bagfile_reader.readNextRGBDMessage()) {
    }
    
    return 0;
}

