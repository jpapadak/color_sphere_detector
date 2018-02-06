/* 
 * File:   SphereDetector.hpp
 * Author: John Papadakis
 *
 * Created on January 25, 2018, 3:15 PM
 */

#include <cstddef>
#include <cassert>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#ifndef SPHEREDETECTOR_HPP
#define SPHEREDETECTOR_HPP

enum class Color {
    RED, GREEN, BLUE, YELLOW, MAGENTA, CYAN, OTHER
};

class SphereDetector {
public:
    
    SphereDetector() {    
    }
    
    virtual ~SphereDetector() {
    }
    
    std::vector<std::pair<cv::Vec3f, Color>> detect(const cv::Mat& rgb_input) {
        assert(rgb_input.channels() == 3);
        
        cv::Mat rgb_image;
        if (rgb_input.depth() != CV_32F) {
            rgb_input.convertTo(rgb_image, CV_32F);
        } else {
            rgb_image = rgb_input;
        }
        
        cv::medianBlur(rgb_image, rgb_image, 3);
        
        const size_t& rows = rgb_image.rows;
        const size_t& cols = rgb_image.cols;
        
        bool visualize = true;
        float colorful_threshold = .25;
        float color_likelihood_threshold = .7;
        float eccentricity_threshold = .35;
        float detection_min_pixels = 50;
        
        cv::Mat color_classified_image = this->classifyPixelColors(rgb_image, colorful_threshold, color_likelihood_threshold);
        
        if (visualize) {
            cv::Mat false_color = this->imagesc(color_classified_image);
            cv::imshow("Color Classification", false_color);
            cv::waitKey(1);
        }
        
        std::unordered_map<Color, std::vector<cv::Point2f>, EnumClassHash> color_locations_map;
        assert(color_classified_image.isContinuous() and color_classified_image.type() == CV_8UC1);
        for (size_t row = 0; row < color_classified_image.rows; ++row) {
            int8_t* p = color_classified_image.ptr<int8_t>(row);
            for (size_t col = 0; col < color_classified_image.cols; ++col) {
                
                Color color = toColor(p[col]);
                if (color != Color::OTHER) {
                    color_locations_map[color].emplace_back(col, row);
                }
                
            }
        }
        
        std::vector<std::pair<cv::Vec3f, Color>> detections;
        for (std::pair<Color, std::vector<cv::Point2f>> entry : color_locations_map) {
            Color color = entry.first;
            std::vector<cv::Point2f>& locations = entry.second;
            
            if (locations.size() < detection_min_pixels) {
                continue;
            }
            
            cv::Mat locations_mat = cv::Mat(locations.size(), 2, CV_32FC1, locations.data());
            cv::Mat mean, cov;
            cv::calcCovarMatrix(locations_mat, cov, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);
            cov = cov/(locations_mat.rows - 1);
            
            cv::Mat eigenvalues;
            cv::eigen(cov, eigenvalues);
            float eccentricity = std::abs(1 - eigenvalues.at<float>(1)/eigenvalues.at<float>(0));
            
            if (eccentricity < eccentricity_threshold) {
                float radius = 2*std::sqrt(eigenvalues.at<float>(0));
                detections.emplace_back(cv::Vec3f(mean.at<float>(0), mean.at<float>(1), radius), color);
            }
            
        }
        
        if (visualize) {
            cv::Mat output = rgb_input.clone();
            output.convertTo(output, CV_8UC3);
            for (const std::pair<cv::Vec3f, Color>& detection : detections) {
                const cv::Vec3f& xyrad = detection.first;
                Color color = detection.second;
                cv::Vec3f colorvec = 255*colormap.at(color);
                cv::circle(output, cv::Point2f(xyrad[0], xyrad[1]), xyrad[2], cv::Scalar(colorvec[0], colorvec[1], colorvec[2]), 2, 1);
            }
            cv::imshow("Detections", output);
            cv::waitKey(1);
        }
        
        return detections;

    }
    
    const std::vector<std::pair<cv::Vec3f, Color>>& getDetections() {
        return this->detections;
    }
    
    void rgbd_callback(const cv::Mat& rgb_input, const cv::Mat& depth_input, const cv::Mat& rgb_distortion_coeffs, const cv::Mat& rgb_camera_matrix) {
        this->detections = this->detect(rgb_input);
    }
    
private:
    
    std::vector<std::pair<cv::Vec3f, Color>> detections;
    
    struct EnumClassHash {
        template <typename T>
        std::size_t operator()(T t) const {
            return static_cast<std::size_t>(t);
        }
    };
    
    const std::unordered_map<Color, cv::Vec3f, EnumClassHash> colormap = {
        {Color::RED, cv::Vec3f(1, 0, 0)},
        {Color::GREEN, cv::Vec3f(0, 1, 0)},
        {Color::BLUE, cv::Vec3f(0, 0, 1)},
        {Color::YELLOW, cv::Vec3f(1, 1, 0)},
        {Color::MAGENTA, cv::Vec3f(1, 0, 1)},
        {Color::CYAN, cv::Vec3f(0, 1, 1)}
    };
    
    size_t toInteger(Color color) const {
        return static_cast<size_t>(color);
    }
    
    Color toColor(size_t integer) const {
        return static_cast<Color>(integer);
    }
    
    cv::Mat classifyPixelColors(const cv::Mat& rgb_image, const float& colorful_threshold, const float& color_likelihood_threshold) const {
        
        cv::Mat colorful(rgb_image.rows, rgb_image.cols, CV_8UC1);
        cv::Mat color_classes(rgb_image.rows, rgb_image.cols, CV_8UC1);
        
        cv::Vec3f intensity_vector(1, 1, 1);
        cv::Vec3f w = cv::normalize(intensity_vector);
        cv::Vec3f v1_perp = (1/sqrt(2.0))*cv::Vec3f(-1, 1, 0);
        cv::Vec3f v2_perp = (1/sqrt(6.0))*cv::Vec3f(1, 1, -2);
        cv::Matx23f orthogonal_projection(
                v1_perp(0), v1_perp(1), v1_perp(2), 
                v2_perp(0), v2_perp(1), v2_perp(2)
        );
        
        rgb_image.forEach<cv::Vec3f>(
            [&](const cv::Vec3f& pixel, const int* position) -> void {
                size_t row = position[0];
                size_t col = position[1];
                
                cv::Vec2f pixel_vector = orthogonal_projection*(pixel/255.0);
                float radius = cv::norm(pixel_vector);
//                float angle = std::atan2<float>(pixel_vector(1), pixel_vector(0));

                if (radius > colorful_threshold) {
                    colorful.at<int8_t>(row, col) = 255;
                    
                    Color pixel_color;
                    float max_color_likelihood = 0;
                    for (const std::pair<Color, cv::Vec3f>& color : colormap) {
                        cv::Vec2f color_vector = orthogonal_projection*color.second;
//                        float color_angle = std::atan2<float>(color_vector(1), color_vector(0));
                        float color_likelihood = pixel_vector.dot(color_vector)/(cv::norm(pixel_vector)*cv::norm(color_vector));
                        if (color_likelihood > max_color_likelihood) {
                            max_color_likelihood = color_likelihood;
                            pixel_color = color.first;
                        }
                        
                    }
                    
                    if (max_color_likelihood > color_likelihood_threshold) {
                        color_classes.at<int8_t>(row, col) = toInteger(pixel_color);
                    } else {
                        color_classes.at<int8_t>(row, col) = toInteger(Color::OTHER);
                    }
                    
                } else {
                    colorful.at<int8_t>(row, col) = false;
                    color_classes.at<int8_t>(row, col) = toInteger(Color::OTHER);
                }
            }
        );
        
        cv::imshow("Colorful", colorful);
        cv::waitKey(1);
        
        return color_classes;
    }
    
    cv::Vec3f vectorProjection(const cv::Vec3f& a, const cv::Vec3f& b) {
        return a.dot(b)*cv::normalize(b);
    }
    
    cv::Vec3f vectorRejection(const cv::Vec3f& a, const cv::Vec3f& b) {
        return a - vectorProjection(a, b);
    }
    
    cv::Mat imagesc(const cv::Mat& input) const {
        double min;
        double max;
        cv::minMaxIdx(input, &min, &max);
        
        cv::Mat equalized;
        float scale = 255.0/(max-min);
        input.convertTo(equalized, CV_8UC1, scale, min*scale);

        cv::Mat false_color;
        cv::applyColorMap(equalized, false_color, cv::COLORMAP_JET);
        
        return false_color;
    }
    
};

#endif /* SPHEREDETECTOR_HPP */

