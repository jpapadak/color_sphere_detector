/* 
 * File:   SphereDetector.hpp
 * Author: John Papadakis
 *
 * Created on January 25, 2018, 3:15 PM
 */

#include <cmath>
#include <vector>
#include <map>
#include <cassert>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#ifndef SPHEREDETECTOR_HPP
#define SPHEREDETECTOR_HPP

enum class Color {
    OTHER, RED, GREEN, BLUE, YELLOW, ORANGE
};

class SphereDetector {
public:
    
    SphereDetector() {    
    }
    
    virtual ~SphereDetector() {
    }
    
    std::vector<std::pair<cv::Vec3f, Color>> detect(const cv::Mat& rgb_input) {
        // Assumes rgb_input is has channels RGB in order
        
        const size_t xmargin = 100; // pixels
        const size_t ymargin = 75; // pixels
        cv::Mat rgb_image = rgb_input(cv::Rect(xmargin, ymargin, rgb_input.cols - xmargin, rgb_input.rows - ymargin)).clone();
        cv::medianBlur(rgb_image, rgb_image, 5);
        
        if (rgb_image.depth() != CV_32F) {
            rgb_image.convertTo(rgb_image, CV_32F);
        }
        
        const size_t& rows = rgb_image.rows;
        const size_t& cols = rgb_image.cols;
        
        bool visualize = true;
        float colorful_threshold = .12; // higher is more restrictive
        float color_likelihood_threshold = .98; // -1 to 1, higher is more restrictive
        float eccentricity_threshold = .37; // 0 to 1, higher is more relaxed
        float detection_min_pixels = 75; // higher is more restrictive
        
        cv::Mat color_classified_image = this->classifyPixelColors(rgb_image, colorful_threshold, color_likelihood_threshold);
        
//        cv::Mat labels;
//        cv::Mat stats;
//        cv::Mat centroids;
//        cv::connectedComponentsWithStats(color_classified_image, labels, stats, centroids);
//        cv::imshow("CC", this->imagesc(labels));
//        cv::waitKey(1);
        
        // Collect classified pixels
        std::map<Color, std::vector<cv::Point2f>> color_locations_map;
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
        
        // Assess data for detections
        std::vector<std::pair<cv::Vec3f, Color>> detections;
        for (std::pair<Color, std::vector<cv::Point2f>> entry : color_locations_map) {
            Color color = entry.first;
            std::vector<cv::Point2f>& locations = entry.second;
            
            if (locations.size() > detection_min_pixels) {
            
                cv::Mat locations_mat = cv::Mat(locations.size(), 2, CV_32FC1, locations.data());
                cv::Mat mean, cov;
                cv::calcCovarMatrix(locations_mat, cov, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);
                cov = cov/(locations_mat.rows - 1);

                cv::Mat eigenvalues;
                cv::eigen(cov, eigenvalues);
                float eccentricity = std::abs(1.0 - eigenvalues.at<float>(1)/eigenvalues.at<float>(0));

                if (eccentricity < eccentricity_threshold) {
                    float radius = 2*std::sqrt(eigenvalues.at<float>(0));
                    detections.emplace_back(cv::Vec3f(mean.at<float>(0) + xmargin, mean.at<float>(1) + ymargin, radius), color);
                }
                
            }
            
        }
        
        if (visualize) {
            cv::imshow("Color Classification", this->imagesc(color_classified_image));
            cv::waitKey(1);
            
            cv::Mat output = rgb_input.clone();
            if (output.type() != CV_8UC3) {
                output.convertTo(output, CV_8UC3);
            }
            for (const std::pair<cv::Vec3f, Color>& detection : detections) {
                const cv::Vec3f& xyrad = detection.first;
                Color color = detection.second;
                cv::Vec3f colorvec = 255*colormap.at(color);
                cv::circle(output, cv::Point2f(xyrad[0], xyrad[1]), xyrad[2], cv::Scalar(colorvec[0], colorvec[1], colorvec[2]), 2, 1);
            }
            cv::cvtColor(output, output, CV_RGB2BGR);
            cv::imshow("Detections", output);
            cv::waitKey(1);
        }
        
        return detections;

    }
    
    const std::vector<std::pair<cv::Vec3f, Color>>& getDetections() {
        return this->detections;
    }
    
    void rgbd_callback(const cv::Mat& color_input, const cv::Mat& depth_input, const cv::Mat& color_distortion_coeffs, const cv::Mat& color_camera_matrix) {
        cv::Mat rgb_input;
        cv::cvtColor(color_input, rgb_input, CV_BGR2RGB);
        this->detections = this->detect(rgb_input);
    }
    
private:
    
    std::vector<std::pair<cv::Vec3f, Color>> detections;
    
    const std::map<Color, cv::Vec3f> colormap = {
        {Color::RED, cv::Vec3f(.6860, .1381, .1757)},
        {Color::GREEN, cv::Vec3f(.1722, .4837, .34)},
        {Color::BLUE, cv::Vec3f(.0567, .3462, .6784)},
        {Color::YELLOW, cv::Vec3f(.8467, .8047, .2646)},
        {Color::ORANGE, cv::Vec3f(.7861, .1961, .0871)},
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
        
        // Create orthonormal basis in the (1, 1, 1) plane
        cv::Vec3f v1_perp = (1/std::sqrt(2.0))*cv::Vec3f(-1, 1, 0);
        cv::Vec3f v2_perp = (1/std::sqrt(6.0))*cv::Vec3f(1, 1, -2);
        cv::Matx23f orthogonal_projection = {
                v1_perp(0), v1_perp(1), v1_perp(2), 
                v2_perp(0), v2_perp(1), v2_perp(2)
        };
        std::map<Color, cv::Vec2f> projected_colormap = this->projectColormap(orthogonal_projection);
        
        rgb_image.forEach<cv::Vec3f>(
            [&](const cv::Vec3f& pixel, const int* position) -> void {
                size_t row = position[0];
                size_t col = position[1];
                
                cv::Vec2f pixel_vector = orthogonal_projection*(pixel/255.0);
                float pixel_magnitude = cv::norm<float, 2, 1>(pixel_vector);
                
                if (pixel_magnitude > colorful_threshold) {
                    colorful.at<int8_t>(row, col) = true;
                    
                    Color pixel_color = Color::OTHER;
                    float max_color_likelihood = 0;
                    for (const std::pair<Color, cv::Vec2f>& color : projected_colormap) {
                        const cv::Vec2f& color_vector = color.second;
                        
                        float color_likelihood = cv::normalize<float, 2>(pixel_vector).dot(color_vector);
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
        
        cv::imshow("Colorful", this->imagesc(colorful));
        cv::waitKey(1);
        
        return color_classes;
    }
    
    std::map<Color, cv::Vec2f> projectColormap(const cv::Matx23f& projection_matrix) const {
        std::map<Color, cv::Vec2f> projected_colormap;
        
        for (const std::pair<Color, cv::Vec3f>& color : colormap) {
            projected_colormap[color.first] = cv::normalize<float, 2>(projection_matrix*color.second);
        }
        
        return projected_colormap;
    }
    
    cv::Vec3f vectorProjection(const cv::Vec3f& a, const cv::Vec3f& b) {
        return a.dot(b)*cv::normalize(b);
    }
    
    cv::Vec3f vectorRejection(const cv::Vec3f& a, const cv::Vec3f& b) {
        return a - vectorProjection(a, b);
    }
    
    cv::Matx<float, 3, 5> colorsToMatrix() const {
        // Matrix of column vectors
        cv::Matx<float, 3, 5> result;
        
        size_t col = 0;
        for (const std::pair<Color, cv::Vec3f>& color : colormap) {
            result(0, col) = color.second(0);
            result(1, col) = color.second(1);
            result(2, col) = color.second(2);
            col++;
        }
        
        return result;
    }
    
    template <typename NumericType, int rows, int cols>
    cv::Matx<NumericType, rows, cols> normalizeColumns(const cv::Matx<NumericType, rows, cols>& input_matrix) const {
        cv::Matx<NumericType, rows, cols> normalized;
        
        for (int col = 0; col < cols; ++col) {
            // would love a simple Matx -> Vec conversion
            cv::Matx<NumericType, rows, 1> output_col = (1.0/cv::norm(input_matrix.col(col)))*input_matrix.col(col);
            for (int row = 0; row < rows; ++row) {
                normalized(row, col) = output_col(row);
            }
        }
        
        return normalized;
    }
    
    cv::Mat imagesc(const cv::Mat& input) const {
        double min;
        double max;
        cv::minMaxIdx(input, &min, &max);
        
        cv::Mat scaled;
        float scale = 255.0/(max - min);
        input.convertTo(scaled, CV_8UC1, scale, -min*scale);

        cv::Mat false_color;
        cv::applyColorMap(scaled, false_color, cv::COLORMAP_JET);
        
        return false_color;
    }
    
};

#endif /* SPHEREDETECTOR_HPP */

