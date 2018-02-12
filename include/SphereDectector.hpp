/* 
 * File:   SphereDetector.hpp
 * Author: John Papadakis
 *
 * Created on January 25, 2018, 3:15 PM
 */

#ifndef SPHEREDETECTOR_HPP
#define SPHEREDETECTOR_HPP

#include <cmath>
#include <tuple>
#include <vector>
#include <map>
#include <cassert>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>


enum class Color {
    OTHER, RED, GREEN, BLUE, YELLOW, ORANGE
};

constexpr double pi = std::acos(-1.0);

class SphereDetector {
public:
    
    struct Configuration {
        
        bool visualize = false;
        
        // Margin of pixels ignored on input image
        size_t margin_x = 0; 
        size_t margin_y = 0;
        
        // Color classification parameters
        float colorful_threshold = .10; // magnitude of the vector rejection of the pixel color vector onto the intensity vector (1, 1, 1)
        float color_likelihood_threshold = .98; // scaled dot product between the pixel color vector and the class color vectors, range 0 to 1
        
        // Circle detection parameters
        float bounding_box_ratio_threshold = .93; // ratio between the shortest side to the longest side of the bounding box, range 0 to 1
        float min_radius_threshold = 10; // minimum radius of candidate circle in pixels
        float max_radius_threshold = 50; // maximum radius of candidate circle in pixels
        float circular_area_ratio_threshold = .75; // ratio of number of pixels within candidate circle and expected circle area, range 0 to 1
        
    } config;
    
    SphereDetector() {    
    }
    
    virtual ~SphereDetector() {
    }
    
    void setConfiguration(const SphereDetector::Configuration& config) {
        this->config = config;
    }
    
    std::vector<std::pair<cv::Vec3f, Color>> detect(const cv::Mat& rgb_input) {
        // Assumes rgb_input is has channels RGB in order
        assert(rgb_input.channels() == 3);
        
        const size_t& margin_x = config.margin_x;
        const size_t& margin_y = config.margin_y;
        const float& colorful_threshold = config.colorful_threshold;
        const float& color_likelihood_threshold = config.color_likelihood_threshold;
        const float& bounding_box_ratio_threshold = config.bounding_box_ratio_threshold;
        const float& min_radius_threshold = config.min_radius_threshold;
        const float& max_radius_threshold = config.max_radius_threshold;
        const float& circular_area_ratio_threshold = config.circular_area_ratio_threshold;
        
        // Trim down input image by margin, median blur, convert to float if needed
        cv::Mat rgb_image = rgb_input(cv::Rect(margin_x, margin_y, rgb_input.cols - margin_x, rgb_input.rows - margin_y)).clone();
        cv::medianBlur(rgb_image, rgb_image, 5);
        if (rgb_image.depth() != CV_32F) {
            rgb_image.convertTo(rgb_image, CV_32F);
        }
        
        cv::Mat color_classified_image = this->classifyPixelColors(rgb_image, colorful_threshold, color_likelihood_threshold);
        
        cv::Mat class_and_components;
        if (visualize) {
            class_and_components = color_classified_image.clone();
        }
        
        std::vector<std::pair<cv::Vec3f, Color>> detections;
        for (const std::pair<Color, cv::Vec3f>& entry : colormap) {
            // For each color class, compute color mask and run connected components on mask
            
            Color color = entry.first;
            cv::Mat color_mask = color_classified_image == toInteger(color);
            cv::Mat labeled_image;
            cv::Mat stats;
            cv::Mat centroids;
            cv::connectedComponentsWithStats(color_mask, labeled_image, stats, centroids);
            
            for (size_t label = 1; label < stats.rows; ++label) {
                // Validate each connected component bounding box
                
                int component_area =  stats.at<int>(label, cv::CC_STAT_AREA);
                int bb_width = stats.at<int>(label, cv::CC_STAT_WIDTH);
                int bb_height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
                float bb_ratio = (bb_width > bb_height) ? static_cast<float>(bb_height)/bb_width : static_cast<float>(bb_width)/bb_height;

                if (bb_ratio > bounding_box_ratio_threshold) {
                    // Bounding box is square enough, compute candidate circle and check data against it
                    
                    float circle_radius = (bb_width + bb_height)/4.0f;
                    
                    if (circle_radius > min_radius_threshold and circle_radius < max_radius_threshold) {
                    
                        cv::Vec2f circle_center(bb_width/2.0f, bb_height/2.0f);
                        float circle_radius_sq = std::pow(circle_radius, 2);
                        float circle_area = pi*circle_radius_sq;

                        // Get nonzero (colorful) pixel locations within bounding box
                        int bb_x = stats.at<int>(label, cv::CC_STAT_LEFT);
                        int bb_y = stats.at<int>(label, cv::CC_STAT_TOP);
                        cv::Rect bb_roi = cv::Rect(bb_x, bb_y, bb_width, bb_height);
                        cv::Mat xypoints;
                        cv::findNonZero(color_mask(bb_roi), xypoints);
                        xypoints = xypoints.reshape(1);
                        xypoints.convertTo(xypoints, CV_32F);

                        // Check that the number of pixels inside circle is close to area of the circle
                        cv::Mat zero_centered_points(xypoints.rows, xypoints.cols, CV_32FC1);
                        for (size_t r = 0; r < xypoints.rows; ++r) {
                            zero_centered_points.row(r) = xypoints.row(r) - circle_center;
                        }

                        cv::Mat point_radii_sq;
                        cv::reduce(zero_centered_points.mul(zero_centered_points), point_radii_sq, 1, CV_REDUCE_SUM);
                        float area_points_inside_circle = cv::countNonZero(point_radii_sq <= circle_radius_sq);

                        if (area_points_inside_circle/circle_area > circular_area_ratio_threshold) {
                            cv::Vec3f detection(circle_center(0) + bb_x + margin_x, circle_center(1) + bb_y + margin_y, circle_radius);
                            detections.emplace_back(std::move(detection), color);
                        }

                        if (visualize) {
                            cv::rectangle(class_and_components, bb_roi, toInteger(color));
                        }
                        
                    }
                    
                }
                
            }
        
        }
        
        if (visualize) {
            
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
            cv::imshow("Color Classification & Components", this->imagesc(class_and_components));
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
    bool visualize = true;
    
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
        std::map<Color, cv::Vec2f> projected_colormap = this->projectColormap(orthogonal_projection, true);
        
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
                        
                        float color_likelihood = 0.5*(cv::normalize<float, 2>(pixel_vector).dot(color_vector) + 1);
                        
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
        
        return color_classes;
    }
    
    std::map<Color, cv::Vec2f> projectColormap(const cv::Matx23f& projection_matrix, bool normalize) const {
        std::map<Color, cv::Vec2f> projected_colormap;
        
        for (const std::pair<Color, cv::Vec3f>& color : colormap) {
            if (normalize) {
                projected_colormap[color.first] = cv::normalize<float, 2>(projection_matrix*color.second);
            } else {
                projected_colormap[color.first] = projection_matrix*color.second;
            }
        }
        
        return projected_colormap;
    }
    
    std::tuple<cv::Mat, cv::Mat, cv::Mat> computeMeanAndCovariance(const cv::Mat& points) const {
        size_t n = points.rows;
        
        cv::Mat mean;
        cv::reduce(points, mean, 0, CV_REDUCE_AVG);
        cv::Mat zero_centered_points = points - cv::Mat1f::ones(n, 1)*mean;
        cv::Mat cov = (1.0f/(n - 1))*zero_centered_points*zero_centered_points.t();
        
        return std::make_tuple(mean, cov, zero_centered_points);
    }
    
    template <typename NumericType, int size>
    static cv::Vec<NumericType, size> vectorProjection(const cv::Vec<NumericType, size>& a, const cv::Vec<NumericType, size>& b) {
        return a.dot(b)*cv::normalize(b);
    }
    
    template <typename NumericType, int size>
    static cv::Vec<NumericType, size> vectorRejection(const cv::Vec<NumericType, size>& a, const cv::Vec<NumericType, size>& b) {
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