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

#include <boost/make_shared.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_sphere.h>


enum class Color {
    OTHER, RED, GREEN, BLUE, YELLOW, ORANGE
};

template <typename NumericType, int size>
class Gaussian {

    cv::Vec<NumericType, size> mean;
    cv::Matx<NumericType, size, size> covariance;
    cv::Matx<NumericType, size, size> inv_covariance;
    NumericType det_covariance;
    NumericType log_det_covariance;
    NumericType normalization_factor;
    static constexpr NumericType pi = std::acos(-1);
    static constexpr NumericType nlog2pi = size*std::log(2*pi);

    void precomputeTerms() {
        this->inv_covariance = covariance.inv();
        this->det_covariance = cv::determinant(covariance);
        this->normalization_factor = 1/std::sqrt(std::pow(2*pi, size)*this->det_covariance);
    }

public:

    Gaussian() {
    }

    Gaussian(const cv::Vec<NumericType, size>& mean, const cv::Matx<NumericType, size, size>& covariance) {
        this->mean = mean;
        this->covariance = covariance;
        this->precomputeTerms();
    }

    NumericType evaluate(const cv::Vec<NumericType, size>& x) const {
        cv::Vec<NumericType, size> x_minus_mu = x - mean;
        return normalization_factor*std::exp(-0.5*(x_minus_mu.t()*inv_covariance*x_minus_mu)[0]);
    }

    NumericType evaluateLogLikelihood(const cv::Vec<NumericType, size>& x) const {
        cv::Vec<NumericType, size> x_minus_mu = x - mean;
        return -0.5*(nlog2pi + log_det_covariance + (x_minus_mu.t()*inv_covariance*x_minus_mu)[0]);
    }

    template <typename T, int n, int m>
    static Gaussian<T, n> transform(const Gaussian<T, m>& input, const cv::Matx<T, n, m>& transformation_matrix) {
        return Gaussian<T, n>(transformation_matrix*input.getMean(), transformation_matrix*input.getCovariance()*transformation_matrix.t());
    }

    template <int newsize>
    void transform(const cv::Matx<NumericType, newsize, size>& transformation_matrix) {
        *this = Gaussian::transform(*this, transformation_matrix);
    }

    void setMean(const cv::Vec<NumericType, size>& mean) {
        this->mean = mean;
    }

    const cv::Vec<NumericType, size>& getMean() const {
        return mean;
    }

    void setCovariance(const cv::Matx<NumericType, size, size>& covariance) {
        this->covariance = covariance;
        this->precomputeTerms();
    }

    const cv::Matx<NumericType, size, size>& getCovariance() const {
        return covariance;
    }

    const cv::Matx<NumericType, size, size>& getInverseCovariance() const {
        return inv_covariance;
    }

};

class PixelColorClassifier {
    
public:
    
    std::map<Color, Gaussian<float, 3>> color_class_map = {
            {Color::RED, Gaussian<float, 3>(cv::Vec3f(0.6762, 0.1513, 0.1850), cv::Matx33f(
                    0.0134, 0.0052, 0.0064, 
                    0.0052, 0.0038, 0.0042, 
                    0.0064, 0.0042, 0.0054))},
            {Color::GREEN, Gaussian<float, 3>(cv::Vec3f(0.1387, 0.4116, 0.2718), cv::Matx33f(
                    0.0066, 0.0080, 0.0080, 
                    0.0080, 0.0193, 0.0152, 
                    0.0080, 0.0152, 0.0134))},
            {Color::BLUE, Gaussian<float, 3>(cv::Vec3f(0.0659, 0.3986, 0.7374), cv::Matx33f(
                    0.0113, 0.0083, 0.0034,
                    0.0083, 0.0193, 0.0186, 
                    0.0034, 0.0186, 0.0230))},
            {Color::YELLOW, Gaussian<float, 3>(cv::Vec3f(0.8320, 0.7906, 0.2898), cv::Matx33f(
                    0.0154, 0.0174, 0.0073,
                    0.0174, 0.0202, 0.0088,
                    0.0073, 0.0088, 0.0149))},
            {Color::ORANGE, Gaussian<float, 3>(cv::Vec3f(0.8017, 0.2349, .1267), cv::Matx33f(
                    0.0133, 0.0070, 0.0019,
                    0.0070, 0.0070, 0.0042,
                    0.0019, 0.0042, 0.0041))},
        };
        
    float colorful_threshold = .09; // minimum magnitude of the vector rejection of the pixel color vector onto the intensity vector (1, 1, 1)
    float color_likelihood_threshold = -8; // minimum likelihood value for pixel to not be classified as other
    
    cv::Mat classifyPixelColors(const cv::Mat& rgb_image) {
        return this->classifyPixelColors(rgb_image, color_class_map, colorful_threshold, color_likelihood_threshold);
    }
    
    static cv::Mat classifyPixelColors(const cv::Mat& rgb_image, 
            const std::map<Color, Gaussian<float, 3>>& color_class_map,
            float colorful_threshold, 
            float color_likelihood_threshold) {
                
        cv::Mat colorful(rgb_image.rows, rgb_image.cols, CV_8UC1);
        cv::Mat color_classes(rgb_image.rows, rgb_image.cols, CV_8UC1);
        
        // Create orthonormal basis in the (1, 1, 1) plane
        cv::Vec3f v1_perp = (1/std::sqrt(2.0))*cv::Vec3f(-1, 1, 0);
        cv::Vec3f v2_perp = (1/std::sqrt(6.0))*cv::Vec3f(1, 1, -2);
        cv::Matx23f orthogonal_projection = {
                v1_perp(0), v1_perp(1), v1_perp(2), 
                v2_perp(0), v2_perp(1), v2_perp(2)
        };
        
        std::map<Color, Gaussian<float, 2>> projected_color_class_map;
        for (const std::pair<Color, Gaussian<float, 3>>& color : color_class_map) {
            projected_color_class_map[color.first] = Gaussian<float, 3>::transform(color.second, orthogonal_projection);
        }
        
        rgb_image.forEach<cv::Vec3f>(
            [&](const cv::Vec3f& pixel, const int* position) -> void {
                size_t row = position[0];
                size_t col = position[1];
                
                cv::Vec2f pixel_vector = orthogonal_projection*(pixel/255.0);
                float pixel_magnitude = cv::norm<float, 2, 1>(pixel_vector);
                
                if (pixel_magnitude > colorful_threshold) {
                    colorful.at<int8_t>(row, col) = true;
                    
                    Color pixel_color = Color::OTHER;
                    float max_color_likelihood = -std::numeric_limits<float>::infinity();
                    for (const std::pair<Color, Gaussian<float, 2>>& color : projected_color_class_map) {
                        const Gaussian<float, 2>& color_class = color.second;
                        
                        float color_likelihood = color_class.evaluateLogLikelihood(pixel_vector);
//                        
                        
                        if (color_likelihood > max_color_likelihood) {
                            max_color_likelihood = color_likelihood;
                            pixel_color = color.first;
                        }
                        
                    }
                    
                    if (max_color_likelihood > color_likelihood_threshold) {
                        color_classes.at<int8_t>(row, col) = static_cast<size_t>(pixel_color);
                    } else {
                        color_classes.at<int8_t>(row, col) = static_cast<size_t>(Color::OTHER);
                    }
                    
                } else {
                    colorful.at<int8_t>(row, col) = false;
                    color_classes.at<int8_t>(row, col) = static_cast<size_t>(Color::OTHER);
                }
            }
        );
        
        return color_classes;
    }
    
};

class CircleDetector {
    
    static constexpr double pi = std::acos(-1.0);
    
public:
    
    float bounding_box_ratio_threshold = .92; // ratio between the shortest side to the longest side of the bounding box, range 0 to 1
    float min_circle_radius = 6; // minimum radius of candidate circle in pixels
    float max_circle_radius = 50; // maximum radius of candidate circle in pixels
    float circular_fill_ratio_threshold = .8; // ratio of number of pixels within candidate circle and expected circle area, range 0 to 1
    float component_area_ratio_threshold = .9; // ratio of number of pixels within candidate circle and total component area, range 0 to 1
    
    struct CircleDetection {
        float x;
        float y;
        float radius;
        std::vector<cv::Point2i> locations;
    };
    
    std::vector<CircleDetection> detectCircles(const cv::Mat1b& binary_image) {
        return std::move(this->detectCircles(binary_image, 
                    bounding_box_ratio_threshold, 
                    min_circle_radius, max_circle_radius, 
                    circular_fill_ratio_threshold, 
                    component_area_ratio_threshold)
                );
    }
    
    static std::vector<CircleDetection> detectCircles(const cv::Mat1b& binary_image, 
            float bounding_box_ratio_threshold, float min_radius, float max_radius, 
            float circular_fill_ratio_threshold, float component_area_ratio_threshold) {
        
        std::vector<CircleDetection> detections;
        cv::Mat labeled_image;
        cv::Mat stats;
        cv::Mat centroids;
        cv::connectedComponentsWithStats(binary_image, labeled_image, stats, centroids);

        for (size_t label = 1; label < stats.rows; ++label) {
            // Validate each connected component bounding box

            int component_area =  stats.at<int>(label, cv::CC_STAT_AREA);
            int bb_width = stats.at<int>(label, cv::CC_STAT_WIDTH);
            int bb_height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
            float bb_ratio = (bb_width > bb_height) ? 
                static_cast<float>(bb_height)/bb_width : static_cast<float>(bb_width)/bb_height;

            if (bb_ratio > bounding_box_ratio_threshold) {
                // Bounding box is square enough, compute candidate circle and check data against it

                float circle_radius = (bb_width + bb_height)/4.0f;

                if (circle_radius > min_radius and circle_radius < max_radius) {

                    cv::Point2f circle_center(bb_width/2.0f, bb_height/2.0f);
                    float circle_radius_sq = std::pow(circle_radius, 2);
                    float circle_area = pi*circle_radius_sq;

                    // Get nonzero (colorful) pixel locations within bounding box
                    int bb_x = stats.at<int>(label, cv::CC_STAT_LEFT);
                    int bb_y = stats.at<int>(label, cv::CC_STAT_TOP);
                    cv::Rect bb_roi = cv::Rect(bb_x, bb_y, bb_width, bb_height);
                    std::vector<cv::Point2i> xypoints;
                    cv::findNonZero(binary_image(bb_roi), xypoints);
                    cv::Mat xypoints_mat(xypoints, false); // note: points to same data as xypoints
                    xypoints_mat = xypoints_mat.reshape(1);
                    cv::Mat xypoints_float;
                    xypoints_mat.convertTo(xypoints_float, CV_32F);

                    // Check that the number of pixels inside circle is close to area of the circle,
                    // also check that enough of component pixels are inside circle vs out
                    cv::Mat zero_centered_points(xypoints_float.rows, xypoints_float.cols, CV_32FC1);
                    zero_centered_points.col(0) = xypoints_float.col(0) - circle_center.x;
                    zero_centered_points.col(1) = xypoints_float.col(1) - circle_center.y;
                    cv::Mat point_radii_sq;
                    cv::reduce(zero_centered_points.mul(zero_centered_points), point_radii_sq, 1, CV_REDUCE_SUM);
                    float area_points_inside_circle = cv::countNonZero(point_radii_sq <= circle_radius_sq);
                    float circular_fill_ratio = area_points_inside_circle/circle_area;
                    float component_area_ratio = area_points_inside_circle/component_area;

                    if (circular_fill_ratio > circular_fill_ratio_threshold and 
                            component_area_ratio > component_area_ratio_threshold) {

                        CircleDetection circle;
                        circle.x = circle_center.x + bb_x;
                        circle.y = circle_center.y + bb_y;
                        circle.radius = circle_radius;
                        xypoints_mat.col(0) = xypoints_mat.col(0) + bb_x;
                        xypoints_mat.col(1) = xypoints_mat.col(1) + bb_y;
                        circle.locations = std::move(xypoints);
                        detections.push_back(std::move(circle));

                    }

                }

            }

        }
        
        return detections;

    }
    
};

class SphereDetector {
public:
    
    cv::Point2f focal_length = cv::Point2f(500, 500);
    cv::Point2f image_center = cv::Point2f(320, 240);
    size_t min_points_for_fitting = 10;
    float ransac_model_distance_threshold = .008; // distance from the spherical model within which point is considered an inlier
    float min_sphere_radius = .02; // meters
    float max_sphere_radius = .045; // meters
    float inlier_percentage_threshold = .6; // percentage of data within distance threshold of the refined model, used to accept or reject detection
    
    struct SphereDetection {
        float x;
        float y;
        float z;
        float radius;
        float confidence;
    };
    
    SphereDetection fitSphericalModelRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr points) {
        return this->fitSphericalModelRANSAC(points, min_sphere_radius, 
                max_sphere_radius, ransac_model_distance_threshold);
    }
    
    static SphereDetection fitSphericalModelRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr points, 
            float min_radius, float max_radius, float ransac_model_distance_threshold) {
        
        pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr sphere_model = 
                boost::make_shared<pcl::SampleConsensusModelSphere<pcl::PointXYZ>>(points);
        sphere_model->setRadiusLimits(min_radius, max_radius);
        
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(sphere_model);
        
        ransac.setDistanceThreshold(ransac_model_distance_threshold);
//        ransac.setMaxIterations(10);
        ransac.computeModel();
        Eigen::VectorXf coeffs;
        ransac.getModelCoefficients(coeffs);
        
        // Percentage of points within distance threshold of model
        float inlier_ratio = ransac.inliers_.size()/static_cast<float>(points->size());
        
//        std::cout << "Sphere coeffs: " << coeffs.transpose() << ", confidence: " << inlier_ratio << "\n";
        
        SphereDetection sphere;
        sphere.x = coeffs[0];
        sphere.y = coeffs[1];
        sphere.z = coeffs[2];
        sphere.radius = coeffs[3];
        sphere.confidence = inlier_ratio;
        
        return sphere;
    }
    
};

class ColoredSphereDetector {
    PixelColorClassifier pixel_color_classifier;
    CircleDetector circle_detector;
    SphereDetector sphere_detector;
    
public:
    
    struct Configuration {
        
        bool visualize = false;
        
        // Margin of pixels ignored on input image
        size_t margin_x = 0; 
        size_t margin_y = 0;
        
        // Color classification parameters
        
        std::map<Color, Gaussian<float, 3>> color_class_map = {
            {Color::RED, Gaussian<float, 3>(cv::Vec3f(0.6762, 0.1513, 0.1850), cv::Matx33f(
                    0.0134, 0.0052, 0.0064, 
                    0.0052, 0.0038, 0.0042, 
                    0.0064, 0.0042, 0.0054))},
            {Color::GREEN, Gaussian<float, 3>(cv::Vec3f(0.1387, 0.4116, 0.2718), cv::Matx33f(
                    0.0066, 0.0080, 0.0080, 
                    0.0080, 0.0193, 0.0152, 
                    0.0080, 0.0152, 0.0134))},
            {Color::BLUE, Gaussian<float, 3>(cv::Vec3f(0.0659, 0.3986, 0.7374), cv::Matx33f(
                    0.0113, 0.0083, 0.0034,
                    0.0083, 0.0193, 0.0186, 
                    0.0034, 0.0186, 0.0230))},
            {Color::YELLOW, Gaussian<float, 3>(cv::Vec3f(0.8320, 0.7906, 0.2898), cv::Matx33f(
                    0.0154, 0.0174, 0.0073,
                    0.0174, 0.0202, 0.0088,
                    0.0073, 0.0088, 0.0149))},
            {Color::ORANGE, Gaussian<float, 3>(cv::Vec3f(0.8017, 0.2349, .1267), cv::Matx33f(
                    0.0133, 0.0070, 0.0019,
                    0.0070, 0.0070, 0.0042,
                    0.0019, 0.0042, 0.0041))},
        };
        
        float colorful_threshold = .09; // minimum magnitude of the vector rejection of the pixel color vector onto the intensity vector (1, 1, 1)
        float color_likelihood_threshold = -8; // minimum likelihood value for pixel to not be classified as other
        
        // Circle detection parameters
        
        float bounding_box_ratio_threshold = .92; // ratio between the shortest side to the longest side of the bounding box, range 0 to 1
        float min_circle_radius = 6; // minimum radius of candidate circle in pixels
        float max_circle_radius = 50; // maximum radius of candidate circle in pixels
        float circular_fill_ratio_threshold = .8; // ratio of number of pixels within candidate circle and expected circle area, range 0 to 1
        float component_area_ratio_threshold = .9; // ratio of number of pixels within candidate circle and total component area, range 0 to 1
        
        // Sphere fitting parameters
        
        cv::Point2f focal_length = cv::Point2f(500, 500);
        cv::Point2f image_center = cv::Point2f(320, 240);
        size_t min_points_for_fitting = 10;
        float ransac_model_distance_threshold = .008; // distance from the spherical model within which point is considered an inlier
        float min_sphere_radius = .02; // meters
        float max_sphere_radius = .045; // meters
        float inlier_percentage_threshold = .6; // percentage of data within distance threshold of the refined model, used to accept or reject detection
        
    } config;
    
    struct ColoredCircleDetection : CircleDetector::CircleDetection {
        Color color;
        ColoredCircleDetection() = default;
        explicit ColoredCircleDetection(const CircleDetector::CircleDetection& circle, Color color) : 
            CircleDetector::CircleDetection(circle), color(color) {}
        
    };
    
    struct ColoredSphereDetection : SphereDetector::SphereDetection {
        Color color;
        ColoredSphereDetection() = default;
        explicit ColoredSphereDetection(const SphereDetector::SphereDetection& sphere, Color color) : 
            SphereDetector::SphereDetection(sphere), color(color) {}
        
    };
    
    ColoredSphereDetector() {    
        pixel_color_classifier.color_class_map = config.color_class_map;
        pixel_color_classifier.colorful_threshold = config.colorful_threshold;
        pixel_color_classifier.color_likelihood_threshold = config.color_likelihood_threshold;
        
        circle_detector.bounding_box_ratio_threshold = config.bounding_box_ratio_threshold;
        circle_detector.min_circle_radius = config.min_circle_radius;
        circle_detector.max_circle_radius = config.max_circle_radius;
        circle_detector.component_area_ratio_threshold = config.component_area_ratio_threshold;
        circle_detector.circular_fill_ratio_threshold = config.circular_fill_ratio_threshold;
        
        sphere_detector.focal_length = config.focal_length;
        sphere_detector.image_center = config.image_center;
        sphere_detector.inlier_percentage_threshold = config.inlier_percentage_threshold;
        sphere_detector.min_sphere_radius = config.min_sphere_radius;
        sphere_detector.max_sphere_radius = config.max_sphere_radius;
        sphere_detector.min_points_for_fitting = config.min_points_for_fitting;
        sphere_detector.ransac_model_distance_threshold = config.ransac_model_distance_threshold;
    }
    
    virtual ~ColoredSphereDetector() {
    }
    
    std::vector<ColoredSphereDetection> rgbd_callback(const cv::Mat& color_input, const cv::Mat& depth_input, const cv::Matx33f& camera_matrix) {
        
        cv::Mat rgb_input;
        cv::cvtColor(color_input, rgb_input, CV_BGR2RGB);
        cv::Point2f focal_length(camera_matrix(0, 0), camera_matrix(1, 1));
        cv::Point2f image_center(camera_matrix(0, 2), camera_matrix(1, 2));
        config.focal_length = focal_length;
        config.image_center = image_center;
        
        return this->detect(rgb_input, depth_input);
        
    }
    
    std::vector<ColoredSphereDetection> detect(const cv::Mat& rgb_input, const cv::Mat& depth_input) {
        // Assumes rgb_input is has channels RGB in order
        assert(rgb_input.channels() == 3);
        
        // Trim down input image by margin, median blur, convert to float if needed
        cv::Rect roi(config.margin_x, config.margin_y, rgb_input.cols - config.margin_x, rgb_input.rows - config.margin_y);
        cv::Mat rgb_image = rgb_input(roi).clone();
        cv::medianBlur(rgb_image, rgb_image, 5);
        if (rgb_image.depth() != CV_32F) {
            rgb_image.convertTo(rgb_image, CV_32F);
        }
        
        cv::Mat color_classified_image = 
                pixel_color_classifier.classifyPixelColors(rgb_image,  config.color_class_map, config.colorful_threshold, config.color_likelihood_threshold);
        
//        cv::Mat class_and_components;
//        if (config.visualize) {
//            class_and_components = color_classified_image.clone();
//        }
        
        std::vector<ColoredCircleDetection> colored_circles;
        for (const std::pair<Color, Gaussian<float, 3>>& entry : config.color_class_map) {
            // For each color class, compute color mask and run connected components on mask
            
            Color color = entry.first;
            cv::Mat color_mask = color_classified_image == toInteger(color);
            std::vector<CircleDetector::CircleDetection> circles = circle_detector.detectCircles(
                    color_mask, config.bounding_box_ratio_threshold, config.min_circle_radius, 
                    config.max_circle_radius, config.circular_fill_ratio_threshold, config.component_area_ratio_threshold);
            
            for (const CircleDetector::CircleDetection& circle : circles) {
                colored_circles.emplace_back(circle, color);
            }
            
//            if (visualize) {
//                cv::rectangle(class_and_components, bb_roi, toInteger(color));
//                cv::putText(class_and_components, 
//                        "Fill: " + std::to_string(circular_fill_ratio).substr(0, 4), 
//                        cv::Point(bb_x, bb_y - 2), cv::FONT_HERSHEY_PLAIN, 
//                        0.6, toInteger(color));
//                cv::putText(class_and_components, 
//                        "Area: " + std::to_string(component_area_ratio).substr(0, 4), 
//                        cv::Point(bb_x, bb_y + bb_height + 6), cv::FONT_HERSHEY_PLAIN, 
//                        0.6, toInteger(color));
//            }
            
        }
        
         if (config.visualize) {
            
            cv::Mat output = rgb_input.clone();
            if (output.type() != CV_8UC3) {
                output.convertTo(output, CV_8UC3);
            }
            
            for (const ColoredCircleDetection& detection : colored_circles) {
                cv::Vec3f colorvec = 255*config.color_class_map.at(detection.color).getMean();
                cv::circle(output, cv::Point2f(detection.x, detection.y), detection.radius, 
                        cv::Scalar(colorvec[0], colorvec[1], colorvec[2]), 2, 1);
                cv::putText(output, toString(detection.color) + ", " + std::to_string(detection.radius), 
                        cv::Point2i(detection.x - detection.radius, detection.y - detection.radius - 3), 
                        cv::FONT_HERSHEY_PLAIN, 0.6, cv::Scalar(colorvec[0], colorvec[1], colorvec[2]));
            }
            
            cv::cvtColor(output, output, CV_RGB2BGR);
//            cv::imshow("Color Classification & Components", SphereDetector::imagesc(class_and_components));
            cv::imshow("Detections", output);
            cv::waitKey(1);
        }
        
        std::vector<ColoredSphereDetection> colored_spheres;
        colored_spheres.reserve(colored_circles.size());
        
        for (const ColoredCircleDetection& colored_circle : colored_circles) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr points = 
                    this->reprojectToCloud(colored_circle.locations, depth_input, config.focal_length, config.image_center);
            
            if (points->size() > config.min_points_for_fitting) {
                
                SphereDetector::SphereDetection sphere = sphere_detector.fitSphericalModelRANSAC(
                        points, config.min_sphere_radius, config.max_sphere_radius, config.ransac_model_distance_threshold);
                
                if (sphere.confidence > config.inlier_percentage_threshold) {
                    colored_spheres.emplace_back(sphere, colored_circle.color);
                }
                
            }
            
        }
        
        colored_sphere_detections = colored_spheres;
        
        return colored_spheres;
        
    }
    
    static std::vector<cv::Point3f> reproject(const std::vector<cv::Point2i>& pixel_locations, 
            const cv::Mat& depth_image, const cv::Point2f& focal_length, const cv::Point2f& center) {
        
        const float& fx = focal_length.x;
        const float& fy = focal_length.y;
        const float& cx = center.x;
        const float& cy = center.y;
        
        std::vector<cv::Point3f> points;
        points.reserve(pixel_locations.size());
        
        for (const cv::Point2i& pixel : pixel_locations) {
            
            const float& z = depth_image.at<float>(pixel.y, pixel.x);
            if (not std::isnan(z)) {
                float x = z*(pixel.x - cx)/fx;
                float y = z*(pixel.y - cy)/fy;
                points.emplace_back(x, y, z);
            }
            
        }
        
        return points;
        
    }
    
    static pcl::PointCloud<pcl::PointXYZ>::Ptr reprojectToCloud(const std::vector<cv::Point2i>& pixel_locations, 
            const cv::Mat& depth_image, const cv::Point2f& focal_length, const cv::Point2f& image_center) {
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->is_dense = true;
        cloud->reserve(pixel_locations.size());
        
        for (const cv::Point2i& pixel : pixel_locations) {
            
            const float& z = depth_image.at<float>(pixel.y, pixel.x);
            if (not std::isnan(z)) {
                float x = z*(pixel.x - image_center.x)/focal_length.x;
                float y = z*(pixel.y - image_center.y)/focal_length.y;
                cloud->points.emplace_back(x, y, z);
            }
            
        }
        
        return cloud;
        
    }
    
    static pcl::PointCloud<pcl::PointXYZ>::Ptr reprojectToCloudParallelized(
            const std::vector<cv::Point2i>& pixel_locations, const cv::Mat& depth_image, 
            const cv::Point2f& focal_length, const cv::Point2f& image_center) {
        // Not yet working
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->is_dense = false;
        cloud->reserve(pixel_locations.size());
        cv::Mat locations_mat(pixel_locations, false); // convert from vector without copy
        
        locations_mat.forEach<cv::Point2i>(
            [&focal_length, &image_center, &depth_image, &cloud](const cv::Point2i& pixel, const int* position) -> void {
                size_t index = position[0];
            
                const float& z = depth_image.at<float>(pixel.y, pixel.x);
                float x = z*(pixel.x - image_center.x)/focal_length.x;
                float y = z*(pixel.y - image_center.y)/focal_length.y;
                cloud->points.emplace(cloud->begin() + index, x, y, z);
//                cloud[index] = pcl::PointXYZ(x, y, z);
                
            }
            
        );
        
        return cloud;
    }
    
    const std::vector<ColoredSphereDetection>& getColoredSphereDetections() {
        return colored_sphere_detections;
    }
    
private:
    
    static constexpr double pi = std::acos(-1.0);
    
    std::vector<ColoredSphereDetection> colored_sphere_detections;
    
    static size_t toInteger(Color color) {
        return static_cast<size_t>(color);
    }
    
    static Color toColor(size_t integer) {
        return static_cast<Color>(integer);
    }
    
    static std::string toString(Color color) {
        
        switch(color) {
            
            case Color::RED:
                return "red";
                break;
            case Color::GREEN:
                return "green";
                break;
            case Color::BLUE:
                return "blue";
                break;
            case Color::ORANGE:
                return "orange";
                break;
            case Color::YELLOW:
                return "yellow";
                break;
            case Color::OTHER:
                return "other";
                break;
            default:
                throw std::invalid_argument("Invalid color to string conversion requested!");
                
        }
        
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
    
    static cv::Mat imagesc(const cv::Mat& input) {
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