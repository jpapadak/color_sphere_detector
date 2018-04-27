clear;
close all;

[I, ~, alpha] = imread('green_real.png');
[rows, cols, channels] = size(I);
subplot(1, 2, 1), imshow(I);

pixels = double(reshape(I, [rows*cols, 3]));
pixels(alpha == 0, :) = 0;
pixels = pixels/255;

ortho_projection = [ 
    % perpendicular vectors in the (1, 1, 1) plane
    -0.707106781186548, 0.707106781186548, 0; 
    0.408248290463863, 0.408248290463863, -0.816496580927726
];

proj_pixels = (ortho_projection*pixels')';
radius = sqrt(sum(proj_pixels.^2, 2));
angles = atan2(proj_pixels(:, 2), proj_pixels(:, 1));
colorful_threshold = .04;

Icolorful = pixels;
Icolorful(radius < colorful_threshold, :) = 0;
Icolorful = reshape(Icolorful, [rows, cols, 3]);
subplot(1, 2, 2), imshow(Icolorful, []);

colorful_pixels = pixels(radius > colorful_threshold, :);
colorful_proj_pixels = proj_pixels(radius > colorful_threshold, :);

color_mean = mean(colorful_pixels)
color_cov = cov(colorful_pixels)
color_radius = radius(radius > colorful_threshold);
color_angles = angles(radius > colorful_threshold);

avg = mean(colorful_proj_pixels)
var = cov(colorful_proj_pixels)
figure, plot(colorful_proj_pixels(:,1), colorful_proj_pixels(:,2), '.', 'color', color_mean), hold on; plot_gaussian_ellipsoid(avg, var), axis([-1, 1, -1, 1]);

% red, 0.6920, 0.2688, 0.2720, mu 2.6243, var 2.6838e-05
% green, 0.2264, 0.4306, 0.3068, mu 0.1280, var 6.2498e-04
% blue, 0.1520, 0.3446, 0.5130, mu -1.0072, var 7.1333e-05
% yellow, 0.7520, 0.7206, 0.2898, mu 1.6315, var 1.1887e-05
% orange, 0.6528, 0.2496, 0.1616, mu 2.4483, var 2.8503e-05

