
[rows, cols, channels] = size(maskedRGBImage);
indices = find(BW);
pixels = double(reshape(maskedRGBImage, [rows*cols, 3]));
pixels = pixels(indices, :)/255;
mean(pixels)

ortho_projection = [ 
    % perpendicular vectors in the (1, 1, 1) plane
    -0.707106781186548, 0.707106781186548, 0; 
    0.408248290463863, 0.408248290463863, -0.816496580927726
];

proj_pixels = ortho_projection*pixels';
angles = atan2(proj_pixels(2, :), proj_pixels(1, :));

avg = mean(angles)
var = cov(angles)

% red, 0.6920, 0.2688, 0.2720, mu 2.6243, var 2.6838e-05
% green, 0.2264, 0.4306, 0.3068, mu 0.1280, var 6.2498e-04
% blue, 0.1520, 0.3446, 0.5130, mu -1.0072, var 7.1333e-05
% yellow, 0.7520, 0.7206, 0.2898, mu 1.6315, var 1.1887e-05
% orange, 0.6528, 0.2496, 0.1616, mu 2.4483, var 2.8503e-05

