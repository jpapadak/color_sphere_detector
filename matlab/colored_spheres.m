clear;
close all;

rgb = imread('/home/jpapadak/Desktop/spheres.png');
[rows, cols, channels] = size(rgb);
pixels = double(reshape(rgb, [rows*cols, channels]));
pixels_normalized = pixels/255.0;
colors = [ ...
    0 0;
    -0.895103510000000, 0.445858360000000; % red 
    0.999003890000000, -0.0446237210000000; % green
    0.465293020000000, -0.885156690000000; % blue
    -0.0646888170000000, 0.997905490000000; % yellow
    -0.784423710000000, 0.620225310000000; % orange
    ];
ortho_projection = [ 
    % perpendicular vectors in the (1, 1, 1) plane
    -0.707106781186548, 0.707106781186548, 0; 
    0.408248290463863, 0.408248290463863, -0.816496580927726
    ];

projected_pixels = (ortho_projection*pixels_normalized')';
pixel_magnitudes = sqrt(sum(projected_pixels.^2, 2));

projected_pixels(pixel_magnitudes < .12, :) = 0;

[vals, indices] = max(colors*projected_pixels');

classified_pixels = reshape(indices', [rows, cols, 1]);


hold on; 
plot(projected_pixels(:, 1), projected_pixels(:, 2), '.');
plot(colors(:, 1), colors(:, 2), 'ro');
hold off;
axis equal;

figure;
imagesc(classified_pixels)