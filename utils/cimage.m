%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cimage - Image utils
%
%   Functions:
%       im_new = to_rgb(im)             - Convert gray-scale image to RGB
%       im_new = to_gray(im)            - Convert RGB image to gray-scale
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef cimage
  
methods (Static)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function im_new = to_rgb(im)
    % Change a image to RGB image
    %   the pixel value is 0~255, the order is RGB
    %
    % Parameters:
    %   im          - input image
    %
    % Output:
    %   im_new      - RGB image
    %

    % check image dimension
    dims = size(im);
    if( length(dims) == 3 )        
        im_new = im;
        return;
    elseif( length(dims) ~= 2 )
        error('input image format is wrong!\n');
    end

    img_w = dims(2);
    img_h = dims(1);

    im_new = zeros(img_h, img_w, 3);
    im_new(:, :, 1) = im;
    im_new(:, :, 2) = im;
    im_new(:, :, 3) = im;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function im_new = to_gray(im)
    % Change a image to gray-scale image
    %   the pixel value is 0~255
    %
    % Parameters:
    %   im          - input image (RGB or gray-scale)
    %
    % Output:
    %   im_new      - gray-scale image
    %

    % check image dimension
    dims = size(im);
    if( length(dims) == 2 )
        im_new = im;
        return;
    elseif( length(dims) ~= 3 )
        error('input image format is wrong!\n');
    end

    im_new = rgb2gray(im);
end



end % end of methods
end % end of classdef

