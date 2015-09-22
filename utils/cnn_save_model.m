%  Save matconvnet CNN model to binary files
%     1. Display network configures
%     2. Save network configure to file
%
%  The code use the matconvnet, please download it from 
%       http://www.vlfeat.org/matconvnet/
%

% save Matlab data (0) or C data order (1)
dataOrder = 1;

% CNN model name
cnn_model_name = 'imagenet-vgg-f';

% save file path
save_path = '../data';
save_fname_base = fullfile(save_path, cnn_model_name);

% CNN model settings (please change the CNN model path to your path)
cnn_model_path = './cnn_models';
cnn_model = sprintf('%s.mat', cnn_model_name);

% load pre-trained CNN net
net = load(fullfile(cnn_model_path, cnn_model));


% load & preprocess an image
im_ = imread('../data/test_01.jpg');
im = cimage.to_rgb(im_);
im_ = single(im); % note: 255 range
im_ = imresize(im_, net.normalization.imageSize(1:2), 'nearest');
imwrite(im_/255, [save_fname_base, '_img.png']);

% reload the image
im_ = single(imread([save_fname_base, '_img.png']));
imCNN = im_ - net.normalization.averageImage;

% run the CNN
res = vl_simplenn(net, imCNN);


% get network layer number
n = numel(net.layers) ;


% save CNN results
for i=1:n+1
    fname = sprintf('%s_l%d_res', save_fname_base, i-1);

    if( dataOrder == 0 )
        cfile.save_farrayEx(fname,  res(i).x);
    else
        cfile.save_farrayEx(fname,  permute(res(i).x, [2, 1, 3]) );
    end
end


% save model normalization
modelInfo = struct();
modelInfo.layers        = num2str(n);
modelInfo.keepAspect    = num2str(net.normalization.keepAspect);
modelInfo.border        = num2str(net.normalization.border);
modelInfo.imageSize     = num2str(net.normalization.imageSize);
modelInfo.interpolation = num2str(net.normalization.interpolation);
cfile.save_struct(sprintf('%s_info', save_fname_base), modelInfo);

fname = sprintf('%s_averageImage', save_fname_base);
if( dataOrder == 0 )
    cfile.save_farrayEx(fname, net.normalization.averageImage);
else
    cfile.save_farrayEx(fname, permute(net.normalization.averageImage, [2, 1, 3]) );
end


% display & save CNN model
fprintf('CNN model: %s\n\n', cnn_model_name);

% for each layer
for i=1:n
    l = net.layers{i} ;

    % get layer file name
    fn_base = sprintf('%s_l%d_', save_fname_base, i);

    % generate layer info 
    layerInfo = struct();
    layerInfo.type = l.type;
    layerInfo.name = l.name;

    switch l.type
        case 'conv'
            fprintf('layer [%3d]:\n', i);
            fprintf('    type   : %s\n', l.type);
            fprintf('    name   : %s\n', l.name);
            
            fprintf('    filters: [%s (%s)]\n', num2str(size(l.filters)), class(l.filters));
            fprintf('    biases : [%s (%s)]\n', num2str(size(l.biases)), class(l.biases));
            fprintf('    stride : %s\n', num2str(l.stride));
            fprintf('    pad    : [%s]\n', num2str(l.pad));
            
            fprintf('    ----------------------------\n');
            fprintf('    input  : [%s (%s)]\n', num2str(size(res(i).x)), class(res(i).x));
            fprintf('    output : [%s (%s)]\n', num2str(size(res(i+1).x)), class(res(i+1).x));

            stride  = l.stride;
            pad     = l.pad;
            filters = l.filters;

            if( dataOrder == 1 ) 
                stride  = [stride(2), stride(1)];
                pad     = [pad(3), pad(4), pad(1), pad(2)];
                filters = permute(l.filters, [2, 1, 3, 4]);
            end

            cfile.save_farrayEx([fn_base, 'stride'],    stride);
            cfile.save_farrayEx([fn_base, 'pad'],       pad);
            cfile.save_farrayEx([fn_base, 'biases'],    l.biases);
            cfile.save_farrayEx([fn_base, 'filters'],   filters);

            %res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, ...
            %                       'pad', l.pad, 'stride', l.stride) ;

        case 'pool'
            fprintf('layer [%3d]:\n', i);
            fprintf('    type   : %s\n', l.type);
            fprintf('    name   : %s\n', l.name);

            fprintf('    stride : %s\n', num2str(l.stride));
            fprintf('    pad    : [%s]\n', num2str(l.pad));
            fprintf('    method : %s\n', l.method);
            fprintf('    pool   : %s\n', num2str(l.pool));

            fprintf('    ----------------------------\n');
            fprintf('    input  : [%s (%s)]\n', num2str(size(res(i).x)), class(res(i).x));
            fprintf('    output : [%s (%s)]\n', num2str(size(res(i+1).x)), class(res(i+1).x));
            
            
            layerInfo.method = l.method;

            stride  = l.stride;
            pad     = l.pad;
            pool    = l.pool;

            if( dataOrder == 1 ) 
                stride  = [stride(2), stride(1)];
                pad     = [pad(3), pad(4), pad(1), pad(2)];
                pool    = [pool(2), pool(1)];
            end
            
            cfile.save_farrayEx([fn_base, 'stride'],    stride);
            cfile.save_farrayEx([fn_base, 'pad'],       pad);
            cfile.save_farrayEx([fn_base, 'pool'],      pool);


            %res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
            %                       'pad', l.pad, 'stride', l.stride, 'method', l.method) ;

        case 'normalize'
            fprintf('layer [%3d]:\n', i);
            fprintf('    type   : %s\n', l.type);
            fprintf('    name   : %s\n', l.name);
            
            fprintf('    param  : %s\n', num2str(l.param));

            fprintf('    ----------------------------\n');
            fprintf('    input  : [%s (%s)]\n', num2str(size(res(i).x)), class(res(i).x));
            fprintf('    output : [%s (%s)]\n', num2str(size(res(i+1).x)), class(res(i+1).x));
            
            
            cfile.save_farrayEx([fn_base, 'param'],     l.param);


            %res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;

        case 'softmax'
            fprintf('layer [%3d]:\n', i);
            fprintf('    type   : %s\n', l.type);
            fprintf('    name   : %s\n', l.name);
            
            fprintf('    ----------------------------\n');
            fprintf('    input  : [%s (%s)]\n', num2str(size(res(i).x)), class(res(i).x));
            fprintf('    output : [%s (%s)]\n', num2str(size(res(i+1).x)), class(res(i+1).x));


            %res(i+1).x = vl_nnsoftmax(res(i).x) ;

        case 'loss'
            fprintf('layer [%3d]:\n', i);
            fprintf('    type   : %s\n', l.type);
            fprintf('    name   : %s\n', l.name);
            
            fprintf('    class  : %d\n', length(l.class));

            fprintf('    ----------------------------\n');
            fprintf('    input  : [%s (%s)]\n', num2str(size(res(i).x)), class(res(i).x));
            fprintf('    output : [%s (%s)]\n', num2str(size(res(i+1).x)), class(res(i+1).x));

            
            %res(i+1).x = vl_nnloss(res(i).x, l.class) ;

        case 'softmaxloss'
            fprintf('layer [%3d]:\n', i);
            fprintf('    type   : %s\n', l.type);
            fprintf('    name   : %s\n', l.name);

            fprintf('    class  : %d\n', length(l.class));

            fprintf('    ----------------------------\n');
            fprintf('    input  : [%s (%s)]\n', num2str(size(res(i).x)), class(res(i).x));
            fprintf('    output : [%s (%s)]\n', num2str(size(res(i+1).x)), class(res(i+1).x));


            %res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;

        case 'relu'
            fprintf('layer [%3d]:\n', i);
            fprintf('    type   : %s\n', l.type);
            fprintf('    name   : %s\n', l.name);
            
            fprintf('    ----------------------------\n');
            fprintf('    input  : [%s (%s)]\n', num2str(size(res(i).x)), class(res(i).x));
            fprintf('    output : [%s (%s)]\n', num2str(size(res(i+1).x)), class(res(i+1).x));


            %res(i+1).x = vl_nnrelu(res(i).x) ;

        case 'noffset'
            fprintf('layer [%3d]:\n', i);
            fprintf('    type   : %s\n', l.type);
            fprintf('    name   : %s\n', l.name);
            fprintf('    param  : %s\n', num2str(l.param));

            fprintf('    ----------------------------\n');
            fprintf('    input  : [%s (%s)]\n', num2str(size(res(i).x)), class(res(i).x));
            fprintf('    output : [%s (%s)]\n', num2str(size(res(i+1).x)), class(res(i+1).x));


            %res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;

        case 'dropout'
            fprintf('layer [%3d]:\n', i);
            fprintf('    type   : %s\n', l.type);
            fprintf('    name   : %s\n', l.name);
            
            fprintf('    rate   : %s\n', num2str(l.rate));
            
            fprintf('    ----------------------------\n');
            fprintf('    input  : [%s (%s)]\n', num2str(size(res(i).x)), class(res(i).x));
            fprintf('    output : [%s (%s)]\n', num2str(size(res(i+1).x)), class(res(i+1).x));


            cfile.save_farrayEx([fn_base, 'rate'],      l.rate);


            %{
            if opts.disableDropout
                res(i+1).x = res(i).x ;
            elseif opts.freezeDropout
                [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux) ;
            else
                [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
            end
            %}
            
        case 'custom'
            fprintf('layer [%3d]:\n', i);
            fprintf('    type   : %s\n', l.type);
            fprintf('    name   : %s\n', l.name);


            %res(i+1) = l.forward(l, res(i), res(i+1)) ;
        otherwise
            error('Unknown layer type %s', l.type) ;
    end

    fprintf('\n');

    % save layer information to file
    cfile.save_struct([fn_base, 'info'], layerInfo);

    % save input & output data
    if( dataOrder == 0 )
        cfile.save_farrayEx([fn_base, 'xin'],  res(i).x);
        cfile.save_farrayEx([fn_base, 'xout'], res(i+1).x);
    else
        cfile.save_farrayEx([fn_base, 'xin'],  permute(res(i).x,   [2, 1, 3]) );
        cfile.save_farrayEx([fn_base, 'xout'], permute(res(i+1).x, [2, 1, 3]) );
    end

end
