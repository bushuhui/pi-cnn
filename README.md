# PI-CNN - high-performance convolutional neural network toolbox for C++

This program is a C++ toolbox for extracting CNN feature map from image. For achieving high-performance computation, it support CUDA acceleration. The average computation of 3 layers feature maps of a 640x480 image is 20 ms. You can easly integrated the code to you embedded program. The core functions are extracted from matconvnet (http://www.vlfeat.org/matconvnet/)


## Requirements:
* OpenCV 2.4.9 (or above)
* OpenBLAS (include in the package at ./Thirdparty/OpenBLAS)
* gfortran (sudo apt-get install gfortran)
* CUDA 5.0 (or above)
* PIL (included in the code at ./Thirdparty/PIL)

## Compile:

*1. build OpenBLAS*

```cd ./Thirdparty/OpenBLAS
tar xzf OpenBLAS-0.2.14.tar.gz
make 
sudo make install
```

*2. build PIL*
```
cd ./Thirdparty/PIL
make
```

*3. build pi-cnn*
```
cd cnn_models/
wget http://www.adv-ci.com/download/pi-cnn/imagenet-vgg-f.cm 
cd ..
make
```



## Usage:

```
# GPU calculation
./test_CNN useGPU=1

# CPU calculation
./test_CNN useGPU=0

# show feature maps
./test_CNN act=showFeatureMap

# match feature points
./test_CNN act=matchWholeImage
```

## Plateform:
Only test on LinuxMint 17.1 64-bit, may be other distributions are also support. 


## Screenshot:
-![alt text](http://www.adv-ci.com/blog/wp-content/uploads/2015/09/screenshot_2-275x300.png "Screenshot 1")
-![alt text](http://www.adv-ci.com/blog/wp-content/uploads/2015/09/screenshot_1-1024x559.png "Screenshot 2")



## Project homepage:
http://www.adv-ci.com/blog/source/pi-cnn/
 

