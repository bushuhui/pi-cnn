###############################################################################
###############################################################################
CC                  = gcc
CXX                 = g++

ENABLE_GPU         ?= true

###############################################################################
###############################################################################
DEFAULT_CFLAGS      = -D_GNU_SOURCE -O3 -DNDEBUG -fPIC
DEFAULT_CFLAGS     += -g -rdynamic
DEFAULT_LDFLAGS     = -lstdc++ -lpthread 

#DEFAULT_CFLAGS     += -fopenmp -pthread
#DEFAULT_LDFLAGS    += -fopenmp


################################################################################
# OpenCV settings
# run following command first:
#   export PKG_CONFIG_PATH=/opt/opencv-2.4/lib/pkgconfig
################################################################################
OPENCV_CFLAGS       = $(shell pkg-config --cflags opencv)
OPENCV_LDFLAGS      = $(shell pkg-config --libs   opencv) 


###############################################################################
###############################################################################
CUDA_DIR            = /usr/local/cuda
CUDA_NVCC           = $(CUDA_DIR)/bin/nvcc
CUDA_CFLAGS         = -I$(CUDA_DIR)/include 
CUDA_LDFLAGS        = -L$(CUDA_DIR)/lib64 -lcublas -lcudart


###############################################################################
###############################################################################
OPENBLAS_DIR        = /opt/OpenBLAS
OPENBLAS_CFLAGS     = -I$(OPENBLAS_DIR)/include 
OPENBLAS_LDFLAGS    = -L$(OPENBLAS_DIR)/lib -lopenblas

###############################################################################
###############################################################################
PIL_DIR             = ./Thirdparty/PIL
PIL_CFLAGS          = -I$(PIL_DIR)/src -DPIL_LINUX 
PIL_LDFLAGS         = -L$(PIL_DIR)/libs -lpi_base \
                      -Wl,-rpath=$(PIL_DIR)/libs

###############################################################################
###############################################################################
CUDAUTILS_DIR       = ./Thirdparty/cuda_utils_5.0
CUDAUTILS_CFLAGS    = -I$(CUDAUTILS_DIR)/inc


###############################################################################
###############################################################################
LIBS_CFLAGS         = $(OPENBLAS_CFLAGS)  $(PIL_CFLAGS)  $(OPENCV_CFLAGS)
LIBS_LDFLAGS        = $(OPENBLAS_LDFLAGS) $(PIL_LDFLAGS) $(OPENCV_LDFLAGS)

CFLAGS              = $(DEFAULT_CFLAGS)  $(LIBS_CFLAGS)
LDFLAGS             = $(DEFAULT_LDFLAGS) $(LIBS_LDFLAGS)


ifneq ($(ENABLE_GPU),)

LIBS_CFLAGS        += $(CUDAUTILS_CFLAGS)
LIBS_LDFLAGS       += $(CUDAUTILS_LDFLAGS)

CFLAGS             +=  $(CUDA_CFLAGS) $(CUDAUTILS_CFLAGS) \
                        -DENABLE_GPU 
LDFLAGS            += $(CUDA_LDFLAGS) $(CUDAUTILS_LDFLAGS)


CUFLAGS             = -ccbin $(CXX) \
                        --compiler-options=-fPIC \
                        $(LIBS_CFLAGS) \
                        -DENABLE_GPU 

CUFLAGS			   += -arch=sm_21 
endif


###############################################################################
###############################################################################

cpp_src             = src/bits/im2col.cpp src/bits/pooling.cpp src/bits/normalize.cpp src/bits/subsample.cpp
cpp_src            += src/PI_Tensor.cpp src/PI_CNN.cpp src/cnnFMextractor.cpp

ifeq ($(ENABLE_GPU),)
cpp_src            += src/nn_conv.cpp src/nn_relu.cpp src/nn_normalize.cpp src/nn_pool.cpp src/nn_softmax.cpp
cpp_src            += src/bits/misc_utils.cpp
else
cpp_src            += src/nn_conv_gpu.cu src/nn_relu_gpu.cu src/nn_normalize_gpu.cu src/nn_pool_gpu.cu src/nn_softmax_gpu.cu
cpp_src            += src/bits/im2col_gpu.cu src/bits/pooling_gpu.cu src/bits/normalize_gpu.cu \
                      src/bits/subsample_gpu.cu src/bits/misc_utils_gpu.cu
endif

cpp_tgt            := $(patsubst %.cpp,%.o,$(cpp_src))
cpp_tgt            := $(patsubst %.cu,%.o,$(cpp_tgt))


target              = test_CNN libpi_cnn.so

###############################################################################
###############################################################################

all : $(cpp_tgt) $(target)


%.o : %.cpp
	$(CXX) -c $? -o $(@) $(CFLAGS) 

# CUDA codes
ifneq ($(ENABLE_GPU),)
%.o : %.cu
	$(CUDA_NVCC) -c $? -o $(@) $(CUFLAGS)
endif


libpi_cnn.so : $(cpp_tgt)
	$(CXX) -o $@ $(cpp_tgt) -shared $(LDFLAGS)
	
test_CNN : test_CNN.cpp libpi_cnn.so 
	$(CXX) $< -o $(@) $(CFLAGS) $(LDFLAGS) -L. -lpi_cnn -Wl,-rpath=.


clean:
	rm -f $(target) src/*.o src/bits/*.o 

