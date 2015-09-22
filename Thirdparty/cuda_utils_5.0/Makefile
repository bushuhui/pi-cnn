###############################################################################
###############################################################################
TOPDIR := .

#inclue system configure
include $(TOPDIR)/make.conf

###############################################################################
###############################################################################

CFLAGS             += -I./inc

lib_src             =   src/multithreading.cpp 
cu_src              = 

target_obj          = $(patsubst %.cpp,%.o,$(lib_src)) $(patsubst %.cu,%.o,$(cu_src))
target              = lib/libcuda_utils.a


###############################################################################
###############################################################################

all : $(target)


# Standard code
src/%.o : src/%.cpp
	$(CXX) -c $? -o $(@) $(CFLAGS) 

src/%.o : src/%.cu
	$(CUDA_NVCC) -c $? -o $(@) $(CUFLAGS) 


$(target) : $(target_obj)
	$(AR) rcs $@ $(target_obj)

clean:
	rm -f src/*.o lib/$(target)
	
