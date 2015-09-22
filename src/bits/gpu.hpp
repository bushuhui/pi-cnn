/** @file    gpu.h
 ** @brief   GPU helper functions.
 ** @author  Andrea Vedaldi
 **/

#ifndef VL_GPU_H
#define VL_GPU_H

#include <iostream>

#ifdef ENABLE_GPU
#include <cuda.h>
#endif


#if __CUDA_ARCH__ >= 200
#define VL_CUDA_NUM_THREADS 1024
#else
#define VL_CUDA_NUM_THREADS 512
#endif


inline int divideUpwards(int a, int b)
{
    return (a + b - 1) / b ;
}

#endif
