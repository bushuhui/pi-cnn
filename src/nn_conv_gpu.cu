/** @file vl_nnconv.cu
 ** @brief Convolution block
 ** @author Andrea Vedaldi
 **/

/*
 Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg.
 All rights reserved.

 This file is part of the VLFeat library and is made available under
 the terms of the BSD license (see the COPYING file).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <f77blas.h>
#ifdef ENABLE_GPU
#include <cublas_v2.h>
#endif

//#include <base/time/Global_Timer.h>

#include "bits/im2col.hpp"
#include "bits/subsample.hpp"
#include "PI_CNN.h"

#ifdef ENABLE_GPU
cublasHandle_t    thisCublasHandle = NULL;
#endif

////////////////////////////////////////////////////////////////////////////////
/// Dispatcher functions
////////////////////////////////////////////////////////////////////////////////

static void
sgemv_dispatch(bool gpuMode,
               char op,
               int m, int n,
               float alpha,
               float const * a, int lda,
               float const * x, int incx,
               float beta,
               float * y, int incy)
{
    if (!gpuMode) {
        sgemv_(&op,
              &m, &n, &alpha,
              (float*)a, &lda,
              (float*)x, &incx,
              &beta,
              y, &incy) ;
    } else {
#ifdef ENABLE_GPU
        cublasSgemv(thisCublasHandle,
                    (op == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                    (int)m, (int)n,
                    &alpha,
                    a, lda,
                    x, (int)incx,
                    &beta,
                    y, (int)incy) ;
#endif
    }
}

static void
sgemm_dispatch(bool gpuMode,
               char op1, char op2,
               int m, int n, int k,
               float alpha,
               float const * a, int lda,
               float const * b, int ldb,
               float beta,
               float * c, int ldc)
{
    if (!gpuMode) {
        sgemm_(&op1, &op2,
              &m, &n, &k,
              &alpha,
              (float*)a, &lda,
              (float*)b, &ldb,
              &beta,
              c, &ldc) ;
    } else {
#ifdef ENABLE_GPU
        cublasSgemm(thisCublasHandle,
                    (op1 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                    (op2 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                    (int)m, (int)n, (int)k,
                    &alpha,
                    a, (int)lda,
                    b, (int)ldb,
                    &beta,
                    c, (int)ldc);
#endif
    }
}

static void
copy_dispatch(bool gpuMode,
              float * dest,
              float const * src,
              size_t numElements)
{
    if (!gpuMode) {
        memcpy(dest, src, numElements * sizeof(float)) ;
    } else {
#ifdef ENABLE_GPU
        cudaMemcpy(dest, src, numElements * sizeof(float), cudaMemcpyDeviceToDevice) ;
#endif
    }
}

static void
subsample_dispatch(bool gpuMode,
                   float* subsampled,
                   float const* data,
                   size_t width,
                   size_t height,
                   size_t depth,
                   size_t strideX,
                   size_t strideY,
                   size_t padLeft,
                   size_t padRight,
                   size_t padTop,
                   size_t padBottom)
{
    if (!gpuMode) {
        subsample_cpu(subsampled,
                      data,
                      width,
                      height,
                      depth,
                      strideX,
                      strideY,
                      padLeft,
                      padRight,
                      padTop,
                      padBottom) ;
    } else {
#ifdef ENABLE_GPU
        subsample_gpu(subsampled,
                      data,
                      width,
                      height,
                      depth,
                      strideX,
                      strideY,
                      padLeft,
                      padRight,
                      padTop,
                      padBottom) ;
#endif
    }
}

static void
subsampleBackward_dispatch(bool gpuMode,
                           float* dzdx,
                           float const* dzdy,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t strideX,
                           size_t strideY,
                           size_t padLeft,
                           size_t padRight,
                           size_t padTop,
                           size_t padBottom)
{
    if (!gpuMode) {
        subsampleBackward_cpu(dzdx,
                              dzdy,
                              width,
                              height,
                              depth,
                              strideX,
                              strideY,
                              padLeft,
                              padRight,
                              padTop,
                              padBottom) ;
    } else {
#ifdef ENABLE_GPU
        subsampleBackward_gpu(dzdx,
                              dzdy,
                              width,
                              height,
                              depth,
                              strideX,
                              strideY,
                              padLeft,
                              padRight,
                              padTop,
                              padBottom) ;
#endif
    }
}


static void
im2col_dispatch(bool gpuMode,
                float* stacked,
                float const* data,
                size_t width,
                size_t height,
                size_t depth,
                size_t windowWidth,
                size_t windowHeight,
                size_t strideX,
                size_t strideY,
                size_t padLeft,
                size_t padRight,
                size_t padTop,
                size_t padBottom)
{
    if (!gpuMode) {
        im2col_cpu<float>(stacked,
                          data,
                          width,
                          height,
                          depth,
                          windowWidth,
                          windowHeight,
                          strideX,
                          strideY,
                          padLeft,
                          padRight,
                          padTop,
                          padBottom) ;
    } else {
#ifdef ENABLE_GPU
        im2col_gpu<float>(stacked,
                          data,
                          width,
                          height,
                          depth,
                          windowWidth,
                          windowHeight,
                          strideX,
                          strideY,
                          padLeft,
                          padRight,
                          padTop,
                          padBottom) ;
#endif
    }
}

static void
col2im_dispatch(bool gpuMode,
                float* data,
                float const* stacked,
                size_t width,
                size_t height,
                size_t depth,
                size_t windowWidth,
                size_t windowHeight,
                size_t strideX,
                size_t strideY,
                size_t padLeft,
                size_t padRight,
                size_t padTop,
                size_t padBottom)
{
    if (!gpuMode) {
        col2im_cpu<float>(data,
                          stacked,
                          width,
                          height,
                          depth,
                          windowWidth,
                          windowHeight,
                          strideX,
                          strideY,
                          padLeft,
                          padRight,
                          padTop,
                          padBottom) ;
    } else {
#ifdef ENABLE_GPU
        col2im_gpu<float>(data,
                          stacked,
                          width,
                          height,
                          depth,
                          windowWidth,
                          windowHeight,
                          strideX,
                          strideY,
                          padLeft,
                          padRight,
                          padTop,
                          padBottom) ;
#endif
    }
}


////////////////////////////////////////////////////////////////////////////////
/// convolution layer
////////////////////////////////////////////////////////////////////////////////

int nn_conv(PI_CNN_Layer *l, PI_Tensor<float> *xin, PI_Tensor<float> *xout)
{
    PI_Tensor<float>    &filters    = l->filters;
    PI_Tensor<float>    &biases     = l->biases;
    PI_Tensor<float>    &allOnes    = l->convAllOnes;
    PI_Tensor<float>    temp        = l->convTemp;

    int                 gpuMode     = l->isGPU();

    int                 strideX     = l->stride[1],
                        strideY     = l->stride[0],
                        padTop      = l->pad[0],
                        padBottom   = l->pad[1],
                        padLeft     = l->pad[2],
                        padRight    = l->pad[3];

    int                 numGroups   = 1;


    // create cublas handle
#ifdef ENABLE_GPU
    if( gpuMode && thisCublasHandle == NULL ) {
        cublasCreate_v2(&thisCublasHandle);
    }
#endif

    // create output tensor
    xout->resize(
                (xin->height + (padTop+padBottom) - filters.height)/strideY + 1,
                (xin->width + (padLeft+padRight) - filters.width)/strideX + 1,
                filters.size,
                xin->size,
                gpuMode);

    // grouped filters
    numGroups = xin->depth / filters.depth;

    // all ones & temp tensor
    allOnes.resize(xout->height, xout->width, 1, 1, gpuMode);
    allOnes.fill(1.0);

    temp.resize(xout->height, xout->width,
                filters.height * filters.width * filters.depth * numGroups, 1,
                gpuMode);

    if( 0 ) {
        dbg_pt("layerType = %s, layerName = %s, useGPU = %d, numGroups = %d, images = %d",
               l->layerTypeStr().c_str(), l->name.c_str(), gpuMode,
               numGroups, xin->size);
    }

    // for each image
    for (int image = 0 ; image < xin->size ; ++image) {
        /*
           temp (phi(x)): m x k
           filters, derFilters: k x n (for one group of filters)
           derOutput (dzdy) : m x n (for one group of filters)
           res (y) : m x n (for one group of filters)
        */
        size_t dataOffset   = (xin->height*xin->width*xin->depth) * image ;
        size_t outputOffset = (xout->height*xout->width*xout->depth) * image ;

        int m = temp.height * temp.width ;                      // number of output pixels
        int n = filters.size/numGroups ;                        // number of filters per group
        int k = filters.height*filters.width*filters.depth ;    // filter volume

        im2col_dispatch(gpuMode,
                        temp.data,
                        xin->data + dataOffset,
                        xin->height, xin->width, xin->depth,
                        filters.height, filters.width,
                        strideY, strideX,
                        padTop, padBottom, padLeft, padRight) ;

        for (int g = 0 ; g < numGroups ; ++ g) {
            size_t filterGrpOffset = k * n * g ;
            size_t tempGrpOffset = m * k * g ;
            size_t outputGrpOffset = m * n * g  ;
            float alpha = 1 ;
            float beta = 0 ;

            sgemm_dispatch(gpuMode, 'n', 'n',
                           m, n, k,
                           alpha,
                           temp.data + tempGrpOffset, m,
                           filters.data + filterGrpOffset, k,
                           beta,
                           xout->data + outputOffset + outputGrpOffset, m) ;
        }

        {
            float alpha = 1 ;
            float beta = 1 ;
            int   q = 1 ;

            sgemm_dispatch(gpuMode, 'n', 'n',
                           m, biases.numElements, q,
                           alpha,
                           allOnes.data, m,
                           biases.data, q,
                           beta,
                           xout->data + outputOffset, m) ;
        }
    }

    return 0;
}
