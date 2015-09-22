/** @file vl_nnpool.cu
 ** @brief Pooling block
 ** @author Andrea Vedaldi
 ** @author Karel Lenc
 **/

/*
Copyright (C) 2014 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <assert.h>

#include <base/debug/debug_config.h>

#include "bits/pooling.hpp"
#include "PI_CNN.h"
#include "nn_pool.h"


int nn_pool(PI_CNN_Layer *l, PI_Tensor<float> *xin, PI_Tensor<float> *xout)
{
    int gpuMode             = l->isGPU();

    int poolWidth           = l->poolSize[1];
    int poolHeight          = l->poolSize[0];

    int strideX             = l->stride[1];
    int strideY             = l->stride[0];

    int padLeft             = l->pad[2];
    int padRight            = l->pad[3];
    int padTop              = l->pad[0];
    int padBottom           = l->pad[1];

    PoolMethod method       = NN_POOL_MAX;

    // get parameters
    if( l->poolMethod == PI_CNN_Layer::POOL_AVERAGE ) method = NN_POOL_AVG;

    // print info
    if( 0 ) {
        dbg_pt("stride : %d %d\n", strideY, strideX);
        dbg_pt("pad    : %d %d %d %d\n", padTop, padBottom, padLeft, padRight);
        dbg_pt("method : %d\n", method);
        dbg_pt("pool   : %d %d\n", poolHeight, poolWidth);
    }

    // check parameters
    if (strideX < 1 || strideY < 1) {
        dbg_pe("At least one element of STRIDE is smaller than one.") ;
    }

    // create output tensor
    xout->resize((xin->height + (padTop+padBottom) - poolHeight)/strideY + 1,
                 (xin->width  + (padLeft+padRight) - poolWidth)/strideX + 1,
                 xin->depth,
                 xin->size,
                 gpuMode);
    //xout->fill(0);


    // run the pooling
    if (gpuMode) {
#ifdef ENABLE_GPU
        pooling_gpu<float>(xout->data,
                           xin->data,
                           method,
                           xin->height, xin->width,
                           xin->depth * xin->size,
                           poolHeight,
                           poolWidth,
                           strideY,
                           strideX,
                           padTop,
                           padBottom,
                           padLeft,
                           padRight) ;
#else
        dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
#endif
    } else {
        pooling_cpu<float>(xout->data,
                           xin->data,
                           method,
                           xin->height, xin->width,
                           xin->depth * xin->size,
                           poolHeight,
                           poolWidth,
                           strideY,
                           strideX,
                           padTop,
                           padBottom,
                           padLeft,
                           padRight) ;
    }

    return 0;
}
