/** @file gnormalize.cu
 ** @brief Normalization block
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <assert.h>

#include "bits/normalize.hpp"
#include "PI_CNN.h"


////////////////////////////////////////////////////////////////////////////////
/// normalized layer
////////////////////////////////////////////////////////////////////////////////

int nn_normalize(PI_CNN_Layer *l, PI_Tensor<float> *xin, PI_Tensor<float> *xout)
{
    int         gpuMode = l->isGPU();

    size_t      normDepth = (size_t) l->param.data[0];
    float       normKappa = l->param.data[1];
    float       normAlpha = l->param.data[2];
    float       normBeta  = l->param.data[3];

    if( 0 ) {
        printf("nn_normalize: \n");
        printf("    normDepth   = %d\n", normDepth);
        printf("    normAlpha   = %f\n", normAlpha);
        printf("    normKappa   = %f\n", normKappa);
        printf("    normBeta    = %f\n", normBeta);
    }

    // create output tensor
    xout->resize(xin->dims[0], xin->dims[1], xin->dims[2], xin->dims[3], gpuMode);


    // forward
    if (gpuMode) {
#ifdef ENABLE_GPU
        normalize_gpu<float>(xout->data,
                             xin->data,
                             xin->height, xin->width, xin->depth, xin->size,
                             normDepth, normKappa, normAlpha, normBeta) ;
#else
        dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
#endif
    } else {
        normalize_cpu<float>(xout->data,
                             xin->data,
                             xin->height, xin->width, xin->depth, xin->size,
                             normDepth, normKappa, normAlpha, normBeta) ;
    }

    return 0;
}
