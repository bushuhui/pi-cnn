
#include "bits/misc_utils.h"
#include "nn_relu.h"


int nn_relu(PI_CNN_Layer *l, PI_Tensor<float> *xin, PI_Tensor<float> *xout)
{
    if( 0 ) {
        dbg_pt("layerType = %s, layerName = %s, useGPU = %d",
               l->layerTypeStr().c_str(), l->name.c_str(), l->isGPU());
    }

    // create output tensor
    xout->resize(xin->dimN, xin->dims, l->isGPU());

    // do RELU
    if( l->isGPU() ) {
        nn_relu_gpu(xout->data, xin->data, xin->numElements);
    } else {
        nn_relu_cpu(xout->data, xin->data, xin->numElements);
    }

    return 0;
}
