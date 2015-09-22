#ifndef __NN_RELU_H__
#define __NN_RELU_H__

#include "PI_Tensor.h"
#include "PI_CNN.h"

int nn_relu(PI_CNN_Layer *l, PI_Tensor<float> *xin, PI_Tensor<float> *xout);


#endif // end of __NN_RELU_H__
