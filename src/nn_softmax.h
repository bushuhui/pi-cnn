#ifndef __NN_SOFTMAX_H__
#define __NN_SOFTMAX_H__

#include "PI_Tensor.h"
#include "PI_CNN.h"

int nn_softmax(PI_CNN_Layer *l, PI_Tensor<float> *xin, PI_Tensor<float> *xout);


#endif // end of __NN_SOFTMAX_H__

