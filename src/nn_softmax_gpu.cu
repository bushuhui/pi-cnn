
#include <math.h>

#include "bits/misc_utils.h"
#include "nn_softmax.h"


int nn_softmax(PI_CNN_Layer *l, PI_Tensor<float> *xin, PI_Tensor<float> *xout)
{
    if( 0 ) {
        dbg_pt("layerType = %s, layerName = %s, useGPU = %d",
               l->layerTypeStr().c_str(), l->name.c_str(), l->isGPU());
    }

    // create output tensor
    int nFea   = l->nnW.dims[0];
    int nClass = l->nnW.dims[1];
    int nData  = xin->dims[1];

    xout->resize(nData, 1, 1, 1, l->isGPU());

    // do softmax
    if( l->isGPU() ) {
        dbg_pe("Currently do not support GPU!");
        return -1;
    } else {
        PI_Tensor<float> M;
        M.resize(nClass, nData, 1, 1, l->isGPU());

        // do M = exp(w * xin)
        //    M = bsxfun(@rdivide, M, sum(M));
        //    [p,pred] = max(M, [], 1);
        #pragma omp parallel
        for(int j=0; j<nData; j++) {
            float *p1 = l->nnW.data;
            float *p2 = xin->data + j*nFea;
            float *p3 = M.data + j*nClass;

            float clsSum = 0;

            for(int i=0; i<nClass; i++) {
                float s = 0;

                for(int k=0; k<nFea; k++) {
                    s += p1[k] * p2[k];
                }

                float v = exp(s);
                clsSum += v;
                p3[i] = v;

                p1 += nFea;
            }

            float vMax = 0;
            float iMax = 0;
            for(int i=0; i<nClass; i++) {
                p3[i] /= clsSum;

                if( p3[i] > vMax ) {
                    vMax = p3[i];
                    iMax = i;
                }
            }

            xout->data[j] = iMax;
        }
    }

    return 0;
}
