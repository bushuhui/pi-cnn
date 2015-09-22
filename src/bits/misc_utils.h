#ifndef __MISC_UTILS_H__
#define __MISC_UTILS_H__

#include <stdio.h>
#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void nn_relu_cpu(float* data_out,
                 float const* data_in,
                 size_t N);
void nn_relu_gpu(float* data_out,
                 float const* data_in,
                 size_t N);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<typename T> void fill_array_gpu(T *data_out,  T v,  size_t N);
template<typename T> void fill_array_cpu(T *data_out,  T v,  size_t N);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void imgConvert2CNN_cpu(uint8_t *imgInput, float *imgOutput, float *avgImg,
               int imgW, int imgH,
               int cnnPatchW, int cnnPatchH, int cnnPatchC,
               int imgPatchW, int imgPatchH,
               int imgOverlapSize);

void imgConvert2CNN_gpu(uint8_t *imgInput, float *imgOutput, float *avgImg,
               int imgW, int imgH,
               int cnnPatchW, int cnnPatchH, int cnnPatchC,
               int imgPatchW, int imgPatchH,
               int imgOverlapSize);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int featureMapConvert_cpu(float *imgIn, float *imgOut, size_t *dims);
int featureMapConvert_gpu(float *imgIn, float *imgOut, size_t *dims);


#endif // end of __MISC_UTILS_H__

