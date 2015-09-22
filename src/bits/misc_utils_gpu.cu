
#include <base/debug/debug_config.h>

#ifdef ENABLE_GPU
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <helper_cuda.h>            // helper functions for CUDA error checking and initialization
#endif

#include "gpu.hpp"
#include "misc_utils.h"


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define CUDA_STRIDE         32
#define IMUL(a,b)           __mul24(a,b)


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_GPU

#if 0
__global__ void nn_relu_kernel(
        float* data_out,
        const float* data_in,
        size_t N)
{
    size_t n_elem_per_thread = (N + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    size_t block_start_idx = n_elem_per_thread * blockDim.x * blockIdx.x;

    size_t thread_start_idx = block_start_idx
            + (threadIdx.x / CUDA_STRIDE) * n_elem_per_thread * CUDA_STRIDE
            + (threadIdx.x % CUDA_STRIDE);
    size_t thread_end_idx = thread_start_idx + n_elem_per_thread * CUDA_STRIDE;
    if(thread_end_idx > N) thread_end_idx = N;

    for(size_t idx=thread_start_idx; idx < thread_end_idx; idx+=CUDA_STRIDE)
    {
        if( data_in[idx] < 0.0 ) data_out[idx] = 0.0;
        else                     data_out[idx] = data_in[idx];
    }
}
#endif

#if 1
__global__ void nn_relu_kernel(
        float* data_out,
        const float* data_in,
        size_t N)
{
    size_t idxBeg    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t blockSize = blockDim.x * gridDim.x;

    for(size_t idx=idxBeg; idx < N; idx+=blockSize)
    {
        if( data_in[idx] < 0.0 ) data_out[idx] = 0.0;
        else                     data_out[idx] = data_in[idx];
    }
}
#endif

#if 0
__global__ void nn_relu_kernel(
        float* data_out,
        const float* data_in,
        size_t N)
{
    size_t n_elem_per_thread = (N + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);

    size_t idxBeg = (blockIdx.x * blockDim.x + threadIdx.x ) * n_elem_per_thread;
    size_t idxEnd = idxBeg + n_elem_per_thread;

    if( idxBeg > N ) idxBeg = N;
    if( idxEnd > N ) idxEnd = N;

    for(size_t idx=idxBeg; idx < idxEnd; idx++)
    {
        if( data_in[idx] < 0.0 ) data_out[idx] = 0.0;
        else                     data_out[idx] = data_in[idx];
    }
}
#endif

#endif // end of ENABLE_GPU

void nn_relu_gpu(float* data_out,
                 float const* data_in,
                 size_t N)
{
#ifdef ENABLE_GPU
    int blockSize = divideUpwards(N, VL_CUDA_NUM_THREADS);
    if( blockSize > 1024 ) blockSize = 1024;

    //dbg_pt("blockSize = %d, threadSize = %d\n", blockSize, VL_CUDA_NUM_THREADS);

    nn_relu_kernel <<< blockSize, VL_CUDA_NUM_THREADS >>>
                    (data_out, data_in, N);

    if (cudaGetLastError() != cudaSuccess) {
        std::cout
            << "nn_relu_kernel error ("
            << cudaGetErrorString(cudaGetLastError())
            << ")" << std::endl ;
    }

    cudaThreadSynchronize();  // Wait all thread finish

    return;
#else
    dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
    return;
#endif
}


void nn_relu_cpu(float*         data_out,
                 float const*   data_in,
                 size_t         N)
{
    for(size_t i=0; i<N; i++) {
        if( data_in[i] < 0.0 ) data_out[i] = 0.0;
        else                   data_out[i] = data_in[i];
    }
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_GPU
template<typename T>
__global__ void fill_array_kernel(T* data_out, T v, size_t N)
{
    size_t idxBeg    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t blockSize = blockDim.x * gridDim.x;

    for(size_t idx=idxBeg; idx < N; idx+=blockSize)
        data_out[idx] = v;
}
#endif

template<typename T>
void fill_array_gpu(T *data_out, T v, size_t N)
{
#ifdef ENABLE_GPU
    int blockSize = divideUpwards(N, VL_CUDA_NUM_THREADS);
    if( blockSize > 1024 ) blockSize = 1024;

    fill_array_kernel<T> <<< blockSize, VL_CUDA_NUM_THREADS >>>
                    (data_out, v, N);

    if (cudaGetLastError() != cudaSuccess) {
        std::cout
            << "fill_array_kernel error ("
            << cudaGetErrorString(cudaGetLastError())
            << ")" << std::endl ;
    }

    cudaThreadSynchronize();  // Wait all thread finish

    return;
#else
    dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
    return;
#endif
}

template<typename T>
void fill_array_cpu(T *data_out,  T v,  size_t N)
{
    for(size_t i=0; i<N; i++) data_out[i] = v;
}


template void fill_array_gpu<float>  (float  *data_out,  float v,  size_t N);
template void fill_array_gpu<double> (double *data_out,  double v, size_t N);
template void fill_array_gpu<int>    (int    *data_out,  int v,    size_t N);

template void fill_array_cpu<float>  (float  *data_out,  float v,  size_t N);
template void fill_array_cpu<double> (double *data_out,  double v, size_t N);
template void fill_array_cpu<int>    (int    *data_out,  int v,    size_t N);



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


#ifdef ENABLE_GPU
__global__ void imgConvert2CNN_kernel(uint8_t *imgInput, float *imgOutput, float *avgImg,
                                      int imgW, int imgH,
                                      int cnnPatchW, int cnnPatchH, int cnnPatchC,
                                      int imgPatchW, int imgPatchH,
                                      int imgOverlapSize,
                                      int N)
{
    int cnnPatchBlockSize = cnnPatchW * cnnPatchH;
    int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if( pixIdx < N ) {
        int subImgIdx    = pixIdx / cnnPatchBlockSize;            // subimage index
        int subImgPixIdx = pixIdx % cnnPatchBlockSize;            // pixel index in a subimage

        float *pD = imgOutput + subImgIdx*cnnPatchBlockSize*cnnPatchC;

        // get sub-image x, y & offsets
        int subImgX = subImgIdx % imgPatchW;
        int subImgY = subImgIdx / imgPatchW;
        int offX    = subImgX * (cnnPatchW - imgOverlapSize);
        int offY    = subImgY * (cnnPatchH - imgOverlapSize);
        if( offX + cnnPatchW > imgW ) offX = imgW - cnnPatchW;
        if( offY + cnnPatchH > imgH ) offY = imgH - cnnPatchH;

        // get x,y in sub-image and x,y in input image
        int ix   = subImgPixIdx % cnnPatchW;
        int iy   = subImgPixIdx / cnnPatchW;
        int imgX = offX + ix;
        int imgY = offY + iy;

        // copy image & do normalization
        pD[0*cnnPatchBlockSize + subImgPixIdx] = (float)(imgInput[(imgY*imgW + imgX)*3 + 0]) - avgImg[0*cnnPatchBlockSize + iy*cnnPatchW + ix];
        pD[1*cnnPatchBlockSize + subImgPixIdx] = (float)(imgInput[(imgY*imgW + imgX)*3 + 1]) - avgImg[1*cnnPatchBlockSize + iy*cnnPatchW + ix];
        pD[2*cnnPatchBlockSize + subImgPixIdx] = (float)(imgInput[(imgY*imgW + imgX)*3 + 2]) - avgImg[2*cnnPatchBlockSize + iy*cnnPatchW + ix];
    }
}
#endif


void imgConvert2CNN_gpu(uint8_t *imgInput, float *imgOutput, float *avgImg,
                        int imgW, int imgH,
                        int cnnPatchW, int cnnPatchH, int cnnPatchC,
                        int imgPatchW, int imgPatchH,
                        int imgOverlapSize)
{
#ifdef ENABLE_GPU
    int N          = imgPatchW * imgPatchH * cnnPatchW * cnnPatchH;
    int blockSize  = divideUpwards(N, VL_CUDA_NUM_THREADS);

    // create device image buffer
    uint8_t *dImg;
    checkCudaErrors( cudaMalloc((void**) &dImg, sizeof(uint8_t)*imgW*imgH*cnnPatchC) );
    checkCudaErrors( cudaMemcpy(dImg, imgInput, sizeof(uint8_t)*imgW*imgH*cnnPatchC, cudaMemcpyHostToDevice) );

    // call kernel
    imgConvert2CNN_kernel <<< blockSize, VL_CUDA_NUM_THREADS >>>
                    (dImg, imgOutput, avgImg,
                     imgW, imgH,
                     cnnPatchW, cnnPatchH, cnnPatchC,
                     imgPatchW, imgPatchH,
                     imgOverlapSize,
                     N);

    if (cudaGetLastError() != cudaSuccess) {
        std::cout
            << "imgConvert2CNN_kernel error ("
            << cudaGetErrorString(cudaGetLastError())
            << ")" << std::endl ;
    }

    cudaThreadSynchronize();  // Wait all thread finish

    checkCudaErrors( cudaFree(dImg) );

    return;
#else
    dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
    return;
#endif
}

void imgConvert2CNN_cpu(uint8_t *imgInput, float *imgOutput, float *avgImg,
               int imgW, int imgH,
               int cnnPatchW, int cnnPatchH, int cnnPatchC,
               int imgPatchW, int imgPatchH,
               int imgOverlapSize)
{
    int blockCNN = cnnPatchW * cnnPatchH;

    for(int i=0; i<imgPatchW*imgPatchH; i++) {
        float *pD = imgOutput + i*blockCNN*cnnPatchC;

        int patchX = i % imgPatchW;
        int patchY = i / imgPatchW;

        int offX = patchX * (cnnPatchW-imgOverlapSize);
        int offY = patchY * (cnnPatchH-imgOverlapSize);
        if( offX + cnnPatchW > imgW ) offX = imgW - cnnPatchW;
        if( offY + cnnPatchH > imgH ) offY = imgH - cnnPatchH;

        for(int ii=0; ii<cnnPatchW*cnnPatchH; ii++) {
            int ix = ii % cnnPatchW;
            int iy = ii / cnnPatchW;

            int imgX = offX + ix;
            int imgY = offY + iy;

            pD[0*blockCNN + ii] = (float)(imgInput[(imgY*imgW + imgX)*3 + 0]) - avgImg[0*blockCNN + iy*cnnPatchW + ix];
            pD[1*blockCNN + ii] = (float)(imgInput[(imgY*imgW + imgX)*3 + 1]) - avgImg[1*blockCNN + iy*cnnPatchW + ix];
            pD[2*blockCNN + ii] = (float)(imgInput[(imgY*imgW + imgX)*3 + 2]) - avgImg[2*blockCNN + iy*cnnPatchW + ix];
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_GPU
__global__ void featureMapConvert_kernel(float *imgIn, float *imgOut,
                                         int w, int h, int c, int n)
{
    int pixIdx    = blockIdx.x * blockDim.x + threadIdx.x;
    int blockSize = w*h*c;
    int N         = blockSize*n;

    int sN   = pixIdx / blockSize;
    int sIdx = pixIdx % blockSize;
    int sY   = sIdx / (w*c);
    int sXC  = sIdx % (w*c);
    int sX   = sXC / c;
    int sC   = sXC % c;

    if( pixIdx < N ) {
        imgOut[pixIdx] = imgIn[sN*blockSize + sC*w*h + sY*w + sX];
    }
}
#endif


int featureMapConvert_gpu(float *imgIn, float *imgOut, size_t *dims)
{
#ifdef ENABLE_GPU
    int  w = dims[0],
         h = dims[1],
         c = dims[2],
         n = dims[3];

    int N          = w*h*c*n;
    int blockSize  = divideUpwards(N, VL_CUDA_NUM_THREADS);

    // create device image buffer
    float *dImgOut;
    checkCudaErrors( cudaMalloc((void**) &dImgOut, sizeof(float)*N) );

    // call kernel
    featureMapConvert_kernel <<< blockSize, VL_CUDA_NUM_THREADS >>>
                    (imgIn, dImgOut,
                     w,  h,  c,  n);

    if (cudaGetLastError() != cudaSuccess) {
        std::cout
            << "imgConvert2CNN_kernel error ("
            << cudaGetErrorString(cudaGetLastError())
            << ")" << std::endl ;
    }

    cudaThreadSynchronize();  // Wait all thread finish

    checkCudaErrors( cudaMemcpy(imgOut, dImgOut, sizeof(float)*N, cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaFree(dImgOut) );

    return 0;
#else
    dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
    return;
#endif
}

int featureMapConvert_cpu(float *imgIn, float *imgOut, size_t *dims)
{
    int  w = dims[0],
         h = dims[1],
         c = dims[2],
         n = dims[3];

    float *pd, *ps;

    for(int l=0; l<n; l++) {
        for(int k=0; k<c; k++) {
            for(int j=0; j<h; j++) {
                ps = imgIn  + l*w*h*c + k*w*h + j*w;
                pd = imgOut + l*c*w*h + j*c*w + k;

                for(int i=0; i<w; i++) pd[i*c] = ps[i];
            }
        }
    }

    return 0;
}
