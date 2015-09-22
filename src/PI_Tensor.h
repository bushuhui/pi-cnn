#ifndef __PI_TENSOR_H__
#define __PI_TENSOR_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef ENABLE_GPU
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <helper_cuda.h>            // helper functions for CUDA error checking and initialization
#endif

#include <base/Svar/DataStream.h>
#include <base/debug/debug_config.h>

#include "bits/misc_utils.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class T>
class PI_Tensor
{
    #define dimNMax 8               // maximum dimension size

public:
    PI_Tensor() { init(); }
    PI_Tensor(int gpuMode) {
        init();
        useGPU = gpuMode;
    }
    virtual ~PI_Tensor() { release(); }

    int clear(void) {
        release();
        return 0;
    }


    ///
    /// \brief copy from Tensor
    ///
    /// \param s            - [in] source tensor
    /// \param toGPU        - [in] use GPU (1) or not (0)
    ///
    /// \return             -   0 : success
    ///                        -1 : failed
    ///
    int copyFrom(PI_Tensor<T> *s, int toGPU = 0) {
        resize(s->dimN, s->dims, toGPU);

        if( isEmpty() ) return -1;

        if( toGPU == 0 && s->isGPU() == toGPU ) {
            memcpy(data, s->data, numElements*sizeof(T));
        } else {
#ifdef ENABLE_GPU
            if( toGPU == 1 && s->isGPU() == 1 ) {
                checkCudaErrors( cudaMemcpy(data, s->data, numElements*sizeof(T), cudaMemcpyDeviceToDevice) );
            } else if ( toGPU == 1 && s->isGPU() == 0 ) {
                checkCudaErrors( cudaMemcpy(data, s->data, numElements*sizeof(T), cudaMemcpyHostToDevice) );
            } else if ( toGPU == 0 && s->isGPU() == 1 ) {
                checkCudaErrors( cudaMemcpy(data, s->data, numElements*sizeof(T), cudaMemcpyDeviceToHost) );
            }
#else
            dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
            return -1;
#endif
        }

        return 0;
    }


    int isEmpty(void) {
        if( dimN == 0 || data == NULL ) return 1;
        else return 0;
    }

    int resize(int ds, size_t *d, int gpuMode = 0) {
        // check new size is equal to old size
        if( gpuMode == useGPU && data != NULL ) {
            bool e = true;

            for(int i=0; i<ds; i++)
                if( d[i] != dims[i] ) e = false;

            if( e ) return 0;
        }

        // set and update dimensions
        dimN = ds;
        for(int i=0; i<dimN; i++) dims[i] = d[i];

        updateDims();
        malloc(gpuMode);

        return 0;
    }

    int resize(size_t h, size_t w, size_t d, size_t s=1, int gpuMode = 0) {
        size_t da[4];

        da[0] = h;
        da[1] = w;
        da[2] = d;
        da[3] = s;

        resize(4, da, gpuMode);

        return 0;
    }


    void fill(T v) {
        if( data == NULL ) return;

        if( useGPU ) {
            fill_array_gpu(data, v, numElements);
        } else {
            fill_array_cpu(data, v, numElements);
        }
    }

    void ones(void) {
        fill(1.0);
    }

    void zeros(void) {
        fill(0.0);
    }


    int setGPUMode(int isGPU) {
        if( isGPU ) return toGPU();
        else        return toCPU();
    }

    int isGPU(void) {
        return useGPU;
    }


    int toCPU(void) {
        if( data == NULL ) {
            useGPU = 0;
            return 0;
        }

        if( useGPU ) {
#ifdef ENABLE_GPU
            T *newData = new T[numElements];
            checkCudaErrors( cudaMemcpy(newData, data, numElements*sizeof(T), cudaMemcpyDeviceToHost) );
            checkCudaErrors( cudaFree(data) );
            data = newData;
#else
            dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
            return -1;
#endif
        }

        useGPU = 0;

        return 0;
    }

    int toCPU(PI_Tensor<T> &t) {
        t.resize(dimN, dims);
        if( isEmpty() ) return 0;

        if( useGPU ) {
#ifdef ENABLE_GPU
            checkCudaErrors( cudaMemcpy(t.data, data, numElements*sizeof(T), cudaMemcpyDeviceToHost) );
#else
            dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
            return -1;
#endif
        } else {
            memcpy(t.data, data, numElements*sizeof(T));
        }

        return 0;
    }

    int toGPU(void) {
        if( data == NULL ) {
            useGPU = 1;
            return 0;
        }

        if( !useGPU ) {
#ifdef ENABLE_GPU
            T *newData;

            checkCudaErrors( cudaMalloc((void**) &newData, sizeof(T)*numElements) );
            checkCudaErrors( cudaMemcpy(newData, data, numElements*sizeof(T), cudaMemcpyHostToDevice) );

            delete [] data;
            data = newData;
#else
            dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
            return -1;
#endif
        }

        useGPU = 1;

        return 0;
    }

    int toGPU(PI_Tensor<T> &t) {
#ifdef ENABLE_GPU
        t.resize(dimN, dims, 1);
        if( isEmpty() ) return 0;

        if( !useGPU ) {
            checkCudaErrors( cudaMemcpy(t.data, data, numElements*sizeof(T), cudaMemcpyHostToDevice) );
        } else {
            checkCudaErrors( cudaMemcpy(t.data, data, numElements*sizeof(T), cudaMemcpyDeviceToDevice) );
        }

        return 0;
#else
        dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
        return -1;
#endif
    }


    int toStream(pi::RDataStream &ds) {
        // write dimension number & each dimension
        ds.write(dimN);
        for(int i=0; i<dimNMax; i++) ds.write(dims[i]);

        // write array
        if( !isEmpty() ) ds.write((uint8_t*) data, sizeof(T)*numElements);

        return 0;
    }

    int fromStream(pi::RDataStream &ds) {
        // clear old contents
        clear();

        // read dimension number & each dimension
        ds.read(dimN);
        for(int i=0; i<dimNMax; i++) ds.read(dims[i]);

        updateDims();
        malloc();

        // read array
        if( !isEmpty() ) ds.read((uint8_t*) data, sizeof(T)*numElements);

        return 0;
    }

    ///
    /// \brief load Tensor from file
    ///
    /// \param fname        - [in] data file name
    ///
    /// \return             -   0 : success
    ///                        -1 : can not open file
    ///
    int load(const char *fname) {
        FILE        *fp = NULL;

        // free old resource
        clear();

        // open data file
        fp = fopen(fname, "rb");
        if( fp == NULL ) {
            dbg_pe("Can not open file: %s\n", fname);
            return -1;
        }

        // read dimN
        fread(&dimN, sizeof(size_t), 1, fp);
        if( dimN > dimNMax ) {
            dbg_pe("Input data file dimenson is too large! (%d)\n", dimN);
            return -2;
        }

        // read each dimension
        fread(dims, sizeof(size_t), dimN, fp);

        // set tensor sizes & alloc data array
        // FIXME: when useGPU is on then, this fail
        updateDims();
        malloc();

        // read data array
        fread(data, sizeof(T), numElements, fp);

        fclose(fp);

        return 0;
    }

    ///
    /// \brief save Tensor to file
    ///
    /// \param fname        - [in] data file name
    ///
    /// \return             -   0 : success
    ///                        -1 : can not open file
    ///
    int save(const char *fname) {
        FILE        *fp = NULL;

        // open file
        fp = fopen(fname, "wb");
        if( fp == NULL ) {
            dbg_pe("Can not open file: %s\n", fname);
            return -1;
        }

        // write dimN
        fwrite(&dimN, sizeof(size_t), 1, fp);

        // write each dimension
        for(int i=0; i<dimN; i++)
            fwrite(&dims[i], sizeof(size_t), 1, fp);

        // write array
        // FIXME: when useGPU is on, this will fail
        fwrite(data, sizeof(T), numElements, fp);

        fclose(fp);

        return 0;
    }

    int loadDims(const char *fname) {
        FILE        *fp = NULL;

        // free old resource
        clear();

        // open data file
        fp = fopen(fname, "rb");
        if( fp == NULL ) {
            dbg_pe("Can not open file: %s\n", fname);
            return -1;
        }

        // read dimN
        fread(&dimN, sizeof(size_t), 1, fp);
        if( dimN > dimNMax ) {
            dbg_pe("Input data file dimenson is too large! (%d)\n", dimN);
            return -2;
        }

        // read each dimension
        fread(dims, sizeof(size_t), dimN, fp);

        // set tensor sizes & alloc data array
        // FIXME: when useGPU is on then, this fail
        updateDims();

        fclose(fp);

        return 0;
    }

    int print(void) {
        printf("PI_Tensor, dimN = %lld, useGPU = %d\n", dimN, useGPU);
        printf("    dims        = ");
        for(int i=0; i<dimN; i++) printf("%lld ", dims[i]);
        printf("\n");

        printf("    height      = %lld\n", height);
        printf("    width       = %lld\n", width);
        printf("    depth       = %lld\n", depth);
        printf("    size        = %lld\n", size);
        printf("    numElements = %lld\n", numElements);
        printf("    sizeof(T)   = %d\n", sizeof(T));
        printf("    data        = 0x%016llX\n", data);
    }

public:
    size_t          dimN;                       ///< dimension size
    size_t          dims[dimNMax];              ///< each dimension size

    size_t          height;                     ///< 1st dimension (height, row)
    size_t          width;                      ///< 2nd dimension (width, column)
    size_t          depth;                      ///< 3rd dimension (color, or filters)
    size_t          size;                       ///< 4th dimension (instance index)
    size_t          numElements;                ///< totoal data number

    T               *data;                      ///< totoal data


protected:
    int             useGPU;                     ///< use GPU array or not


protected:
    void init(void) {
        useGPU = 0;

        dimN = 0;
        for(int i=0; i<dimNMax; i++) dims[i] = 1;
        updateDims();

        data        = NULL;
    }

    void release(void) {
        free();
        init();
    }

    int malloc(int gpuMode = 0) {
        // free old array
        free();

        useGPU = gpuMode;

        if( dimN == 0 ) return 0;

        //dbg_pt("gpuMode = %d, numElements = %d", gpuMode, numElements);

        // create new array        
        if( useGPU ) {
#ifdef ENABLE_GPU
            checkCudaErrors( cudaMalloc((void**) &data, sizeof(T)*numElements) );
#else
            dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
            return -1;
#endif
        } else {
            data = new T[numElements];
            //data = ::malloc(sizeof(T)*numElements);
        }

        return 0;
    }

    void free(void) {
        //dbg_pt("data = 0x%016llX, useGPU = %d", data, useGPU);

        if( data == NULL ) return;

        // free old array
        if( useGPU ) {
#ifdef ENABLE_GPU
            checkCudaErrors( cudaFree(data) );
            data = NULL;
#else
            dbg_pe("Please enable GPU by set defination: ENABLE_GPU");
#endif

        } else {
            delete [] data;
            //::free(data);
            data = NULL;
        }
    }

    void updateDims(void) {
        height = (dimN >= 1) ? dims[0] : 1;
        width  = (dimN >= 2) ? dims[1] : 1;
        depth  = (dimN >= 3) ? dims[2] : 1;
        size   = (dimN >= 4) ? dims[3] : 1;

        if( dimN == 0 ) {
            numElements = 0;
        } else {
            numElements = 1;
            for(int i=0; i<dimN; i++) numElements *= dims[i];
        }
    }
};

#endif // end of __PI_ARRAY_H__
