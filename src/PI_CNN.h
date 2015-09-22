#ifndef __PI_CNN_H__
#define __PI_CNN_H__

#include <string>
#include <vector>
#include <map>
#include <set>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "PI_Tensor.h"


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// \brief The PI_CNN_Layer class
///
class PI_CNN_Layer
{
public:
    ///
    /// \brief The supported layer type enum
    ///
    enum LayerType {
        NONE,                   ///< none layer
        CONV,                   ///< Convolution layer
        RELU,                   ///< RELU layer
        NORMALIZE,              ///< Normalize layer
        POOL,                   ///< Pooling layer
        SOFTMAX,                ///< Softmax layer
        LOSS,                   ///< Loss layer
        SOFTMAXLOSS,            ///< Softmax loss layer
        NOFFSET,                ///< noffset layer
        DROPOUT                 ///< Dropout layer
    };

    ///
    /// \brief The pooling method enum
    ///
    enum PoolMethod {
        POOL_MAX,               ///< maximum pooling
        POOL_AVERAGE            ///< average pooling
    };


public:
    PI_CNN_Layer() { init(); }
    virtual ~PI_CNN_Layer() { release(); }

    int clear(void) {
        release();
        return 0;
    }

    int isGPU(void) {
        return useGPU;
    }

    int setMode(int isGPU) {
        /*
        dbg_pt("type=%s, name=%s, layerLoaded = %d, gpuMode = %d",
               layerTypeStr().c_str(), name.c_str(),
               layerLoaded, isGPU);
        */

        if( layerLoaded ) {
            if( isGPU ) {
                filters.toGPU();
                biases.toGPU();
            } else {
                filters.toCPU();
                biases.toCPU();
            }
        }

        useGPU = isGPU;

        return 0;
    }

    int load(const char *fname);
    int save(const char *fname);

    int print(void);

    size_t size(void) {
        size_t s = 1024 + (filters.numElements + biases.numElements)*sizeof(float);
        return s;
    }

    std::string layerType2Str(LayerType lt);
    LayerType   layerType2ID(const std::string &s);
    std::string layerTypeStr(void) { return layerType2Str(type); }

    int toStream(pi::RDataStream &ds) {
        // write layer info
        pi::ri32 lt = (pi::ri32) type;
        ds.write(lt);
        ds.write(name);

        // write filters & biases
        filters.toStream(ds);
        biases.toStream(ds);

        // write stride & pad
        for(int i=0; i<8; i++) ds.write(stride[i]);
        for(int i=0; i<8; i++) ds.write(pad[i]);

        // write parameters
        param.toStream(ds);

        // write pool
        pi::ri32 pm = (pi::ri32) poolMethod;
        ds.write(pm);
        for(int i=0; i<8; i++) ds.write(poolSize[i]);

        return 0;
    }

    int fromStream(pi::RDataStream &ds) {
        // read layer info
        pi::ri32 lt;
        ds.read(lt);  type = (LayerType) lt;
        ds.read(name);

        // read filters & biases
        filters.fromStream(ds);
        biases.fromStream(ds);

        // read stride & pad
        for(int i=0; i<8; i++) ds.read(stride[i]);
        for(int i=0; i<8; i++) ds.read(pad[i]);

        // read parameters
        param.fromStream(ds);

        // read pool
        pi::ri32 pm;
        ds.read(pm); poolMethod = (PoolMethod) pm;
        for(int i=0; i<8; i++) ds.read(poolSize[i]);

        // set flags
        useGPU = 0;
        layerLoaded = 1;
    }

public:
    LayerType           type;               ///< layer type
    std::string         name;               ///< layer name

    PI_Tensor<float>    filters;            ///< filters
    PI_Tensor<float>    biases;             ///< biases
    PI_Tensor<float>    convAllOnes;        ///< all one tensor (for convolution)
    PI_Tensor<float>    convTemp;           ///< temp tensor (for convolution)

    int32_t             stride[8];          ///< stride
    int32_t             pad[8];             ///< pad

    PI_Tensor<float>    param;              ///< normalize parameters

    PoolMethod          poolMethod;         ///< pool method
    int32_t             poolSize[8];        ///< pool size


    PI_Tensor<float>    nnW, nnBiases;      ///< NN weight & bias

protected:
    int32_t             useGPU;             ///< use GPU or not
    int32_t             layerLoaded;        ///< layer configures loaded or not

protected:
    void init(void) {
        useGPU = 0;
        layerLoaded = 0;

        type = NONE;
        name = "";

        for(int i=0; i<8; i++) {
            stride[i]   = 0;
            pad[i]      = 0;
            poolSize[i] = 0;
        }

        poolMethod = POOL_MAX;
        poolSize[0] = 3;
        poolSize[1] = 3;
    }

    void release(void) {
        init();

        filters.clear();
        biases.clear();
        param.clear();
    }
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

typedef std::vector< PI_Tensor<float> > CNN_FeatureMap;     ///< CNN feature map container

///
/// \brief The PI_CNN class
///
class PI_CNN
{
public:
    PI_CNN() { init(); }
    virtual ~PI_CNN() {
        release();
    }

    int clear(void) {
        release();
        return 0;
    }

    int isGPU(void) {
        return useGPU;
    }

    int setMode(int isGPU) {
        if( cnnModelLoaded ) {
            if( isGPU ) cnnAverageImage.toGPU();
            else        cnnAverageImage.toCPU();

            for(int i=0; i<layerNum; i++) cnnLayers[i].setMode(isGPU);
        } else {
            dbg_pe("CNN model has not loaded yet!");
        }

        useGPU = isGPU;

        return 0;
    }

    int load(const char *modelFN);
    int save(const char *modelFN);
    int loadFast(const char *modelFN);
    int saveFast(const char *modelFN);


    int print(void);

    int getLayerNum(void) { return layerNum; }

    int cnnForward(cv::Mat *img);

    int setFeatureMapLayer(std::set<int> *fmLayer) {
        cnnFeatureMapLayer = *fmLayer;

        // get calculate CNN layer number
        std::set<int>::iterator it;
        calcToLayer = 0;
        for(it=cnnFeatureMapLayer.begin(); it!=cnnFeatureMapLayer.end(); it++) {
            int l = *it;
            if( l > calcToLayer ) calcToLayer = l;
        }

        // resize feature map array
        cnnFeatureMap.resize(cnnFeatureMapLayer.size());
        cnnFMScaleFactor.resize(cnnFeatureMapLayer.size());

        return 0;
    }

    int getFeatureMapLayer(std::set<int> *fmLayer) {
        *fmLayer = cnnFeatureMapLayer;
        return 0;
    }

    int setFeatureMapLayer(std::vector<int> *fmLayer) {
        // copy feature map layers
        {
            cnnFeatureMapLayer.clear();
            std::vector<int>::iterator it;

            for(it=fmLayer->begin(); it!=fmLayer->end(); it++) {
                int l = *it;
                cnnFeatureMapLayer.insert(l);
            }
        }

        // get calculate CNN layer number
        {
            std::set<int>::iterator it;
            calcToLayer = 0;
            for(it=cnnFeatureMapLayer.begin(); it!=cnnFeatureMapLayer.end(); it++) {
                int l = *it;
                if( l > calcToLayer ) calcToLayer = l;
            }
        }

        // resize feature map array
        cnnFeatureMap.resize(cnnFeatureMapLayer.size());
        cnnFMScaleFactor.resize(cnnFeatureMapLayer.size());

        return 0;
    }

    int getFeatureMapLayer(std::vector<int> *fmLayer) {
        fmLayer->clear();

        for(int i=0; i<layerNum; i++) {
            std::set<int>::iterator it;
            it = cnnFeatureMapLayer.find(i+1);
            if( it != cnnFeatureMapLayer.end() ) fmLayer->push_back(i+1);
        }

        return 0;
    }


    CNN_FeatureMap* getFeatureMap(void) {
        return &cnnFeatureMap;
    }

    CNN_FeatureMap* getResults(void) {
        return &cnnRes;
    }

    int getModelLoaded(void) {
        return cnnModelLoaded;
    }

    int getFeatures(std::vector<int> *arrX, std::vector<int> *arrY, PI_Tensor<float>  *feas);
    int getFeatures(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor);


    int toStream(pi::RDataStream &ds) {
        // write CNN model info
        ds.write(cnnModelName);
        ds.write(layerNum);
        ds.write(calcToLayer);
        ds.write(cnnKeepAspect);

        ds.write(cnnBorder[0]);
        ds.write(cnnBorder[1]);
        ds.write(cnnPatchW);
        ds.write(cnnPatchH);
        ds.write(cnnPatchC);

        // average image
        cnnAverageImage.toStream(ds);

        // each layer
        for(int i=0; i<layerNum; i++) cnnLayers[i].toStream(ds);

        return 0;
    }

    int fromStream(pi::RDataStream &ds) {
        // read CNN model info
        ds.read(cnnModelName);
        ds.read(layerNum);
        ds.read(calcToLayer);
        ds.read(cnnKeepAspect);

        ds.read(cnnBorder[0]);
        ds.read(cnnBorder[1]);
        ds.read(cnnPatchW);
        ds.read(cnnPatchH);
        ds.read(cnnPatchC);

        // average image
        cnnAverageImage.fromStream(ds);

        // read each layer
        cnnLayers.resize(layerNum);
        cnnRes.resize(layerNum+1);

        for(int i=0; i<layerNum; i++) cnnLayers[i].fromStream(ds);

        // set loaded flag
        cnnModelLoaded = 1;

        return 0;
    }

protected:
    int32_t                             useGPU;             ///< use GPU or not

    std::string                         cnnModelName;       ///< CNN model name
    int32_t                             cnnModelLoaded;     ///< model loaded or not

    int32_t                             layerNum;           ///< layer number
    int32_t                             calcToLayer;        ///< calculated to layer

    int32_t                             cnnKeepAspect;      ///< keep image aspect
    int32_t                             cnnBorder[2];       ///< borders
    PI_Tensor<float>                    cnnAverageImage;    ///< average image
                                                            ///<   for input image normalization
    int32_t                             cnnPatchW,          ///< CNN input patch width
                                        cnnPatchH,          ///< CNN input patch height
                                        cnnPatchC;          ///< CNN input patch channel

    std::vector<PI_CNN_Layer>           cnnLayers;          ///< CNN layers
    CNN_FeatureMap                      cnnRes;             ///< each layer's result

    std::set<int>                       cnnFeatureMapLayer; ///< Feature map layers
    CNN_FeatureMap                      cnnFeatureMap;      ///< Feature map
    int32_t                             cnnFeatureMapDepth; ///< Feature map dimension
    std::vector<int>                    cnnFMScaleFactor;   ///< Feature map scale factor
    std::vector<int>                    cnnFM_offX,         ///< Feature map offset x
                                        cnnFM_offY;         ///< Feature map offset y


    int32_t                             imgOverlapSize;     ///< image overlap size

    int32_t                             imgW, imgH, imgC;   ///< image width, height, channel
    int32_t                             imgPatchW,          ///< image patch size in cols
                                        imgPatchH;          ///< image patch size in rows

protected:
    void init(void) {
        useGPU          = 0;
        cnnModelLoaded  = 0;

        layerNum        = 21;
        calcToLayer     = 21;

        cnnPatchW       = 224;
        cnnPatchH       = 224;
        cnnPatchC       = 3;

        cnnKeepAspect   = 1;
        cnnBorder[0]    = 0;
        cnnBorder[1]    = 0;

        imgOverlapSize  = 64;
    }

    void release(void) {
        cnnAverageImage.clear();

        cnnLayers.clear();
        cnnRes.clear();

        cnnFeatureMapLayer.clear();
        cnnFeatureMap.clear();

        init();
    }

    int cnnForwardCore(void);
};

#endif // end of __PI_CNN_H__
