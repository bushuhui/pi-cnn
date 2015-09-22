#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <base/utils/utils_str.h>
#include <base/utils/utils_misc.h>
#include <base/time/Global_Timer.h>

#include "nn_conv.h"
#include "nn_normalize.h"
#include "nn_pool.h"
#include "nn_relu.h"
#include "bits/misc_utils.h"

#include "PI_CNN.h"

using namespace std;
using namespace pi;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// \brief remove empty item in string array
///
/// \param saIn     - [in] input string array
///
/// \return         - filtered string array
///
StringArray StringArrayRemoveEmpty(StringArray saIn)
{
    StringArray r;

    for(int i=0; i<saIn.size(); i++) {
        string s = trim(saIn[i]);
        if( s.length() > 0 ) r.push_back(saIn[i]);
    }

    return r;
}

///
/// \brief calculate best image scale factor which make imgSize/scaleFactor = <int>
///
/// \param imgSize      - [in] input image size
/// \param imgSizeNew   - [in] original scaled image size
///
/// \return             - best scale factor that imgSize/scaleFactor = <int>, and
///                            also make scaleFactor = 2^n
///
int calcBestScaleFactor(int imgSize, int imgSizeNew)
{
    float       r = 1.0 * imgSize / imgSizeNew;
    float       e, eMin;
    int         n, nBest;

    n     = 1;
    nBest = 1;
    eMin  = 1e99;

    for(int i=0; i<10; i++) {
        e = fabs(r - n);
        if( e < eMin ) {
            eMin = e;
            nBest = n;
        }

        n = n*2;
    }

    return nBest;
}


int imgScale2CPU_cpu(float  *imgIn, float  *imgOut,
                     size_t *dimIn, size_t *dimOut)
{
    int  w = dimIn[0],   h = dimIn[1],   c = dimIn[2],   n = dimIn[3];
    int nw = dimOut[1], nh = dimOut[2], nc = dimOut[0], nn = dimOut[3];

    // get scale index
    int *idxX = new int[nw], *idxY = new int[nh];

    for(int i=0; i<nw; i++) {
        idxX[i] = (int)( round(1.0 * (w-1) * i / (nw-1)) );
    }

    for(int i=0; i<nh; i++) {
        idxY[i] = (int)( round(1.0 * (h-1) * i / (nh-1)) );
    }

    // do image scale
    float *pd, *ps;

    for(int l=0; l<nn; l++) {
        for(int k=0; k<nc; k++) {
            for(int j=0; j<nh; j++) {
                ps = imgIn  + l*w*h*c    + k*w*h   + idxY[j]*w;
                pd = imgOut + l*nc*nw*nh + j*nc*nw + k;

                for(int i=0; i<nw; i++) {
                    pd[i*nc] = ps[idxX[i]];
                }
            }
        }
    }

    delete [] idxX;
    delete [] idxY;

    return 0;
}

///
/// \brief do image scale
///
/// \param imgIn        - [in] input image (GPU, CPU array) (w x h x c x n)
/// \param imgOut       - [out] scaled image (CPU only)     (c x w x h x n)
///
/// \return             -  0 : success
///                       -1 : failed
///
/// NOTE: output image only support CPU tensor
///
int imageScale(PI_Tensor<float> *imgIn, PI_Tensor<float> *imgOut)
{
    int  w = imgIn->dims[0],   h = imgIn->dims[1],   c = imgIn->dims[2],   n = imgIn->dims[3];
    int nw = imgOut->dims[1], nh = imgOut->dims[2], nc = imgOut->dims[0], nn = imgOut->dims[3];

    int *idxX, *idxY;

    PI_Tensor<float> imgTemp;

    // convert tensor to CPU if necessary
    if( imgIn->isGPU() ) {
        imgIn->toCPU(imgTemp);
        imgIn = &imgTemp;
    }

    // do image scale
    {
        idxX = new int[nw];
        idxY = new int[nh];

        // get scale index
        for(int i=0; i<nw; i++) idxX[i] = (int)( round(1.0 * (w-1) * i / (nw-1)) );
        for(int i=0; i<nh; i++) idxY[i] = (int)( round(1.0 * (h-1) * i / (nh-1)) );

        // do image scale
        float *pd, *ps;

        for(int l=0; l<nn; l++) {
            for(int k=0; k<nc; k++) {
                for(int j=0; j<nh; j++) {
                    ps = imgIn->data  + l*w*h*c    + k*w*h   + idxY[j]*w;
                    pd = imgOut->data + l*nc*nw*nh + j*nc*nw + k;

                    for(int i=0; i<nw; i++) {
                        *(pd + i*nc) = *(ps + idxX[i]);
                    }
                }
            }
        }

        delete [] idxX;
        delete [] idxY;
    }

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int PI_CNN_Layer::load(const char *fname)
{
    char buf[256];

    // clear old data
    clear();

    // load layer type
    {
        string      layerType;
        CParamArray pa;

        sprintf(buf, "%s_info", fname);
        if( 0 != pa.load(buf) ) {
            dbg_pe("Can not read file: %s\n", buf);
            return -1;
        }

        layerType = "conv";
        name      = "conv1";

        pa.s("type", layerType);
        pa.s("name", name);

        type = layerType2ID(layerType);

        if( type == POOL ) {
            poolMethod = POOL_MAX;

            string pm = "max";
            pa.s("method", pm);
            if( pm == "average" ) poolMethod = POOL_AVERAGE;
        }
    }

    switch( type ) {
    case CONV:
    {
        PI_Tensor<float> _stride, _pad;

        sprintf(buf, "%s_filters", fname);
        filters.load(buf);

        sprintf(buf, "%s_biases", fname);
        biases.load(buf);

        sprintf(buf, "%s_stride", fname);
        _stride.load(buf);

        sprintf(buf, "%s_pad", fname);
        _pad.load(buf);

        for(int i=0; i<_stride.numElements; i++) stride[i] = _stride.data[i];
        for(int i=0; i<_pad.numElements; i++)    pad[i]    = _pad.data[i];

        break;
    }

    case RELU:
        break;

    case NORMALIZE:
        sprintf(buf, "%s_param", fname);
        param.load(buf);

        break;

    case POOL:
    {
        PI_Tensor<float> _stride, _pad, _pool;

        sprintf(buf, "%s_stride", fname);
        _stride.load(buf);

        sprintf(buf, "%s_pad", fname);
        _pad.load(buf);

        sprintf(buf, "%s_pool", fname);
        _pool.load(buf);

        for(int i=0; i<_stride.numElements; i++) stride[i]   = _stride.data[i];
        for(int i=0; i<_pad.numElements; i++)    pad[i]      = _pad.data[i];
        for(int i=0; i<_pool.numElements; i++)   poolSize[i] = _pool.data[i];

        break;
    }

    case SOFTMAX:
        sprintf(buf, "%s_W", fname);
        nnW.load(buf);

        break;

    default:
        dbg_pe("Unsupported layer type! (%d)", type);
        break;
    }

    layerLoaded = 1;

    return 0;
}

int PI_CNN_Layer::save(const char *fname)
{
    char buf[256];

    // save layer type
    {
        string      layerType;
        CParamArray pa;

        layerType = layerTypeStr();
        pa.set_s("type", layerType);
        pa.set_s("name", name);

        if( type == POOL ) {
            poolMethod = POOL_MAX;

            string pm = "max";
            if( poolMethod == POOL_AVERAGE ) pm = "average";

            pa.set_s("method", pm);
        }

        sprintf(buf, "%s_info", fname);
        if( 0 != pa.save(buf) ) {
            dbg_pe("Can not read file: %s\n", buf);
            return -1;
        }
    }

    switch( type ) {
    case CONV:
    {
        PI_Tensor<float> _stride, _pad;

        sprintf(buf, "%s_filters", fname);
        filters.save(buf);

        sprintf(buf, "%s_biases", fname);
        biases.save(buf);

        _stride.resize(8, 1, 1, 1);
        _pad.resize(8, 1, 1, 1);
        for(int i=0; i<8; i++) {
            _stride.data[i] = stride[i];
            _pad.data[i] = pad[i];
        }

        sprintf(buf, "%s_stride", fname);
        _stride.save(buf);

        sprintf(buf, "%s_pad", fname);
        _pad.save(buf);

        break;
    }

    case RELU:
        break;

    case NORMALIZE:
        sprintf(buf, "%s_param", fname);
        param.save(buf);

        break;

    case POOL:
    {
        PI_Tensor<float> _stride, _pad, _pool;

        _stride.resize(8, 1, 1, 1);
        _pad.resize(8, 1, 1, 1);
        _pool.resize(8, 1, 1, 1);

        for(int i=0; i<8; i++) {
            _stride.data[i] = stride[i];
            _pad.data[i]    = pad[i];
            _pool.data[i]   = poolSize[i];
        }

        sprintf(buf, "%s_stride", fname);
        _stride.save(buf);

        sprintf(buf, "%s_pad", fname);
        _pad.save(buf);

        sprintf(buf, "%s_pool", fname);
        _pool.save(buf);

        break;
    }

    case SOFTMAX:
        break;

    default:
        dbg_pe("Unsupported layer type! (%d)", type);
        break;
    }

    return 0;
}


int PI_CNN_Layer::print(void)
{
    printf("CNN layer: type=%s, name=%s, useGPU=%d\n",
           layerTypeStr().c_str(), name.c_str(), useGPU);

    if( type == CONV ) {
        printf("    filter   : [%d %d %d %d]\n",
                                filters.dims[0], filters.dims[1],
                                filters.dims[2], filters.dims[3]);
        printf("    biases   : [%d %d]\n", biases.dims[0], biases.dims[1]);
        printf("    stride   : %d %d\n", stride[0], stride[1]);
        printf("    pad      : %d %d %d %d\n", pad[0], pad[1],
                                               pad[2], pad[3]);
    } else if ( type == RELU ) {

    } else if ( type == NORMALIZE ) {
        if( !param.isEmpty() )
            printf("    param    : %f %f %f %f\n",
                                    param.data[0], param.data[1],
                                    param.data[2], param.data[3]);
    } else if ( type == POOL ) {
        printf("    stride   : %d %d\n", stride[0], stride[1]);
        printf("    pad      : %d %d %d %d\n", pad[0], pad[1],
                                               pad[2], pad[3]);
        printf("    method   : %s\n", poolMethod==POOL_MAX ? "max" : "average");
        printf("    pool     : %d %d\n", poolSize[0], poolSize[1]);
    }

    return 0;
}


std::string PI_CNN_Layer::layerType2Str(LayerType lt)
{
    string layerType = "none";

    if( type == NONE )              layerType = "none";
    else if( type == CONV )         layerType = "conv";
    else if( type == RELU )         layerType = "relu";
    else if( type == NORMALIZE )    layerType = "normalize";
    else if( type == POOL )         layerType = "pool";
    else if( type == SOFTMAX )      layerType = "softmax";

    return layerType;
}

PI_CNN_Layer::LayerType PI_CNN_Layer::layerType2ID(const std::string &s)
{
    LayerType lt = NONE;

    if( s == "none" )               lt = NONE;
    else if( s == "conv" )          lt = CONV;
    else if( s == "relu" )          lt = RELU;
    else if( s == "normalize" )     lt = NORMALIZE;
    else if( s == "pool" )          lt = POOL;
    else if( s == "softmax" )       lt = SOFTMAX;

    return lt;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

const pi::ru32      PI_CNN_MagicNum = 0x4589;


int PI_CNN::load(const char *modelFN)
{
    string              fname;
    CParamArray         pa;

    StringArray         sa;
    string              s_temp;

    // release old data
    release();

    // save CNN model file
    cnnModelName = modelFN;

    // get network configure
    {
        fname = fmt::sprintf("%s_info", modelFN);
        if( 0 != pa.load(fname) ) {
            dbg_pe("Can not open file: %s\n", fname.c_str());
        }

        pa.i("layers",      layerNum);
        pa.i("keepAspect",  cnnKeepAspect);

        s_temp = "0 0";
        pa.s("border", s_temp);
        sa = StringArrayRemoveEmpty(split_text(s_temp, " "));
        cnnBorder[0] = str_to_int(trim(sa[0]));
        cnnBorder[1] = str_to_int(trim(sa[1]));

        s_temp = "224 224 3";
        pa.s("imageSize", s_temp);
        sa = StringArrayRemoveEmpty(split_text(s_temp, " "));
        cnnPatchW = str_to_int(trim(sa[0]));
        cnnPatchH = str_to_int(trim(sa[1]));
        cnnPatchC = str_to_int(trim(sa[2]));

        // load average image
        fname = fmt::sprintf("%s_averageImage", modelFN);
        cnnAverageImage.load(fname.c_str());

        // create layers & results array
        if( layerNum > 0 ) {
            cnnLayers.resize(layerNum);
            cnnRes.resize(layerNum+1);
        }
    }

    // load each layer
    for(int layer = 0; layer < layerNum; layer ++) {
        // layer configure base name
        fname = fmt::sprintf("%s_l%d", modelFN, layer+1);

        // load layer config
        PI_CNN_Layer &l = cnnLayers[layer];
        l.load(fname.c_str());
    }

    // set loaded flag
    cnnModelLoaded = 1;

    return 0;
}

int PI_CNN::save(const char *modelFN)
{
    string              fname;
    CParamArray         pa;
    string              s_temp;

    if( useGPU ) {
        dbg_pe("Please convert CNN model to GPU array first!");
        return -1;
    }

    // save network configure
    {
        pa.set_i("layers",     layerNum);
        pa.set_i("keepAspect", cnnKeepAspect);

        s_temp = fmt::sprintf("%d %d", cnnBorder[0], cnnBorder[1]);
        pa.set_s("border", s_temp);

        s_temp = fmt::sprintf("%d %d %d", cnnPatchW, cnnPatchH, cnnPatchC);
        pa.set_s("imageSize", s_temp);

        // save average image
        fname = fmt::sprintf("%s_averageImage", modelFN);
        cnnAverageImage.save(fname.c_str());

        // save to network configure file
        fname = fmt::sprintf("%s_info", modelFN);
        if( 0 != pa.save(fname) ) {
            dbg_pe("Can not save to file: %s\n", fname.c_str());
        }
    }

    // save each layer
    for(int layer = 0; layer < layerNum; layer ++) {
        // layer configure base name
        fname = fmt::sprintf("%s_l%d", modelFN, layer+1);

        // load layer config
        PI_CNN_Layer &l = cnnLayers[layer];
        l.save(fname.c_str());
    }

    return 0;
}

int PI_CNN::loadFast(const char *modelFN)
{
    // release old data
    release();

    // load from file
    FILE *fp = fopen(modelFN, "rb");

    if( fp == NULL ) {
        dbg_pe("Can not open file: %s", modelFN);
        return -1;
    }

    size_t fileLength = filelength(fp);
    uint8_t *buf = new uint8_t[fileLength];

    fread(buf, fileLength, 1, fp);
    fclose(fp);

    // parse data stream
    RDataStream ds(buf, fileLength);

    ru32 magic, ver;
    ds.getHeader(magic, ver);

    if( magic != PI_CNN_MagicNum ) {
        dbg_pe("Input file format error! magic number = %08x, filename = %s", magic, modelFN);
        if( buf != NULL ) delete [] buf;
        return -2;
    }

    fromStream(ds);

    if( buf != NULL ) delete [] buf;

    return 0;
}

int PI_CNN::saveFast(const char *modelFN)
{
    // predict memory size
    size_t memsize = 0;
    memsize += 1024 + cnnAverageImage.numElements * sizeof(float);
    for(int i=0; i<layerNum; i++) memsize += cnnLayers[i].size();

    // generate data stream
    RDataStream ds;
    ds.reserve(memsize);
    ds.setHeader(PI_CNN_MagicNum, 1);

    toStream(ds);

    // save to file
    FILE *fp = fopen(modelFN, "wb");
    if( fp == NULL ) {
        dbg_pe("Can not open file: %s", modelFN);
        return -1;
    }

    fwrite(ds.data(), ds.size(), 1, fp);

    fclose(fp);

    return 0;
}



int PI_CNN::print(void)
{
    printf("============================================================\n");

    printf("CNN mode: %s\n", cnnModelName.c_str());
    printf("    layers      = %d\n", layerNum);
    printf("    calcToLayer = %d\n", calcToLayer);
    printf("    keepAspect  = %d\n", cnnKeepAspect);
    printf("    border      = %d %d\n", cnnBorder[0], cnnBorder[1]);
    printf("    imageSize   = %d %d %d\n", cnnPatchW, cnnPatchH, cnnPatchC);
    printf("\n");

    printf("averageImage: "); cnnAverageImage.print(); printf("\n");

    for(int i=0; i< layerNum; i++) {
        PI_CNN_Layer &l = cnnLayers[i];

        printf("layer[%3d] ", i+1);  l.print(); printf("\n");
    }

    printf("============================================================\n");

    return 0;
}



int PI_CNN::cnnForward(cv::Mat *img)
{
    timer.enter("PI_CNN::cnnForward");

    // check CNN model loaded or not
    if( !cnnModelLoaded ) {
        dbg_pe("CNN model have not loaded yet!\n");
        return -1;
    }

    // check and prepare input image
    imgW  = img->cols;
    imgH  = img->rows;
    imgC  = img->channels();

    cv::Mat imgColor;

    if( imgC != 3 ) {
        cv::cvtColor(*img, imgColor, cv::COLOR_GRAY2RGB);       // convert gray image to color (RGB)
    } else {
        cv::cvtColor(*img, imgColor, cv::COLOR_BGR2RGB);        // convert BGR image to RGB image
    }
    img = &imgColor;


    // calculate image patch number
    imgPatchW = (int)( ceil( 1.0*(imgW - cnnPatchW)/(cnnPatchW - imgOverlapSize) ) ) + 1;
    imgPatchH = (int)( ceil( 1.0*(imgH - cnnPatchH)/(cnnPatchH - imgOverlapSize) ) ) + 1;

    {
        cnnFM_offX.resize(imgPatchW);
        cnnFM_offY.resize(imgPatchH);

        for(int i=0; i<imgPatchW; i++) {
            int offX = i * (cnnPatchW-imgOverlapSize);
            if( offX + cnnPatchW > imgW ) offX = imgW - cnnPatchW;
            cnnFM_offX[i] = offX;
        }

        for(int i=0; i<imgPatchH; i++) {
            int offY = i * (cnnPatchH-imgOverlapSize);
            if( offY + cnnPatchH > imgH ) offY = imgH - cnnPatchH;
            cnnFM_offY[i] = offY;
        }
    }

    // create CNN input data & convert to it
    PI_Tensor<float> &imgT = cnnRes[0];
    imgT.resize(cnnPatchW, cnnPatchH, cnnPatchC, imgPatchW*imgPatchH,
                useGPU);

    if( useGPU ) {
        imgConvert2CNN_gpu(img->data, imgT.data, cnnAverageImage.data,
                           imgW, imgH,
                           cnnPatchW, cnnPatchH, cnnPatchC,
                           imgPatchW, imgPatchH,
                           imgOverlapSize);
    } else {
        imgConvert2CNN_cpu(img->data, imgT.data, cnnAverageImage.data,
                           imgW, imgH,
                           cnnPatchW, cnnPatchH, cnnPatchC,
                           imgPatchW, imgPatchH,
                           imgOverlapSize);
    }

    // call CNN forward
    cnnForwardCore();

    timer.leave("PI_CNN::cnnForward");

    return 0;
}


int PI_CNN::cnnForwardCore(void)
{
    int fmN = 0;
    cnnFeatureMapDepth = 0;

    timer.enter("PI_CNN::cnnForwardCore");

    for(int li=0; li<calcToLayer; li++) {
        //printf("\n");
        //dbg_pt("layer [%3d] =========================================begin!", li);

        PI_Tensor<float>    *xin  = &(cnnRes[li]);
        PI_Tensor<float>    *xout = &(cnnRes[li+1]);
        PI_CNN_Layer        *l    = &(cnnLayers[li]);

        //printf("CNN xin[%3d] : ", li); xin->print();  printf("\n");
        //printf("CNN xout[%3d]: ", li); xout->print(); printf("\n");

        switch(l->type) {
        case PI_CNN_Layer::CONV:
            timer.enter("nn_conv");
            nn_conv(l, xin, xout);
            timer.leave("nn_conv");
            break;

        case PI_CNN_Layer::RELU:
            timer.enter("nn_relu");
            nn_relu(l, xin, xout);
            timer.leave("nn_relu");
            break;

        case PI_CNN_Layer::NORMALIZE:
            timer.enter("nn_normalize");
            nn_normalize(l, xin, xout);
            timer.leave("nn_normalize");
            break;

        case PI_CNN_Layer::POOL:
            timer.enter("nn_pool");
            nn_pool(l, xin, xout);
            timer.leave("nn_pool");
            break;

        default:
            dbg_pw("Unsupported layer type: %s\n", l->layerTypeStr().c_str());
            break;
        }

        // copy feature map
        std::set<int>::iterator fmi = cnnFeatureMapLayer.find(li+1);
        if( fmi != cnnFeatureMapLayer.end() ) {
            timer.enter("PI_CNN::cnnForwardCore::fmConvert");

            PI_Tensor<float> &fm = cnnFeatureMap[fmN];

            if( 0 ) {
                // get best scale factor
                // FIXME: only support 224 CNN kernel
                int scaleFactor = calcBestScaleFactor(cnnAverageImage.width, xout->width);
                int bestSize = cnnAverageImage.width / scaleFactor;
                cnnFMScaleFactor[fmN] = scaleFactor;

                // create feature map & scale it
                //  NOTE: feature map storage order is [C x nW x nH x N]
                fm.resize(xout->depth, bestSize, bestSize, xout->dims[3]);
                imageScale(xout, &fm);
            } else {
                // [C x W x H x N]
                fm.resize(xout->dims[2], xout->dims[0], xout->dims[1], xout->dims[3]);

                if( xout->isGPU() ) {
                    featureMapConvert_gpu(xout->data, fm.data, xout->dims);
                } else {
                    featureMapConvert_cpu(xout->data, fm.data, xout->dims);
                }
            }

            cnnFeatureMapDepth += fm.dims[0];
            fmN ++;

            timer.leave("PI_CNN::cnnForwardCore::fmConvert");
        }

        //dbg_pt("=============================================end!");
        //printf("\n\n");
    }

    timer.leave("PI_CNN::cnnForwardCore");

    return 0;
}

int PI_CNN::getFeatures(std::vector<int> *arrX, std::vector<int> *arrY,
                PI_Tensor<float> *feas)
{
    timer.enter("PI_CNN::getFeatures");

    // create output feature map
    int npoint = arrX->size();
    feas->resize(cnnFeatureMapDepth, npoint, 1, 1);

    // for each point
    for(int p=0; p<npoint; p++) {
        int x = arrX->at(p), y = arrY->at(p);
        int px, py, xx, yy, patchIdx, patchSize;
        int ix, iy;

        px = x / cnnPatchW;
        py = y / cnnPatchH;
        xx = x - cnnFM_offX[px];
        yy = y - cnnFM_offY[py];
        patchIdx = py*imgPatchW + px;

        float *pD = feas->data + p*cnnFeatureMapDepth;

        // for each feature map
        for(int fi=0; fi<cnnFeatureMapLayer.size(); fi++) {
            PI_Tensor<float> &fm = cnnFeatureMap[fi];

            int fmD = fm.dims[0], fmW = fm.dims[1], fmH = fm.dims[2];
            patchSize = fmD * fmW * fmH;

            //ix = xx / cnnFMScaleFactor[fi];
            //iy = yy / cnnFMScaleFactor[fi];

            ix = (int)( round(1.0 * (fmW-1) * xx / (cnnPatchW -1)) );
            iy = (int)( round(1.0 * (fmH-1) * yy / (cnnPatchH -1)) );

            float *pS = fm.data + patchIdx*patchSize + fmD * (iy*fmW + ix);

            memcpy_fast(pD, pS, sizeof(float)*fmD);
            pD += fmD;
        }
    }

    timer.leave("PI_CNN::getFeatures");

    return 0;
}

int PI_CNN::getFeatures(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor)
{
    timer.enter("PI_CNN::getFeatures");

    // create output feature map
    int npoint = keypoints.size();
    descriptor.create(npoint, cnnFeatureMapDepth, CV_32FC1);
    float *pDescriptor = (float*) descriptor.data;

    // for each point
    for(int p=0; p<npoint; p++) {
        cv::KeyPoint &kp = keypoints[p];

        int x = (int)(kp.pt.x + 0.5), y = (int)(kp.pt.y + 0.5);
        int px, py, xx, yy, patchIdx, patchSize;
        int ix, iy;

        if( x < 0 ) x = 0; if( x > imgW ) x = imgW - 1;
        if( y < 0 ) y = 0; if( y > imgH ) y = imgH - 1;

        if( 0 ) {
            px = x / (cnnPatchW - imgOverlapSize);
            py = y / (cnnPatchH - imgOverlapSize);
            xx = x - cnnFM_offX[px];
            yy = y - cnnFM_offY[py];

            if( xx > imgOverlapSize/2 ) if( px > 0 ) px --;
            if( yy > imgOverlapSize/2 ) if( py > 0 ) py --;
        } else {
            px = (x - imgOverlapSize/2) / (cnnPatchW - imgOverlapSize);
            py = (y - imgOverlapSize/2) / (cnnPatchH - imgOverlapSize);

            if( px >= imgPatchW ) px --;
            if( py >= imgPatchH ) py --;
        }

        xx = x - cnnFM_offX[px];
        yy = y - cnnFM_offY[py];

        /*
        printf("p[%5d] x, y = %5d, %5d, px, py = %4d, %4d,  xx, yy = %5d, %5d\n",
               p, x, y, px, py, xx, yy);
        */


        patchIdx = py*imgPatchW + px;

        float *pD = pDescriptor + p*cnnFeatureMapDepth;

        // for each feature map
        for(int fi=0; fi<cnnFeatureMapLayer.size(); fi++) {
            PI_Tensor<float> &fm = cnnFeatureMap[fi];

            int fmD = fm.dims[0], fmW = fm.dims[1], fmH = fm.dims[2];
            patchSize = fmD * fmW * fmH;

            //ix = xx / cnnFMScaleFactor[fi];
            //iy = yy / cnnFMScaleFactor[fi];

            ix = (int)( round(1.0 * (fmW-1) * xx / (cnnPatchW -1)) );
            iy = (int)( round(1.0 * (fmH-1) * yy / (cnnPatchH -1)) );

            float *pS = fm.data + patchIdx*patchSize + fmD * (iy*fmW + ix);

            memcpy_fast(pD, pS, sizeof(float)*fmD);
            pD += fmD;
        }
    }

    timer.leave("PI_CNN::getFeatures");

    return 0;
}
