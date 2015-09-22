#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>


#ifdef ENABLE_GPU
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <helper_functions.h>       // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>            // helper functions for CUDA error checking and initialization
#include <helper_cuda_drvapi.h>     // helper functions for drivers
#endif

#include <base/time/Time.h>
#include <base/debug/debug_config.h>
#include <base/utils/utils.h>

#include "src/PI_Tensor.h"
#include "src/PI_CNN.h"
#include "src/cnnFMextractor.h"

#include "src/nn_conv.h"
#include "src/nn_relu.h"
#include "src/nn_normalize.h"
#include "src/nn_pool.h"

#include "src/bits/misc_utils.h"

using namespace pi;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int test_CUDA(CParamArray *pa)
{
    int     nTest = 10;
    int     N;
    float   *hData, *hRes1, *hRes2;
    float   *gData, *gRes1;

    N = 1 << 24;

    // create test data
    hData = new float[N];
    hRes1 = new float[N];
    hRes2 = new float[N];

    for(int i=0; i<N; i++) {
        hData[i] = (rand() % 1000 - 500) * 1.0 / 5.0;
    }

    // run CPU RELU
    for(int i=0; i<nTest; i++) {
        timer.enter("nn_relu_cpu");
        nn_relu_cpu(hRes1, hData, N);
        timer.leave("nn_relu_cpu");
    }

#ifdef ENABLE_GPU
    int nDev = 0, devID = 0;

    cudaGetDeviceCount(&nDev);
    printf("dev num = %d\n", nDev);

    // init GPU device
    if( cudaSuccess != cudaSetDevice(devID) ) {
        dbg_pe("Can not open CUDA device: %d", devID);
    }

    checkCudaErrors( cudaMalloc( (void **)&gData, sizeof(float)*N) );
    checkCudaErrors( cudaMalloc( (void **)&gRes1, sizeof(float)*N) );

    checkCudaErrors( cudaMemcpy(gData, hData, sizeof(float)*N, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(gRes1, hData, sizeof(float)*N, cudaMemcpyHostToDevice) );

    // run GPU RELU
    for(int i=0; i<nTest; i++) {
        timer.enter("nn_relu_gpu");
        nn_relu_gpu(gRes1, gData, N);
        timer.leave("nn_relu_gpu");
    }

    checkCudaErrors( cudaMemcpy(hRes2, gRes1, sizeof(float)*N, cudaMemcpyDeviceToHost) );

    checkCudaErrors( cudaFree(gData) );
    checkCudaErrors( cudaFree(gRes1) );

    float e = 0;
    for(int i=0; i<N; i++) {
        float d = hRes1[i] - hRes2[i];
        //printf("%12f -> %12f <-> %12f -> %12f\n", hData[i], hRes1[i], hRes2[i], d);

        e += fabs(d);
    }

    printf("\ndifference = %f\n", e/N);

#endif


    delete [] hData;
    delete [] hRes1;
    delete [] hRes2;

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int cv_featureDescriptor(CParamArray *pa)
{
    using namespace cv;

    // read image
    string imageFN = svar.GetString("image", "./test.png");
    Mat img = imread(imageFN);

    // extract keypoints & descriptors
    Ptr<FeatureDetector>    detector;
    SiftDescriptorExtractor extractor;

    vector<KeyPoint>        keypoints;
    Mat                     descriptors;

    detector = new SiftFeatureDetector;

    detector->detect(img, keypoints);
    extractor.compute(img, keypoints, descriptors);

    // print keypoints
    for(int i=0; i<keypoints.size(); i++) {
        KeyPoint &p = keypoints[i];

        printf("kp[%6d] x, y = %12f, %12f\n", i, p.pt.x, p.pt.y);
        printf("           size = %12f, angle = %12f\n", p.size, p.angle);
        printf("           response = %12f, octave = %3d, class_id = %4d\n", p.response, p.octave, p.class_id);
    }
    printf("\n");

    // print descriptors
    //      type: CV_MAT_TYPE, CV_32F
    printf("descriptor: \n");
    printf("    cols     = %d\n", descriptors.cols);
    printf("    rows     = %d\n", descriptors.rows);
    printf("    channels = %d\n", descriptors.channels());
    printf("    type     = %d\n", descriptors.type());

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int CNN_ModelConvert(CParamArray *pa)
{
    int testNum = svar.GetInt("testNum", 1);

    // load CNN model
    string modelFN = svar.GetString("model", "./data/imagenet-vgg-f");

    PI_CNN cnn;
    for(int i=0; i<testNum; i++) {
        timer.enter("cnn::load");
        cnn.load(modelFN.c_str());
        timer.leave("cnn::load");
    }
    cnn.print();

    // save with fast format
    string modelFN_out = svar.GetString("model_out", "./cnn_models/imagenet-vgg-f.cm");
    for(int i=0; i<testNum; i++) {
        timer.enter("cnn::saveFast");
        cnn.saveFast(modelFN_out.c_str());
        timer.leave("cnn::saveFast");
    }

    // load with fast format
    PI_CNN cnn2;
    for(int i=0; i<testNum; i++) {
        timer.enter("cnn::loadFast");
        cnn2.loadFast(modelFN_out.c_str());
        timer.leave("cnn::loadFast");
    }
    cnn2.print();


    // load image
    string imgFN = fmt::sprintf("%s_img.png", modelFN);
    imgFN = svar.GetString("image", imgFN);
    cv::Mat img = cv::imread(imgFN);

    // call CNN
    cnn.cnnForward(&img);
    cnn2.cnnForward(&img);

    // load ground-truth results
    int cnnLayerNum = cnn.getLayerNum();
    CNN_FeatureMap *cnnRes1 = cnn.getResults();
    CNN_FeatureMap *cnnRes2 = cnn2.getResults();

    for(int i=0; i<=cnnLayerNum; i++) {
        PI_Tensor<float>    resGT;
        PI_Tensor<float>    res1, res2;

        // get ground truth result
        string fname = fmt::sprintf("%s_l%d_res", modelFN, i);
        resGT.load(fname.c_str());

        // get calculated result
        cnnRes1->at(i).toCPU(res1);
        cnnRes2->at(i).toCPU(res2);

        if( res1.isEmpty() ) continue;

        // calculate difference & error
        float   d1 = 0, e1 = 0, dMax1 = 0;
        float   d2 = 0, e2 = 0, dMax2 = 0;

        for(int j=0; j<resGT.numElements; j++) {
            d1 = fabs(resGT.data[j] - res1.data[j]);
            e1 += d1;
            if( d1 > dMax1 ) dMax1 = d1;

            d2 = fabs(resGT.data[j] - res2.data[j]);
            e2 += d2;
            if( d2 > dMax2 ) dMax2 = d2;

            /*
            printf("[%8d] %12f  |  %12f -> %12f | %12f -> %12f\n",
                   j, resGT.data[j],
                   res1.data[j], d1,
                   res2.data[j], d2);
            */
        }

        printf("## layer results [%3d]: e1 = %12f, dMax1 = %12f | e2 = %12f, dMax2 = %12f\n\n",
               i,
               e1, dMax1,
               e2, dMax2);
    }

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int test_CNN(CParamArray *pa)
{
    int testNum = svar.GetInt("testNum", 1);
    int useGPU = svar.GetInt("useGPU", 0);

    // init cuda
    if( useGPU ) {
        int nDev = 0, devID = 0;

        cudaGetDeviceCount(&nDev);
        printf("GPU devNum = %d\n", nDev);

        if( nDev > 0 ) {
            // init GPU device
            if( cudaSuccess != cudaSetDevice(devID) ) {
                dbg_pe("Can not open CUDA device: %d", devID);
            }
        } else {
            useGPU = 0;
        }
    }

    // load CNN model
    string modelFN = svar.GetString("model", "./cnn_models/imagenet-vgg-f.cm");

    PI_CNN cnn;
    cnn.loadFast(modelFN.c_str());
    cnn.setMode(useGPU);

    std::set<int> fmLayers;
    fmLayers.insert(4);
    fmLayers.insert(12);
    fmLayers.insert(15);
    cnn.setFeatureMapLayer(&fmLayers);

    cnn.print();

    // load image
    string fnTestData = "./data/imagenet-vgg-f";
    fnTestData = svar.GetString("testData", fnTestData);
    string imgFN = fmt::sprintf("%s_img.png", fnTestData);

    cv::Mat img = cv::imread(imgFN);

    int imgW, imgH, imgC, block;
    imgW  = img.cols;
    imgH  = img.rows;
    imgC  = img.channels();
    block = imgW * imgH;

    dbg_pt("useGPU = %d", useGPU);
    dbg_pt("input img W, H, C = %3d, %3d, %3d\n", imgW, imgH, imgC);


    // call CNN
    double t1 = pi::tm_getTimeStamp(), dt;
    for(int i=0; i<testNum; i++)
        cnn.cnnForward(&img);
    dt = pi::tm_getTimeStamp() - t1;
    printf("\n\nCalculation time = %f s\n", dt);


    // load ground-truth results
    int cnnLayerNum = cnn.getLayerNum();
    CNN_FeatureMap *cnnRes = cnn.getResults();

    for(int i=0; i<=cnnLayerNum; i++) {
        PI_Tensor<float>    resGT;
        PI_Tensor<float>    res1;

        // get ground truth result
        string fname = fmt::sprintf("%s_l%d_res", fnTestData, i);
        resGT.load(fname.c_str());

        // get calculated result
        cnnRes->at(i).toCPU(res1);
        if( res1.isEmpty() ) continue;

        // get output file
        string fnRes = fmt::sprintf("%s_l%d_dResNN", fnTestData, i);
        FILE *fp = fopen(fnRes.c_str(), "wt");

        // calculate difference & error
        float   d = 0, e = 0, dMax = 0;

        for(int j=0; j<resGT.numElements; j++) {
            d = fabs(resGT.data[j] - res1.data[j]);
            e += d;

            if( d > dMax ) dMax = d;

            fprintf(fp, "[%8d] %12f %12f -> %12f\n", j, resGT.data[j], res1.data[j], d);
        }

        fclose(fp);

        printf("layer results [%3d]: e = %12f, dMax = %12f\n", i, e, dMax);
    }


    // get point feature map
    PI_Tensor<float>    featureMap;
    vector<int>         arrX, arrY;
    int                 nPoint;

    nPoint = 2000;
    arrX.resize(nPoint);
    arrY.resize(nPoint);
    for(int i=0; i<nPoint; i++) {
        arrX[i] = rand() % imgW;
        arrY[i] = rand() % imgH;
    }

    cnn.getFeatures(&arrX, &arrY, &featureMap);

    //printf("\nFeatures: "); featureMap.print(); printf("\n");

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int test_cnnFMextractor(CParamArray *pa)
{
    int testNum = svar.GetInt("testNum", 1);
    int useGPU = svar.GetInt("useGPU", 0);

    // init cuda
    if( useGPU ) {
        int nDev = 0, devID = 0;

        cudaGetDeviceCount(&nDev);
        printf("GPU devNum = %d\n", nDev);

        if( nDev > 0 ) {
            // init GPU device
            if( cudaSuccess != cudaSetDevice(devID) ) {
                dbg_pe("Can not open CUDA device: %d", devID);
            }
        } else {
            useGPU = 0;
        }
    }

    // load CNN model
    string modelFN = svar.GetString("model", "./cnn_models/imagenet-vgg-f.cm");

    std::vector<int> fmLayers;
    fmLayers.push_back(1);
    fmLayers.push_back(5);
    fmLayers.push_back(9);
    //cnnFMextractor extractor(modelFN.c_str(), useGPU, &fmLayers);
    cnnFMextractor extractor(modelFN.c_str(), useGPU);

    // load image
    string imgFN = svar.GetString("image", "./data/test640.png");
    cv::Mat img = cv::imread(imgFN);
    int imgW = img.cols, imgH = img.rows;

    // generate keypoints
    std::vector<cv::KeyPoint>   kps;
    cv::Mat                     desc1, desc2;
    int                         nPoint;

    nPoint = 2000;
    kps.resize(nPoint);
    for(int i=0; i<nPoint; i++) {
        cv::KeyPoint &kp = kps[i];

        kp.pt.x = rand() % imgW;
        kp.pt.y = rand() % imgH;
    }

    // extract CNN descriptors
    for(int i=0; i<testNum; i++) {
        (extractor)(img, cv::Mat(), kps, desc1);
        extractor.compute(img, kps, desc2);
    }

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int showFeatureMap(CParamArray *pa)
{
    int useGPU = svar.GetInt("useGPU", 0);

    // init cuda
    if( useGPU ) {
        int nDev = 0, devID = 0;

        cudaGetDeviceCount(&nDev);
        printf("GPU devNum = %d\n", nDev);

        if( nDev > 0 ) {
            // init GPU device
            if( cudaSuccess != cudaSetDevice(devID) ) {
                dbg_pe("Can not open CUDA device: %d", devID);
            }
        } else {
            useGPU = 0;
        }
    }

    // load CNN model
    string modelFN = svar.GetString("model", "./cnn_models/imagenet-vgg-f.cm");

    std::vector<int> fmLayers;
    fmLayers.push_back(2);
    fmLayers.push_back(6);
    fmLayers.push_back(10);
    cnnFMextractor extractor(modelFN.c_str(), useGPU, &fmLayers);
    //cnnFMextractor extractor(modelFN.c_str(), useGPU);

    // load image
    cv::Mat img1 = cv::imread(svar.GetString("testCNN_Matching.image1", "./data/test.png"));
    cerr<<"Loading "<<svar.GetString("testCNN_Matching.image1", "./data/test.png")<<endl;

    int imgW = img1.cols;
    int imgH = img1.rows;

    // get feature map
    vector<cv::KeyPoint>    keyPoints1;
    cv::Mat                 desc1;

    keyPoints1.clear();
    keyPoints1.reserve(imgW*imgH);
    for(int y=0;y<imgH;y++)
        for(int x=0;x<imgW;x++)
        {
            keyPoints1.push_back(cv::KeyPoint(cv::Point2f(x,y),0));
        }

    extractor.compute(img1, keyPoints1, desc1);



    float *d1 = (float*) desc1.data;
    int fmd = desc1.cols;
    float *fm = new float[imgW * imgH];

    cv::Mat img_result(imgH, imgW, CV_8UC1);

    cv::imshow("img1",img1);

    for(int i=0; i<fmd; i++) {
        printf("feature map: %4d\n", i);

        float   vmax = -9e99;
        float   vmin =  9e99;

        for(int j=0; j < imgW*imgH; j++) {
            float v = d1[j*fmd + i];

            fm[j] = v;
            if( v > vmax ) vmax = v;
            if( v < vmin ) vmin = v;
        }

        for(int j=0; j<imgW*imgH; j++) {
            img_result.data[j] = (uint8_t) (fm[j] - vmin) / (vmax - vmin) * 255;
        }

        cv::imshow("featureMap", img_result);
        cv::waitKey(0);
    }

    delete [] fm;

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int testCNN_Matching(CParamArray *pa)
{
    int useGPU = svar.GetInt("useGPU", 0);

    // init cuda
    if( useGPU ) {
        int nDev = 0, devID = 0;

        cudaGetDeviceCount(&nDev);
        printf("GPU devNum = %d\n", nDev);

        if( nDev > 0 ) {
            // init GPU device
            if( cudaSuccess != cudaSetDevice(devID) ) {
                dbg_pe("Can not open CUDA device: %d", devID);
            }
        } else {
            useGPU = 0;
        }
    }

    // load CNN model
    string modelFN = svar.GetString("model", "./cnn_models/imagenet-vgg-f.cm");

    std::vector<int> fmLayers;
    fmLayers.push_back(1);
    fmLayers.push_back(5);
    fmLayers.push_back(9);
    //cnnFMextractor extractor(modelFN.c_str(), useGPU, &fmLayers);
    cnnFMextractor extractor(modelFN.c_str(), useGPU);

    // load image
    cv::Mat img1 = cv::imread(svar.GetString("testCNN_Matching.image1", "./data/test.png"));
    cerr<<"Loading "<<svar.GetString("testCNN_Matching.image1", "./data/test.png")<<endl;
    cv::Mat img2 = cv::imread(svar.GetString("testCNN_Matching.image2", "./data/test.png"));
    cerr<<"Loading "<<svar.GetString("testCNN_Matching.image2", "./data/test.png")<<endl;

    vector<cv::KeyPoint> keyPoints1,keyPoints2;
    cv::Mat desc1,desc2;
    timer.enter("FeatureDetect");
    cv::FeatureDetector* Detect=new cv::FastFeatureDetector(svar.GetInt("FastThreshold",27));
    Detect->detect(img1,keyPoints1);
    Detect->detect(img2,keyPoints2);
    timer.leave("FeatureDetect");

    timer.enter("FeatureDescripter");
    extractor.compute(img1, keyPoints1, desc1);
    extractor.compute(img2, keyPoints2, desc2);
    timer.leave("FeatureDescripter");

    cv::BFMatcher matcher(cv::NORM_L2,true);//NORM_HAMMING

    vector<cv::DMatch> matches;

    matcher.match(desc1,desc2,matches);

    cv::Mat img_matches;
    cv::drawMatches(img1, keyPoints1, img2, keyPoints2,
                matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                vector<char>(), cv::DrawMatchesFlags::DEFAULT );
    cv::imshow( "Match", img_matches);
    cv::waitKey(0);

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int matchWholeImage(CParamArray *pa)
{
    int useGPU = svar.GetInt("useGPU", 0);

    // init cuda
    if( useGPU ) {
        int nDev = 0, devID = 0;

        cudaGetDeviceCount(&nDev);
        printf("GPU devNum = %d\n", nDev);

        if( nDev > 0 ) {
            // init GPU device
            if( cudaSuccess != cudaSetDevice(devID) ) {
                dbg_pe("Can not open CUDA device: %d", devID);
            }
        } else {
            useGPU = 0;
        }
    }

    // load CNN model
    string modelFN = svar.GetString("model", "./cnn_models/imagenet-vgg-f.cm");

    std::vector<int> fmLayers;
    fmLayers.push_back(1);
    fmLayers.push_back(5);
    fmLayers.push_back(9);
    //cnnFMextractor extractor(modelFN.c_str(), useGPU, &fmLayers);
    cnnFMextractor extractor(modelFN.c_str(), useGPU);

    // load image
    cv::Mat img1 = cv::imread(svar.GetString("testCNN_Matching.image1", "./data/test.png"));
    cerr<<"Loading "<<svar.GetString("testCNN_Matching.image1", "./data/test.png")<<endl;
    cv::Mat img2 = cv::imread(svar.GetString("testCNN_Matching.image2", "./data/test.png"));
    cerr<<"Loading "<<svar.GetString("testCNN_Matching.image2", "./data/test.png")<<endl;

    vector<cv::KeyPoint> keyPoints1,keyPoints2;
    cv::Mat desc1,desc2;
    timer.enter("FeatureDetect");
    cv::FeatureDetector* Detect=new cv::FastFeatureDetector(svar.GetInt("FastThreshold",27));
//    Detect->detect(img1,keyPoints1);
    Detect->detect(img2,keyPoints2);
    timer.leave("FeatureDetect");
    keyPoints1.clear();
    keyPoints1.reserve(img2.rows*img2.cols);
    for(int y=0;y<img2.rows;y++)
        for(int x=0;x<img2.cols;x++)
        {
            keyPoints1.push_back(cv::KeyPoint(cv::Point2f(x,y),0));
        }

    timer.enter("FeatureDescripter");
    extractor.compute(img1, keyPoints1, desc1);
    extractor.compute(img2, keyPoints2, desc2);
    timer.leave("FeatureDescripter");
    cv::SL2<float> sl2;
    cv::Mat img_result(img1.rows,img1.cols,CV_32FC1);
    cv::imshow("img1",img1);
    cv::imshow("img2",img2);
    for(int i=0,iend=keyPoints2.size();i<iend;i++)
    {
        float maxVal=0;
        float minVal=1e10;
        cv::KeyPoint minLocation;
        float result;
        for(int j=0,jend=keyPoints1.size();j<jend;j++)
        {
            img_result.at<float>(j)=sl2((float*)desc1.row(j).data,
                                        (float*)desc2.row(i).data,desc1.cols);
            result=img_result.at<float>(j);
            if(result>maxVal) maxVal=result;
            if(result<minVal)
            {
                minVal=result;
                minLocation=keyPoints1.at(j);
            }

        }
        cout<<"val:"<<img_result.at<float>(0)<<endl;
        cv::Mat clone1=img1.clone();
        cv::circle(img_result,minLocation.pt,8,cv::Scalar(1.),2);
        cv::circle(clone1,minLocation.pt,8,cv::Scalar(0,0,255),2);
        cv::imshow( "Match", img_result*(1/maxVal));
        cv::imshow("img1",clone1);
        cv::Mat img_clone=img2.clone();
        cv::circle(img_clone,keyPoints2[i].pt,8,cv::Scalar(255,0,0),2);
        cv::imshow("img2",img_clone);
        cv::waitKey(0);
    }

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int matchWholeImageORB(CParamArray *pa)
{
    // load image
    cv::Mat img1 = cv::imread(svar.GetString("testCNN_Matching.image1", "./data/test.png"));
    cerr<<"Loading "<<svar.GetString("testCNN_Matching.image1", "./data/test.png")<<endl;
    cv::Mat img2 = cv::imread(svar.GetString("testCNN_Matching.image2", "./data/test.png"));
    cerr<<"Loading "<<svar.GetString("testCNN_Matching.image2", "./data/test.png")<<endl;

    vector<cv::KeyPoint> keyPoints1,keyPoints2;
    cv::Mat desc1,desc2;
    timer.enter("FeatureDetect");
    cv::FeatureDetector* Detect=new cv::FastFeatureDetector(svar.GetInt("FastThreshold",27));
//    Detect->detect(img1,keyPoints1);
    Detect->detect(img2,keyPoints2);
    timer.leave("FeatureDetect");
    keyPoints1.clear();
    keyPoints1.reserve(img2.rows*img2.cols);
    for(int y=0;y<img2.rows;y++)
        for(int x=0;x<img2.cols;x++)
        {
            keyPoints1.push_back(cv::KeyPoint(cv::Point2f(x,y),0));
        }

    timer.enter("FeatureDescripter");
    cv::OrbDescriptorExtractor extractor;
    extractor.compute(img1, keyPoints1, desc1);
    extractor.compute(img2, keyPoints2, desc2);
    cout<<"keyPoints1.size:"<<keyPoints1.size()<<endl;
    timer.leave("FeatureDescripter");
    cv::Hamming sl2;
    cv::Mat img_result(img1.rows,img1.cols,CV_32FC1);
    cv::imshow("img1",img1);
    cv::imshow("img2",img2);
    for(int i=0,iend=keyPoints2.size();i<iend;i++)
    {
        float maxVal=0;
        float minVal=1e10;
        cv::KeyPoint minLocation;
        float result;
        for(int j=0,jend=keyPoints1.size();j<jend;j++)
        {
            img_result.at<float>(j)=sl2(desc1.row(j).data,
                                        desc2.row(i).data,desc1.cols);
            result=img_result.at<float>(j);
            if(result>maxVal) maxVal=result;
            if(result<minVal)
            {
                minVal=result;
                minLocation=keyPoints1.at(j);
            }

        }
        cout<<"val:"<<img_result.at<float>(0)<<endl;
        cv::Mat clone1=img1.clone();
        cv::circle(img_result,minLocation.pt,8,cv::Scalar(1.),2);
        cv::circle(clone1,minLocation.pt,8,cv::Scalar(0,0,255),2);
        cv::imshow( "Match", img_result*(1/maxVal));
        cv::imshow("img1",clone1);
        cv::Mat img_clone=img2.clone();
        cv::circle(img_clone,keyPoints2[i].pt,8,cv::Scalar(255,0,0),2);
        cv::imshow("img2",img_clone);
        cv::waitKey(0);
    }

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct TestFunctionArray g_fa[] =
{
    TEST_FUNC_DEF(test_CUDA,                "Test CUDA"),
    TEST_FUNC_DEF(cv_featureDescriptor,     "Test OpenCV keypoint & descriptor extraction"),

    TEST_FUNC_DEF(CNN_ModelConvert,         "Convert CNN model"),

    TEST_FUNC_DEF(test_CNN,                 "Test CNN"),
    TEST_FUNC_DEF(test_cnnFMextractor,      "Test CNN FeatureMap extractor"),

    TEST_FUNC_DEF(showFeatureMap,           "Show feature map"),

    TEST_FUNC_DEF(testCNN_Matching,         "Test CNN FeatureMap extractor"),
    TEST_FUNC_DEF(matchWholeImage,          "Test CNN FeatureMap extractor"),
    TEST_FUNC_DEF(matchWholeImageORB,       "Test CNN FeatureMap extractor"),


    {NULL,  "NULL",  "NULL"},
};


int main(int argc, char *argv[])
{
    // setup debug trace
    dbg_stacktrace_setup();

    // run function
    return svar_main(argc, argv, g_fa);
}
