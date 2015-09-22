#ifndef __CNNFMEXTRACTOR_H__
#define __CNNFMEXTRACTOR_H__

#include <opencv2/features2d/features2d.hpp>

#include "PI_CNN.h"


class cnnFMextractor
{
public:
    cnnFMextractor() { }

    cnnFMextractor(const char *cnnModelFN, int useGPU = 0, const std::vector<int> *fmLayers = NULL) {
        setupCNN(cnnModelFN, useGPU, fmLayers);
    }

    virtual ~cnnFMextractor() {}

    int setupCNN(const char *cnnModelFN, int useGPU = 0, const std::vector<int> *fmLayers = NULL) {
        // load CNN model
        m_cnnModel.loadFast(cnnModelFN);

        // set GPU mode
        if( useGPU ) m_cnnModel.setMode(useGPU);

        // set default feature map layer
        if( fmLayers == NULL ) {
            m_cnnFeatureMapLayers.push_back(4);
            m_cnnFeatureMapLayers.push_back(12);
            m_cnnFeatureMapLayers.push_back(15);
        } else {
            m_cnnFeatureMapLayers = *fmLayers;
        }

        m_cnnModel.setFeatureMapLayer(&m_cnnFeatureMapLayers);
    }

    ///
    /// \brief Compute the CNN feature map on an image
    ///
    /// \param image            - [in] image
    /// \param mask             - [in] mask
    /// \param keypoints        - [in] keypoints
    /// \param descriptors      - [out] CNN feature map descriptor
    ///
    void operator()( cv::InputArray image, cv::InputArray mask,
                     std::vector<cv::KeyPoint>& keypoints,
                     cv::OutputArray descriptors) {
        cv::Mat img  = image.getMat();
        cv::Mat &des = descriptors.getMatRef();

        if( 0 != m_cnnModel.cnnForward(&img) ) return;
        m_cnnModel.getFeatures(keypoints, des);
    }

    ///
    /// \brief Compute the CNN feature map on an image
    ///
    /// \param image            - [in] image
    /// \param mask             - [in] mask
    /// \param keypoints        - [in] keypoints
    /// \param descriptors      - [out] CNN feature map descriptor
    ///
    void compute( const cv::Mat& image,
                  CV_IN_OUT std::vector<cv::KeyPoint>& keypoints,
                  CV_OUT cv::Mat& descriptors ) {
        cv::Mat *img = (cv::Mat*) &image;

        if( 0 != m_cnnModel.cnnForward(img) ) return;
        m_cnnModel.getFeatures(keypoints, descriptors);
    }


    ///
    /// \brief Get CNN model
    ///
    /// \return                 - CNN model
    ///
    PI_CNN* getCNN(void) { return &m_cnnModel; }

    ///
    /// \brief Get feature map layers
    ///
    /// \return                 - feature map layers
    ///
    std::vector<int> getFeatureMapLayers(void) { return m_cnnFeatureMapLayers; }


protected:
    PI_CNN              m_cnnModel;                         ///< CNN model
    std::vector<int>    m_cnnFeatureMapLayers;              ///< feature map layers
};


#endif // end of __CNNFMEXTRACTOR_H__
