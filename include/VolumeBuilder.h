#pragma once

#include <opencv2/core/mat.hpp>
#include <sophus/se3.hpp>

namespace HandySLAM {
    class VolumeBuilder {
    public:
        VolumeBuilder();
        void integrateFrame(const cv::Mat& im, const cv::Mat& depthmap, const Sophus::SE3f& pose);
    private:
    };
}