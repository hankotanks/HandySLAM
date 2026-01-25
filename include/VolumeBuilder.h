#pragma once

#include <filesystem>

#include <opencv2/core/mat.hpp>
#include <sophus/se3.hpp>

#include "Profile.h"

namespace HandySLAM {
    class VolumeBuilder {
    public:
        VolumeBuilder(const Intrinsics& intrinsics, double voxelSize, double maxDepth);
        void integrateFrame(const cv::Mat& im, const cv::Mat& depthmap, const Sophus::SE3f& pose);
        bool save(const std::filesystem::path& pathOut) const;
    private:
        std::size_t frameIdx_;
        Intrinsics intrinsics_;
        double voxelSize_;
        double maxDepth_;
        struct Volume;
        std::shared_ptr<Volume> volume_;
    };
}