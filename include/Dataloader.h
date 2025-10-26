#pragma once

#include <filesystem>

#include <opencv2/core/mat.hpp>

#include <System.h>

namespace HandySLAM {
    struct Frame {
        double timestamp;
        cv::Mat im, depthmap;
        std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    };

    class Dataloader {
    public:
        Dataloader(const std::filesystem::path& pathScene) : pathScene_(pathScene) { /* STUB */ };
        virtual const std::optional<Frame> next() = 0;
        virtual const std::filesystem::path& pathSettings() = 0;
    protected:
        std::filesystem::path pathScene_;
    };
}