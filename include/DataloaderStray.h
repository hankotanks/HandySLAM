#pragma once

#include <filesystem>
#include <fstream>
#include <optional>

#include <opencv2/core/mat.hpp>

#include "Dataloader.h"
#include "Initializer.h"

namespace HandySLAM {
    class DataloaderStray : public Dataloader {
    public:
        DataloaderStray(Initializer& init);
        ~DataloaderStray();
        const std::optional<Frame> next() noexcept final override;
    private:
        const std::optional<Frame> nextInternal() noexcept;
        std::optional<double> timestamp() noexcept;
        std::optional<cv::Mat> im() noexcept;
        std::optional<cv::Mat> depthmap() noexcept;
        std::vector<ORB_SLAM3::IMU::Point> vImuMeas(double timestamp) noexcept;
    private:
        std::size_t frameIdx_;
        std::size_t frameCount_;
        std::filesystem::path pathRGB_;
        std::filesystem::path pathDepth_;
        std::filesystem::path pathOdom_;
        std::filesystem::path pathIMU_;
        std::filesystem::path pathCameraMatrix_;
        cv::VideoCapture cap_;
        std::size_t fps_;
        std::ifstream readerOdom_;
        std::ifstream readerIMU_;
        std::optional<Frame> carryOverFrame_;
    };
}