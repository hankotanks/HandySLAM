#pragma once

#include <filesystem>
#include <fstream>
#include <optional>

#include <opencv2/core/mat.hpp>

#include "Dataloader.h"

namespace HandySLAM {
    class DataloaderStray : public Dataloader {
    public:
        DataloaderStray(const std::filesystem::path& pathScene);
        DataloaderStray(const std::filesystem::path& pathScene, cv::Size sizeInternal);
        ~DataloaderStray();
        const std::optional<Frame> next() final override;
        const std::filesystem::path& pathSettings() final override;
    private:
        DataloaderStray(const std::filesystem::path& pathScene, bool generateSettings);
    private:
        void generateSettingsFile(const std::filesystem::path& pathCameraMatrix);
        double getFrequencyIMU();
        std::optional<double> nextTimestampGeneric(std::ifstream& reader);
        std::optional<double> nextTimestamp();
        std::optional<cv::Mat> nextDepthFrame();
        std::vector<ORB_SLAM3::IMU::Point> nextIMU(double timestamp);
    private:
        cv::Size sizeInternal_;
        std::size_t frameIdx_;
        std::filesystem::path pathSettings_;
        std::filesystem::path pathRGB_;
        std::filesystem::path pathDepth_;
        std::filesystem::path pathOdom_;
        std::filesystem::path pathIMU_;
        cv::VideoCapture cap_;
        std::size_t fps_;
        cv::Size sizeOriginal_;
        std::ifstream readerOdom_;
        std::ifstream readerIMU_;
        double freq_;
        std::optional<ORB_SLAM3::IMU::Point> carryOverSensorData_;
    };
}