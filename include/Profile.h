#pragma once

#include <filesystem>
#include <optional>

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

namespace HandySLAM {
    class Profile {
    public:
        std::filesystem::path pathSettings;
    public:
        Profile(const std::string& profile, std::size_t fps);
        Profile(const std::string& profile, std::size_t fps, cv::Size sizeInternal);
    private:
        Profile(const std::string& profile, std::size_t fps, std::optional<cv::Size> sizeInternal);
    private:
        Eigen::Matrix4d imu2cam_;
        double noiseGyr_;
        double noiseAcc_;
        double rwalkGyr_;
        double rwalkAcc_;
        double updateRate_;
        double fx_, fy_, cx_, cy_;
        std::size_t fps_;
        cv::Size sizeOriginal_;
        cv::Size sizeInternal_;
    };
}
