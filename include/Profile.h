#pragma once

#include <filesystem>
#include <optional>

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

namespace HandySLAM {
    class Profile {
    public:
        Profile(const std::string& profile, cv::Size sizeDepthmap, std::size_t fps);
        const std::string strSettingsFile() const; 
    private:
    private:
        std::string profile_;
        Eigen::Matrix4d cam2imu_;
        double noiseGyr_;
        double noiseAcc_;
        double rwalkGyr_;
        double rwalkAcc_;
        double updateRate_;
        double fx_, fy_, cx_, cy_;
        std::size_t fps_;
        cv::Size sizeIm_;
        cv::Size sizeDepthmap_;
    };
}
