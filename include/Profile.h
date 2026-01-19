#pragma once

#include <optional>

#include <opencv2/core/mat.hpp>
#include <Eigen/Core>

namespace HandySLAM {
    struct DistortionParams {
        double k1, k2, p1, p2;
    };

    struct Intrinsics {
        double fx;
        double fy;
        double cx;
        double cy;
        std::optional<DistortionParams> distortion;
    };

    struct Profile {
        std::string name;
        Eigen::Matrix4d T_cam_imu;
        double update_rate;
        double gyroscope_noise_density;
        double gyroscope_random_walk;   
        double accelerometer_noise_density;
        double accelerometer_random_walk;  
        double timeshift_cam_imu;
        // parser
        Profile(const std::string& profileName);
    private:
        struct {
            Intrinsics intrinsics;
            cv::Size resolution;
        } unused;
    };
}
