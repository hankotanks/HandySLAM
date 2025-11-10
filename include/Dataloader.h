#pragma once

#include <filesystem>
#include <optional>

#include <opencv2/core/mat.hpp>

#include <System.h>

namespace HandySLAM {
    struct Frame {
        std::size_t index;
        double timestamp;
        cv::Mat im;
        cv::Mat depthmap;
        std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
        // validation
        bool vImuMeasValidate(double timestampPrev) const;
    };

    struct Intrinsics {
        double fx;
        double fy;
        double cx;
        double cy;
    };

    struct Profile {
        std::string name;
        Eigen::Matrix4d T_cam_imu;
        Intrinsics intrinsics;
        std::size_t fps;
        double update_rate;
        double gyroscope_noise_density;
        double gyroscope_random_walk;   
        double accelerometer_noise_density;
        double accelerometer_random_walk;  
        double timeshift_cam_imu;
        cv::Size resolution;
        // parser
        Profile(const std::string& profile);
    };

    class Dataloader {
    public:
        Dataloader(const std::filesystem::path& pathScene, const std::string& profileName);
        const std::string strSettingsFile();
        virtual const std::optional<Frame> next() = 0;
    protected:
        std::filesystem::path pathScene_;
        Profile profile_;
        // THIS MUST BE SET INSIDE THE DERIVED CLASS'S CONSTRUCTOR
        cv::Size sizeDepthmap;
        // THIS MUST BE SET INSIDE THE DERIVED CLASS'S CONSTRUCTOR
        std::size_t fps;
        // THIS MUST BE SET INSIDE THE DERIVED CLASS'S CONSTRUCTOR
        Intrinsics intrinsics;

    public:
        class DataloaderIterator {
        public:
            using value_type = Frame;
            using reference = Frame&;
            using pointer = Frame*;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::input_iterator_tag;
            DataloaderIterator(Dataloader* loader = nullptr) : loader_(loader) { next(); }
            DataloaderIterator& operator++() { next(); return (*this); }
            Frame& operator*() { return frameCurr_; }
            Frame* operator->() { return &frameCurr_; }
            bool operator==(const DataloaderIterator& other) const { return loader_ == other.loader_; }
            bool operator!=(const DataloaderIterator& other) const { return !(*this == other); }
        private:
            void next() {
                if(loader_ == nullptr) return;
                std::optional<Frame> frameNext = loader_->next();
                if(frameNext) frameCurr_ = std::move(*frameNext);
                else loader_ = nullptr;
            }
        private:
            Dataloader* loader_ = nullptr;
            Frame frameCurr_;
        };
        DataloaderIterator begin() { return DataloaderIterator(this); }
        DataloaderIterator end() { return DataloaderIterator(); }
    private:
        std::optional<std::filesystem::path> strSettingsFile_;
    };
}