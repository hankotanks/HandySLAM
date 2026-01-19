#pragma once

#include <filesystem>
#include <optional>

#include <opencv2/core/mat.hpp>

#include <System.h>

#include "Profile.h"
#include "util.h"

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

    struct SceneMetadata {
        cv::Size sizeIm;
        cv::Size sizeDepthmap;
        std::size_t fps;
        Intrinsics intrinsics;
    };

    class Dataloader {
    public:
        Dataloader(const std::filesystem::path& pathScene);
        Dataloader(const std::filesystem::path& pathScene, const std::string& profileName);
        const std::string strSettingsFile();
        const Intrinsics& intrinsics() const { return metadata_->intrinsics; }
        const std::filesystem::path& pathScene() const { return pathScene_; }
        virtual const std::optional<Frame> next() noexcept = 0;
    protected:
        std::filesystem::path pathScene_;
        // THESE MUST BE SET INSIDE THE CONSTRUCTOR OF ANY DERIVED CLASS
        SetOnce<Profile> profile_;
        SetOnce<SceneMetadata> metadata_;
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
