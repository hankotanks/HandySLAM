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
    };

    class Dataloader {
    public:
        Dataloader(const std::filesystem::path& pathScene) : pathScene_(pathScene) { /* STUB */ };
        virtual const std::optional<Frame> next() = 0;
        virtual const std::filesystem::path& pathSettings() = 0;
    protected:
        std::filesystem::path pathScene_;
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
    };
}