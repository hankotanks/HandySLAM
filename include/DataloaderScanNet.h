#pragma once

#include <fstream>

#include "Dataloader.h"
#include "Initializer.h"

#include "json.h"

namespace HandySLAM {
    class DataloaderScanNet : public Dataloader {
    public:
        DataloaderScanNet(Initializer& init);
        ~DataloaderScanNet();
        const std::optional<Frame> next() noexcept final override;
    private:
        std::optional<cv::Mat> im() noexcept;
        std::optional<cv::Mat> depthmap() noexcept;
    private:
        std::size_t frameIdx_;
        std::filesystem::path pathJSON_;
        std::filesystem::path pathRGB_;
        std::filesystem::path pathDepth_;
        std::filesystem::path pathCameras_;
        std::ifstream readerJSON_;
        nlohmann::json parserJSON_;
        nlohmann::json::const_iterator iterJSON_;
        cv::VideoCapture cap_;
        std::ifstream readerDepth_;
    };
}