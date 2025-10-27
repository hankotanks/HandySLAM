#include <iostream>

#include <filesystem>
#include <optional>
#include <algorithm>

#include <System.h>

#include "Config.h"
#include "DataloaderStray.h"

#define ASSERT_PATH_EXISTS(path_) if(!std::filesystem::exists(path_)) { \
        std::cout << "Path does not exist [" << path_ << "]." << std::endl; \
        exit(1); \
    }

#define SIZE_INTERNAL cv::Size(256, 192)

int main(int argc, char* argv[]) {
    if(argc != 2) {
        std::cout << "Must provide scene path." << std::endl;
        exit(1);
    }

    std::filesystem::path pathScene(argv[1]);
    ASSERT_PATH_EXISTS(pathScene);

    HandySLAM::DataloaderStray data(pathScene, SIZE_INTERNAL);

    {
        ORB_SLAM3::System SLAM(VOCAB_PATH, data.pathSettings(), ORB_SLAM3::System::IMU_RGBD);

        std::optional<HandySLAM::Frame> frame;
        data.next();
        while((frame = data.next())) {
            std::sort(frame->vImuMeas.begin(), frame->vImuMeas.end(), [](ORB_SLAM3::IMU::Point a, ORB_SLAM3::IMU::Point b) { return a.t < b.t; });
#if 1
            std::cout << "timestamps [" << std::fixed << std::setprecision(4) << frame->timestamp << "]: ";
            for(const ORB_SLAM3::IMU::Point& pt : frame->vImuMeas) {
                std::cout << std::fixed << std::setprecision(4) << pt.t << ", ";
            }
            std::cout << std::endl;
#endif
            SLAM.TrackRGBD(frame->im, frame->depthmap, frame->timestamp, frame->vImuMeas);
        }

        std::cout << "Press ENTER to close viewer." << std::endl;
        std::cin.get();

        SLAM.Shutdown();
    }

    return 0;
}
