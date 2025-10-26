#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <filesystem>
#include <optional>

#include <System.h>

#include "Config.h"
#include "DataloaderStray.h"

#define ASSERT_PATH_EXISTS(path_) if(!std::filesystem::exists(path_)) { \
        std::cout << "Path does not exist [" << path_ << "]." << std::endl; \
        exit(1); \
    }

int main(int argc, char* argv[]) {
    if(argc != 2) {
        std::cout << "Must provide scene path." << std::endl;
        exit(1);
    }

    std::filesystem::path pathScene(argv[1]);
    ASSERT_PATH_EXISTS(pathScene);

    HandySLAM::DataloaderStray data(pathScene);

    {
        ORB_SLAM3::System SLAM(VOCAB_PATH, data.pathSettings(), ORB_SLAM3::System::IMU_RGBD);

        std::optional<HandySLAM::Frame> frame;
        while((frame = data.next())) {
            SLAM.TrackRGBD(frame->im, frame->depthmap, frame->timestamp, frame->vImuMeas);
        }

        SLAM.Shutdown();
    }

    return 0;
}
