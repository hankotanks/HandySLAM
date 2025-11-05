#include <iostream>

#include <filesystem>
#include <optional>
#include <algorithm>

#include <System.h>

#include "Config.h"
#include "Profile.h"
#include "DataloaderStray.h"

#define ASSERT_PATH_EXISTS(path_) if(!std::filesystem::exists(path_)) { \
        std::cout << "Path does not exist [" << path_ << "]." << std::endl; \
        exit(1); \
    }

// TODO: Better handling of size parity between RGB and depth maps
#define SIZE_INTERNAL cv::Size(256, 192)

int main(int argc, char* argv[]) {
    if(argc != 3) {
        std::cout << "Usage: " << argv[0] << " <scene_path> <profile_name>" << std::endl;
        exit(1);
    }
    // parse scene path
    std::filesystem::path pathScene(argv[1]);
    ASSERT_PATH_EXISTS(pathScene);
    // load scene data
    HandySLAM::DataloaderStray data(pathScene);
    // parse profile
    HandySLAM::Profile profile(argv[2], data.fps(), SIZE_INTERNAL);
    {
        ORB_SLAM3::System SLAM(VOCAB_PATH, profile.pathSettings, ORB_SLAM3::System::IMU_RGBD);
        // iterate through frames
        std::size_t maps = 0;
        std::size_t stateCurr, statePrev = 0;
        std::optional<ORB_SLAM3::IMU::Point> meas;
        for(HandySLAM::Frame& frameCurr : data) {
            // sort IMU measurements
            std::sort(frameCurr.vImuMeas.begin(), frameCurr.vImuMeas.end(), [](ORB_SLAM3::IMU::Point a, ORB_SLAM3::IMU::Point b) { return a.t < b.t; });
            // process frame
            SLAM.TrackRGBD(frameCurr.im, frameCurr.depthmap, frameCurr.timestamp, frameCurr.vImuMeas); 
            // check if SLAM broke and we had to make a new map
            stateCurr = SLAM.GetTrackingState();
            if(stateCurr == ORB_SLAM3::Tracking::RECENTLY_LOST && stateCurr != statePrev) ++maps; 
            statePrev = stateCurr;
            // store this frame's final IMU reading for the next iteration
            if(!frameCurr.vImuMeas.empty()) meas = frameCurr.vImuMeas.back();
        }
        // wait for input before closing the visualizer
        std::cout << "Info: Generated " << (maps ? maps : 1) << " maps." << std::endl;
        std::cout << "Press ENTER to close viewer." << std::endl;
        std::cin.get();
        // shutdown the SLAM system
        SLAM.Shutdown();
        while(!SLAM.isShutDown()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return 0;
}
