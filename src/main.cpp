#include <iostream>
#include <filesystem>
#include <limits>
#include <optional>
#include <algorithm>
#include <thread>
#include <chrono>

#include <System.h>

#include "Config.h"
#include "Dataloader.h"
#include "DataloaderStray.h"
#include "util.h"

int main(int argc, char* argv[]) {
    // TODO: Actually CLI parsing that support every Dataloader
    if(argc != 3 && argc != 4) {
        std::cout << "Usage: " << argv[0] << " <scene_path> <profile_name> [--imu]" << std::endl;
        exit(1);
    }
    // parse scene path
    std::filesystem::path pathScene(argv[1]);
    ASSERT_PATH_EXISTS(pathScene);
    // load scene data
    HandySLAM::DataloaderStray data(pathScene, argv[2]);
    // parse profile
    const std::string strSettingsFile = data.strSettingsFile();
    bool usingImu = (argc == 4 && std::string(argv[3]) == "--imu");
    enum ORB_SLAM3::System::eSensor sensor = usingImu ? ORB_SLAM3::System::IMU_RGBD : ORB_SLAM3::System::RGBD;
    {
        ORB_SLAM3::System SLAM(VOCAB_PATH, strSettingsFile, sensor);
        // iterate through frames
        double timestampPrev = std::numeric_limits<double>::max() * -1.0;
        for(HandySLAM::Frame& frameCurr : data) {
            if(usingImu && !frameCurr.vImuMeasValidate(timestampPrev)) break;
            // process frame
            SLAM.TrackRGBD(frameCurr.im, frameCurr.depthmap, frameCurr.timestamp, frameCurr.vImuMeas); 
            timestampPrev = frameCurr.timestamp;
        }
        SLAM.Shutdown();
        while(!SLAM.isShutDown()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // wait for input before closing the visualizer
        std::cout << "Press ENTER to save trajectory. [^C] to exit without saving." << std::endl;
        std::cin.get();
        // save trajectory
        SLAM.SaveTrajectoryTUM(pathScene / "trajectory.txt");
    }
    return 0;
}
