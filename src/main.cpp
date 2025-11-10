#include <iostream>

#include <filesystem>
#include <limits>
#include <optional>
#include <algorithm>

#include <System.h>

#include "Config.h"
#include "Dataloader.h"
#include "DataloaderStray.h"
#include "util.h"

int main(int argc, char* argv[]) {
    if(argc != 3 && argc != 4) {
        std::cout << "Usage: " << argv[0] << " <scene_path> <profile_name> --imu" << std::endl;
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
#ifdef DEBUG_FRAMES
            std::cout << "frame " << frameCurr.index << ", t: " << frameCurr.timestamp << ", meas: [ ";
            for(const ORB_SLAM3::IMU::Point& meas : frameCurr.vImuMeas) {
                std::cout << std::fixed << std::setprecision(4) << "{ "
                    << "t: " << meas.t << ", "
                    << "a: [" << meas.a.x() << ", " << meas.a.y() << ", " << meas.a.z() << "], "
                    << "w: [" << meas.w.x() << ", " << meas.w.y() << ", " << meas.w.z() << "] "
                    << "}, ";
            }
            std::cout << "]" << std::endl;
#endif
            // process frame
            SLAM.TrackRGBD(frameCurr.im, frameCurr.depthmap, frameCurr.timestamp, frameCurr.vImuMeas); 
            timestampPrev = frameCurr.timestamp;
        }
        // wait for input before closing the visualizer
        std::cout << "Press ENTER to close viewer." << std::endl;
        std::cin.get();
        // shut down the SLAM system and save trajectory
        SLAM.Shutdown();
        SLAM.SaveTrajectoryTUM(pathScene / "trajectory.csv");
    }
    return 0;
}
