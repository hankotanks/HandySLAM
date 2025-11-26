#include <iostream>
#include <filesystem>
#include <limits>
#include <optional>
#include <algorithm>
#include <thread>
#include <chrono>
#include <unordered_map>

#include <System.h>

#include "Config.h"
#include "Dataloader.h"
#include "DataloaderStray.h"
#include "Initializer.h"

int main(int argc, char* argv[]) {
    // register dataloaders with the initializers
    HandySLAM::Initializer::add<HandySLAM::DataloaderStray>("stray");
    // build dataloader
    HandySLAM::Dataloader* data = HandySLAM::Initializer::init(argc, argv);
    // check that the dataloader was successfully constructed
    if(!data) {
        log_err("Failed to initialize Dataloader.");
        return 1;
    }
    // generate settings file
    const std::string strSettingsFile = data->strSettingsFile();
    // configure sensor
    enum ORB_SLAM3::System::eSensor sensor = HandySLAM::Initializer::get().usingImu ? 
        ORB_SLAM3::System::IMU_RGBD : 
        ORB_SLAM3::System::RGBD;
    // perform SLAM
    {
        ORB_SLAM3::System SLAM(VOCAB_PATH, strSettingsFile, sensor);
        // iterate through frames
        double timestampPrev = std::numeric_limits<double>::max() * -1.0;
        for(HandySLAM::Frame& frameCurr : *data) {
            if(HandySLAM::Initializer::get().usingImu && !frameCurr.vImuMeasValidate(timestampPrev)) break;
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
        SLAM.SaveTrajectoryTUM(HandySLAM::Initializer::get().pathScene / "trajectory.txt");
    }
    return 0;
}
