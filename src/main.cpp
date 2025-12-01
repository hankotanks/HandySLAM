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
    if(!data) {
        log_err("Failed to initialize Dataloader.");
        return 1;
    }
    const HandySLAM::Initializer& init = HandySLAM::Initializer::get();
    // perform SLAM
    {
        ORB_SLAM3::System SLAM(VOCAB_PATH, data->strSettingsFile(), init.sensor());
        // iterate through frames
        for(HandySLAM::Frame& frameCurr : *data) {
            if(init.usingMono) {
                SLAM.TrackMonocular(frameCurr.im, frameCurr.timestamp, frameCurr.vImuMeas);
            } else {
                SLAM.TrackRGBD(frameCurr.im, frameCurr.depthmap, frameCurr.timestamp, frameCurr.vImuMeas); 
            }
        }
        SLAM.Shutdown();
        while(!SLAM.isShutDown()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // wait for input before closing the visualizer
        std::cout << "Press ENTER to save trajectory. [^C] to exit without saving." << std::endl;
        std::cin.get();
        // save trajectory
        SLAM.SaveTrajectoryTUM(init.pathScene / "trajectory.txt");
    }
    return 0;
}
