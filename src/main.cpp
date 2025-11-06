#include <iostream>

#include <filesystem>
#include <optional>
#include <algorithm>

#include <System.h>

#include "Config.h"
#include "Profile.h"
#include "DataloaderStray.h"
#include "util.h"

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
    cv::Size sizeDepthmap = data.sizeDepthmap();
    double fps = data.fps();
    HandySLAM::Profile profile(argv[2], sizeDepthmap, fps);
    std::string strSettingsFile = profile.strSettingsFile();
    {
        ORB_SLAM3::System SLAM(VOCAB_PATH, strSettingsFile, ORB_SLAM3::System::IMU_RGBD);
        // iterate through frames
        std::size_t maps = 1;
        int stateCurr;
        int statePrev = static_cast<int>(ORB_SLAM3::Tracking::NO_IMAGES_YET);
        for(HandySLAM::Frame& frameCurr : data) {
            // process frame
            SLAM.TrackRGBD(frameCurr.im, frameCurr.depthmap, frameCurr.timestamp, frameCurr.vImuMeas); 
            // check if SLAM broke and we had to make a new map
            stateCurr = SLAM.GetTrackingState();
            if(stateCurr == static_cast<int>(ORB_SLAM3::Tracking::RECENTLY_LOST) 
                && stateCurr != statePrev) ++maps; 
            statePrev = stateCurr;
        }
        // wait for input before closing the visualizer
        std::cout << "Info: Generated " << maps << " maps." << std::endl;
        std::cout << "Press ENTER to close viewer." << std::endl;
        std::cin.get();
        // shutdown the SLAM system
        SLAM.Shutdown();
        while(!SLAM.isShutDown()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return 0;
}
