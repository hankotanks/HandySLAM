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

    std::locale::global(std::locale::classic());

    std::filesystem::path pathScene(argv[1]);
    ASSERT_PATH_EXISTS(pathScene);

    HandySLAM::DataloaderStray data(pathScene, SIZE_INTERNAL);
    {
        ORB_SLAM3::System SLAM(VOCAB_PATH, data.pathSettings(), ORB_SLAM3::System::IMU_RGBD);

        std::size_t maps = 0;
        std::size_t stateCurr, statePrev = 0;
        std::optional<HandySLAM::Frame> frameCurr;
        std::optional<ORB_SLAM3::IMU::Point> meas;
        while((frameCurr = data.next())) {
            // sort IMU measurements
            std::sort(frameCurr->vImuMeas.begin(), frameCurr->vImuMeas.end(), [](ORB_SLAM3::IMU::Point a, ORB_SLAM3::IMU::Point b) { return a.t < b.t; });
            // remove non-contiguous entries
            if(meas) {
                while(!frameCurr->vImuMeas.empty() && frameCurr->vImuMeas.front().t < meas->t) {
                    frameCurr->vImuMeas.erase(frameCurr->vImuMeas.cbegin());
                }                 
            }
            // std::cout << std::fixed << std::setprecision(4) << frameCurr->timestamp << std::endl;
            // process frame
            SLAM.TrackRGBD(frameCurr->im, frameCurr->depthmap, frameCurr->timestamp, frameCurr->vImuMeas); 
            // check if SLAM broke and we had to make a new map
            stateCurr = SLAM.GetTrackingState();
            if(stateCurr == ORB_SLAM3::Tracking::RECENTLY_LOST && stateCurr != statePrev) ++maps; 
            statePrev = stateCurr;
            // store this frame's final IMU reading for the next iteration
            if(!frameCurr->vImuMeas.empty()) meas = frameCurr->vImuMeas.back();
        }

        std::cout << "Generated " << maps << " maps." << std::endl;
        std::cout << "Press ENTER to close viewer." << std::endl;
        std::cin.get();

        SLAM.Shutdown();
        while(!SLAM.isShutDown());
    }

    return 0;
}

#if 0
    std::cout << "timestamps [" << std::fixed << std::setprecision(4) << frame->timestamp << "]: ";
    for(const ORB_SLAM3::IMU::Point& pt : frame->vImuMeas) {
        std::cout << std::fixed << std::setprecision(4) << pt.t << ", ";
    }
    std::cout << std::endl;
#endif