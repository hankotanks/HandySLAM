#include <iostream>
#include <filesystem>

#include <System.h>

#include "Config.h"
#include "Dataloader.h"
#include "DataloaderStray.h"
#include "DataloaderScanNet.h"
#include "Initializer.h"
#include "VolumeBuilder.h"

int main(int argc, char* argv[]) {
    // register dataloaders with the initializers
    HandySLAM::Initializer::add<HandySLAM::DataloaderStray>("stray");
    HandySLAM::Initializer::add<HandySLAM::DataloaderScanNet>("scannetpp");
    // build dataloader
    HandySLAM::Dataloader* data;
    try {
        data = HandySLAM::Initializer::init(argc, argv);
        if(!data) {
            log_err("Failed to initialize Dataloader.");
            return 1;
        }
    } catch(...) {
        log_err("Failed to initialize Dataloader.");
        return 1;
    }
    // get handle to initializer
    const HandySLAM::Initializer& init = HandySLAM::Initializer::get();
    // perform SLAM
    {
        ORB_SLAM3::System SLAM(VOCAB_PATH, data->strSettingsFile(), init.sensor());
        // iterate through frames
        for(HandySLAM::Frame& frame : *data) {
            if(init.usingMono) {
                SLAM.TrackMonocular(frame.im, frame.timestamp, frame.vImuMeas);
            } else {
                SLAM.TrackRGBD(frame.im, frame.depthmap, frame.timestamp, frame.vImuMeas); 
            }
        }
        // shutdown threads (including viewer)
        SLAM.ShutdownAndWait();
        // save a TSDF volume if it was requested
        if(init.saveVolume) {
            data = HandySLAM::Initializer::restart();
            if(!data) {
                log_err("Failed to initialize Dataloader.");
                return 1;
            }
            // start volume construction
            HandySLAM::VolumeBuilder out(data->intrinsics(), init.voxelSize, init.depthCutoff);
            // iterate through the poses of all frames
            Sophus::SE3f framePose;
            for(const HandySLAM::Frame& frame : *data) {
                if(!SLAM.GetPose(framePose, frame.timestamp)) {
                    std::cout << frame.index << " skipped" << std::endl;
                    continue;
                }
                // add frame to volume
                out.integrateFrame(frame.im, frame.depthmap, framePose);
            }
            // write the completed TSDF
            std::filesystem::path pathOut(data->pathScene() / "out.ply");
            if(!out.save(pathOut)) {
                log_err("Failed to build TSDF Volume");
                return 1;
            }
        }
    }
    return 0;
}

