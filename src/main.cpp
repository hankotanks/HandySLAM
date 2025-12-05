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

bool extract_edges_from_atlas(ORB_SLAM3::Atlas* atlas, std::set<std::pair<double, double>>& edge_set);

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
        std::cout << "Press ENTER to save camera trajectory and graph edges. [^C] to exit without saving" << std::endl;
        std::cin.get();
        // save trajectory
        SLAM.SaveTrajectoryTUM(init.pathScene / "trajectory.txt");
        std::cout << "Finished saving camera trajectory" << std::endl;
        // save edges
        std::set<std::pair<double, double>> edges;
        if(!extract_edges_from_atlas(SLAM.mpAtlas, edges)) {
            log_err("Failed to extract graph edges from Atlas");
            return 1;
        }
        std::filesystem::path pathEdges = init.pathScene / "edges.txt";
        std::ofstream writer(pathEdges);
        if(!writer) {
            log_err("Failed to write to [", pathEdges, "].");
            return 1;
        }
        std::cout << "Saving graph edges to " << pathEdges << " ..." << std::endl;
        writer.imbue(std::locale::classic());
        writer << std::fixed << std::setprecision(10);
        for(const auto& e : edges) writer << e.first << " " << e.second << std::endl;
        writer.close();
        std::cout << "Finished saving graph edges" << std::endl;
    }
    return 0;
}

bool extract_edges_from_atlas(ORB_SLAM3::Atlas* atlas, std::set<std::pair<double, double>>& edges) {
    std::vector<ORB_SLAM3::Map*> maps = atlas->GetAllMaps();
    for(ORB_SLAM3::Map* map : maps) {
        std::vector<ORB_SLAM3::KeyFrame*> keyframes = map->GetAllKeyFrames();
        
        for(ORB_SLAM3::KeyFrame* kf : keyframes) {
            int fi = kf->mnFrameId;
            double fi_timestamp = kf->mTimeStamp;
            ORB_SLAM3::KeyFrame* parent = kf->GetParent();
            if(parent) {
                int fj = parent->mnFrameId;
                double fj_timestamp = parent->mTimeStamp;
                if (fi != fj)
                {
                    auto p = std::minmax(fi_timestamp, fj_timestamp);
                    edges.insert(p);
                }
            }
            for(ORB_SLAM3::KeyFrame* loopKF : kf->GetLoopEdges()) {
                int fj = loopKF->mnFrameId;
                double fj_timestamp = loopKF->mTimeStamp;
                if (fi != fj) {
                    auto p = std::minmax(fi_timestamp, fj_timestamp);
                    edges.insert(p);
                }
            }

            for(ORB_SLAM3::KeyFrame* connKF : kf->GetConnectedKeyFrames()) {
                int fj = connKF->mnFrameId;
                double fj_timestamp = connKF->mTimeStamp;
                if (fi != fj) {
                    auto p = std::minmax(fi_timestamp, fj_timestamp);
                    edges.insert(p);
                }
            }
        }
    }
    return true;
}
