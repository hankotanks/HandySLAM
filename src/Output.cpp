#include "Output.h"

#include <exception>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "Initializer.h"
#include "VolumeBuilder.h"
#include "util.h"

namespace HandySLAM {
    Output::Output(const ORB_SLAM3::System& SLAM, int argc, char* argv[]) {
        HandySLAM::Dataloader* data = HandySLAM::Initializer::init(argc, argv);
        if(!data) {
            log_err("Failed to initialize Dataloader.");
            throw std::exception();
        }

        Sophus::SE3f framePose;
        for(const HandySLAM::Frame& frame : *data) {
            if(!SLAM.GetPose(framePose, frame.timestamp)) {
                std::cout << frame.index << " skipped" << std::endl;
                continue;
            }
#if 0
            // Resize color to match depth
            cv::Mat color_resized;
            cv::resize(frame.im, color_resized, frame.depthmap.size());

            // Convert to RGB
            cv::Mat color_rgb;
            cv::cvtColor(color_resized, color_rgb, cv::COLOR_BGR2RGB);

            // Create Open3D images
            auto o3d_color = std::make_shared<open3d::geometry::Image>();
            o3d_color->Prepare(color_rgb.cols, color_rgb.rows, 3, 1);
            std::memcpy(o3d_color->data_.data(), color_rgb.data, color_rgb.total() * 3);

            auto o3d_depth = std::make_shared<geometry::Image>();
            cv::Mat depth_float;
            if (frame.depthmap.type() != CV_32F) {
                frame.depthmap.convertTo(depth_float, CV_32F, 1.0 / 1000.0); // mm -> meters
            } else {
                depth_float = frame.depthmap;
            }
            o3d_depth->Prepare(depth_float.cols, depth_float.rows, 1, 4);
            std::memcpy(o3d_depth->data_.data(), depth_float.data, depth_float.total() * sizeof(float));

            // Pose: Sophus::SE3f -> Eigen::Matrix4d
            Eigen::Matrix4d Tcw = pose.matrix().cast<double>().inverse();

            // Integrate into TSDF
            tsdf_volume->Integrate(*o3d_depth, *o3d_color, intrinsic, Tcw);
#endif
        }
    }
}