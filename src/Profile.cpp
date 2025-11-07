#include "Profile.h"

#include <fstream>
#include <iostream>

#include "fkyaml.h"

#include "util.h"

namespace {
    std::string readFile(const std::filesystem::path& path) {
        std::ifstream file(path);
        std::ostringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
}

namespace HandySLAM {
    // TODO: All as_float calls in this constructor
    // should be redundant and check for integer values where relevant
    // also, we need to check support for 1e3 notation
    Profile::Profile(const std::string& profile, cv::Size sizeDepthmap, std::size_t fps) {
        // fps
        fps_ = fps;
        // profile
        std::filesystem::path pathProfile = std::filesystem::path(PROJECT_ROOT) / "profiles";
        std::filesystem::path pathTransform = pathProfile / (profile + "-camchain-imucam.yaml");
        std::filesystem::path pathNoise = pathProfile / (profile + "-imu.yaml");
        ASSERT_PATH_EXISTS(pathTransform);
        ASSERT_PATH_EXISTS(pathNoise);
        // parse IMU noise parameters
        auto nodeRoot = fkyaml::node::deserialize(readFile(pathNoise));
        auto nodeImu = nodeRoot["imu0"];
        noiseGyr_ = static_cast<double>(nodeImu["gyroscope_noise_density"].as_float());
        noiseAcc_ = static_cast<double>(nodeImu["accelerometer_noise_density"].as_float());
        rwalkGyr_ = static_cast<double>(nodeImu["gyroscope_random_walk"].as_float());
        rwalkAcc_ = static_cast<double>(nodeImu["accelerometer_random_walk"].as_float());
        // parse update_rate
        updateRate_ = static_cast<double>(nodeImu["update_rate"].as_float());
        // parse intrinsics
        nodeRoot = fkyaml::node::deserialize(readFile(pathTransform));
        auto nodeCam = nodeRoot["cam0"];
        auto nodeIntrinsics = nodeCam["intrinsics"].as_seq();
        if(nodeIntrinsics.size() != 4) {
            log_err("Failed to parse intrinsics [", pathTransform, "].");
            exit(1);
        }
        fx_ = static_cast<double>(nodeIntrinsics[0].as_float());
        fy_ = static_cast<double>(nodeIntrinsics[1].as_float());
        cx_ = static_cast<double>(nodeIntrinsics[2].as_float());
        cy_ = static_cast<double>(nodeIntrinsics[3].as_float());
        // parse resolution
        auto nodeResolution = nodeCam["resolution"].as_seq();
        if(nodeResolution.size() != 2) {
            log_err("Failed to prase resolution [", pathTransform, "].");
            exit(1);
        }
        sizeIm_.width = nodeResolution[0].as_int();
        sizeIm_.height = nodeResolution[1].as_int();
        sizeDepthmap_ = sizeDepthmap;
        // parse IMU calibration matrix
        Eigen::Matrix4d imu2cam;
        std::size_t row = 0, col;
        for(auto& nodeRow : nodeCam.at("T_cam_imu")) {
            col = 0;
            for(auto& nodeVal : nodeRow) {
                imu2cam(row, col) = static_cast<double>(nodeVal.as_float());
                col++;
            }
            if(col != 4) {
                log_err("Failed to parse IMU-to-camera transform [", pathTransform, "].");
                exit(1);
            }
            row++;
        }
        if(row != 4) {
            log_err("Failed to parse IMU-to-camera transform [", pathTransform, "].");
            exit(1);
        }
        // invert matrix to get camera-to-IMU transform
        cam2imu_ = imu2cam.inverse();
    }

    const std::string Profile::strSettingsFile() const {
        // create a temporary file
        std::filesystem::path pathSettings = std::filesystem::temp_directory_path() / (profile_ + ".yaml");
        std::ofstream writer(pathSettings);
        if(!writer) {
            log_err("Failed to write to [", pathSettings, "].");
            exit(1);
        }
        // write YAML settings to file
        writer << std::fixed << std::setprecision(4);
        writer << "%YAML:1.0" << std::endl;
        writer << "File.version: \"1.0\"" << std::endl;
        // camera
        writer << "Camera.type: \"PinHole\"" << std::endl;
        writer << "Camera1.fx: " << fx_ << std::endl;
        writer << "Camera1.fy: " << fy_ << std::endl;
        writer << "Camera1.cx: " << cx_ << std::endl;
        writer << "Camera1.cy: " << cy_ << std::endl;
        writer << "Camera1.k1: 0.0" << std::endl;
        writer << "Camera1.k2: 0.0" << std::endl;
        writer << "Camera1.p1: 0.0" << std::endl;
        writer << "Camera1.p2: 0.0" << std::endl;
        writer << "Camera.width: "     << sizeIm_.width << std::endl;
        writer << "Camera.height: "    << sizeIm_.height << std::endl;
        writer << "Camera.newWidth: "  << sizeDepthmap_.width << std::endl;
        writer << "Camera.newHeight: " << sizeDepthmap_.height << std::endl;
        writer << "Camera.fps: " << fps_ << std::endl;
        writer << "Camera.RGB: 1" << std::endl;
        writer << "Stereo.ThDepth: 40.0" << std::endl;
        writer << "Stereo.b: 0.0745" << std::endl;
        writer << "RGBD.MinDepth: 0.1" << std::endl;
        writer << "RGBD.MaxDepth: 5.0" << std::endl;
        writer << "RGBD.DepthMapFactor: 1000.0" << std::endl;
        // IMU
        writer << "IMU.T_b_c1: !!opencv-matrix" << std::endl;
        writer << "   rows: 4" << std::endl;
        writer << "   cols: 4" << std::endl;
        writer << "   dt: f" << std::endl;
        writer << "   data: [ ";
        for(std::size_t i = 0; i < cam2imu_.rows(); ++i) {
            for(std::size_t j = 0; j < cam2imu_.cols(); ++j) {
                writer << cam2imu_(i, j);
                if (!(i == cam2imu_.rows() - 1 && j == cam2imu_.cols() - 1)) {
                    writer << ", ";
                }
            }
        }
        writer << " ]" << std::endl;
        writer << "IMU.InsertKFsWhenLost: 0" << std::endl;
        writer << "IMU.NoiseGyro: " << noiseGyr_ << std::endl;
        writer << "IMU.NoiseAcc: "  << noiseAcc_ << std::endl; 
        writer << "IMU.GyroWalk: "  << rwalkGyr_ << std::endl; 
        writer << "IMU.AccWalk: "   << rwalkAcc_ << std::endl;  
        writer << "IMU.Frequency: " << updateRate_ << std::endl;
        // generic parameters
        writer << "ORBextractor.nFeatures: 1250" << std::endl;
        writer << "ORBextractor.scaleFactor: 1.2" << std::endl;
        writer << "ORBextractor.nLevels: 8" << std::endl;
        writer << "ORBextractor.iniThFAST: 20" << std::endl;
        writer << "ORBextractor.minThFAST: 7" << std::endl;
        writer << "Viewer.KeyFrameSize: 0.05" << std::endl;
        writer << "Viewer.KeyFrameLineWidth: 1.0" << std::endl;
        writer << "Viewer.GraphLineWidth: 0.9" << std::endl;
        writer << "Viewer.PointSize: 2.0" << std::endl;
        writer << "Viewer.CameraSize: 0.08" << std::endl;
        writer << "Viewer.CameraLineWidth: 3.0" << std::endl;
        writer << "Viewer.ViewpointX: 0.0" << std::endl;
        writer << "Viewer.ViewpointY: -0.7" << std::endl;
        writer << "Viewer.ViewpointZ: -3.5" << std::endl;
        writer << "Viewer.ViewpointF: 500.0" << std::endl;
        // close reader
        writer.close();
        // return path to the settings file
        return pathSettings.string();
    }
}
