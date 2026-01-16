#include "Dataloader.h"

#include <exception>

#include "Initializer.h"
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
    bool Frame::vImuMeasValidate(double timestampPrev) const {
        if(vImuMeas.empty()) {
            log_err("No IMU measurements for current frame.");
            return false;
        }
        if(vImuMeas.size() > 1) {
            for(std::size_t i = 1; i < vImuMeas.size(); ++i) {
                if(vImuMeas[i].t < vImuMeas[i - 1].t) {
                    log_err("IMU measurements are not monotonically increasing.");
                    return false;
                }
            }
        }
        if(vImuMeas.front().t < timestampPrev || vImuMeas.back().t > timestamp) {
            log_err("IMU measurements are not bounded by imagery timestamps.");
            return false;
        }
        return true;
    }

    Profile::Profile(const std::string& profileName) {
        name = profileName;
        // profile
        std::filesystem::path pathProfile = std::filesystem::path(PROJECT_ROOT) / "profiles";
        std::filesystem::path pathTransform = pathProfile / (name + "-camchain-imucam.yaml");
        std::filesystem::path pathNoise = pathProfile / (name + "-imu.yaml");
        ASSERT_PATH_EXISTS(pathTransform);
        ASSERT_PATH_EXISTS(pathNoise);
        // parse IMU noise parameters
        auto nodeRoot = fkyaml::node::deserialize(readFile(pathNoise));
        auto nodeImu = nodeRoot["imu0"];
        gyroscope_noise_density = static_cast<double>(nodeImu["gyroscope_noise_density"].as_float());
        gyroscope_random_walk = static_cast<double>(nodeImu["gyroscope_random_walk"].as_float());
        accelerometer_noise_density = static_cast<double>(nodeImu["accelerometer_noise_density"].as_float());
        accelerometer_random_walk = static_cast<double>(nodeImu["accelerometer_random_walk"].as_float());
        // parse update_rate
        update_rate = static_cast<double>(nodeImu["update_rate"].as_float());
        // parse intrinsics
        nodeRoot = fkyaml::node::deserialize(readFile(pathTransform));
        auto nodeCam = nodeRoot["cam0"];
        auto nodeIntrinsics = nodeCam["intrinsics"].as_seq();
        if(nodeIntrinsics.size() != 4) {
            log_err("Failed to parse intrinsics [", pathTransform, "].");
            throw std::exception();
        }
        unused.intrinsics.fx = static_cast<double>(nodeIntrinsics[0].as_float());
        unused.intrinsics.fy = static_cast<double>(nodeIntrinsics[1].as_float());
        unused.intrinsics.cx = static_cast<double>(nodeIntrinsics[2].as_float());
        unused.intrinsics.cy = static_cast<double>(nodeIntrinsics[3].as_float());
        // parse timeshift_cam_imu as seconds
        timeshift_cam_imu = static_cast<double>(nodeCam["timeshift_cam_imu"].as_float()) * 1.0e-9;
        // parse resolution
        auto nodeResolution = nodeCam["resolution"].as_seq();
        if(nodeResolution.size() != 2) {
            log_err("Failed to parse resolution [", pathTransform, "].");
            throw std::exception();
        }
        unused.resolution.width = nodeResolution[0].as_int();
        unused.resolution.height = nodeResolution[1].as_int();
        // parse IMU calibration matrix
        std::size_t row = 0, col;
        for(auto& nodeRow : nodeCam.at("T_cam_imu")) {
            col = 0;
            for(auto& nodeVal : nodeRow) {
                T_cam_imu(row, col) = static_cast<double>(nodeVal.as_float());
                col++;
            }
            if(col != 4) {
                log_err("Failed to parse IMU-to-camera transform [", pathTransform, "].");
                throw std::exception();
            }
            row++;
        }
        if(row != 4) {
            log_err("Failed to parse IMU-to-camera transform [", pathTransform, "].");
            throw std::exception();
        }
    }

    Dataloader::Dataloader(const std::filesystem::path& pathScene) : 
        pathScene_(pathScene) { /* STUB */ }

    Dataloader::Dataloader(const std::filesystem::path& pathScene, const std::string& profileName) :
        pathScene_(pathScene), profile_(profileName) { /* STUB */ }

    const std::string Dataloader::strSettingsFile() {
        // ensure invariants are held
        if(!profile_.has_value())
            throw std::logic_error("profile_ was not set by derived Dataloader.");
        if(!metadata_.has_value())
            throw std::logic_error("metadata_ was not set by derived Dataloader.");
        // don't generate the file a second time
        if(strSettingsFile_) return strSettingsFile_->string();
        // create file
        std::filesystem::path pathSettings = std::filesystem::temp_directory_path() / (profile_->name + ".yaml");
        strSettingsFile_.emplace(pathSettings);
        std::ofstream writer(pathSettings);
        if(!writer) {
            log_err("Failed to write to [", pathSettings, "].");
            throw std::exception();
        }
        writer.imbue(std::locale::classic());
        // write YAML settings to file
        writer << std::fixed << std::setprecision(4);
        writer << "%YAML:1.0" << std::endl;
        writer << "File.version: \"1.0\"" << std::endl;
        // camera
        writer << "Camera.type: \"PinHole\"" << std::endl;
        writer << "Camera1.fx: " << metadata_->intrinsics.fx << std::endl;
        writer << "Camera1.fy: " << metadata_->intrinsics.fy << std::endl;
        writer << "Camera1.cx: " << metadata_->intrinsics.cx << std::endl;
        writer << "Camera1.cy: " << metadata_->intrinsics.cy << std::endl;
        if(metadata_->intrinsics.distortion) {
            writer << "Camera1.k1: " << metadata_->intrinsics.distortion->k1 << std::endl;
            writer << "Camera1.k2: " << metadata_->intrinsics.distortion->k2 << std::endl;
            writer << "Camera1.k3: 0.0" << std::endl;
            writer << "Camera1.p1: " << metadata_->intrinsics.distortion->p1 << std::endl;
            writer << "Camera1.p2: " << metadata_->intrinsics.distortion->p2 << std::endl;
        }
        writer << "Camera.width: "     << metadata_->sizeIm.width << std::endl;
        writer << "Camera.height: "    << metadata_->sizeIm.height << std::endl;
        if(!Initializer::get().usingMono) {
            writer << "Camera.newWidth: "  << metadata_->sizeDepthmap.width << std::endl;
            writer << "Camera.newHeight: " << metadata_->sizeDepthmap.height << std::endl;
        }
        writer << "Camera.fps: " << metadata_->fps << std::endl;
        writer << "Camera.RGB: 1" << std::endl;
        writer << "Stereo.ThDepth: 5.0" << std::endl;
        writer << "Stereo.b: 1.0" << std::endl;
        writer << "RGBD.DepthMapFactor: 1000.0" << std::endl;
        // IMU
        writer << "IMU.T_b_c1: !!opencv-matrix" << std::endl;
        writer << "   rows: 4" << std::endl;
        writer << "   cols: 4" << std::endl;
        writer << "   dt: f" << std::endl;
        writer << "   data: [ ";
        Eigen::Matrix4d T_imu_cam = profile_->T_cam_imu.inverse();
        for(std::size_t i = 0; i < T_imu_cam.rows(); ++i) {
            for(std::size_t j = 0; j < T_imu_cam.cols(); ++j) {
                // remove translation
                writer << ((j != 3 || i == 3) ? T_imu_cam(i, j) : 0.0);
                if(!(i == T_imu_cam.rows() - 1 && j == T_imu_cam.cols() - 1)) writer << ", ";
            }
        }
        writer << " ]" << std::endl;
        writer << "IMU.InsertKFsWhenLost: 0" << std::endl;
        writer << "IMU.NoiseGyro: " << profile_->gyroscope_noise_density << std::endl;
        writer << "IMU.NoiseAcc: "  << profile_->accelerometer_noise_density << std::endl; 
        writer << "IMU.GyroWalk: "  << profile_->gyroscope_random_walk << std::endl; 
        writer << "IMU.AccWalk: "   << profile_->accelerometer_random_walk << std::endl;  
        writer << "IMU.Frequency: " << profile_->update_rate << std::endl;
        // generic parameters
        writer << "ORBextractor.nFeatures: 2500" << std::endl;
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
        // return path
        return pathSettings.string();
    }
}