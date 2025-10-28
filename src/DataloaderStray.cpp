#include "DataloaderStray.h"

#include <filesystem>
#include <optional>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <opencv2/core/mat.hpp>
#include <sstream>

#include "Dataloader.h"

#define ASSERT_PATH_EXISTS(path_) if(!std::filesystem::exists(path_)) { \
        std::cout << "Path does not exist [" << path_ << "]." << std::endl; \
        exit(1); \
    }

namespace HandySLAM {
    DataloaderStray::DataloaderStray(const std::filesystem::path& pathScene, bool generateSettings) : Dataloader(pathScene) {
        // initialize frame index
        frameIdx_ = 0;
        // rgb.mp4
        pathRGB_ = pathScene / "rgb.mp4";
        ASSERT_PATH_EXISTS(pathRGB_);
        // open RGB video capture
        cap_ = cv::VideoCapture(pathRGB_);
        if(!cap_.isOpened()) {
            std::cout << "Failed to open [" << pathRGB_ << "]." << std::endl;
            exit(1);
        }
        fps_  = static_cast<std::size_t>(cap_.get(cv::CAP_PROP_FPS));
        std::size_t w, h;
        w = static_cast<std::size_t>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        h = static_cast<std::size_t>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        sizeOriginal_ = cv::Size(w, h);
        sizeInternal_ = sizeOriginal_;
        // depth/*
        pathDepth_ = pathScene / "depth";
        ASSERT_PATH_EXISTS(pathDepth_);
        // odometry.csv
        pathOdom_ = pathScene / "odometry.csv";
        ASSERT_PATH_EXISTS(pathOdom_);
        readerOdom_ = std::ifstream(pathOdom_, std::ios::binary);
        if(!readerOdom_.is_open()) {
            std::cout << "Failed to open [" << pathOdom_ << "]." << std::endl;
            exit(1);
        }
        std::string temp;
        std::getline(readerOdom_, temp);
        // imu.csv
        pathIMU_ = pathScene / "imu.csv";
        ASSERT_PATH_EXISTS(pathIMU_);
        readerIMU_ = std::ifstream(pathIMU_, std::ios::binary);
        if(!readerIMU_.is_open()) {
            std::cout << "Failed to open [" << pathIMU_ << "]." << std::endl;
            exit(1);
        }
        std::getline(readerIMU_, temp);
        // frequency
        freq_ = DataloaderStray::getFrequencyIMU();
        // break early if settings don't need to be generated
        if(!generateSettings) return; 
        // camera_matrix.csv
        std::filesystem::path pathCameraMatrix(pathScene / "camera_matrix.csv");
        ASSERT_PATH_EXISTS(pathCameraMatrix);
        // generate iphone.yaml
        DataloaderStray::generateSettingsFile(pathCameraMatrix);
    }

    DataloaderStray::DataloaderStray(const std::filesystem::path& pathScene) : DataloaderStray(pathScene, true) { /* STUB */ }

    DataloaderStray::DataloaderStray(const std::filesystem::path& pathScene, cv::Size sizeInternal) : DataloaderStray(pathScene, false) {
        sizeInternal_ = sizeInternal;
        // generate settings
        std::filesystem::path pathCameraMatrix(pathScene / "camera_matrix.csv");
        ASSERT_PATH_EXISTS(pathCameraMatrix);
        DataloaderStray::generateSettingsFile(pathCameraMatrix);
    }

    DataloaderStray::~DataloaderStray() {
        cap_.release();
        readerOdom_.close();
        readerIMU_.close();
    }

    const std::filesystem::path& DataloaderStray::pathSettings() {
        return pathSettings_;
    }

    const std::optional<Frame> DataloaderStray::next() {
        Frame curr;
        curr.index = frameIdx_;
        // read RGB frame
        if(!cap_.read(curr.im)) {
            std::cout << "Failed to get RGB imagery on frame " << frameIdx_ << std::endl;
            return std::nullopt;
        }
        // depth frame
        std::optional<cv::Mat> depthmap = DataloaderStray::nextDepthFrame();
        if(!depthmap) {
            std::cout << "Failed to get depth map on frame " << frameIdx_ << std::endl;
            return std::nullopt;
        }
        cv::resize(*depthmap, *depthmap, sizeInternal_);
        curr.depthmap = std::move(*depthmap);
        // timestamp
        std::optional<double> timestamp = DataloaderStray::nextTimestamp();
        if(!timestamp) {
            std::cout << "Failed to get timestamp on frame " << frameIdx_ << std::endl;
            return std::nullopt;
        }
        curr.timestamp = *timestamp;
        // vImuMeas
        curr.vImuMeas = DataloaderStray::nextIMU(curr.timestamp);
        // return frame
        frameIdx_++;
        return curr;
    }

    std::optional<cv::Mat> DataloaderStray::nextDepthFrame() {
        // build filename
        std::ostringstream fname;
        fname << std::setw(6) << std::setfill('0') << frameIdx_ << ".png";
        // check validity of path
        std::filesystem::path pathDepthFrame = pathDepth_ / fname.str();
        if(!std::filesystem::exists(pathDepthFrame)) return std::nullopt;
        // read depth map
        cv::Mat depthmap = cv::imread(pathDepthFrame, cv::IMREAD_UNCHANGED);
        if(depthmap.data == nullptr) {
            std::cout << "Failed to read depth frame." << std::endl;
            return std::nullopt;
        }
        return depthmap;
    }

    std::optional<double> DataloaderStray::nextTimestampGeneric(std::ifstream& reader) {
        std::string line;
        if(!std::getline(reader, line)) {
            std::cout << "Failed to get timestamp." << std::endl;
            return std::nullopt;
        }
        std::stringstream lineStream(line);
        std::string elem;
        if(std::getline(lineStream, elem, ',')) return static_cast<double>(std::stold(elem));
        std::cout << "Failed to get timestamp." << std::endl;
        return std::nullopt;
    }

    std::optional<double> DataloaderStray::nextTimestamp() {
        return DataloaderStray::nextTimestampGeneric(readerOdom_);
    }

    std::vector<ORB_SLAM3::IMU::Point> DataloaderStray::nextIMU(double timestamp) {
        std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
        // loop until we catch up
        double timestampCurr;
        do {
            // save position
            fpos<__mbstate_t> pos = readerIMU_.tellg();
            // get line
            std::string line;
            if(!std::getline(readerIMU_, line)) {
                std::cout << "Failed to get IMU frame data." << std::endl;
                exit(1);
            }
            // read the current timestamp
            std::stringstream lineStream(line);
            std::string elem;
            if(!std::getline(lineStream, elem, ',')) exit(1);
            timestampCurr = static_cast<double>(std::stold(elem));
            // check if we need to break
            if(timestampCurr >= timestamp) {
                readerIMU_.clear();
                readerIMU_.seekg(pos);
                return vImuMeas;
            }
            // read accelerometry and gyro data
            float acc_x, acc_y, acc_z, ang_vel_x, ang_vel_y, ang_vel_z;
            if(!std::getline(lineStream, elem, ',')) exit(1);
            acc_x = static_cast<float>(std::stold(elem));
            if(!std::getline(lineStream, elem, ',')) exit(1);
            acc_y = static_cast<float>(std::stold(elem));
            if(!std::getline(lineStream, elem, ',')) exit(1);
            acc_z = static_cast<float>(std::stold(elem));
            if(!std::getline(lineStream, elem, ',')) exit(1);
            ang_vel_x = static_cast<float>(std::stold(elem));
            if(!std::getline(lineStream, elem, ',')) exit(1);
            ang_vel_y = static_cast<float>(std::stold(elem));
            if(!std::getline(lineStream, elem)) exit(1);
            ang_vel_z = static_cast<float>(std::stold(elem));
            // add the new point
            vImuMeas.emplace_back(acc_x, acc_y, acc_z, ang_vel_x, ang_vel_y, ang_vel_z, timestampCurr);
        } while(true);
        return std::vector<ORB_SLAM3::IMU::Point>{};
    }

    double DataloaderStray::getFrequencyIMU() {
        // save current position
        auto pos = readerIMU_.tellg();
        if(pos == -1) exit(1);
        // get two adjacent timestamps
        std::optional<double> t1 = DataloaderStray::nextTimestampGeneric(readerIMU_);
        std::optional<double> t2 = DataloaderStray::nextTimestampGeneric(readerIMU_);
        // reset stream back to where it was
        readerIMU_.clear();
        readerIMU_.seekg(pos);
        if(!t1 || !t2) exit(1);
        // return frequency
        return std::floor(1.0 / (*t2 - *t1));
    }

    void DataloaderStray::generateSettingsFile(const std::filesystem::path& pathCameraMatrix) {
        // open the input stream
        std::ifstream reader(pathCameraMatrix);
        if(!reader.is_open()) {
            std::cout << "Failed to open [" << pathCameraMatrix << "]." << std::endl;
            exit(1);
        }
        // get first row
        std::string line;
        if(!std::getline(reader, line)) {
            std::cout << "Failed to get IMU frame data." << std::endl;
            exit(1);
        }
        std::stringstream lineStream(line);
        std::string elem;
        // read the relevant intrinsics
        double fx, fy, cx, cy;
        // row: 0, col: 0
        if(!std::getline(lineStream, elem, ',')) exit(1);
        fx = static_cast<double>(std::stold(elem));
        // row: 0, col: 1
        if(!std::getline(lineStream, elem, ',')) exit(1);
        // row: 0, col: 2
        if(!std::getline(lineStream, elem)) exit(1);
        cx = static_cast<double>(std::stold(elem));
        // get second row
        if(!std::getline(reader, line)) {
            std::cout << "Failed to get IMU frame data." << std::endl;
            exit(1);
        }
        lineStream = std::move(std::stringstream(line));
        // row: 1, col: 0
        if(!std::getline(lineStream, elem, ',')) exit(1);
        // row: 1, col: 1
        if(!std::getline(lineStream, elem, ',')) exit(1);
        fy = static_cast<double>(std::stold(elem));
        // row: 1, col: 2
        if(!std::getline(lineStream, elem)) exit(1);
        cy = static_cast<double>(std::stold(elem));
        // get third row
        if(!std::getline(reader, line)) {
            std::cout << "Failed to get IMU frame data." << std::endl;
            exit(1);
        }
        lineStream = std::move(std::stringstream(line));
        // row: 2, col: 0
        if(!std::getline(lineStream, elem, ',')) exit(1);
        // row: 2, col: 1
        if(!std::getline(lineStream, elem, ',')) exit(1);
        // row: 2, col: 2
        if(!std::getline(lineStream, elem)) exit(1);
        double scalar = static_cast<double>(std::stold(elem));
        // adjust by scalar (although should always be 1.0)
        fx /= scalar;
        fy /= scalar;
        cx /= scalar;
        cy /= scalar;
        // create a temporary file
        pathSettings_ = std::filesystem::temp_directory_path() / "iphone.yaml";
        std::ofstream writer(pathSettings_);
        if(!writer) {
            std::cout << "Failed to write to [" << pathSettings_ << "]." << std::endl;
            exit(1);
        }
        // write YAML settings to file
        writer << std::fixed << std::setprecision(4);
        writer << "%YAML:1.0" << std::endl;
        writer << "File.version: \"1.0\"" << std::endl;
        // camera
        writer << "Camera.type: \"PinHole\"" << std::endl;
        writer << "Camera1.fx: " << fx << std::endl;
        writer << "Camera1.fy: " << fy << std::endl;
        writer << "Camera1.cx: " << cx << std::endl;
        writer << "Camera1.cy: " << cy << std::endl;
        writer << "Camera1.k1: 0.0" << std::endl;
        writer << "Camera1.k2: 0.0" << std::endl;
        writer << "Camera1.p1: 0.0" << std::endl;
        writer << "Camera1.p2: 0.0" << std::endl;
        writer << "Camera.width: "     << sizeOriginal_.width << std::endl;
        writer << "Camera.height: "    << sizeOriginal_.height << std::endl;
        writer << "Camera.newWidth: "  << sizeInternal_.width << std::endl;
        writer << "Camera.newHeight: " << sizeInternal_.height << std::endl;
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
        writer << "   data: [1.0,  0.0,  0.0, 0.0," << std::endl;
        writer << "          0.0, -1.0,  0.0, 0.0," << std::endl;
        writer << "          0.0,  0.0, -1.0, 0.0," << std::endl;
        writer << "          0.0,  0.0,  0.0, 1.0]" << std::endl;
        writer << "IMU.InsertKFsWhenLost: 0" << std::endl;
        writer << "IMU.NoiseGyro: 5.1e-3" << std::endl;  // TODO: Consider these values
        writer << "IMU.NoiseAcc:  1.4e-2" << std::endl;  // TODO: Consider these values
        writer << "IMU.GyroWalk:  5.0e-4" << std::endl;  // TODO: Consider these values
        writer << "IMU.AccWalk:   2.5e-3" << std::endl;  // TODO: Consider these values
        writer << "IMU.Frequency: " << freq_ << std::endl;
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
    }
}
