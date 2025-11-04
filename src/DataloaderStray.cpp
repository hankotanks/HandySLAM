#include "DataloaderStray.h"

#include <filesystem>
#include <optional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <opencv2/core/mat.hpp>

#include "Dataloader.h"

template <typename... Args>
void log_err(Args&&... args) {
    std::cout << "[ERROR] ";
    int dummy[] = { 0, ((std::cout << std::forward<Args>(args) << ' '), 0) ... };
    (void) dummy;
    std::cout << std::endl;
}

#define ASSERT_PATH_EXISTS(path_) if(!std::filesystem::exists(path_)) { \
        log_err("Path does not exist [", path_, "]."); \
        exit(1); \
    }

namespace {
    bool readLine(std::ifstream& reader, std::stringstream& stream) {
        std::string line;
        if(!std::getline(reader, line)) {
            log_err("Failed to read line.");
            return false;
        }
        stream = std::move(std::stringstream(line));
        stream.imbue(std::locale::classic());
        return true;
    }

    template <typename T>
    bool parse(std::stringstream& stream, T& val, bool final) {
        std::string elem;
        if(!std::getline(stream, elem, final ? '\n' : ',')) {
            log_err("Failed to parse value.");
            return false;
        }
        std::stringstream temp(elem);
        temp.imbue(std::locale::classic());
        temp >> val;
        return !stream.fail();
    }
} // private scope

namespace HandySLAM {
    DataloaderStray::DataloaderStray(const std::filesystem::path& pathScene, std::optional<cv::Size> sizeInternal) : Dataloader(pathScene) {
        // rgb.mp4
        pathRGB_ = pathScene / "rgb.mp4";
        ASSERT_PATH_EXISTS(pathRGB_);
        // open RGB video capture
        cap_ = cv::VideoCapture(pathRGB_);
        if(!cap_.isOpened()) {
            log_err("Failed to open ", pathRGB_, "].");
            exit(1);
        }
        fps_  = static_cast<std::size_t>(cap_.get(cv::CAP_PROP_FPS));
        std::size_t w, h;
        w = static_cast<std::size_t>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        h = static_cast<std::size_t>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        sizeOriginal_ = cv::Size(w, h);
        sizeInternal_ = sizeInternal.value_or(sizeOriginal_);
        // initialize frame index and count
        frameIdx_ = 0;
        frameCount_ = static_cast<std::size_t>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
        // depth/*
        pathDepth_ = pathScene / "depth";
        ASSERT_PATH_EXISTS(pathDepth_);
        // odometry.csv
        pathOdom_ = pathScene / "odometry.csv";
        ASSERT_PATH_EXISTS(pathOdom_);
        readerOdom_ = std::ifstream(pathOdom_, std::ios::binary);
        if(!readerOdom_.is_open()) {
            log_err("Failed to open ", pathOdom_, "].");
            exit(1);
        }
        std::string temp;
        std::getline(readerOdom_, temp);
        // imu.csv
        pathIMU_ = pathScene / "imu.csv";
        ASSERT_PATH_EXISTS(pathIMU_);
        readerIMU_ = std::ifstream(pathIMU_, std::ios::binary);
        if(!readerIMU_.is_open()) {
            log_err("Failed to open ", pathIMU_, "].");
            exit(1);
        }
        std::getline(readerIMU_, temp);
        // frequency
        auto pos = readerIMU_.tellg();
        if(pos == -1) {
            log_err("Failed to calculate IMU frequency.");
            exit(1);
        }
        // get two adjacent timestamps
        std::optional<double> t1 = DataloaderStray::timestampGeneric(readerIMU_);
        std::optional<double> t2 = DataloaderStray::timestampGeneric(readerIMU_);
        // reset stream back to where it was
        readerIMU_.clear();
        readerIMU_.seekg(pos);
        if(!t1 || !t2) {
            log_err("Failed to calculate IMU frequency.");
            exit(1);
        }
        // set frequency
        freq_ = std::floor(1.0 / (*t2 - *t1));
        // camera_matrix.csv
        std::filesystem::path pathCameraMatrix(pathScene / "camera_matrix.csv");
        ASSERT_PATH_EXISTS(pathCameraMatrix);
        // generate iphone.yaml
        DataloaderStray::generateSettingsFile(pathCameraMatrix);
        // skip initial frames if IMU data is missing
        std::size_t advanced = 0;
        while((carryOverFrame_ = DataloaderStray::nextInternal())) {
            if(carryOverFrame_ && !carryOverFrame_->vImuMeas.empty()) break;
            advanced++;
        }
        std::cout << "Advanced " << advanced << " frames to the start of IMU measurements." << std::endl;
    }

    DataloaderStray::DataloaderStray(const std::filesystem::path& pathScene) : 
        DataloaderStray(pathScene, std::nullopt) { /* STUB */ }

    DataloaderStray::DataloaderStray(const std::filesystem::path& pathScene, cv::Size sizeInternal) : 
        DataloaderStray(pathScene, std::optional(sizeInternal)) { /* STUB */ }

    DataloaderStray::~DataloaderStray() {
        cap_.release();
        readerOdom_.close();
        readerIMU_.close();
    }

    const std::filesystem::path& DataloaderStray::pathSettings() {
        return pathSettings_;
    }

    // this internal version permits empty IMU measurements
    const std::optional<Frame> DataloaderStray::nextInternal() {
        Frame curr;
        curr.index = frameIdx_;
        if(curr.index + 1 >= frameCount_) return std::nullopt;
        // read RGB frame
        std::optional<cv::Mat> im = DataloaderStray::im();
        if(!im) return std::nullopt;
        curr.im = std::move(*im);
        // depth frame
        std::optional<cv::Mat> depthmap = DataloaderStray::depthmap();
        if(!depthmap) return std::nullopt;
        cv::resize(*depthmap, *depthmap, sizeInternal_);
        curr.depthmap = std::move(*depthmap);
        // timestamp
        std::optional<double> timestamp = DataloaderStray::timestamp();
        if(!timestamp) return std::nullopt;
        curr.timestamp = *timestamp;
        // vImuMeas
        curr.vImuMeas = DataloaderStray::vImuMeas(curr.timestamp);
        // return frame
        frameIdx_++;
        return curr;
    }

    const std::optional<Frame> DataloaderStray::next() {
        if(carryOverFrame_) {
            Frame carryOverFrame = std::move(*carryOverFrame_);
            carryOverFrame_.reset();
            return carryOverFrame;
        }
        std::optional<Frame> frame = DataloaderStray::nextInternal();
        if(frame && frame->vImuMeas.empty()) return std::nullopt;
        return frame;
    }

    std::optional<cv::Mat> DataloaderStray::im() {
        cv::Mat im;
        if(!cap_.read(im)) {
            log_err("Failed to get RGB imagery on frame", frameIdx_, ".");
            return std::nullopt;
        }
        return im;
    }

    std::optional<cv::Mat> DataloaderStray::depthmap() {
        // build filename
        std::ostringstream fname;
        fname << std::setw(6) << std::setfill('0') << frameIdx_ << ".png";
        // check validity of path
        std::filesystem::path pathDepthFrame = pathDepth_ / fname.str();
        if(!std::filesystem::exists(pathDepthFrame)) return std::nullopt;
        // read depth map
        cv::Mat depthmap = cv::imread(pathDepthFrame, cv::IMREAD_UNCHANGED);
        if(depthmap.data == nullptr) {
            log_err("Failed to get depth map on frame", frameIdx_, ".");
            return std::nullopt;
        }
        return depthmap;
    }

    std::optional<double> DataloaderStray::timestampGeneric(std::ifstream& reader) {
        std::stringstream stream;
        double timestamp;
        if(!readLine(reader, stream) || !parse<double>(stream, timestamp, false)) {
            log_err("Failed to get timestamp.");
            return std::nullopt;
        }
        return timestamp;
    }

    std::optional<double> DataloaderStray::timestamp() {
        return DataloaderStray::timestampGeneric(readerOdom_);
    }

    std::vector<ORB_SLAM3::IMU::Point> DataloaderStray::vImuMeas(double timestamp) {
        std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
        // loop until we catch up
        do {
            // save position
            auto pos = readerIMU_.tellg();
            // get line
            std::stringstream stream;
            if(!readLine(readerIMU_, stream)) {
                log_err("Failed to get IMU data.");
                return vImuMeas;
            }
            // read the current timestamp
            double timestampCurr;
            if(!parse<double>(stream, timestampCurr, false)) 
                return std::vector<ORB_SLAM3::IMU::Point>{};
            // check if we need to break
            if(timestampCurr > timestamp) {
                readerIMU_.clear();
                readerIMU_.seekg(pos);
                return vImuMeas;
            }
            // read accelerometry and gyro data
            cv::Point3f acc, gyro;
            if(!parse<float>(stream, acc.x, false)) 
                return std::vector<ORB_SLAM3::IMU::Point>{};
            if(!parse<float>(stream, acc.y, false)) 
                return std::vector<ORB_SLAM3::IMU::Point>{};
            if(!parse<float>(stream, acc.z, false)) 
                return std::vector<ORB_SLAM3::IMU::Point>{};
            if(!parse<float>(stream, gyro.x, false)) 
                return std::vector<ORB_SLAM3::IMU::Point>{};
            if(!parse<float>(stream, gyro.y, false)) 
                return std::vector<ORB_SLAM3::IMU::Point>{};
            if(!parse<float>(stream, gyro.z, true)) 
                return std::vector<ORB_SLAM3::IMU::Point>{};
            // add the new point
            vImuMeas.emplace_back(acc, gyro, timestampCurr);
        } while(true);
        return std::vector<ORB_SLAM3::IMU::Point>{};
    }

    void DataloaderStray::generateSettingsFile(const std::filesystem::path& pathCameraMatrix) {
        // open the input stream
        std::ifstream reader(pathCameraMatrix);
        if(!reader.is_open()) {
            log_err("Failed to open [", pathCameraMatrix, "].");
            exit(1);
        }
        // get first row
        std::stringstream stream;
        if(!readLine(reader, stream)) exit(1);
        // read the relevant intrinsics
        double fx, fy, cx, cy, temp;
        // row: 0, col: 0
        if(!parse<double>(stream, fx, false)) exit(1);
        // row: 0, col: 1
        if(!parse<double>(stream, temp, false)) exit(1);
        // row: 0, col: 2
        if(!parse<double>(stream, cx, true)) exit(1);
        // get second row
        if(!readLine(reader, stream)) exit(1);
        // row: 1, col: 0
        if(!parse<double>(stream, temp, false)) exit(1);
        // row: 1, col: 1
        if(!parse<double>(stream, fy, false)) exit(1);
        // row: 1, col: 2
        if(!parse<double>(stream, cy, true)) exit(1);
        // get third row
        if(!readLine(reader, stream)) exit(1);
        // row: 2, col: 0
        if(!parse<double>(stream, temp, false)) exit(1);
        // row: 2, col: 1
        if(!parse<double>(stream, temp, false)) exit(1);
        // row: 2, col: 2
        double scalar;
        if(!parse<double>(stream, scalar, false)) exit(1);
        // adjust by scalar (although should always be 1.0)
        fx /= scalar;
        fy /= scalar;
        cx /= scalar;
        cy /= scalar;
        // create a temporary file
        pathSettings_ = std::filesystem::temp_directory_path() / "iphone.yaml";
        std::ofstream writer(pathSettings_);
        if(!writer) {
            log_err("Failed to write to [", pathSettings_, "].");
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
#if 0
        writer << "   data: [1.0,  0.0,  0.0, 0.0," << std::endl;
        writer << "          0.0,  1.0,  0.0, 0.0," << std::endl;
        writer << "          0.0,  0.0,  1.0, 0.0," << std::endl;
        writer << "          0.0,  0.0,  0.0, 1.0]" << std::endl;
#else
        // temporary data matrix for imu-to-camera calibration
        writer << "   data: [ 0.01313266, -0.99986337,  0.01003839,  0.05819003," << std::endl;
        writer << "          -0.99973402, -0.01293929,  0.01909088,  0.08413739," << std::endl;
        writer << "          -0.01895838, -0.01028643, -0.99976736, -0.22258164," << std::endl;
        writer << "           0.0       ,  0.0       ,  0.0       ,  1.0       ]" << std::endl;
#endif
        writer << "IMU.InsertKFsWhenLost: 0" << std::endl;
        writer << "IMU.NoiseGyro: 5.1e-3" << std::endl;  // TODO: Add the platform as a configuration option
        writer << "IMU.NoiseAcc:  1.4e-2" << std::endl;  // TODO: Add the platform as a configuration option
        writer << "IMU.GyroWalk:  5.0e-4" << std::endl;  // TODO: Add the platform as a configuration option
        writer << "IMU.AccWalk:   2.5e-3" << std::endl;  // TODO: Add the platform as a configuration option
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

#if 0
    std::size_t accumulated = 0;
    std::vector<ORB_SLAM3::IMU::Point> vImuMeasPre = std::move(carryOverFrame_->vImuMeas);
    while((carryOverFrame_ = DataloaderStray::nextInternal()) && (vImuMeasPre.back().t - vImuMeasPre.front().t) < vImuMeasPreThreshold) {
        vImuMeasPre.insert(vImuMeasPre.end(), carryOverFrame_->vImuMeas.begin(), carryOverFrame_->vImuMeas.end());
        accumulated++;
    }
    carryOverFrame_->vImuMeas.insert(carryOverFrame_->vImuMeas.begin(), vImuMeasPre.begin(), vImuMeasPre.end());
    std::optional<ORB_SLAM3::IMU::Point> vImuMeasOvershoot = DataloaderStray::vImuMeasOvershoot();
    if(!vImuMeasOvershoot) {
        log_err("Failed to read past the first preintegration frame.");
        exit(1);
    }
    carryOverFrame_->vImuMeas.push_back(*vImuMeasOvershoot);
    std::cout << std::fixed << std::setprecision(4) << carryOverFrame_->timestamp << ", " << carryOverFrame_->vImuMeas.front().t << ", " << carryOverFrame_->vImuMeas.back().t << std::endl;
    for(ORB_SLAM3::IMU::Point& pt : carryOverFrame_->vImuMeas) {
        std::cout << std::fixed << std::setprecision(4) << pt.t << ", ";
    }
    std::cout << std::endl;
    std::cout << "Accumulated IMU measurements from " << accumulated << " frames for preintegration." << std::endl;

    std::optional<ORB_SLAM3::IMU::Point> DataloaderStray::vImuMeasOvershoot() {
        // save position
        auto pos = readerIMU_.tellg();
        // get line
        std::stringstream stream;
        if(!readLine(readerIMU_, stream)) {
            log_err("Failed to get IMU data.");
            return std::nullopt;
        }
        // read the current timestamp
        double timestampCurr;
        if(!parse<double>(stream, timestampCurr, false)) return std::nullopt;
        // read accelerometry and gyro data
        cv::Point3f acc, gyro;
        if(!parse<float>(stream, acc.x, false))  return std::nullopt;
        if(!parse<float>(stream, acc.y, false))  return std::nullopt;
        if(!parse<float>(stream, acc.z, false))  return std::nullopt;
        if(!parse<float>(stream, gyro.x, false)) return std::nullopt;
        if(!parse<float>(stream, gyro.y, false)) return std::nullopt;
        if(!parse<float>(stream, gyro.z, true))  return std::nullopt;
        // add the new point
        readerIMU_.clear();
        readerIMU_.seekg(pos);
        // return
        return ORB_SLAM3::IMU::Point(acc, gyro, timestampCurr);
    }
#endif
