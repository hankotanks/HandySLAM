#include "DataloaderStray.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <opencv2/core/mat.hpp>

#include "Dataloader.h"

#include "util.h"

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
    bool parse(std::stringstream& stream, T& val, bool final = false) {
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

namespace {
    HandySLAM::Intrinsics parseCameraMatrix(const std::filesystem::path& pathCameraMatrix) {
        std::ifstream reader(pathCameraMatrix);
        if(!reader.is_open()) {
            log_err("Failed to open [", pathCameraMatrix, "].");
            throw std::exception();
        }
        // get first row
        std::stringstream stream;
        if(!readLine(reader, stream))
            goto parse_camera_matrix_fail;
        // read the relevant intrinsics
        HandySLAM::Intrinsics intrinsics;
        // row: 0, col: 0
        if(!parse<double>(stream, intrinsics.fx))
            goto parse_camera_matrix_fail;
        // row: 0, col: 1
        double dummy;
        if(!parse<double>(stream, dummy))
            goto parse_camera_matrix_fail;
        // row: 0, col: 2
        if(!parse<double>(stream, intrinsics.cx, true))
            goto parse_camera_matrix_fail;
        // get second row
        if(!readLine(reader, stream))
            goto parse_camera_matrix_fail;
        // row: 1, col: 0
        if(!parse<double>(stream, dummy))
            goto parse_camera_matrix_fail;
        // row: 1, col: 1
        if(!parse<double>(stream, intrinsics.fy))
            goto parse_camera_matrix_fail;
        // row: 1, col: 2
        if(!parse<double>(stream, intrinsics.cy, true))
            goto parse_camera_matrix_fail;
        // get third row
        if(!readLine(reader, stream))
            goto parse_camera_matrix_fail;
        // row: 2, col: 0
        if(!parse<double>(stream, dummy))
            goto parse_camera_matrix_fail;
        // row: 2, col: 1
        if(!parse<double>(stream, dummy))
            goto parse_camera_matrix_fail;
        // row: 2, col: 2
        double scalar;
        if(!parse<double>(stream, scalar))
            goto parse_camera_matrix_fail;
        // adjust by scalar (although should always be 1.0)
        intrinsics.fx /= scalar;
        intrinsics.fy /= scalar;
        intrinsics.cx /= scalar;
        intrinsics.cy /= scalar;
        // return intrinsics
        return intrinsics;
parse_camera_matrix_fail:
        log_err("Failed to parse [", pathCameraMatrix, "].");
        throw std::exception();
    }
} // private scope

namespace HandySLAM {
    const std::string DataloaderStray::flag = "stray";

    DataloaderStray::DataloaderStray(Initializer& init) : Dataloader(init.pathScene) {
        // parse args
        std::string profileName;
        clipp::group options = {
            clipp::option("-p", "--profile").required(true).doc("name of the calibration profile") &
            clipp::value("profile_name", profileName, [&](const std::string& profileNameRaw) {
                std::filesystem::path pathProfile = std::filesystem::path(PROJECT_ROOT) / "profiles";
                std::filesystem::path pathTransform = pathProfile / (profileNameRaw + "-camchain-imucam.yaml");
                std::filesystem::path pathNoise = pathProfile / (profileNameRaw + "-imu.yaml");
                return std::filesystem::exists(pathTransform) && std::filesystem::exists(pathNoise);
            })
        };
        init.parse(options);
        // configure profile
        profile_ = Profile(profileName);
        // rgb.mp4
        pathRGB_ = pathScene_ / "rgb.mp4";
        ASSERT_PATH_EXISTS(pathRGB_);
        // open RGB video capture
        cap_ = cv::VideoCapture(pathRGB_);
        if(!cap_.isOpened()) {
            log_err("Failed to open ", pathRGB_, "].");
            throw std::exception();
        }
        // read RGB video's fps
        std::size_t fps = static_cast<std::size_t>(cap_.get(cv::CAP_PROP_FPS));
        // and resolution
        cv::Size sizeIm;
        sizeIm.width  = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        sizeIm.height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        // initialize frame index and count
        frameIdx_ = 0;
        frameCount_ = static_cast<std::size_t>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
        // depth/*
        pathDepth_ = pathScene_ / "depth";
        ASSERT_PATH_EXISTS(pathDepth_);
        // odometry.csv
        pathOdom_ = pathScene_ / "odometry.csv";
        ASSERT_PATH_EXISTS(pathOdom_);
        readerOdom_ = std::ifstream(pathOdom_, std::ios::binary);
        if(!readerOdom_.is_open()) {
            log_err("Failed to open ", pathOdom_, "].");
            throw std::exception();
        }
        std::string temp;
        std::getline(readerOdom_, temp);
        // imu.csv
        pathIMU_ = pathScene_ / "imu.csv";
        ASSERT_PATH_EXISTS(pathIMU_);
        readerIMU_ = std::ifstream(pathIMU_, std::ios::binary);
        if(!readerIMU_.is_open()) {
            log_err("Failed to open ", pathIMU_, "].");
            throw std::exception();
        }
        std::getline(readerIMU_, temp);
        // skip initial frames if IMU data is missing
        std::size_t advanced = 0;
        while((carryOverFrame_ = DataloaderStray::nextInternal())) {
            if(carryOverFrame_ && !carryOverFrame_->vImuMeas.empty()) break;
            advanced++;
        }
        std::cout << "Advanced " << advanced << " frames to the start of IMU measurements." << std::endl;
        // read depthmap resolution
        cv::Mat depthmap = cv::imread(pathDepth_ / "000000.png", cv::IMREAD_UNCHANGED);
        cv::Size sizeDepthmap(depthmap.cols, depthmap.rows);
        // parse camera_matrix.csv
        pathCameraMatrix_ = pathScene_ / "camera_matrix.csv";
        ASSERT_PATH_EXISTS(pathCameraMatrix_);
        // read the relevant intrinsics
        Intrinsics intrinsics = parseCameraMatrix(pathCameraMatrix_);
        // set SceneMetadata
        metadata_ = (SceneMetadata) { sizeIm, sizeDepthmap, fps, intrinsics };
    }

    DataloaderStray::~DataloaderStray() {
        cap_.release();
        readerOdom_.close();
        readerIMU_.close();
    }

    // this internal version permits empty IMU measurements
    const std::optional<Frame> DataloaderStray::nextInternal() noexcept {
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

    const std::optional<Frame> DataloaderStray::next() noexcept {
        if(carryOverFrame_) {
            Frame carryOverFrame = std::move(*carryOverFrame_);
            carryOverFrame_.reset();
            return carryOverFrame;
        }
        std::optional<Frame> frame = DataloaderStray::nextInternal();
        if(frame && frame->vImuMeas.empty()) return std::nullopt;
        return frame;
    }

    std::optional<cv::Mat> DataloaderStray::im() noexcept {
        cv::Mat im;
        if(!cap_.read(im)) {
            log_err("Failed to get RGB imagery on frame", frameIdx_, ".");
            return std::nullopt;
        }
        return im;
    }

    std::optional<cv::Mat> DataloaderStray::depthmap() noexcept {
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

    std::optional<double> DataloaderStray::timestamp() noexcept {
        std::stringstream stream;
        double timestamp;
        if(!readLine(readerOdom_, stream) || !parse<double>(stream, timestamp)) {
            log_err("Failed to get timestamp.");
            return std::nullopt;
        }
        return timestamp;
    }

    std::vector<ORB_SLAM3::IMU::Point> DataloaderStray::vImuMeas(double timestamp) noexcept {
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
            if(!parse<double>(stream, timestampCurr)) return {};
            timestampCurr += profile_->timeshift_cam_imu;
            // check if we need to break
            if(timestampCurr > timestamp) {
                readerIMU_.clear();
                readerIMU_.seekg(pos);
                return vImuMeas;
            }
            // read accelerometry and gyro data
            cv::Point3f acc, gyro;
            if(!parse<float>(stream, acc.x))  return {};
            if(!parse<float>(stream, acc.y))  return {};
            if(!parse<float>(stream, acc.z))  return {};
            if(!parse<float>(stream, gyro.x)) return {};
            if(!parse<float>(stream, gyro.y)) return {};
            if(!parse<float>(stream, gyro.z, true))  return {};
            // add the new point
            vImuMeas.emplace_back(acc, gyro, timestampCurr);
        } while(true);
        return {};
    }
}
