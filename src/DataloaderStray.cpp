#include "DataloaderStray.h"

#include <filesystem>
#include <optional>
#include <fstream>
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
    DataloaderStray::DataloaderStray(const std::filesystem::path& pathScene) : Dataloader(pathScene) {
        // rgb.mp4
        pathRGB_ = pathScene_ / "rgb.mp4";
        ASSERT_PATH_EXISTS(pathRGB_);
        // open RGB video capture
        cap_ = cv::VideoCapture(pathRGB_);
        if(!cap_.isOpened()) {
            log_err("Failed to open ", pathRGB_, "].");
            exit(1);
        }
        fps_  = static_cast<std::size_t>(cap_.get(cv::CAP_PROP_FPS));
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
            exit(1);
        }
        std::string temp;
        std::getline(readerOdom_, temp);
        // imu.csv
        pathIMU_ = pathScene_ / "imu.csv";
        ASSERT_PATH_EXISTS(pathIMU_);
        readerIMU_ = std::ifstream(pathIMU_, std::ios::binary);
        if(!readerIMU_.is_open()) {
            log_err("Failed to open ", pathIMU_, "].");
            exit(1);
        }
        std::getline(readerIMU_, temp);
        // skip initial frames if IMU data is missing
        std::size_t advanced = 0;
        while((carryOverFrame_ = DataloaderStray::nextInternal())) {
            if(carryOverFrame_ && !carryOverFrame_->vImuMeas.empty()) break;
            advanced++;
        }
        std::cout << "Advanced " << advanced << " frames to the start of IMU measurements." << std::endl;
    }

    DataloaderStray::~DataloaderStray() {
        cap_.release();
        readerOdom_.close();
        readerIMU_.close();
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

    const double DataloaderStray::fps() const {
        return fps_;
    }

    const cv::Size DataloaderStray::sizeDepthmap() const {
        cv::Mat depthmap = cv::imread(pathDepth_ / "000000.png", cv::IMREAD_UNCHANGED);
        return cv::Size(depthmap.cols, depthmap.rows);
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
