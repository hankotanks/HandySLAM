#include "DataloaderScanNet.h"

#include <exception>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <zlib.h>
#include <lz4.h>

#include "Dataloader.h"
#include "util.h"

namespace {
    constexpr int WIDTH  = 256;
    constexpr int HEIGHT = 192;
    constexpr size_t FRAME_FLOAT_BYTES = WIDTH * HEIGHT * sizeof(float);
    constexpr size_t FRAME_UINT16_BYTES = WIDTH * HEIGHT * sizeof(uint16_t);
} // private scope

namespace HandySLAM {
    DataloaderScanNet::DataloaderScanNet(Initializer& init) : Dataloader(init.pathScene) {
        // parse args
        bool upscaled = false;
        clipp::group options = clipp::group {
            clipp::option("-u", "--upscaled").set(upscaled).doc("use larger depthmaps")
        };
        init.parse(options);

        // set frame index
        frameIdx_ = 0;

        std::filesystem::path pathData = pathScene_ / "iphone";
        ASSERT_PATH_EXISTS(pathData);

        pathJSON_ = pathData / "pose_intrinsic_imu.json";
        ASSERT_PATH_EXISTS(pathJSON_);

        pathRGB_ = pathData / "rgb.mkv";
        ASSERT_PATH_EXISTS(pathRGB_);

        pathDepth_ = upscaled ? std::filesystem::path(pathData / "depth_upscaled") : (pathData / "depth.bin");
        ASSERT_PATH_EXISTS(pathDepth_);

        std::filesystem::path pathColmap = pathData / "colmap";
        ASSERT_PATH_EXISTS(pathColmap);

        pathCameras_ = pathColmap / "cameras.txt";
        ASSERT_PATH_EXISTS(pathCameras_);

        // create reader for pose_intrinsic_imu.json
        readerJSON_ = std::ifstream(pathJSON_);
        if(!readerJSON_.is_open()) {
            log_err("Failed to open:", pathJSON_);
            throw std::exception();
        }
        // parse the JSON
        readerJSON_ >> parserJSON_;
        // iterator for DataloaderScanNet::next
        iterJSON_ = parserJSON_.begin();
        
        // RGB
        cap_ = cv::VideoCapture(pathRGB_);
        if(!cap_.isOpened()) {
            log_err("Failed to open:", pathRGB_);
            throw std::exception();
        }
        // read RGB video's fps
        std::size_t fps = static_cast<std::size_t>(cap_.get(cv::CAP_PROP_FPS));
        // and resolution
        cv::Size sizeIm;
        sizeIm.width  = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        sizeIm.height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));

        // depth stream (to be read with DataloaderScanNet::depthmap)
        if(!std::filesystem::is_directory(pathDepth_)) readerDepth_ = std::ifstream(pathDepth_);

        // metadata
        SceneMetadata meta;
        meta.fps = fps;
        // depthmap size is fixed if we are unpacking the binary ourselves
        if(std::filesystem::is_directory(pathDepth_)) {
            std::filesystem::path pathIm = pathDepth_ / "000000.png";
            cv::Mat im = cv::imread(pathIm, cv::IMREAD_UNCHANGED);
            if(im.empty()) {
                log_err("Failed to query depthmap dimensions:", pathIm);
                throw std::exception();
            }
            meta.sizeDepthmap.width = im.cols;
            meta.sizeDepthmap.height = im.rows;
        } else {
            meta.sizeDepthmap.width = WIDTH;
            meta.sizeDepthmap.height = HEIGHT;
        }

        // profile
        Profile profile("iphone16pro");
        // fix IMU update_rate to fps, since ScanNet++ provides aggregated IMU readings
        profile.update_rate = static_cast<double>(meta.fps);
        // set profile
        profile_ = profile;

        // intrinsics
        std::ifstream readerCameras(pathCameras_);
        if(!readerCameras.is_open()) {
            log_err("Failed to open:", pathCameras_);
            throw std::exception();
        }
        // read camera parameters
        bool success;
        std::string temp;
        while(std::getline(readerCameras, temp)) {
            if(temp.empty() || temp[0] == '#') continue;
            std::istringstream tempStream(temp);
            int cameraId;
            std::string cameraModel; 
            DistortionParams distortion;
            tempStream >> cameraId >> cameraModel >> 
                meta.sizeIm.width >> meta.sizeIm.height >> 
                meta.intrinsics.fx >> meta.intrinsics.fy >>
                meta.intrinsics.cx >> meta.intrinsics.cy >> 
                distortion.k1 >> distortion.k2 >>
                distortion.p1 >> distortion.p2;
            meta.intrinsics.distortion = distortion;
            success = true;
            // TODO: Read distortion parameters, switch camera based on whether they are set
            break;
        }
        if(!success) {
            log_err("Failed to read:", pathCameras_);
            throw std::exception();
        }
        if(sizeIm.width != meta.sizeIm.width || sizeIm.height != meta.sizeIm.height) {
            log_err("Failed to read (RGB dimension disagreement):", pathCameras_);
            throw std::exception();
        }

        // set the SceneMetadata member
        metadata_ = meta;
    }

    std::optional<cv::Mat> DataloaderScanNet::im() noexcept {
        cv::Mat im;
        if(!cap_.read(im)) {
            log_err("Failed to get RGB imagery on frame:", frameIdx_);
            return std::nullopt;
        }
        return im;
    }

    // this function mirrors the python implementation (extract_depth) given here
    // https://github.com/scannetpp/scannetpp/blob/main/iphone/prepare_iphone_data.py
    std::optional<cv::Mat> DataloaderScanNet::depthmap() noexcept {
        if(std::filesystem::is_directory(pathDepth_)) {
            std::ostringstream fname;
            fname << std::setw(6) << std::setfill('0') << frameIdx_ << ".png";
            // check validity of path
            std::filesystem::path pathDepthFrame = pathDepth_ / fname.str();
            if(!std::filesystem::exists(pathDepthFrame)) return std::nullopt;
            // read depth map
            cv::Mat depthmap = cv::imread(pathDepthFrame, cv::IMREAD_UNCHANGED);
            if(depthmap.data == nullptr) {
                log_err("Failed to get depth map from image file:", pathDepthFrame, "| index:", frameIdx_);
                return std::nullopt;
            }
            return depthmap;
        }

        uint32_t size;
        if(!readerDepth_.read(reinterpret_cast<char*>(&size), 4)) return std::nullopt;
        // read compressed data
        std::vector<char> dataCompressed(size);
        if(!readerDepth_.read(dataCompressed.data(), size)) {
            log_err("Failed to read frame:", frameIdx_);
            return std::nullopt;
        }

        std::vector<char> dataRaw(FRAME_UINT16_BYTES * 4); // assumed size

        int decompressed_size = 0;
        bool success = false;

        // try lz4 first
        decompressed_size = LZ4_decompress_safe(dataCompressed.data(), dataRaw.data(), size, dataRaw.size());
        if(decompressed_size > 0) {
            success = true;
        } else {
            // fallback to zlib if it fails
            z_stream stream {};
            stream.next_in = reinterpret_cast<Bytef*>(dataCompressed.data());
            stream.avail_in = size;

            if(inflateInit2(&stream, -MAX_WBITS) == Z_OK) {
                stream.next_out = reinterpret_cast<Bytef*>(dataRaw.data());
                stream.avail_out = dataRaw.size();

                int ret = inflate(&stream, Z_FINISH);
                inflateEnd(&stream);

                if(ret == Z_STREAM_END) {
                    success = true;
                    decompressed_size = stream.total_out;
                }
            }
        }

        if(!success) {
            log_err("Failed to read frame (decompression failed):", frameIdx_);
            return std::nullopt;
        }

        cv::Mat depthmap(HEIGHT, WIDTH, CV_16UC1);
        if(decompressed_size == FRAME_UINT16_BYTES) {
            std::memcpy(depthmap.data, dataRaw.data(), FRAME_UINT16_BYTES);
        } else {
            // convert to millimeters to be consistent with DataloaderStray
            float* depth = reinterpret_cast<float*>(dataRaw.data());
            for(int y = 0; y < HEIGHT; ++y) {
                for(int x = 0; x < WIDTH; ++x) {
                    float val = depth[y * WIDTH + x];
                    depthmap.at<uint16_t>(y, x) = static_cast<uint16_t>(val * 1000.0f);
                }
            }
        }

        return depthmap;
    }

    const std::optional<Frame> DataloaderScanNet::next() noexcept {
        Frame curr;
        curr.index = frameIdx_;
        std::optional<cv::Mat> im, depthmap;
        im = DataloaderScanNet::im();
        if(!im) return std::nullopt;
        curr.im = *im;
        depthmap =  DataloaderScanNet::depthmap();
        if(!depthmap) return std::nullopt;
        curr.depthmap = *depthmap;
        // parse timestamp and IMU from JSON
        if(!iterJSON_.value().contains("timestamp") || !iterJSON_.value().contains("imu")) {
            log_err("Found invalid frame in:", pathJSON_, "| index:", frameIdx_);
            return std::nullopt;
        }
        // assign timestamp
        curr.timestamp = iterJSON_.value()["timestamp"].get<double>();
        // construct IMU reading
        auto imu = iterJSON_.value()["imu"];
        std::vector<double> gyro = imu["rotate_rate"].get<std::vector<double>>();
        std::vector<double> acc = imu["acceleration"].get<std::vector<double>>();
        std::vector<double> grav = imu["gravity"].get<std::vector<double>>();
        if(gyro.size() != 3 || acc.size() != 3 || grav.size() != 3) {
            log_err("Found invalid frame (malformed IMU readings) in:", pathJSON_, "| index:", frameIdx_);
            return std::nullopt;
        }
        // must reintroduce gravity
        curr.vImuMeas.emplace_back(acc[0] + grav[0], acc[1] + grav[1], acc[2] + grav[2], gyro[0], gyro[1], gyro[2], curr.timestamp);
        frameIdx_++;
        iterJSON_++;
        return curr;
    }
}
