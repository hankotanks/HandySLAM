#pragma once

#include <string>
#include <exception>
#include <filesystem>

#include <System.h>

#include "clipp.h"

#include "Dataloader.h"

namespace HandySLAM {
    class Initializer {
    public:
        Initializer(const Initializer& args) = delete;
        Initializer(Initializer&& args) = delete;
        Initializer& operator=(const Initializer&) = delete;
        Initializer& operator=(Initializer&&) = delete;
    private:
        Initializer(int argc, char* argv[]) : 
            argc_(argc), argv_(argv), 
            usingImu(false), usingMono(false), 
            saveVolume(false), voxelSize(0.01), depthCutoff(4.0) {
            std::string loaderDoc = "must be one of [";
            std::size_t i = 0;
            for(const auto& it : Initializer::loaders_) {
                loaderDoc += it.first;
                loaderDoc += (++i == Initializer::loaders_.size()) ? "" : ", ";
            }
            loaderDoc += "]";
            cli_ = clipp::group {
                clipp::value("loader", loaderName)
                    .doc(loaderDoc)
                    .required(true),
                clipp::value("scene_path", pathSceneRaw_, [&](const std::string& pathSceneRaw) {
                    pathScene = std::move(std::filesystem::path(pathSceneRaw));
                    return std::filesystem::exists(pathScene);
                }).doc("path to scene folder")
            };
            clipp::parsing_result res = clipp::parse(std::min(argc_, 3), argv_, cli_);
            cli_.push_back(clipp::option("--imu").set(usingImu).doc("enable IMU-integration"));
            cli_.push_back(clipp::option("--mono").set(usingMono).doc("use only color imagery"));
            cli_.push_back(clipp::option("-o", "--out").set(saveVolume).doc("save TSDF volume"));
            cli_.push_back(clipp::option("--voxel-length").doc("TSDF volume's voxel length (in meters)") & clipp::value("size", voxelSize)),
            cli_.push_back(clipp::option("--depth-cutoff").doc("depth threshold for TSDF") &clipp::value("threshold", depthCutoff));
            if(!res) {
                std::cout << clipp::make_man_page(cli_, argv_[0]) << std::endl;
                throw std::runtime_error("Failed to parse CLI arguments.");
            }
            if(Initializer::loaders_.find(loaderName) == Initializer::loaders_.end()) {
                std::cout << clipp::make_man_page(cli_, argv_[0]) << std::endl;
                throw std::runtime_error("Failed to find loader.");
            }
        }
    public:
        void parse(clipp::group& options) {
            clipp::group cli(cli_);
            if(!clipp::parse(argc_, argv_, cli.push_back(options))) {
                cli[0] = clipp::command(loaderName);
                std::cout << clipp::make_man_page(cli, instance_->argv_[0]) << std::endl;
                throw std::exception();
            }
        }

        static Dataloader* init(int argc, char* argv[]) {
            try {
                instance_ = new Initializer(argc, argv);
            } catch(...) {
                if(instance_ != nullptr) {
                    std::cout << clipp::usage_lines(instance_->cli_, instance_->argv_[0]) << std::endl;
                } else {
                    log_err("Failed to initialize Dataloader.");
                }
                throw;
            }
            return Initializer::loaders_[instance_->loaderName]();
        }

        static Dataloader* restart(void) {
            if(instance_ == nullptr) 
                throw std::exception();

            return Initializer::loaders_[instance_->loaderName]();
        }
        
        template<typename T>
        static void add(const std::string& name) {
            std::function<Dataloader*()> f = [&]() mutable {
                return new T(*instance_);
            };
            Initializer::loaders_.insert(std::make_pair(name, f));
        }

        static const Initializer& get() {
            return (*Initializer::instance_);
        }

        const enum ORB_SLAM3::System::eSensor sensor() const {
            return usingImu ? 
                (usingMono ? ORB_SLAM3::System::IMU_MONOCULAR : ORB_SLAM3::System::IMU_RGBD) : 
                (usingMono ? ORB_SLAM3::System::MONOCULAR : ORB_SLAM3::System::RGBD);
        }

    public:
        std::string loaderName;
        std::filesystem::path pathScene;
        bool usingImu;
        bool usingMono;
        bool saveVolume;
        double voxelSize;
        double depthCutoff;
    private:
        int argc_;
        char** argv_;
        std::string pathSceneRaw_;
        clipp::group cli_;
        static std::unordered_map<std::string, std::function<Dataloader*()>> loaders_;
        static Initializer* instance_;
    };
}