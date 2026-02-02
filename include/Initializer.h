#pragma once

#include <complex>
#include <exception>
#include <stdexcept>
#include <string>
#include <filesystem>
#include <cstdlib>

#include <System.h>

#include "clipp.h"

#include "Dataloader.h"
#include "util.h"

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
            upscale(false),
            saveVolume(false), voxelSize(0.01), maxDepth(4.0) {
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
            cli_.push_back(clipp::option("--imu").set(usingImu).doc("enable IMU-integration"));
            cli_.push_back(clipp::option("--mono").set(usingMono).doc("use only color imagery"));
            cli_.push_back(clipp::option("--upscale").set(upscale).doc("upscale depth imagery with PrompDA"));
            cli_.push_back(clipp::option("-o", "--out").set(saveVolume).doc("save TSDF volume"));
            cli_.push_back(clipp::option("--voxel-length").doc("TSDF volume's voxel length (in meters)") & clipp::value("size", voxelSize)),
            cli_.push_back(clipp::option("--max-depth").doc("depth threshold for TSDF") &clipp::value("threshold", maxDepth));
            clipp::parsing_result res = clipp::parse(argc, argv_, cli_);
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
                throw std::runtime_error("Failed to parse CLI arguments.");
            }
        }

        static Dataloader* init(int argc, char* argv[]) {
            try {
                instance_ = new Initializer(argc, argv);
                if(instance_->upscale) {
                    std::filesystem::path pathRoot(PROJECT_ROOT);
                    std::filesystem::path pathPython = pathRoot / "env" / "bin" / "python3";
                    ASSERT_PATH_EXISTS(pathPython);
                    std::filesystem::path pathPrompt = pathRoot / "PromptDA";
                    ASSERT_PATH_EXISTS(pathPrompt);
                    std::filesystem::path pathSceneAbs = std::filesystem::absolute(instance_->pathScene);
                    ASSERT_PATH_EXISTS(pathSceneAbs);
                    
                    std::string command = "cd \"" + pathPrompt.string() + "\" && " + \
                        "\"" + pathPython.string() + "\" -m promptda.scripts.infer " + instance_->loaderName + " " + pathSceneAbs.string();

                    if(std::system(command.c_str()) != 0) {
                        log_err("Failed to infer upscaled depth using PromptDA.");
                        throw std::exception();
                    }
                }
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
                throw std::logic_error("Can only call Dataloader::restart after Dataloader::init.");

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
        bool upscale;
        bool saveVolume;
        double voxelSize;
        double maxDepth;
    private:
        int argc_;
        char** argv_;
        std::string pathSceneRaw_;
        clipp::group cli_;
        static std::unordered_map<std::string, std::function<Dataloader*()>> loaders_;
        static Initializer* instance_;
    };
}