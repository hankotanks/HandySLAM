#pragma once

#include <string>
#include <exception>
#include <filesystem>

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
        Initializer(int argc, char* argv[]) : argc_(argc), argv_(argv), usingImu(false) {
            std::string loaderDoc = "must be one of [";
            std::size_t i = 0;
            for(const auto& it : Initializer::loaders_) {
                loaderDoc += it.first;
                loaderDoc += (++i == Initializer::loaders_.size()) ? "" : ", ";
            }
            loaderDoc += "]";
            cli_ = clipp::group{
                clipp::value("loader", loaderName)
                    .doc(loaderDoc)
                    .required(true)
            };
            clipp::parsing_result res = clipp::parse(std::min(argc_, 2), argv_, cli_);
            cli_.push_back(clipp::value("scene_path", pathSceneRaw_, 
                [&](const std::string& pathSceneRaw) {
                    pathScene = std::move(std::filesystem::path(pathSceneRaw));
                    return std::filesystem::exists(pathScene);
                }).doc("path to scene folder"));
            cli_.push_back(clipp::option("--imu").set(usingImu).doc("enable IMU-integration"));
            if(!res) {
                std::cout << clipp::make_man_page(cli_, argv_[0]) << std::endl;
                throw std::exception();
            }
            if(Initializer::loaders_.find(loaderName) == Initializer::loaders_.end()) {
                std::cout << clipp::make_man_page(cli_, argv_[0]) << std::endl;
                throw std::exception();
            }
        }
    public:
        void parse(clipp::group& options) {
            if(!clipp::parse(argc_, argv_, cli_.push_back(options))) {
                cli_[0] = clipp::command(loaderName);
                std::cout << clipp::make_man_page(instance_->cli_, instance_->argv_[0]) << std::endl;
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
                throw std::exception();
            }
            return Initializer::loaders_[instance_->loaderName]();
        }
        
        template<typename T, typename... Args>
        static void add(const std::string& name, Args&&... args) {
            std::function<Dataloader*()> f = [&, ...args = std::forward<Args>(args)]() mutable {
                return new T(*instance_, std::forward<Args>(args)...);
            };
            Initializer::loaders_.insert(std::make_pair(name, f));
        }

        static const Initializer& get() {
            return (*Initializer::instance_);
        }

    public:
        std::string loaderName;
        std::filesystem::path pathScene;
        bool usingImu;
    private:
        int argc_;
        char** argv_;
        std::string pathSceneRaw_;
        clipp::group cli_;
        static std::unordered_map<std::string, std::function<Dataloader*()>> loaders_;
        static Initializer* instance_;
    };
}