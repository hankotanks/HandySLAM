#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <filesystem>
#include <optional>

#include <System.h>

#define ASSERT_PATH_EXISTS(path_) if(!std::filesystem::exists(path_)) { \
        std::cout << "Path does not exist [" << path_ << "]." << std::endl; \
        exit(1); \
    }

double nextStamp(std::ifstream& odom) {
    std::string line;
    if(!std::getline(odom, line)) {
        std::cout << "Failed to get timestamp." << std::endl;
        exit(1);
    }
    std::stringstream lineStream(line);
    std::string elem;
    if(std::getline(lineStream, elem, ',')) return std::stod(elem);
    std::cout << "Failed to get timestamp." << std::endl;
    exit(1);
}

void readIMU(std::ifstream& imu, std::vector<ORB_SLAM3::IMU::Point>& vImuMeas, double until) {
    double timestamp;
    do {
        std::string line;
        if(!std::getline(imu, line)) {
            std::cout << "Failed to get IMU frame data." << std::endl;
            exit(1);
        }

        std::stringstream lineStream(line);
        std::string elem;
        if(!std::getline(lineStream, elem, ',')) exit(1);
        timestamp = static_cast<double>(std::stold(elem));

        float acc_x, acc_y, acc_z, ang_vel_x, ang_vel_y, ang_vel_z;
        if(!std::getline(lineStream, elem, ',')) exit(1);
        acc_x = static_cast<float>(std::stold(elem));
        if(!std::getline(lineStream, elem, ',')) exit(1);
        acc_y = static_cast<float>(std::stold(elem));
        if(!std::getline(lineStream, elem, ',')) exit(1);
        acc_z = std::stold(elem);
        if(!std::getline(lineStream, elem, ',')) exit(1);
        ang_vel_x = static_cast<float>(std::stold(elem));
        if(!std::getline(lineStream, elem, ',')) exit(1);
        ang_vel_y = static_cast<float>(std::stold(elem));
        if(!std::getline(lineStream, elem, ',')) exit(1);
        ang_vel_z = static_cast<float>(std::stold(elem));
        
        vImuMeas.emplace_back(acc_x, acc_y, acc_z, ang_vel_x, ang_vel_y, ang_vel_z, timestamp);
    } while(timestamp < until);
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        std::cout << "Must provide scene path." << std::endl;
        exit(1);
    }

    std::filesystem::path strSettingsFile(PROJECT_ROOT);
    strSettingsFile /= "iphone.yaml";
    ASSERT_PATH_EXISTS(strSettingsFile);

    std::filesystem::path pathScene(argv[1]);
    ASSERT_PATH_EXISTS(pathScene);

    ORB_SLAM3::System SLAM(ORB_SLAM3_DIR_VOC, strSettingsFile, ORB_SLAM3::System::IMU_RGBD);

    std::filesystem::path pathRGB(pathScene);
    pathRGB /= "rgb.mp4";
    ASSERT_PATH_EXISTS(pathRGB);

    cv::VideoCapture cap(pathRGB);
    if(!cap.isOpened()) {
        std::cout << "Failed to open [" << pathRGB << "]." << std::endl;
        exit(1);
    }

    std::filesystem::path pathDepth(pathScene);
    pathDepth /= "depth";
    ASSERT_PATH_EXISTS(pathDepth);

    std::filesystem::path pathOdom(pathScene);
    pathOdom /= "odometry.csv";
    ASSERT_PATH_EXISTS(pathOdom);

    std::ifstream fileOdom(pathOdom);
    if(!fileOdom.is_open()) {
        std::cout << "Failed to open [" << pathOdom << "]." << std::endl;
        exit(1);
    }

    std::string line;
    std::getline(fileOdom, line);

    std::filesystem::path pathIMU(pathScene);
    pathIMU /= "imu.csv";
    ASSERT_PATH_EXISTS(pathIMU);

    std::ifstream fileIMU(pathIMU);
    if(!fileIMU.is_open()) {
        std::cout << "Failed to open [" << pathIMU << "]." << std::endl;
        exit(1);
    }

    std::getline(fileIMU, line);

    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;

    cv::Mat im;
    for(int frameCount = 0; cap.read(im); ++frameCount) {
        std::ostringstream fname;
        fname << std::setw(6) << std::setfill('0') << frameCount << ".png";

        cv::Mat depthmap = cv::imread(pathDepth / fname.str(), cv::IMREAD_UNCHANGED);
        if(depthmap.data == nullptr) {
            std::cout << "Failed to read depth frame." << std::endl;
            exit(1);
        }
        cv::resize(depthmap, depthmap, im.size());

        double timestamp = nextStamp(fileOdom);
        readIMU(fileIMU, vImuMeas, timestamp);

        std::optional<ORB_SLAM3::IMU::Point> carryOverSensorData;
        if(!vImuMeas.empty()) {
            carryOverSensorData = vImuMeas.back();
            vImuMeas.pop_back();
        }

        if(!vImuMeas.empty()) {
            SLAM.TrackRGBD(im, depthmap, timestamp, vImuMeas);
            vImuMeas.clear();
        }

        if(carryOverSensorData) vImuMeas.push_back(*carryOverSensorData);
    }
    
    cap.release();

    SLAM.Shutdown();

    SLAM.SaveTrajectoryEuRoC(pathScene / "output.txt");

    return 0;
}
