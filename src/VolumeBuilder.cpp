#include "VolumeBuilder.h"

#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>

// ONLY INCLUDE OPEN3D HERE

#define SDF_TRUNC_SCALAR 4.0

namespace {
    open3d::geometry::Image convertImage(const cv::Mat& mat) {
        CV_Assert(mat.isContinuous());
        open3d::geometry::Image img;
        img.Prepare(mat.cols, mat.rows, mat.channels(), mat.elemSize1());
        std::memcpy(img.data_.data(), mat.data,  mat.rows * mat.cols * mat.channels() * mat.elemSize1());
        return img;
    }
}

namespace HandySLAM {
    struct VolumeBuilder::Volume {
        open3d::pipelines::integration::ScalableTSDFVolume tsdf_;
        Volume(double voxel_length, double sdf_trunc) : 
            tsdf_(voxel_length, sdf_trunc, open3d::pipelines::integration::TSDFVolumeColorType::RGB8) { /* STUB */ }
    };

    VolumeBuilder::VolumeBuilder(const Intrinsics& intrinsics, double voxelSize, double maxDepth) : 
        frameIdx_(0), intrinsics_(intrinsics), voxelSize_(voxelSize), maxDepth_(maxDepth) {

        std::cout << "[INFO] Began TSDF volume construction" << std::endl;

        volume_ = std::make_shared<VolumeBuilder::Volume>(voxelSize, voxelSize * SDF_TRUNC_SCALAR);
    }

    void VolumeBuilder::integrateFrame(const cv::Mat& im, const cv::Mat& depthmap, const Sophus::SE3f& pose) {
        std::cout << "[INFO] Integrating frame " << frameIdx_ << std::endl;
        cv::Mat imResized, imColor;
        cv::resize(im, imResized, depthmap.size());
        cv::cvtColor(imResized, imColor, cv::COLOR_BGR2RGB);

        cv::Mat depthmapFloat(depthmap.rows, depthmap.cols, CV_32F);
        depthmap.convertTo(depthmapFloat, CV_32F, 1.0 / 1000.0);

        cv::Mat depthmapMasked = depthmapFloat.clone();
        depthmapMasked.setTo(0.0, depthmapMasked > maxDepth_);
        depthmapMasked.setTo(0.0, depthmapMasked <= 0.0);

        cv::Mat imFinal, depthmapFinal;
        if(intrinsics_.distortion) {
            cv::Mat K = (cv::Mat_<double>(3,3) <<
                intrinsics_.fx, 0.0, intrinsics_.cx,
                0.0, intrinsics_.fy, intrinsics_.cy,
                0.0,  0.0, 1.0);

            cv::Mat D = (cv::Mat_<double>(1,4) <<
                intrinsics_.distortion->k1, intrinsics_.distortion->k2, intrinsics_.distortion->p1, intrinsics_.distortion->p2);

            cv::undistort(imResized, imFinal, K, D);
            cv::undistort(depthmapMasked, depthmapFinal, K, D);
        } else {
            imFinal = imResized;
            depthmapFinal = depthmapMasked;
        }
        
        open3d::geometry::RGBDImage image(convertImage(imFinal), convertImage(depthmapFinal));

        Intrinsics intrinsics(intrinsics_);
        double sx = static_cast<double>(depthmap.cols) / im.cols;
        double sy = static_cast<double>(depthmap.rows) / im.rows;
        intrinsics.fx *= sx;
        intrinsics.fy *= sy;
        intrinsics.cx *= sx;
        intrinsics.cy *= sy;

        open3d::camera::PinholeCameraIntrinsic intrinsicsCamera(depthmapFinal.cols, depthmapFinal.rows, 
            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy);

        Eigen::Matrix4d T_cw = pose.matrix().cast<double>().inverse();

        volume_->tsdf_.Integrate(image, intrinsicsCamera, T_cw);

        frameIdx_++;
    }

    bool VolumeBuilder::save(const std::filesystem::path& pathOut) const {
        std::cout << "[INFO] Extracting mesh from TSDF" << std::endl;

        auto mesh = volume_->tsdf_.ExtractTriangleMesh();
        mesh->ComputeVertexNormals();

        std::cout << "[INFO] Writing TSDF to " << pathOut << std::endl;

        bool result = open3d::io::WriteTriangleMesh(pathOut, *mesh, false, true);

        std::cout << "[INFO] TSDF volume construction " << (result ? "finished" : "failed") << ": " << pathOut << std::endl;

        return result;
    }
}