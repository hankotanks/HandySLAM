#!/bin/bash
ROS_SETUP_SCRIPT=$(find /opt -name setup.bash 2>/dev/null | grep ros)
source "$ROS_SETUP_SCRIPT"
source "devel/setup.bash"
rosrun kalibr kalibr_bagcreater \
    --folder /data \
    --output-bag /data/calib.bag

rosrun kalibr kalibr_calibrate_imu_camera \
    --bag /data/calib.bag \
    --cam /data/cam0.yaml \
    --imu /data/imu.yaml \
    --imu-models calibrated \
    --target /data/aprilgrid.yaml