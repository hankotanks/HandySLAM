#!/bin/bash

# ensure profiles folder exists
PATH_PROFILES="/profiles"
if [ ! -e "$PATH_PROFILES" ]; then
    echo "Error: $PATH_PROFILES does not exist. Make sure this container was run with calibrate_docker_start.sh."
    exit 1
fi

# ensure utility scripts are accessible
if [ ! -f "/split_frames.py" ]; then
    echo "Error: /split_frames.py does not exist. Make sure this container was run with calibrate_docker_start.sh."
    exit 1
fi

# ensure scratch folder is accessible
PATH_BAG="/tmp/$HANDYSLAM_PROFILE_NAME"
mkdir "$PATH_BAG"
if [ ! -e "$PATH_BAG" ]; then
    echo "Error: Could not create temporary folder. Make sure this container was run with calibrate_docker_start.sh."
    exit 1
fi
PATH_BAG_FILE="/tmp/$HANDYSLAM_PROFILE_NAME.bag"

# run ros setup script
PATH_ROS_SETUP=$(find /opt -name setup.bash 2>/dev/null | grep ros)
source "$PATH_ROS_SETUP"
echo "Info: Ran $PATH_ROS_SETUP."

# run kalibr setup script
source "devel/setup.bash"
echo "Info: Ran Kalibr's setup.bash."

# find rgb video file and confirm it exists
PATH_RGB_MATCHES=($(find "/" -maxdepth 1 -type f -name "rgb.*"))
if [ ${#PATH_RGB_MATCHES[@]} -eq 0 ]; then
    echo "Error: No video file found in root folder."
    exit 1
elif [ ${#PATH_RGB_MATCHES[@]} -gt 1 ]; then
    echo "Error: Multiple matching video files found in root folder."
    exit 1
fi
PATH_RGB="${PATH_RGB_MATCHES[0]}"
if [ ! -f "$PATH_RGB" ]; then
    echo "Error: $PATH_RGB does not exist. Make sure this container was run with calibrate_docker_start.sh."
    exit 1
fi

# ensure timestamps.txt exists at root
PATH_TIMESTAMPS="/timestamps.txt"
if [ ! -f "$PATH_TIMESTAMPS" ]; then
    echo "Error: $PATH_TIMESTAMPS does not exist. Make sure this container was run with calibrate_docker_start.sh."
    exit 1
fi

# ensure imu.csv exists at root
PATH_IMU="/imu.csv"
if [ ! -f "$PATH_IMU" ]; then
    echo "Error: $PATH_IMU does not exist. Make sure this container was run with calibrate_docker_start.sh."
    exit 1
fi

# copy imu.csv to bag
PATH_BAG_IMU="$PATH_BAG/imu.csv"
cp "$PATH_IMU" "$PATH_BAG_IMU"
if [ ! -f "$PATH_BAG_IMU" ]; then
    echo "Error: Failed to copy $PATH_IMU to $PATH_BAG_IMU."
    exit 1
fi

PATH_TARGET="/target.yaml"
if [ ! -f "$PATH_TARGET" ]; then
    echo "Error: $TARGET does not exist. Make sure this container was run with calibrate_docker_start.sh."
    exit 1
fi

HOME="/tmp"

# calculate average imu freq
IMU_DT=$(awk -F',' 'NR>1 {dt = $1 - prev; if(dt>0) print dt; prev = $1} NR==1 {prev=$1}' "$PATH_IMU")
IMU_DT_MEDIAN=$(echo "$IMU_DT" | sort -n | awk '{a[NR]=$1} END{if(NR%2==1){print a[(NR+1)/2]} else{print (a[NR/2]+a[NR/2+1])/2}}')
update_rate=$(awk -v dt_ns="$IMU_DT_MEDIAN" 'BEGIN {printf "%.2f", 1e9 / dt_ns}')
echo "Info: <update_rate>: $update_rate"

# write IMU calibration file
PATH_TEMP_IMU_CALIB="/tmp/imu.yaml"
cat <<EOF > "$PATH_TEMP_IMU_CALIB"
sensor_model: "imu"
rostopic: "/imu"
update_rate: $update_rate
accelerometer_noise_density: $HANDYSLAM_ACC_NOISE
accelerometer_random_walk: $HANDYSLAM_ACC_RANDOM_WALK
gyroscope_noise_density: $HANDYSLAM_GYRO_NOISE
gyroscope_random_walk: $HANDYSLAM_GYRO_RANDOM_WALK
EOF
echo "Info: Finished writing IMU calibration to $PATH_TEMP_IMU_CALIB."

# extract frames
PATH_BAG_FRAMES="$PATH_BAG/cam0"
mkdir "$PATH_BAG_FRAMES"
if [ ! -e "$PATH_BAG_FRAMES" ]; then
    echo "Error: Unable to create temporary folder: $PATH_BAG_FRAMES."
    exit 1
fi
echo "Info: Created folder for imagery: $PATH_BAG_FRAMES."
read CAMERA_W CAMERA_H < <(python3 /split_frames.py "$PATH_RGB" "$PATH_TIMESTAMPS" "$PATH_BAG_FRAMES")
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Failed to split RGB imagery: $EXIT_CODE."
    exit $EXIT_CODE
fi
echo "Info: Finished splitting image frames."
echo "Info: Resolution of RGB imagery: [${CAMERA_W}, ${CAMERA_H}]."

# write the camera intrinsics file
PATH_TEMP_CAMERA_INTRINSICS="/tmp/cam0.yaml"
cat <<EOF > "$PATH_TEMP_CAMERA_INTRINSICS"
cam0:
  camera_model: pinhole
  resolution: [$CAMERA_W, $CAMERA_H]
  camera_name: cam0
  intrinsics: [$HANDYSLAM_FX, $HANDYSLAM_FY, $HANDYSLAM_CX, $HANDYSLAM_CY]
  distortion_model: radtan
  distortion_coeffs: [0, 0, 0, 0]
  rostopic: /cam0/image_raw
EOF
echo "Info: Finished writing camera intrinsics to $PATH_TEMP_CAMERA_INTRINSICS."

# create rosbag
rosrun kalibr kalibr_bagcreater \
    --folder $PATH_BAG \
    --output-bag $PATH_BAG_FILE

# check success
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Failed to create rosbag: $EXIT_CODE."
    exit $EXIT_CODE
fi
echo "Info: Created $PATH_BAG_FILE."

# remove all the old frames
rm -rf "${PATH_BAG_FRAMES}/*.png"
echo "Info: Removed frames: ${PATH_BAG_FRAMES}."

# perform calibration
rosrun kalibr kalibr_calibrate_imu_camera \
    --bag $PATH_BAG_FILE \
    --cam $PATH_TEMP_CAMERA_INTRINSICS \
    --imu $PATH_TEMP_IMU_CALIB \
    --imu-models calibrated \
    --target $PATH_TARGET \
    --dont-show-report

# check success
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Failed to perform calibration: $EXIT_CODE."
    exit $EXIT_CODE
fi
echo "Info: Found calibration parameters."

# copy results to profile folder
cp "/tmp/$HANDYSLAM_PROFILE_NAME-camchain-imucam.yaml" "$PATH_PROFILES/$HANDYSLAM_PROFILE_NAME-camchain-imucam.yaml"
cp "/tmp/$HANDYSLAM_PROFILE_NAME-imu.yaml" "$PATH_PROFILES/$HANDYSLAM_PROFILE_NAME-imu.yaml"