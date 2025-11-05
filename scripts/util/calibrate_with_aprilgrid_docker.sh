#!/bin/bash

# make sure sufficient arguments are provided
if [ "$#" -ne 10 ]; then
    echo "Usage: $0 <profile_name> <tagCols> <tagRows> <tagSize> <tagSpacing> <accelerometer_noise_density> <accelerometer_random_walk> <gyroscope_noise_density> <gyroscope_random_walk> <update_rate>"
    exit 1
fi

# ensure scene data exists
if [ ! -e "/data" ]; then
    echo "Error: Scene folder does not exist. Make sure this container was run with calibration_start.sh."
    exit 1
fi

# ensure project root is accessible
if [ ! -e "/HandySLAM" ]; then
    echo "Error: Project root does not exist. Make sure this container was run with calibration_start.sh."
    exit 1
fi

# validate target arguments
if ! [[ "$2" =~ ^[0-9]+$ ]]; then
    echo "Error: <tagCols> must be an integer."
    exit 1
fi
if ! [[ "$3" =~ ^[0-9]+$ ]]; then
    echo "Error: <tagRows> must be an integer."
    exit 1
fi
if ! [[ "$4" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: <tagSize> must be a number."
    exit 1
fi
if ! [[ "$5" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: <tagSpacing> must be a number."
    exit 1
fi
if ! [[ "$6" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: <accelerometer_noise_density> must be a number."
    exit 1
fi
if ! [[ "$7" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: <accelerometer_random_walk> must be a number."
    exit 1
fi
if ! [[ "$8" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: <gyroscope_noise_density> must be a number."
    exit 1
fi
if ! [[ "$9" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: <gyroscope_random_walk> must be a number."
    exit 1
fi
if ! [[ "${10}" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: <update_rate> must be a number."
    exit 1
fi

echo "tagCols: $2"
echo "tagRows: $3"
echo "tagSize: $4"
echo "tagSpacing: $5"
echo "accelerometer_noise_density: $6"
echo "accelerometer_random_walk: $7"
echo "gyroscope_noise_density: $8"
echo "gyroscope_random_walk: $9"
echo "update_rate: ${10}"

PATH_RGB="/data/rgb.mp4"
PATH_ODOMETRY="/data/odometry.csv"
PATH_IMU="/data/imu.csv"
PATH_CAMERA_MATRIX="/data/camera_matrix.csv"

# temporary folder
PATH_TEMP="/tmp/scratch"
mkdir "$PATH_TEMP"
if [ ! -e "$PATH_TEMP" ]; then
    echo "Error: Unable to create temporary data folder: $PATH_TEMP."
    exit 1
fi
echo "Info: Created temporary data folder: $PATH_TEMP."

# output path
PATH_TEMP_RESULTS="/results"
PATH_TEMP_BAG="$PATH_TEMP_RESULTS/$1.bag"

# ensure temporary folders are cleaned up
cleanup() {
    rm -rf "$PATH_TEMP"
    rm -rf "$PATH_TEMP_BAG"
}
trap cleanup EXIT

# extract frames
PATH_TEMP_FRAMES="$PATH_TEMP/cam0"
mkdir "$PATH_TEMP_FRAMES"
if [ ! -e "$PATH_TEMP_FRAMES" ]; then
    echo "Error: Unable to create temporary folder: $PATH_TEMP_FRAMES."
    exit 1
fi
echo "Info: Created folder for imagery: $PATH_TEMP_FRAMES."
read CAMERA_W CAMERA_H < <(python3 /HandySLAM/scripts/util/split_frames.py "$PATH_RGB" "$PATH_ODOMETRY" "$PATH_TEMP_FRAMES")
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Failed to split RGB imagery: $EXIT_CODE."
    exit $EXIT_CODE
fi
echo "Info: Finished splitting image frames."
echo "Info: Resolution of RGB imagery: [${CAMERA_W}, ${CAMERA_H}]."

# write the target configuration file
PATH_TEMP_TARGET="$PATH_TEMP/aprilgrid.yaml"
cat <<EOF > "$PATH_TEMP_TARGET"
target_type: 'aprilgrid'
tagCols: $2
tagRows: $3
tagSize: $4
tagSpacing: $5
EOF
echo "Info: Finished writing target configuration to $PATH_TEMP_TARGET."

# write IMU in the format kalibr expects
PATH_TEMP_IMU="$PATH_TEMP/imu.csv"
echo "timestamp,omega_x,omega_y,omega_z,acc_x,acc_y,acc_z" > "$PATH_TEMP_IMU"
# treat timestamp as string literal to avoid losing precision
awk -F',' 'NR>1 {
    gsub(/[^0-9.]/, "", $1)     # remove stray chars
    split($1, t, ".")
    sec = t[1]
    frac = t[2]
    while(length(frac) < 9) frac = frac "0"
    t_nsec = sec "" substr(frac,1,9)
    gsub(/^0+/, "", t_nsec)     # remove leading zeros
    printf "%s,%s,%s,%s,%s,%s,%s\n", t_nsec, $5, $6, $7, $2, $3, $4
}' "$PATH_IMU" >> "$PATH_TEMP_IMU"

# write IMU calibration file
PATH_TEMP_IMU_CALIB="$PATH_TEMP/imu.yaml"
cat <<EOF > "$PATH_TEMP_IMU_CALIB"
sensor_model: "imu"
rostopic: "/imu"
update_rate: ${10}
accelerometer_noise_density: $6
accelerometer_random_walk: $7
gyroscope_noise_density: $8
gyroscope_random_walk: $9
EOF
echo "Info: Finished writing IMU calibration to $PATH_TEMP_IMU_CALIB."

# parse camera intrinsics
read FX _ CX <<<"$(awk -F',' 'NR==1 {print $1, $2, $3}' $PATH_CAMERA_MATRIX)"
read _ FY CY <<<"$(awk -F',' 'NR==2 {print $1, $2, $3}' $PATH_CAMERA_MATRIX)"
echo "Info: Parsed camera instrinsics (fx: $FX, fy: $FY, cx: $CX, cy: $CY)."

# write the camera intrinsics file
PATH_TEMP_CAMERA_INTRINSICS="$PATH_TEMP/cam0.yaml"
cat <<EOF > "$PATH_TEMP_CAMERA_INTRINSICS"
cam0:
  camera_model: pinhole
  resolution: [$CAMERA_W, $CAMERA_H]
  camera_name: cam0
  intrinsics: [$FX, $FY, $CX, $CY]
  distortion_model: radtan
  distortion_coeffs: [0, 0, 0, 0]
  rostopic: /cam0/image_raw
EOF
echo "Info: Finished writing camera intrinsics to $PATH_TEMP_CAMERA_INTRINSICS."

# run ros setup script
PATH_ROS_SETUP=$(find /opt -name setup.bash 2>/dev/null | grep ros)
source "$PATH_ROS_SETUP"
echo "Info: Ran $PATH_ROS_SETUP."

# run kalibr setup script
source "devel/setup.bash"
echo "Info: Ran Kalibr's setup.bash."

# create rosbag
rosrun kalibr kalibr_bagcreater \
    --folder $PATH_TEMP \
    --output-bag $PATH_TEMP_BAG

# check success
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Failed to create rosbag: $EXIT_CODE."
    exit $EXIT_CODE
fi
echo "Info: Created $PATH_TEMP_BAG."

# perform calibration
rosrun kalibr kalibr_calibrate_imu_camera \
    --bag $PATH_TEMP_BAG \
    --cam $PATH_TEMP_CAMERA_INTRINSICS \
    --imu $PATH_TEMP_IMU_CALIB \
    --imu-models calibrated \
    --target $PATH_TEMP_TARGET \
    --dont-show-report

# check success
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Failed to perform calibration: $EXIT_CODE."
    exit $EXIT_CODE
fi
echo "Info: Found calibration parameters."
