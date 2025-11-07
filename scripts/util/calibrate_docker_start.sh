#!/bin/bash

# make sure sufficient arguments are provided
if [ "$#" -ne 13 ]; then
    echo "Usage: $0 <profile_name> <rgb_video_path> <timestamps_path> <target_path> <imu_path> <fx> <fy> <cx> <cy> <accelerometer_noise_density> <accelerometer_random_walk> <gyroscope_noise_density> <gyroscope_random_walk>"
    exit 1
fi

# ensure rgb video file exists
if [ ! -f "$2" ]; then
    echo "Error: <rgb_video_path> not found: $2."
    exit 1
fi
RGB_FNAME="${2##*/}"
if [[ "${RGB_FNAME%.*}" != "rgb" ]]; then
    echo "Error: <rgb_video_path> must point to a video with name 'rgb': $2."
fi

# ensure timestamps.txt exists
if [ ! -f "$3" ]; then
    echo "Error: <timestamps_path> not found: $3."
    exit 1
fi
if [[ "${3##*/}" != "timestamps.txt" ]]; then
    echo "Error: <timestamps_path> must point to timestamps.txt: $3."
fi

# ensure target.yaml exists
if [ ! -f "$4" ]; then
    echo "Error: <target_path> not found: $4."
    exit 1
fi
if [[ "${4##*/}" != "target.yaml" ]]; then
    echo "Error: <target_path> must point to target.yaml: $4."
fi

# ensure imu.csv exists
if [ ! -f "$5" ]; then
    echo "Error: <imu_path> not found: $5."
    exit 1
fi
if [[ "${5##*/}" != "imu.csv" ]]; then
    echo "Error: <imu_path> must point to imu.csv: $5."
fi


# validate target arguments
if ! [[ "$6" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: <fx> must be a number (no scientific notation, ex. 1.3e3)."
    exit 1
fi
if ! [[ "$7" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: <fy> must be a number (no scientific notation, ex. 1.3e3)."
    exit 1
fi
if ! [[ "$8" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: <cx> must be a number (no scientific notation, ex. 1.3e3)."
    exit 1
fi
if ! [[ "$9" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: <cy> must be a number (no scientific notation, ex. 1.3e3)."
    exit 1
fi
if ! [[ "${10}" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: <accelerometer_noise_density> must be a number."
    exit 1
fi
if ! [[ "${11}" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: <accelerometer_random_walk> must be a number."
    exit 1
fi
if ! [[ "${12}" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: <gyroscope_noise_density> must be a number."
    exit 1
fi
if ! [[ "${13}" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: <gyroscope_random_walk> must be a number."
    exit 1
fi

HANDYSLAM_PROFILE_NAME="$1"
HANDYSLAM_FX=$(awk -v v="$6" 'BEGIN { printf "%f\n", v }')
HANDYSLAM_FY=$(awk -v v="$7" 'BEGIN { printf "%f\n", v }')
HANDYSLAM_CX=$(awk -v v="$8" 'BEGIN { printf "%f\n", v }')
HANDYSLAM_CY=$(awk -v v="$9" 'BEGIN { printf "%f\n", v }')
HANDYSLAM_ACC_NOISE=$(awk -v v="${10}" 'BEGIN { printf "%f\n", v }')
HANDYSLAM_ACC_RANDOM_WALK=$(awk -v v="${11}" 'BEGIN { printf "%f\n", v }')
HANDYSLAM_GYRO_NOISE=$(awk -v v="${12}" 'BEGIN { printf "%f\n", v }')
HANDYSLAM_GYRO_RANDOM_WALK=$(awk -v v="${13}" 'BEGIN { printf "%f\n", v }')
echo "Info: <fx>: $HANDYSLAM_FX"
echo "Info: <fy>: $HANDYSLAM_FY"
echo "Info: <cx>: $HANDYSLAM_CX"
echo "Info: <cy>: $HANDYSLAM_CY"
echo "Info: <accelerometer_noise_density>: $HANDYSLAM_ACC_NOISE"
echo "Info: <accelerometer_random_walk>: $HANDYSLAM_ACC_RANDOM_WALK"
echo "Info: <gyroscope_noise_density>: $HANDYSLAM_GYRO_NOISE"
echo "Info: <gyroscope_random_walk>: $HANDYSLAM_GYRO_RANDOM_WALK"

# HandySLAM root path
PATH_PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
echo "Info: Project root: $PATH_PROJECT_ROOT."

# additional paths for mounting
PATH_PROFILES="$PATH_PROJECT_ROOT/profiles"
if [ ! -e "$PATH_PROFILES" ]; then
    echo "Info: $PATH_PROFILES does not exist. Creating."
    mkdir "$PATH_PROFILES"
    if [ ! -e "$PATH_PROFILES" ]; then
        echo "Error: Failed to create $PATH_PROFILES."
        exit 1
    fi
fi
PATH_SCRIPT="$PATH_PROJECT_ROOT/scripts/util/calibrate_docker.sh"
if [ ! -f "$PATH_SCRIPT" ]; then
    echo "Error: $PATH_SCRIPT does not exist."
    exit 1
fi
PATH_SCRIPT_SPLIT_FRAMES="$PATH_PROJECT_ROOT/scripts/util/split_frames.py"
if [ ! -f "$PATH_SCRIPT_SPLIT_FRAMES" ]; then
    echo "Error: $PATH_SCRIPT_SPLIT_FRAMES does not exist."
    exit 1
fi

# enter the container
echo "Info: Entering docker container."
xhost +SI:localuser:$(whoami)
docker run --rm -it \
    -u $(id -u):$(id -g) \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -e "HANDYSLAM_PROFILE_NAME=$HANDYSLAM_PROFILE_NAME" \
    -e "HANDYSLAM_FX=$HANDYSLAM_FX" \
    -e "HANDYSLAM_FY=$HANDYSLAM_FY" \
    -e "HANDYSLAM_CX=$HANDYSLAM_CX" \
    -e "HANDYSLAM_CY=$HANDYSLAM_CY" \
    -e "HANDYSLAM_ACC_NOISE=$HANDYSLAM_ACC_NOISE" \
    -e "HANDYSLAM_ACC_RANDOM_WALK=$HANDYSLAM_ACC_RANDOM_WALK" \
    -e "HANDYSLAM_GYRO_NOISE=$HANDYSLAM_GYRO_NOISE" \
    -e "HANDYSLAM_GYRO_RANDOM_WALK=$HANDYSLAM_GYRO_RANDOM_WALK" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$2:/rgb.mp4" \
    -v "$3:/timestamps.txt" \
    -v "$4:/target.yaml" \
    -v "$5:/imu.csv" \
    -v "$PATH_SCRIPT:/calibrate_docker.sh" \
    -v "$PATH_SCRIPT_SPLIT_FRAMES:/split_frames.py" \
    -v "$PATH_PROFILES:/profiles" \
    --entrypoint /bin/bash \
    kalibr -c '
        export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1
        cd $WORKSPACE
        chmod +x /calibrate_docker.sh
        /calibrate_docker.sh
        EXIT_CODE=$?
        echo "Info: Calibration finished with exit code $EXIT_CODE"
        exit $EXIT_CODE
    '

# check success
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Docker exited with code $EXIT_CODE."
    exit $EXIT_CODE
fi
echo "Info: Closing docker container."