#!/bin/bash

# make sure scene_path is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <scene_path>"
    exit 1
fi

# ensure it exists
if [ -e "$1" ]; then
    echo "Found scene: $1."
else
    echo "Error: Scene folder does not exist: $1."
    exit 1
fi

SCENE_NAME=$(basename "$1")

# ensure rgb.mp4 exists
PATH_RGB="$1/rgb.mp4"
if [ ! -f "$PATH_RGB" ]; then
    echo "Error: Video not found: $PATH_RGB."
    exit 1
fi

# ensure odometry.csv exists
PATH_ODOMETRY="$1/odometry.csv"
if [ ! -f "$PATH_ODOMETRY" ]; then
    echo "Error: Odometry not found: $PATH_ODOMETRY."
    exit 1
fi

# ensure imu.csv exists
PATH_IMU="$1/imu.csv"
if [ ! -f "$PATH_IMU" ]; then
    echo "Error: IMU data not found: $PATH_IMU."
    exit 1
fi

# ensure camera_matrix.csv exists
PATH_CAMERA_MATRIX="$1/camera_matrix.csv"
if [ ! -f "$PATH_CAMERA_MATRIX" ]; then
    echo "Error: Camera matrix file not found: $PATH_CAMERA_MATRIX."
    exit 1
fi

# make result folder
PATH_TEMP_RESULTS=$(mktemp -d "/tmp/${SCENE_NAME}_XXXX_results")
mkdir "$PATH_TEMP_RESULTS"
if [ ! -e "$PATH_TEMP_RESULTS" ]; then
    echo "Error: Unable to create temporary results folder: $PATH_TEMP_RESULTS."
    exit 1
fi
echo "Info: Created temporary results folder: $PATH_TEMP_RESULTS."

# ensure temporary folders are cleaned up
cleanup() {
    rm -rf "$PATH_TEMP_RESULTS"/*.bag
}
trap cleanup EXIT

# enter the container
echo "Info: Entering docker container."
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
sudo docker run -it -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$1:/data" \
    -v "$PROJECT_ROOT:/HandySLAM" \
    -v "$PATH_TEMP_RESULTS:/results"\
    kalibr
echo "Info: Closing docker container."

# check success
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Docker exited with code $EXIT_CODE."
    exit $EXIT_CODE
fi

# TODO: Read temp-camchain-imucam.yaml and create a profile file in HandySLAM/profiles
# the user should additionally pass in a profile name
# DataloaderStray must also be provided a profile, which it will load and use
# to generate the settings file