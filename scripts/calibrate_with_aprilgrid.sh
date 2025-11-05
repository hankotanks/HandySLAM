#!/bin/bash

# make sure scene_path is provided
if [ "$#" -ne 10 ]; then
    echo "Usage: $0 <scene_path> <tagCols> <tagRows> <tagSize> <tagSpacing> <accelerometer_noise_density> <accelerometer_random_walk> <gyroscope_noise_density> <gyroscope_random_walk> <update_rate>"
    exit 1
fi

# ensure it exists
if [ -e "$1" ]; then
    echo "Info: Found scene: $1."
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

# make result folder
PATH_TEMP_RESULTS=$(mktemp -d "/tmp/${SCENE_NAME}_XXXX_results")
if [ ! -e "$PATH_TEMP_RESULTS" ]; then
    echo "Error: Unable to create temporary results folder: $PATH_TEMP_RESULTS."
    exit 1
fi
echo "Info: Created temporary results folder: $PATH_TEMP_RESULTS."

# enter the container
echo "Info: Entering docker container."
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
docker run --rm -it \
    -u $(id -u):$(id -g) \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$1:/data" \
    -v "$PROJECT_ROOT:/HandySLAM" \
    -v "$PATH_TEMP_RESULTS:/results" \
    --entrypoint /bin/bash \
    kalibr -c '
        export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1
        cd $WORKSPACE
        chmod +x /HandySLAM/scripts/util/calibrate_with_aprilgrid_docker.sh
        /HandySLAM/scripts/util/calibrate_with_aprilgrid_docker.sh '"$2 $3 $4 $5 $6 $7 $8 $9 ${10}"'
        EXIT_CODE=$?
        echo "Calibration finished with exit code $EXIT_CODE"
        exit $EXIT_CODE
    '

# check success
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Docker exited with code $EXIT_CODE."
    exit $EXIT_CODE
fi
echo "Info: Closing docker container."

# TODO: Read temp-camchain-imucam.yaml and create a profile file in HandySLAM/profiles
# the user should additionally pass in a profile name
# DataloaderStray must also be provided a profile, which it will load and use
# to generate the settings file