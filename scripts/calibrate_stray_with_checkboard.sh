#!/bin/bash

# make sure sufficient arguments are provided
if [ "$#" -ne 9 ]; then
    echo "Usage: $0 <scene_path> <profile_name> <targetCols> <targetRows> <spacingMeters> <accelerometer_noise_density> <accelerometer_random_walk> <gyroscope_noise_density> <gyroscope_random_walk>"
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
if ! [[ "$3" =~ ^[0-9]+$ ]]; then
    echo "Error: <targetCols> must be an integer."
    exit 1
fi
if ! [[ "$4" =~ ^[0-9]+$ ]]; then
    echo "Error: <targetRows> must be an integer."
    exit 1
fi
if ! [[ "$5" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: <spacingMeters> must be a number (no scientific notation, ex. 1.3e3)."
    exit 1
fi

echo "Info: <targetCols>: $3"
echo "Info: <targetRows>: $4"
echo "Info: <rowSpacingMeters>: $5"
echo "Info: <colSpacingMeters>: $5"

PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")

# make scratch folder
PATH_TEMP="/tmp/$2"
if [ -e "$PATH_TEMP" ]; then
    echo "Info: Removing previous profile's temporary folder: $2."
    rm -rf "$PATH_TEMP"
fi
mkdir "$PATH_TEMP"
if [ ! -e "$PATH_TEMP" ]; then
    echo "Error: Unable to create scratch folder: $PATH_TEMP."
    exit 1
fi

# ensure temporary folders are cleaned up
cleanup() {
    rm -rf "$PATH_TEMP"
}
trap cleanup EXIT

# parse imagery timestamps
PATH_TEMP_TIMESTAMPS="$PATH_TEMP/timestamps.txt"
awk -F',' 'NR>1 {
    gsub(/[^0-9.]/, "", $1)
    split($1, t, ".")
    sec = t[1]
    frac = t[2]
    while(length(frac) < 9) frac = frac "0"
    t_nsec = sec "" substr(frac,1,9)
    gsub(/^0+/, "", t_nsec)
    printf "%s\n", t_nsec
}' "$PATH_ODOMETRY" > "$PATH_TEMP_TIMESTAMPS"
echo "Info: Wrote timestamps to $PATH_TEMP_TIMESTAMPS."

# write IMU in the format kalibr expects
PATH_TEMP_IMU="$PATH_TEMP/imu.csv"
echo "timestamp,omega_x,omega_y,omega_z,acc_x,acc_y,acc_z" > "$PATH_TEMP_IMU"
# treat timestamp as string literal to avoid losing precision
awk -F',' 'NR>1 {
    gsub(/[^0-9.]/, "", $1)
    split($1, t, ".")
    sec = t[1]
    frac = t[2]
    while(length(frac) < 9) frac = frac "0"
    t_nsec = sec "" substr(frac,1,9)
    gsub(/^0+/, "", t_nsec)
    printf "%s,%s,%s,%s,%s,%s,%s\n", t_nsec, $5, $6, $7, $2, $3, $4
}' "$PATH_IMU" >> "$PATH_TEMP_IMU"

# write the target.yaml
PATH_TEMP_TARGET="$PATH_TEMP/target.yaml"
cat <<EOF > "$PATH_TEMP_TARGET"
target_type: 'checkerboard'
targetCols: $3
targetRows: $4
rowSpacingMeters: $5
colSpacingMeters: $5
EOF
echo "Info: Finished writing target configuration to $PATH_TEMP_TARGET."

# parse camera intrinsics
read FX _ CX <<<"$(awk -F',' 'NR==1 {print $1, $2, $3}' $PATH_CAMERA_MATRIX)"
read _ FY CY <<<"$(awk -F',' 'NR==2 {print $1, $2, $3}' $PATH_CAMERA_MATRIX)"
echo "Info: Parsed camera instrinsics (fx: $FX, fy: $FY, cx: $CX, cy: $CY)."

# enter the container
CALIBRATE_DOCKER_START="$(dirname "$(realpath "$0")")/util/calibrate_docker_start.sh"
echo "Info: Running $CALIBRATE_DOCKER_START."
chmod +x "$CALIBRATE_DOCKER_START"
"$CALIBRATE_DOCKER_START" "$2" \
    "$PATH_RGB" "$PATH_TEMP_TIMESTAMPS" "$PATH_TEMP_TARGET" "$PATH_TEMP_IMU" \
    "$FX" "$FY" "$CX" "$CY" "$6" "$7" "$8" "$9"

# check success
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Docker exited with code $EXIT_CODE."
    exit $EXIT_CODE
fi