#!/bin/bash

# make sure scene was provided
if [ "$#" -ne 10 ]; then
    echo "Usage: $0 <scene_path> <tagCols> <tagRows> <tagSize> <tagSpacing> <accelerometer_noise_density> <accelerometer_random_walk> <gyroscope_noise_density> <gyroscope_random_walk> <update_rate>"
    exit 1
fi

# ensure it exists
if [ -e "$1" ]; then
    echo "Found scene: $1."
else
    echo "Error: Scene folder does not exist: $1."
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
if ! [[ "$4" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: <tagSize> must be a number."
    exit 1
fi
if ! [[ "$5" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: <tagSpacing> must be a number."
    exit 1
fi
if ! [[ "$6" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: <accelerometer_noise_density> must be a number."
    exit 1
fi
if ! [[ "$7" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: <accelerometer_random_walk> must be a number."
    exit 1
fi
if ! [[ "$8" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: <gyroscope_noise_density> must be a number."
    exit 1
fi
if ! [[ "$9" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: <gyroscope_random_walk> must be a number."
    exit 1
fi
if ! [[ "${10}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: <update_rate> must be a number."
    exit 1
fi

# ensure rgb imagery exists
RGB_VIDEO="$1/rgb.mp4"
if [ ! -f "$RGB_VIDEO" ]; then
    echo "Error: Video not found: $RGB_VIDEO."
    exit 1
fi

# ensure odometry.csv exists
ODOMETRY_FILE="$1/odometry.csv"
if [ ! -f "$ODOMETRY_FILE" ]; then
    echo "Error: Odometry not found: $ODOMETRY_FILE."
    exit 1
fi

# ensure imu.csv exists
IMU_FILE="$1/imu.csv"
if [ ! -f "$IMU_FILE" ]; then
    echo "Error: IMU data not found: $IMU_FILE."
    exit 1
fi

# ensure camera_matrix.csv exists
CAMERA_MATRIX_FILE="$1/camera_matrix.csv"
if [ ! -f "$CAMERA_MATRIX_FILE" ]; then
    echo "Error: Camera matrix file not found: $CAMERA_MATRIX_FILE."
    exit 1
fi

# make temporary folder
TEMP_FOLDER=$(mktemp -d /tmp/kalibr_data_XXXX)
if [ ! -e "$TEMP_FOLDER" ]; then
    echo "Error: Unable to create temp folder: $TEMP_FOLDER."
    exit 1
fi
echo "Using temporary folder: $TEMP_FOLDER."
trap 'rm -rf "$TEMP_FOLDER"' EXIT

# query frame rate
CAM_RATE=$(ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate "$RGB_VIDEO")
echo "Frame rate: $CAM_RATE fps."

# query dimensions
CAM_WIDTH=$(ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=width "$RGB_VIDEO")
CAM_HEIGHT=$(ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=height "$RGB_VIDEO")
echo "Image size: [${CAM_WIDTH}, ${CAM_HEIGHT}]."

# extract frames
TEMP_FOLDER_FRAMES="$TEMP_FOLDER/cam0"
mkdir "$TEMP_FOLDER_FRAMES"
if [ ! -e "$TEMP_FOLDER_FRAMES" ]; then
    echo "Error: Unable to create temp folder: $TEMP_FOLDER_FRAMES."
    exit 1
fi
echo "Created folder for imagery: $TEMP_FOLDER_FRAMES."
echo "Began splitting image frames."
ffmpeg -i "$RGB_VIDEO" "$TEMP_FOLDER_FRAMES/frame_%06d.png"
echo "Finished splitting image frames."

# write timestamps
TIMESTAMPS_FILE="$TEMP_FOLDER/timestamps.txt"
echo "Began writing frame timestamps to $TIMESTAMPS_FILE."
tail -n +2 "$ODOMETRY_FILE" | cut -d',' -f1 > "$TIMESTAMPS_FILE"
echo "Finished writing frame timestamps to $TIMESTAMPS_FILE."

# rectify frame names with nanosecond timestamps
echo "Began rectifying frame names."
ls "$TEMP_FOLDER_FRAMES"/frame_*.png | sort | \
while read -r frame_file && read -r timestamp_sec <&3; do
    timestamp_nsec=$(awk -v t="$timestamp_sec" 'BEGIN { printf "%.0f", t*1e9 }')
    mv "$frame_file" "$TEMP_FOLDER_FRAMES/${timestamp_nsec}.png"
done 3< "$TIMESTAMPS_FILE"
echo "Finished rectifying frame names."

# write IMU in the format kalibr expects
IMU_FILE_TEMP="$TEMP_FOLDER/imu.csv"
echo "Began parsing $IMU_FILE."
echo "timestamp,omega_x,omega_y,omega_z,acc_x,acc_y,acc_z" > "$IMU_FILE_TEMP"
# treat timestamp as string literal to avoid losing precision
awk -F',' 'NR>1 {
    split($1, t, ".")
    sec = t[1]
    frac = t[2]
    while(length(frac) < 9) frac = frac "0"
    t_nsec = sec "" substr(frac,1,9)
    printf "%s,%s,%s,%s,%s,%s,%s\n", t_nsec, $5, $6, $7, $2, $3, $4
}' "$IMU_FILE" >> "$IMU_FILE_TEMP"
echo "Finished writing into $IMU_FILE_TEMP."

# write IMU calibration file
IMU_CALIB="$TEMP_FOLDER/imu.yaml"
echo "Began writing IMU calibration to $IMU_CALIB."
cat <<EOF > "$IMU_CALIB"
sensor_model: "imu"
rostopic: "/imu"
update_rate: ${10}
accelerometer_noise_density: $6
accelerometer_random_walk: $7
gyroscope_noise_density: $8
gyroscope_random_walk: $9
EOF
echo "Finished writing IMU calibration."

# write the target configuration file
TARGET_FILE="$TEMP_FOLDER/aprilgrid.yaml"
echo "Began writing target configuration file to $TARGET_FILE."
cat <<EOF > "$TARGET_FILE"
target_type: 'aprilgrid'
tagCols: $2
tagRows: $3
tagSize: $4
tagSpacing: $5
EOF
echo "Finished writing target configuration."

# parse camera intrinsics
read FX _ CX <<<"$(awk -F',' 'NR==1 {print $1, $2, $3}' $CAMERA_MATRIX_FILE)"
read _ FY CY <<<"$(awk -F',' 'NR==2 {print $1, $2, $3}' $CAMERA_MATRIX_FILE)"
echo "Parsed camera instrinsics."
echo "fx: $FX"
echo "fy: $FY"
echo "cx: $CX"
echo "cy: $CY"

# write the camera intrinsics file
CAMERA_INTRINSICS_FILE="$TEMP_FOLDER/cam0.yaml"
echo "Began writing camera intrinsics to $CAMERA_INTRINSICS_FILE."
cat <<EOF > "$CAMERA_INTRINSICS_FILE"
cam0:
  camera_model: pinhole
  resolution: [$CAM_WIDTH, $CAM_HEIGHT]
  camera_name: cam0
  intrinsics: [$FX, $FY, $CX, $CY]
  distortion_model: radtan
  distortion_coeffs: [0, 0, 0, 0]
  rostopic: /cam0/image_raw
EOF
echo "Finished writing camera intrinsics."

# copy script to data folder
SCRIPT_DIR=$(dirname "$0")
cp "$SCRIPT_DIR/calibrate_helper.sh" "$TEMP_FOLDER/"
echo "Copied helper script to $TEMP_FOLDER/calibrate_helper.sh."

# enter the container
echo "Entering docker container."
sudo docker run -it -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$TEMP_FOLDER:/data" \
    kalibr

# check success
DOCKER_EXIT_CODE=$?
if [ $DOCKER_EXIT_CODE -ne 0 ]; then
    echo "Error: Calibration failed with exit code: $DOCKER_EXIT_CODE."
    exit $DOCKER_EXIT_CODE
fi

# wait to exit
read -p "Press [Return] to exit..."