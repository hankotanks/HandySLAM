import sys
import os
import cv2

if len(sys.argv) < 3: sys.exit(1)

PATH_RGB = sys.argv[1]
PATH_ODOMETRY = sys.argv[2]
PATH_TEMP_FRAMES = sys.argv[3]

cap = cv2.VideoCapture(PATH_RGB)
if not cap.isOpened():
    print(f"Error: Cannot open video {PATH_RGB}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with open(PATH_ODOMETRY, "r") as f: lines = f.readlines()[1:]

timestamps_sec = [float(line.split(',')[0]) for line in lines]
frame_idx = 0
percent_last = -1
while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= len(timestamps_sec): break
    timestamp_nsec = int(timestamps_sec[frame_idx] * 1e9)
    frame_filename = os.path.join(PATH_TEMP_FRAMES, f"{timestamp_nsec}.png")
    cv2.imwrite(frame_filename, frame)
    frame_idx += 1
    percent = (frame_idx * 100) // total_frames + 1
    if percent != percent_last:
        percent_last = percent
        sys.stderr.write(f"\rInfo: Processing RGB frames [{percent}%].")
        sys.stderr.flush()
        
sys.stderr.write("\n")
sys.stderr.flush()

print(f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

cap.release()