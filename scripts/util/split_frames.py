import sys
import os
import cv2

if len(sys.argv) < 3: sys.exit(1)

PATH_RGB = sys.argv[1]
PATH_TIMESTAMPS = sys.argv[2]
PATH_TEMP_FRAMES = sys.argv[3]

cap = cv2.VideoCapture(PATH_RGB)
if not cap.isOpened():
    sys.stderr.write(f"Error: Cannot open video {PATH_RGB}.\n")
    sys.stderr.flush()
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with open(PATH_TIMESTAMPS, "r") as f: lines = f.readlines()
if not lines:
    sys.stderr.write(f"Error: Cannot read timestamps {PATH_TIMESTAMPS}.\n")
    sys.stderr.flush()
    sys.exit(1)

timestamps_nsec = [int(line.strip()) for line in lines]
frame_idx = 0
percent_last = -1
while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= len(timestamps_nsec): break
    frame_filename = os.path.join(PATH_TEMP_FRAMES, f"{timestamps_nsec[frame_idx]}.png")
    cv2.imwrite(frame_filename, frame)
    frame_idx += 1
    percent = (frame_idx * 100) // total_frames + 1
    if percent != percent_last:
        percent_last = percent
        sys.stderr.write(f"\rInfo: Processing RGB frames [{percent}%].")
        sys.stderr.flush()

sys.stderr.write(f"\nInfo: Finished processing RGB frames [{frame_idx}].\n")
sys.stderr.flush()

print(f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

cap.release()