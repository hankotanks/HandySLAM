
import argparse
import os
import zlib
import numpy as np
import lz4.block
import cv2

from promptda.utils.io_wrapper import to_tensor_func, ensure_multiple_of, save_depth
from promptda.promptda import PromptDA

def extract_color(path_rgb, max_size=1008, multiple_of=14):
    cap = cv2.VideoCapture(path_rgb)
    if not cap.isOpened(): raise RuntimeError("Could not open video")

    while True:
        ret, frame = cap.read()
        if not ret: 
            return None

        image = np.asarray(frame).astype(np.float32)
        image = image / 255.

        max_size = max_size // multiple_of * multiple_of
        if max(image.shape) > max_size:
            h, w = image.shape[:2]
            scale = max_size / max(h, w)
            tar_h = ensure_multiple_of(h * scale)
            tar_w = ensure_multiple_of(w * scale)
            image = cv2.resize(image, (tar_w, tar_h), interpolation=cv2.INTER_AREA)

        yield to_tensor_func(image)

def extract_depth(path_depth):
    height, width = 192, 256
    sample_rate = 1

    try:
        with open(path_depth, 'rb') as infile:
            data = infile.read()
            data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

        for frame_id in range(0, depth.shape[0], sample_rate):
            yield to_tensor_func(depth)

    # per frame compression with lz4/zlib
    except:
        frame_id = 0
        with open(path_depth, 'rb') as infile:
            while True:
                size = infile.read(4)   # 32-bit integer
                if len(size) == 0:
                    break
                size = int.from_bytes(size, byteorder='little')
                if frame_id % sample_rate != 0:
                    infile.seek(size, 1)
                    frame_id += 1
                    continue

                # read the whole file
                data = infile.read(size)
                try:
                    # try using lz4
                    data = lz4.block.decompress(data, uncompressed_size=height * width * 2)  # UInt16 = 2bytes
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width).astype(np.float32)
                    depth /= 1000.
                except:
                    # try using zlib
                    data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width).astype(np.float32)

                yield to_tensor_func(depth)

                frame_id += 1

    return None

def main(args):
    if not os.path.exists(args.scene_path): raise RuntimeError("Scene does not exist")

    path_iphone = os.path.join(args.scene_path, "iphone")
    if not os.path.exists(path_iphone): raise RuntimeError("Scene does not exist")

    path_color = os.path.join(path_iphone, "rgb.mkv")
    if not os.path.exists(path_color): raise RuntimeError("RGB video does not exist")

    path_depth = os.path.join(path_iphone, "depth.bin")
    if not os.path.exists(path_depth): raise RuntimeError("Depth binary does not exist")

    path_out = os.path.join(path_iphone, "depth_upscaled")
    os.makedirs(path_out, exist_ok=True)

    DEVICE = 'cuda'
    model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl").to(DEVICE).eval()      
    
    color = extract_color(path_color)
    depth = extract_depth(path_depth)  

    frame_index = 0
    while True:
        frame_color = next(color, None)
        if frame_color is None:
            print(f"Finished after {frame_index + 1} frames")
            break

        frame_depth = next(depth, None)
        if frame_color is not None and frame_depth is None:
            raise RuntimeError(f"Missing depth data associated with frame {frame_index}")
        
        tensor_color, tensor_depth = frame_color.to(DEVICE), frame_depth.to(DEVICE)
        upscaled = model.predict(tensor_color, tensor_depth)

        save_depth(upscaled.detach().cpu(), output_path = os.path.join(path_out, f'{frame_index:06d}.png'), save_vis = False)

        print(f"Saved frame {frame_index}")
        frame_index += 1

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scene-path', help='path to scene')
    args = p.parse_args()

    main(args)