import os
import cv2
import torch
import glob
from tqdm import tqdm
import argparse
import pandas as pd
from collections import deque
import numpy as np
from PIL import Image
import sys
sys.path.append("../AltFreezing/")
from test_tools.common import detect_all, grab_all_frames
from preprocessing.utils import warp_img, apply_transform, cut_patch


mean_face_landmarks = np.load("preprocessing/20words_mean_face.npy")
print(mean_face_landmarks.shape)
STD_SIZE = (256, 256)
STABLE_POINTS = [33, 36, 39, 42, 45]

def get_mouth_one_video(args):

    input_file = args.video
    target_dir = os.path.join(os.path.dirname(input_file), "mouth", input_file.split("/")[-1].replace(".mp4", ""))
    os.makedirs(target_dir, exist_ok=True)

    max_frame = 768
    cache_file = f"{input_file}_{str(max_frame)}.pth"
    # os.remove(cache_file)
    # if os.path.exists(cache_file):
    #     print("detection result loaded from cache")
    #     _, all_lm68 = torch.load(cache_file)
    #     frames = grab_all_frames(input_file, max_size=max_frame, cvt=True)
    #     if len(frames) != len(all_lm68):
    #         print("detecting")
    #         _, all_lm68, frames = detect_all(input_file, return_frames=True, max_size=max_frame)
    #         print("detect finished")
    # else:
    print("detecting")
    _, all_lm68, frames = detect_all(input_file, return_frames=True, max_size=max_frame)
    print("detect finished")
    print(len(all_lm68), len(frames))

    q_frames, q_landmarks, q_name = deque(), deque(), deque()
    pre_landmarks = None
    for i, frame in enumerate(frames):
        frame_name = str(i).zfill(6) + ".png"
        img = np.asarray(frame, dtype=np.uint8)
        landmarks = all_lm68[i]
        if len(landmarks) > 0:
            landmarks = landmarks[0]
        else:
            if (i==0) or (pre_landmarks is None):
                continue
            landmarks = pre_landmarks
        pre_landmarks = landmarks

        # Add elements to the queues
        q_frames.append(img)
        q_landmarks.append(landmarks)
        q_name.append(frame_name)
        # print(landmarks.shape)
        if len(q_frames) == args.window_margin:  # Wait until queues are large enough
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            # print(smoothed_landmarks.shape)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frames.popleft()
            cur_name = q_name.popleft()

            # Get aligned frame as well as affine transformation that produced it
            trans_frame, trans = warp_img(
                smoothed_landmarks[STABLE_POINTS, :], mean_face_landmarks[STABLE_POINTS, :], cur_frame, STD_SIZE
            )

            # Apply that affine transform to the landmarks
            trans_landmarks = trans(cur_landmarks)

            # Crop mouth region
            cropped_frame = cut_patch(
                trans_frame,
                trans_landmarks[args.start_idx : args.stop_idx],
                args.crop_height // 2,
                args.crop_width // 2,
            )

            # Save image
            target_path = os.path.join(target_dir, cur_name)
            Image.fromarray(cropped_frame.astype(np.uint8)).save(target_path)

    # Process remaining frames in the queue
    while q_frames:
        cur_frame = q_frames.popleft()
        cur_name = q_name.popleft()
        cur_landmarks = q_landmarks.popleft()

        trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
        trans_landmarks = trans(cur_landmarks)

        cropped_frame = cut_patch(
            trans_frame, trans_landmarks[args.start_idx : args.stop_idx], args.crop_height // 2, args.crop_width // 2
        )

        target_path = os.path.join(target_dir, cur_name)
        Image.fromarray(cropped_frame.astype(np.uint8)).save(target_path)

def main(args):

    print(args.folder.split('/')[-2].upper())
    video_path = glob.glob(args.folder + '/**/*.mp4', recursive=True)
    print("number video: ", len(video_path))
    for v in tqdm(video_path):
        args.video = v
        get_mouth_one_video(args)
       
def test_folders(args):

    print("Copping face each video of each method")
    all_datasets = ["Deepfakes-original","Deepfakes-fake"]
    for idx, data_name in enumerate(all_datasets):
        args.folder = f"/FF++/deepfakes/{data_name}/"
        main(args)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, metavar='S',  nargs='+', help='The folder of videos to test the model')
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--crop-width", default=96, type=int, help="Width of mouth ROIs")
    parser.add_argument("--crop-height", default=96, type=int, help="Height of mouth ROIs")
    parser.add_argument("--start-idx", default=48, type=int, help="Start of landmark index for mouth")
    parser.add_argument("--stop-idx", default=68, type=int, help="End of landmark index for mouth")
    parser.add_argument("--window-margin", default=12, type=int, help="Window margin for smoothed_landmarks")
    args = parser.parse_args()
    test_folders(args)
 