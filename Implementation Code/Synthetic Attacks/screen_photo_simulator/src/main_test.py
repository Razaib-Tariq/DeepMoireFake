import cv2
import numpy as np
import os

from image_tools import *
from moire import linear_wave, dither
from basic_shapes import circles
from module import RecaptureModule

import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # File I/O
    parser.add_argument("--datapath", default='/media/',
                    help="Path to the directory containing the source images.")
    parser.add_argument("--savepath", default='/media/NAS/DATASET/Moire-Pattern/Synthetic_Moire/screen_photo_simulator/',
                        help="Path to the output storage directory (automatically generated if not yet there).")
    parser.add_argument("--save-format", type=str, default='png',
                        help="File format of the output file (default: PNG).")

    # Image-related
    parser.add_argument("--canvas-dim", type=int, nargs='+', default=1024,
                        help="Dimensions (height, width) of the canvas to use. Provide a single value to produce a square canvas.")
    parser.add_argument('-e', "--empty", action='store_true',
                        help="Create a white blank canvas, instead of using an image")
    parser.add_argument('-g', "--gamma", type=float, default=1,
                        help="Do gamma correction on the given input (default: 1 => no correction)")
    parser.add_argument('-t', "--type", type=str, default='fixed',
                        help="Type of pattern to generate.")
    parser.add_argument('-rv', "--recapture-verbose", action='store_true',
                        help="Print the log of progress produced as RecaptureModule transforms the input image.")
    parser.add_argument("--psnr", action='store_true',
                        help="Compute the PSNR value of the output image.")

    # Others
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed value for 'np.random.seed'.")
    parser.add_argument('-m', "--show-mask", action='store_true',
                        help="Visualize the inserted nonlinear moire mask.")

    return parser

def process_image(image_path, save_path, args):
    canvas = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if canvas is None:
        print(f"Error loading image: {image_path}")
        return
    original = canvas.copy()
    H, W, _ = canvas.shape

    # Setup RecaptureModule
    dst_H, dst_W, _ = original.shape
    src_pt = np.array([[W // 4, H // 4],
                       [W // 4 * 3, H // 4],
                       [W // 4 * 3, H // 4 * 3],
                       [W // 4, H // 4 * 3]], dtype="float32")
    recap_module = RecaptureModule(dst_H, dst_W,
                                   v_moire=0, v_type='sg', v_skew=[20, 80], v_cont=10, v_dev=3,
                                   h_moire=0, h_type='f', h_skew=[20, 80], h_cont=10, h_dev=3,
                                   nl_moire=True, nl_dir='b', nl_type='sine', nl_skew=0,
                                   nl_cont=10, nl_dev=3, nl_tb=0.15, nl_lr=0.15,
                                   gamma=args.gamma, margins=None, seed=args.seed)
    
    result = recap_module(canvas,
                          new_src_pt=src_pt,
                          verbose=args.recapture_verbose,
                          show_mask=args.show_mask)

    # Check how many values are returned
    if isinstance(result, tuple):
        if len(result) == 2:
            canvas, nl_mask = result
        elif len(result) == 3:
            canvas, nl_mask, _ = result  # If there's an extra value, ignore it
        else:
            raise ValueError("Unexpected number of values returned by recap_module")
    else:
        canvas = result  # If only one value is returned

    # Save output
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(save_path, canvas)
    print(f"Processed and saved: {save_path}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Get the last part of the datapath to create a similar folder structure in savepath
    base_folder_name = os.path.basename(os.path.normpath(args.datapath))
    
    # Prepare the full save path base
    save_base_path = os.path.join(args.savepath, base_folder_name)

    # Walk through the directory and process each image
    for root, _, files in os.walk(args.datapath):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(image_path, args.datapath)
                
                # Prepend the base_folder_name to the savepath structure
                save_path = os.path.join(save_base_path, relative_path)
                save_path = os.path.splitext(save_path)[0] + '.' + args.save_format
                process_image(image_path, save_path, args)

if __name__ == "__main__":
    main()
