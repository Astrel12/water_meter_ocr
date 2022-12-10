#!/usr/bin/env python
"""
Random water image generator
"""

import argparse
import os
import glob
import cv2
import numpy as np


def get_file_list(root_path: str, glob_mask: str) -> list:
    file_list = []
    for path, dirs, files in os.walk(root_path):
        for file in glob.glob(path + os.sep + glob_mask):
            file_list.append(file)
    return file_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes images from dateset folder, '
                                                 'extracts water meter images from them '
                                                 'using given annotations, places it randomly on '
                                                 'background image, and, finally, stores result into output folder.')
    parser.add_argument('--backgrounds_path', metavar='<path_to_backgrounds>', type=str,
                        help='Path to backgrounds folder', required=True)
    parser.add_argument('--dataset_path', metavar='<path_to_dataset>', type=str,
                        help='Path to dataset folder', required=True)
    parser.add_argument('--output_path', metavar='<output_path>', type=str,
                        help='Path to output folder', required=True)
    args = parser.parse_args()

    print(f"Dataset folder: {args.dataset_path}\n"
          f"Background folder: {args.backgrounds_path}\n"
          f"Output folder: {args.output_path}")

    dataset_jsons = get_file_list(args.dataset_path, "*.json")
    background_files = get_file_list(args.backgrounds_path, "*.jpg")

    if len(background_files) != 0:
        bg_image=cv2.imread(background_files[0], cv2.IMREAD_COLOR)
        bg_image_slice = bg_image[10:100, 10:500, :]
        bg_image_slice[:] = (0, 0, 255)
        bg_image[310:400, 10:500, :] = bg_image_slice
        cv2.imshow("Background", bg_image)
        cv2.imshow("Background slice", bg_image_slice)
        cv2.waitKey(0)

    print("Dataset files: ", dataset_jsons)
    print("Background files: ", background_files)
