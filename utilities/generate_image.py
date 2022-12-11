#!/usr/bin/env python
"""
Random water image generator
"""

import argparse
import os
import glob
import cv2
import pathlib
import json
import math
import numpy as np
import random

METER_CLASS_NAME = "meter"
SHAPE_TAG = "shapes"
LABEL_TAG = "label"
POINTS_TAG = "points"
SHAPE_TYPE_TAG = "shape_type"
CIRCLE_SHAPE_TYPE = "circle"
RANDOM_SEED = 13
GENERATOR_IMAGE_SIZE = (1280, 720)
random.seed(13)


def get_file_list(root_path: str, glob_mask: str) -> list:
    file_list = []
    for path, dirs, files in os.walk(root_path):
        for file in glob.glob(path + os.sep + glob_mask):
            file_list.append(file)
    return file_list


def circle_to_rect(circle_points):
    radius = math.dist(circle_points[0], circle_points[1])
    rect_point1 = [int(circle_points[0][0] - radius), int(circle_points[0][1] - radius)]
    rect_point2 = [int(circle_points[0][0] + radius), int(circle_points[0][1] + radius)]
    return [rect_point1, rect_point2]


def clip_rect_by_image_shape(rect, size):
    for point in rect:
        point[0] = np.clip(point[0], 0, size[1])
        point[1] = np.clip(point[1], 0, size[0])


def select_inside_circle(shapes_list: list, circle_points):
    radius = math.dist(circle_points[0], circle_points[1])
    filtered_shapes = []
    for shape in shapes_list:
        inside_flag = True
        for point in shape[POINTS_TAG]:
            if math.dist(point, circle_points[0]) > radius:
                inside_flag = False
                break
        if inside_flag:
            filtered_shapes.append(shape)
    return filtered_shapes


def shift_shape(shape_points: list, vector):
    for point in shape_points:
        point[0] = point[0] - vector[0]
        point[1] = point[1] - vector[1]


def shift_shapes(shapes_list: list, vector):
    for shape in shapes_list:
        shift_shape(shape[POINTS_TAG], vector)


def draw_shapes(image, shape_list: list):
    for shape in shape_list:
        integer_points = [[int(point[0]), int(point[1])] for point in shape[POINTS_TAG]]
        if shape[SHAPE_TYPE_TAG] == "circle":
            radius = math.dist(shape[POINTS_TAG][0], shape[POINTS_TAG][1])
            cv2.circle(image, integer_points[0], int(radius), (255, 0, 0), 2)
        elif shape[SHAPE_TYPE_TAG] == "polygon":
            for i in range(-1, len(integer_points) - 1):
                cv2.line(image, integer_points[i], integer_points[i + 1], (0, 255, 0), 2)

        cv2.putText(image, shape["label"], integer_points[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


def append_meters_image_and_info(file_path: str, image_info_list: list):
    """
    Parses labelme json file, search "meter" objects in it, saves information from json to dict

    :param file_path: path to json file
    :param image_info_list: list with meter images
    """

    with open(file_path) as f:
        json_content = json.load(f)
        if SHAPE_TAG in json_content:
            for shape in json_content[SHAPE_TAG]:
                if shape[LABEL_TAG] == METER_CLASS_NAME:
                    if shape[SHAPE_TYPE_TAG] != CIRCLE_SHAPE_TYPE:
                        continue
                    circle_points = shape[POINTS_TAG]
                    circle_bounding_rect = circle_to_rect(circle_points)
                    jpg_path = str(pathlib.Path(file_path).with_suffix(".jpg"))
                    meter_whole_image = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
                    clip_rect_by_image_shape(circle_bounding_rect, meter_whole_image.shape)
                    meter_image = meter_whole_image[circle_bounding_rect[0][1]:circle_bounding_rect[1][1],
                                                    circle_bounding_rect[0][0]:circle_bounding_rect[1][0]]

                    shapes = select_inside_circle(json_content[SHAPE_TAG], circle_points)
                    shift_shapes(shapes, circle_bounding_rect[0])
                    image_info_list.append((meter_image, circle_points, shapes))

                    # draw_shapes(meter_image, shapes)
                    # cv2.imshow("Test", meter_image)
                    # cv2.waitKey(0)


def generate_random_image(meter_images, background_images, size=GENERATOR_IMAGE_SIZE):
    background_source = random.choice(background_images)
    background = cv2.resize(background_source, size, cv2.INTER_CUBIC)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    meter_image_source, circle, shapes = random.choice(meter_images)

    meter_image = cv2.cvtColor(meter_image_source, cv2.COLOR_BGR2BGRA)
    mask = np.zeros(meter_image_source.shape[:2], np.uint8)
    cv2.circle(mask, (int(circle[0][0]), int(circle[0][1])), int(math.dist(circle[0], circle[1])), 255, -1)
    meter_image[:, :, 3] = mask
    M = np.array([[1., 0., 200.], [0., 1., 200.], [0, 0, 1.]])
    generated_image = cv2.warpPerspective(meter_image, M, size,
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0, 0))
    mask = generated_image[:, :, 3]
    mask = cv2.blur(mask, (10, 10))
    for i in range(3):
        generated_image[:, :, i] = ((np.multiply(background[:, :, i].astype(np.uint16), 255 - mask) +
                                     np.multiply(generated_image[:, :, i].astype(np.uint16), mask)) / 255
                                    ).astype(np.uint8)

    cv2.imshow("Background", background)
    cv2.imshow("Generated image", generated_image)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)

    return generated_image


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
    dataset_jsons.sort()
    background_files = get_file_list(args.backgrounds_path, "*.jpg")
    background_files.sort()
    print("Dataset files: ", dataset_jsons)
    print("Background files: ", background_files)

    backgrounds = []
    meter_images = []

    background_files = [random.choice(background_files)]
    dataset_jsons = [random.choice(dataset_jsons)]

    for file in background_files:
        bg_image = cv2.imread(file, cv2.IMREAD_COLOR)
        caption = "Background " + pathlib.Path(file).name
        backgrounds.append(bg_image)

    for file in dataset_jsons:
        append_meters_image_and_info(file, meter_images)

    generated_image = generate_random_image(meter_images, backgrounds)
