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
from types import SimpleNamespace

from sympy import solve
from sympy.abc import x, y

config_file = str(os.path.dirname(__file__)) + os.path.sep + 'default_generator_config.json'
with open(config_file, 'r') as f:
    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
random.seed(config.RANDOM_SEED)

class FixedGeneratorConfig():

    alpha = 0.
    beta = 0.
    gamma = 0.
    x_displacement = 0.5
    y_displacement = 0.5
    zoom = 1.0

    def __init__(self, alpha, beta, gamma, x_displacement, y_displacement, zoom):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.x_displacement = x_displacement
        self.y_displacement = y_displacement
        self.zoom = zoom


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
        for point in shape[config.POINTS_TAG]:
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
        shift_shape(shape[config.POINTS_TAG], vector)


def draw_shapes(image, shape_list: list):
    for shape in shape_list:
        integer_points = [[int(point[0]), int(point[1])] for point in shape[config.POINTS_TAG]]
        if shape[config.SHAPE_TYPE_TAG] == "circle":
            radius = math.dist(shape[config.POINTS_TAG][0], shape[config.POINTS_TAG][1])
            cv2.circle(image, integer_points[0], int(radius), (255, 0, 0), 2)
        elif shape[config.SHAPE_TYPE_TAG] == "polygon":
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
        if config.SHAPE_TAG in json_content:
            for shape in json_content[config.SHAPE_TAG]:
                if shape[config.LABEL_TAG] == config.METER_CLASS_NAME:
                    if shape[config.SHAPE_TYPE_TAG] != config.CIRCLE_SHAPE_TYPE:
                        continue
                    circle_points = shape[config.POINTS_TAG]
                    circle_bounding_rect = circle_to_rect(circle_points)
                    jpg_path = str(pathlib.Path(file_path).with_suffix(".jpg"))
                    meter_whole_image = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
                    clip_rect_by_image_shape(circle_bounding_rect, meter_whole_image.shape)
                    meter_image = meter_whole_image[circle_bounding_rect[0][1]:circle_bounding_rect[1][1],
                                                    circle_bounding_rect[0][0]:circle_bounding_rect[1][0]]

                    shapes = select_inside_circle(json_content[config.SHAPE_TAG], circle_points)
                    shift_shapes(shapes, circle_bounding_rect[0])
                    image_info_list.append((meter_image, circle_points, shapes))

def show_demo_window(meter_images, background_images, generator_config=config):
    DEMO_WINDOW_NAME = "Demo window"
    cv2.namedWindow(DEMO_WINDOW_NAME)
    controls = [attr for attr in dir(FixedGeneratorConfig)
                if not callable(getattr(FixedGeneratorConfig, attr)) and not attr.startswith("__")]
    def nothing(x):
        None

    for control in controls:
        cv2.createTrackbar(control, DEMO_WINDOW_NAME, 50, 100, nothing)

    def get_pos(bar_name, interval):
        return cv2.getTrackbarPos(bar_name, DEMO_WINDOW_NAME)/ 100.0 * (interval[1] - interval[0]) + interval[0]

    while True:
        fixed_config = FixedGeneratorConfig(
            get_pos("alpha", config.ALPHA_INTERVAL),
            get_pos("beta", config.BETA_INTERVAL),
            get_pos("gamma", config.GAMMA_INTERVAL),
            get_pos("x_displacement", config.X_INTERVAL),
            get_pos("y_displacement", config.Y_INTERVAL),
            get_pos("zoom", config.ZOOM_INTERVAL)
        )

        generated_image = generate_random_image(meter_images, backgrounds, generator_config, fixed_config=fixed_config)
        cv2.imshow(DEMO_WINDOW_NAME, generated_image)
        key = cv2.waitKey(40)
        if key == 27 or key == 20 or key == 13:
            break

def generate_random_image(meter_images, background_images, generator_config=config,
                          fixed_config : FixedGeneratorConfig = None):
    size = tuple(generator_config.GENERATOR_IMAGE_SIZE)
    background_source = background_images[0] #random.choice(background_images)
    background = cv2.resize(background_source, size, cv2.INTER_CUBIC)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    meter_image_source, circle, shapes = meter_images[0] #random.choice(meter_images)

    if fixed_config is None:
        alpha = random.uniform(config.ALPHA_INTERVAL[0], config.ALPHA_INTERVAL[1])
        beta = random.uniform(config.BETA_INTERVAL[0], config.BETA_INTERVAL[1])
        gamma = random.uniform(config.GAMMA_INTERVAL[0], config.GAMMA_INTERVAL[1])
        x_displacement = random.uniform(config.X_INTERVAL[0], config.X_INTERVAL[1]) * size[0]
        y_displacement = random.uniform(config.Y_INTERVAL[0], config.Y_INTERVAL[1]) * size[1]
        zoom_coefficient = random.uniform(config.ZOOM_INTERVAL[0], config.ZOOM_INTERVAL[1])
    else:
        alpha = fixed_config.alpha
        beta = fixed_config.beta
        gamma = fixed_config.gamma
        x_displacement = fixed_config.x_displacement * size[0]
        y_displacement = fixed_config.y_displacement * size[1]
        zoom_coefficient = fixed_config.zoom

    meter_image = cv2.cvtColor(meter_image_source, cv2.COLOR_BGR2BGRA)
    mask = np.zeros(meter_image_source.shape[:2], np.uint8)
    x_c = circle[0][0]
    y_c = circle[0][1]
    R = math.dist(circle[0], circle[1])
    cv2.circle(mask, (int(x_c), int(y_c)), int(R), 255, -1)
    meter_image[:, :, 3] = mask
    M_alpha = np.array([[ math.cos(alpha), math.sin(alpha), 0.],
                        [-math.sin(alpha), math.cos(alpha), 0.],
                        [ 0., 0., 1.]])
    M_beta = np.array([[ math.cos(beta), 0., math.sin(beta)],
                       [ 0.,             1., 0.],
                       [-math.sin(beta), 0., math.cos(beta)]])
    M_gamma = np.array([[ 1., 0., 0.],
                        [ 0., math.cos(gamma), math.sin(gamma)],
                        [ 0.,-math.sin(gamma), math.cos(gamma)]])
    M_rot = np.matmul(M_alpha, np.matmul(M_beta, M_gamma))

    M_2d_rot = M_rot[0:2, 0:2]
    M_2d_b   = M_rot[0:2, 2]
    M_inv    = np.linalg.inv(M_2d_rot)
    b = np.matmul(M_2d_rot, np.array([x_c, y_c])) - M_2d_b

    ellipse = (M_inv[0, 0] * x + M_inv[0, 1] * y) ** 2 + (M_inv[1, 0] * x + M_inv[1, 1] * y) ** 2 - R*R
    tangent_x = M_inv[0, 1] * (M_inv[0, 0] * x + M_inv[0, 1] * y) + M_inv[1, 1] * (M_inv[1, 0] * x + M_inv[1, 1] * y)
    tangent_points_x = solve([tangent_x, ellipse], dict=True)
    tangent_y = M_inv[0, 0] * (M_inv[0, 0] * x + M_inv[0, 1] * y) + M_inv[1, 0] * (M_inv[1, 0] * x + M_inv[1, 1] * y)
    tangent_points_y = solve([tangent_y, ellipse], dict=True)

    x_left   = np.array([float(tangent_points_x[0][x])+b[0], float(tangent_points_x[0][y])+b[1]])
    x_right  = np.array([float(tangent_points_x[1][x])+b[0], float(tangent_points_x[1][y])+b[1]])
    y_top    = np.array([float(tangent_points_y[0][x])+b[0], float(tangent_points_y[0][y])+b[1]])
    y_bottom = np.array([float(tangent_points_y[1][x])+b[0], float(tangent_points_y[1][y])+b[1]])

    meter_image_width = abs(x_right[0] - x_left[0]) + 1
    meter_image_height = abs(y_bottom[1] - y_top[1]) + 1
    x_c_new = (x_left[0] + x_right[0]) / 2
    y_c_new = (y_top[1] + y_bottom[1]) / 2
    background_size = min(size[0], size[1])
    meter_image_size = max(meter_image_width, meter_image_height)
    zoom = zoom_coefficient * background_size / meter_image_size
    x_shift = -x_c_new + x_displacement / zoom
    y_shift = -y_c_new + y_displacement / zoom

    M_rot[2, :] = [0., 0., 1.]
    M_shift_and_zoom = np.array([[1., 0., x_shift],
                                 [0., 1., y_shift],
                                 [0., 0., 1/zoom]])
    M = np.matmul(M_shift_and_zoom, M_rot)
    generated_image = cv2.warpPerspective(meter_image, M, size,
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0, 0))
    mask = generated_image[:, :, 3]
    mask = cv2.blur(mask, (10, 10))
    for i in range(3):
        generated_image[:, :, i] = ((np.multiply(background[:, :, i].astype(np.uint16), 255 - mask) +
                                     np.multiply(generated_image[:, :, i].astype(np.uint16), mask)) / 255
                                    ).astype(np.uint8)
    x_left = (x_left[0] - x_c_new) * zoom + x_displacement
    x_right = (x_right[0] - x_c_new) * zoom + x_displacement
    y_top = (y_top[1] - y_c_new) * zoom + y_displacement
    y_bottom = (y_bottom[1] - y_c_new) * zoom + y_displacement

    cv2.rectangle(generated_image, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (255, 0, 0), 3)

    # cv2.imshow("Background", background)
    # cv2.imshow("Generated image", generated_image)
    # cv2.imshow("Mask", mask)

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

    show_demo_window(meter_images, backgrounds)

