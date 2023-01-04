#!/usr/bin/env python
"""
Random water image generator
"""

import argparse
import os
import glob
# weired form of import for autocompletion
import cv2
import pathlib
import json
import math
import numpy as np
import random
from types import SimpleNamespace
import copy


config_file = str(os.path.dirname(__file__)) + os.path.sep + 'default_generator_config.json'
with open(config_file, 'r') as f:
    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
random.seed(config.RANDOM_SEED)


class FixedGeneratorConfig:
    """
    Class-structure with input parameters for function generate_random image

    Member variables describe water meter position in background
    """
    alpha = 0.  # rotation angle ("Oz axis")
    beta = 0.   # tilt with respect to Ox axis
    gamma = 0.  # tilt with respecto to Oy axis
    x_displacement = 0.5  # horizontal displacement of center in units of ratio of image width
    y_displacement = 0.5  # vertical displacement of center in units of ratio of image height
    zoom = 1.0  # zoom of imprinted to background water meter image

    def __init__(self, alpha, beta, gamma, x_displacement, y_displacement, zoom):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.x_displacement = x_displacement
        self.y_displacement = y_displacement
        self.zoom = zoom

class ImageWithAnnotation:
    """
    Container for image and its description
    """

    def __init__(self, meter_image, circle_points, shapes, water_meter_class):
        self.image = meter_image.copy()
        self.circle_points = circle_points.copy()
        self.shapes = copy.deepcopy(shapes)
        self.class_name = water_meter_class

    def set_meter_rectangle(self, x_left, x_right, y_top, y_bottom):
        for shape in self.shapes:
            if shape[config.LABEL_TAG] == config.METER_CLASS_NAME:
                shape[config.SHAPE_TYPE_TAG] = "rectangle"
                shape[config.POINTS_TAG] = [[x_left, y_top], [x_right, y_bottom]]


def get_file_list(root_path: str, glob_mask: str) -> list:
    """
    Scans folder recursively for files matching GLOB

    :param root_path: path to root folder to scan
    :param glob_mask: GLOB expression for files to search
    :return: list of matched files
    """
    file_list = []
    for path, dirs, files in os.walk(root_path):
        for file in glob.glob(path + os.sep + glob_mask):
            file_list.append(file)
    return file_list


def circle_to_rect(circle_points):
    """
    Transforms circle to its bounding box

    :param circle_points: list or tuple, first element is center of circle, second element is arbitrary point of circle
    :return: [left_top, right_bottom] list of points of bounding box frame
    """
    radius = math.dist(circle_points[0], circle_points[1])
    rect_point1 = [int(circle_points[0][0] - radius), int(circle_points[0][1] - radius)]
    rect_point2 = [int(circle_points[0][0] + radius), int(circle_points[0][1] + radius)]
    return [rect_point1, rect_point2]


def clip_rect_by_image_shape(rect, size):
    """
    Clips all rectangles to image size
    :param rect: list of rectangles
    :param size: [height, width] (or (height, width) of image
    """
    for point in rect:
        point[0] = np.clip(point[0], 0, size[1])
        point[1] = np.clip(point[1], 0, size[0])


def select_inside_circle(shapes_list: list, circle_points) -> list:
    """
    Selects only shapes inside circle
    :param shapes_list: list of lists of points of shapes
    :param circle_points: [center, arbitrary point of circle]
    :return: filtered list of shapes
    """
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
    """
    Shifts all points in list
    :param shape_points: list of shape points
    :param vector size: of displacement
    Procedure changes shape_points inplace
    """
    for point in shape_points:
        point[0] = point[0] - vector[0]
        point[1] = point[1] - vector[1]


def shift_shapes(shapes_list: list, vector):
    """
    Calls shift_shape for all of input shapes
    :param shapes_list: shapes container list
    :param vector: displacement
    Changes shapes_list inplace
    """
    for shape in shapes_list:
        shift_shape(shape[config.POINTS_TAG], vector)


def draw_shapes(image, shape_list: list):
    """
    Draws shapes in image
    :param image: cv2.Mat
    :param shape_list: shapes to draw
    """
    for shape in shape_list:
        integer_points = [[int(point[0]), int(point[1])] for point in shape[config.POINTS_TAG]]
        if shape[config.SHAPE_TYPE_TAG] == "circle":
            radius = math.dist(shape[config.POINTS_TAG][0], shape[config.POINTS_TAG][1])
            cv2.circle(image, integer_points[0], int(radius), (255, 0, 0), 2)
        elif shape[config.SHAPE_TYPE_TAG] == "polygon":
            for i in range(-1, len(integer_points) - 1):
                cv2.line(image, integer_points[i], integer_points[i + 1], (0, 255, 0), 2)
        elif shape[config.SHAPE_TYPE_TAG] == "rectangle":
            cv2.rectangle(image, integer_points[0], integer_points[1], (0, 0, 255), 2)

        cv2.putText(image, shape["label"], integer_points[-1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


def get_water_meter_class(root_path: str, file_path: str) -> str:
    """
    Returns class of water meter
    :param root_path: root dataset folder
    :param file_path: json path with labelme annotation
    :return: eigther first subfolder of dataset root path, or parent folder name of file_path, if root path
    is set directly to folder with json file
    """
    relative_path = os.path.relpath(file_path, root_path)
    path_parts = relative_path.split(os.sep)
    if len(path_parts) > 1:
        return path_parts[0]
    else:
        dirname = os.path.dirname(file_path)
        return os.path.basename(dirname)


def append_meters_image_and_info(root_path: str, file_path: str, image_info_list: list):
    """
    Parses labelme json file, search "meter" objects in it, saves information from json to dict

    :param root_path: root path of dataset, used for water meter class assignment (first subdirectory name)
    :param file_path: path to json file
    :param image_info_list: list with meter images
    """
    water_meter_class = get_water_meter_class(root_path, file_path)
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
                    image_info_list.append(ImageWithAnnotation(meter_image, circle_points, shapes, water_meter_class))


def collect_chars_map(meter_images: list) -> dict:
    """
    Collects character shapes in meter_images for quick random selection of specific characters in generator
    :param meter_images: water meter images with annotation
    :return: map from class and character label of water meter image to list of references to image content and shape
     descriptions
    """
    char_map = dict()
    for image in meter_images:
        for shape in image.shapes:
            label = shape[config.LABEL_TAG]
            if label not in [config.METER_CLASS_NAME, config.VALUE_CLASS_NAME] and '.' not in label:
                if image.class_name not in char_map.keys():
                    char_map[image.class_name] = dict()
                if label not in char_map[image.class_name].keys():
                    char_map[image.class_name][label] = list()
                char_map[image.class_name][label].append((image.image, shape))
    return char_map


def show_demo_window(meter_images: list, background_images: list, character_map: dict, generator_config=config):
    """
    Debug interactive function
    :param meter_images: images for generator random choice
    :param background_images: background images for generator random choice
    :param character_map: mapping of class and character labels to images and shape descriptions with polygons
    :param generator_config: configuration parameters of generator
    """
    DEMO_WINDOW_NAME = "Demo window"
    cv2.namedWindow(DEMO_WINDOW_NAME)
    controls = [attr for attr in dir(FixedGeneratorConfig)
                if not callable(getattr(FixedGeneratorConfig, attr)) and not attr.startswith("__")]

    for control in controls:
        cv2.createTrackbar(control, DEMO_WINDOW_NAME, 50, 100, lambda x: None)

    def get_pos(bar_name, interval):
        """
        Transforms value of trackbar to corespondent value of FixedGeneratorConfig
        :param bar_name: name of trackbar control
        :param interval: [min, max] for conversion of trackbar value which is within [0,100] interval
        :return: value within [min, max] interval
        """
        return cv2.getTrackbarPos(bar_name, DEMO_WINDOW_NAME) / 100.0 * (interval[1] - interval[0]) + interval[0]

    while True:
        fixed_config = FixedGeneratorConfig(
            get_pos("alpha", config.ALPHA_INTERVAL),
            get_pos("beta", config.BETA_INTERVAL),
            get_pos("gamma", config.GAMMA_INTERVAL),
            get_pos("x_displacement", config.X_INTERVAL),
            get_pos("y_displacement", config.Y_INTERVAL),
            get_pos("zoom", config.ZOOM_INTERVAL)
        )
        generated_image = generate_random_image(meter_images, background_images, character_map,
                                                generator_config, fixed_config=fixed_config)
        draw_shapes(generated_image.image, generated_image.shapes)
        cv2.imshow(DEMO_WINDOW_NAME, generated_image.image)
        key = cv2.waitKey(40)
        if key == 27 or key == 20 or key == 13:
            break


def find_tangent_points(M_quad, R, M, b):
    """
    Finds points of ellipse for calculating its bounding box

    Slow sympy solution
    :param M_quad: matrix of quadratic form function (r^T M_quad r - R^2 = 0 is ellipse equation)
    :param R: radius of circle, transformed to ellipse with M matrix
    :param M: transformation matrix, M_quad=M^T M
    :param b: shift due to transformation and circle position
    :return: list with left, right, top and bottom points of ellipse
    """
    from sympy import solve
    from sympy.abc import x, y
    ellipse = M_quad[0, 0] * x ** 2 + (M_quad[0, 1] + M_quad[1, 0]) * x * y + M_quad[1, 1] * y ** 2 - R*R
    tangent_x = M[0, 1] * (M[0, 0] * x + M[0, 1] * y) + M[1, 1] * (M[1, 0] * x + M[1, 1] * y)
    tangent_points_x = solve([tangent_x, ellipse], dict=True)
    tangent_y = M[0, 0] * (M[0, 0] * x + M[0, 1] * y) + M[1, 0] * (M[1, 0] * x + M[1, 1] * y)
    tangent_points_y = solve([tangent_y, ellipse], dict=True)
    x_left   = np.array([float(tangent_points_x[0][x])+b[0], float(tangent_points_x[0][y])+b[1]])
    x_right  = np.array([float(tangent_points_x[1][x])+b[0], float(tangent_points_x[1][y])+b[1]])
    y_top    = np.array([float(tangent_points_y[0][x])+b[0], float(tangent_points_y[0][y])+b[1]])
    y_bottom = np.array([float(tangent_points_y[1][x])+b[0], float(tangent_points_y[1][y])+b[1]])
    return x_left, x_right, y_top, y_bottom


def solve_equations(M_quad, R, D, E):
    """
    Solves system of two linear and quadratic equations

    r = [x, y]^T
    r^T M_quad r - R^2 = 0
    r^T * [D, E] = 0

    :param M_quad: matrix of quadratic form function
    :param R: radius of transformed circle
    :param D: parameter of linear equation, D*x + E*y=0
    :param E: parameter of linear equation, D*x + E*y=0
    :return: list of solutions [x,y]
    """
    A = M_quad[0, 0]
    B = M_quad[0, 1] + M_quad[1, 0]
    C = M_quad[1, 1]
    if abs(D) > abs(E):
        swap_flag = True
        D, E = E, D
        A, C = C, A
    else:
        swap_flag = False
    # y = - D/E x
    A = A - D * B / E + D * D / E / E * C
    C = - R * R
    if -C/A < 0:
        raise ValueError("Equations have no roots")
    x = math.sqrt(-C/A)
    y = - D / E * x
    if swap_flag:
        x, y = y, x
    solution = np.array([x, y])
    return -solution, solution


def find_tangent_points_honest(M_quad, R, M, b):
    """
    Finds points of ellipse for calculating its bounding box

    Direct fast solution
    :param M_quad: matrix of quadratic form function (r^T M_quad r - R^2 = 0 is ellipse equation)
    :param R: radius of circle, transformed to ellipse with M matrix
    :param M: transformation matrix, M_quad=M^T M
    :param b: shift due to transformation and circle position
    :return: list with left, right, top and bottom points of ellipse
    """
    D = M[0, 1] * M[0, 0] + M[1, 1] * M[1, 0]
    E = M[0, 1] * M[0, 1] + M[1, 1] * M[1, 1]
    x_left, x_right = solve_equations(M_quad, R, D, E)
    D = M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0]
    E = M[0, 0] * M[0, 1] + M[1, 0] * M[1, 1]
    y_top, y_bottom = solve_equations(M_quad, R, D, E)
    return x_left + b, x_right + b, y_top + b, y_bottom + b


def calculate_transform_parameters(alpha, beta, gamma, x_displacement, y_displacement, zoom_coefficient,
                                   x_c, y_c, R, size):

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
    M_quad   = np.matmul(M_inv.transpose(), M_inv)
    b = np.matmul(M_2d_rot, np.array([x_c, y_c])) - M_2d_b
    x_left, x_right, y_top, y_bottom = find_tangent_points_honest(M_quad, R, M_inv, b)

    meter_image_width = abs(x_left[0] - x_right[0]) + 1
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
    x_left = (x_left[0] - x_c_new) * zoom + x_displacement
    x_right = (x_right[0] - x_c_new) * zoom + x_displacement
    y_top = (y_top[1] - y_c_new) * zoom + y_displacement
    y_bottom = (y_bottom[1] - y_c_new) * zoom + y_displacement
    return M, x_left, x_right, y_top, y_bottom


def generate_random_image(meter_images: list, background_images: list, character_map: dict,
                          generator_config: dict = config, fixed_config: FixedGeneratorConfig = None):
    """
    Generates random image with water meter
    :param meter_images: list of woter meter images with annotations for random selection to imprint
    :param background_images: list of backgrounds for random choice
    :param character_map: mapping of class and character labels to images and shape descriptions with polygons
    :param generator_config: various parameters with limits for random selection of rotations, displacement, zoom and
    other options of imprint process
    :param fixed_config: FixedGeneratorConfig class, overrides random selection if set
    :return: image with randomly selected background and several randomly placed water meters, and its annotation
    """
    size = tuple(generator_config.GENERATOR_IMAGE_SIZE)
    background_source = random.choice(background_images)
    background = cv2.resize(background_source, size, cv2.INTER_CUBIC)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    image = random.choice(meter_images)
    meter_image_source, circle, shapes = image.image, image.circle_points, image.shapes

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
    M, x_left, x_right, y_top, y_bottom\
        = calculate_transform_parameters(alpha, beta, gamma, x_displacement, y_displacement, zoom_coefficient,
                                         x_c, y_c, R, size)
    generated_image = cv2.warpPerspective(meter_image, M, size,
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0, 0))
    mask = generated_image[:, :, 3]
    mask = cv2.blur(mask, (10, 10))
    for i in range(3):
        generated_image[:, :, i] = ((np.multiply(background[:, :, i].astype(np.uint16), 255 - mask) +
                                     np.multiply(generated_image[:, :, i].astype(np.uint16), mask)) / 255
                                    ).astype(np.uint8)
    generated_image_with_annotations = ImageWithAnnotation(generated_image, circle, shapes, image.class_name)
    #from cv2 import cv2
    for shape in generated_image_with_annotations.shapes:
        points_in = np.array(shape[config.POINTS_TAG]).reshape(-1, 1, 2)
        points_out = cv2.perspectiveTransform(points_in, M)
        shape[config.POINTS_TAG] = points_out.reshape(-1, 2).tolist()
    generated_image_with_annotations.set_meter_rectangle(x_left, x_right, y_top, y_bottom)

    return generated_image_with_annotations


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
    for file in background_files:
        bg_image = cv2.imread(file, cv2.IMREAD_COLOR)
        caption = "Background " + pathlib.Path(file).name
        backgrounds.append(bg_image)

    for file in dataset_jsons:
        append_meters_image_and_info(args.dataset_path, file, meter_images)

    characters_map = collect_chars_map(meter_images)

    show_demo_window(meter_images, backgrounds, characters_map)

