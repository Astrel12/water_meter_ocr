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
import copy
from tqdm import tqdm


def read_config_file(config_file=str(os.path.dirname(__file__)) + os.path.sep + 'default_generator_config.json'):
    """
    Function reads config from json file (should be placed in module folder) to structure with correspondent members
    """

    with open(config_file, 'r') as f:
        config_content = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    if hasattr(config_content, "RANDOM_SEED"):
        random.seed(config_content.RANDOM_SEED)
    return config_content


config = read_config_file()  # called both in __main__ and in __init__


class FixedGeneratorConfig:
    """
    Class-structure with input parameters for function generate_random image

    Member variables describe water meter position in background
    """
    alpha = 0.  # rotation angle ("Oz axis")
    beta = 0.   # tilt with respect to Ox axis
    gamma = 0.  # tilt with respect to Oy axis
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

    def __init__(self, meter_image, circle_points, shapes, water_meter_class, json_file):
        self.image = meter_image.copy()
        self.circle_points = circle_points.copy()
        self.shapes = copy.deepcopy(shapes)
        self.class_name = water_meter_class
        self.json_file = json_file

    def set_meter_rectangle(self, x_left, x_right, y_top, y_bottom):
        for shape in self.shapes:
            if shape[config.LABEL_TAG] == config.METER_CLASS_NAME:
                shape[config.SHAPE_TYPE_TAG] = "rectangle"
                shape[config.LABEL_TAG] = self.class_name
                shape[config.POINTS_TAG] = [[x_left, y_top], [x_right, y_bottom]]


def get_file_list(root_path: str, glob_mask: str) -> list:
    """
    Scans folder recursively for files matching GLOB

    :param root_path: path to root folder for recursive scanning
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

    :param circle_points: list or tuples, first element is the center of circle, second element is the arbitrary
    point of circle
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
    :param vector: size of displacement
    Procedure changes shape_points inplace
    """
    for point in shape_points:
        point[0] = point[0] - vector[0]
        point[1] = point[1] - vector[1]


def shift_shapes(shapes_list: list, vector):
    """
    Calls shift_shape for all input shapes
    :param shapes_list: shapes container list
    :param vector: displacement
    Changes shapes_list inplace
    """
    for shape in shapes_list:
        shift_shape(shape[config.POINTS_TAG], vector)


def put_text_up_right(image, text, point, font, scale, color, thickness):
    """
    Same as putText, but base point is up right text position
    """
    size = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(image, text, (point[0] - size[0][0], point[1] + size[0][1] +
                              size[1]), font, scale, color, thickness)


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
            color = (255, 0, 0)
            cv2.circle(image, integer_points[0], int(radius), color, 2)
        elif shape[config.SHAPE_TYPE_TAG] == "polygon":
            color = (0, 255, 0)
            for i in range(-1, len(integer_points) - 1):
                cv2.line(image, integer_points[i], integer_points[i + 1], color, 2)
        elif shape[config.SHAPE_TYPE_TAG] == "rectangle":
            color = (0, 0, 255)
            cv2.rectangle(image, integer_points[0], integer_points[1], color, 2)
        label = shape[config.LABEL_TAG]
        if label != config.VALUE_CLASS_NAME:
            put_text_up_right(image, label, integer_points[-1], cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.circle(image, integer_points[-1], 4, (0, 0, 0), -1)


def get_water_meter_class(root_path: str, file_path: str) -> str:
    """
    Returns class of water meter
    :param root_path: root dataset folder
    :param file_path: json path with labelme annotation
    :return: either first subfolder of dataset root path, or parent folder name of file_path, if root path
    is set directly to folder with json file
    """
    if not config.ADD_METER_SUBCLASSES:
        return config.METER_CLASS_NAME
    relative_path = os.path.relpath(file_path, root_path)
    path_parts = relative_path.split(os.sep)
    if len(path_parts) > 1:
        return path_parts[0]
    else:
        dir_name = os.path.dirname(file_path)
        return os.path.basename(dir_name)


def distance_to_line(line: np.array, point):
    return np.linalg.norm(np.cross(line[1] - line[0], line[0] - point))/np.linalg.norm(line[1] - line[0])


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise ValueError('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]


def normalize_annotation(image: ImageWithAnnotation):
    """
    Corrects some of user errors in annotation: angles inaccuracies, rectangles instead of polygons, polygons start points
    :param image: container with single water meter image
    """
    value_shape = None

    def rect_to_polygon(shape: dict):
        """
        Changes labelme shape type from rectangle (2 points) to polygon (4 points)

        Order of points: left top, right top, right bottom, left bottom (top based coordinates)
        :param shape: input shape to convert
        """
        if len(shape[config.POINTS_TAG]) != 4:
            print(f'Waring: "{shape[config.LABEL_TAG]}" frame is {shape[config.SHAPE_TYPE_TAG]} with'
                  f' {len(shape[config.POINTS_TAG])} points (annotation file {image.json_file})')
            if shape[config.SHAPE_TYPE_TAG] == "rectangle":
                shape[config.SHAPE_TYPE_TAG] = "polygon"
                left_top = [min(shape[config.POINTS_TAG][0][0], shape[config.POINTS_TAG][1][0]),
                            min(shape[config.POINTS_TAG][0][1], shape[config.POINTS_TAG][1][1])]
                right_bottom = [max(shape[config.POINTS_TAG][0][0], shape[config.POINTS_TAG][1][0]),
                                max(shape[config.POINTS_TAG][0][1], shape[config.POINTS_TAG][1][1])]
                points = [left_top, [right_bottom[0], left_top[1]],
                          right_bottom, [left_top[0], right_bottom[1]]]
                shape[config.POINTS_TAG] = points

    for shape in image.shapes:
        if shape[config.LABEL_TAG] != config.METER_CLASS_NAME:
            rect_to_polygon(shape)
        if shape[config.LABEL_TAG] == config.VALUE_CLASS_NAME:
            if value_shape is not None:
                print(f'Warning: more than one "{config.VALUE_CLASS_NAME}" class in annotation file {image.json_file}')
                continue
            value_shape = shape
    if value_shape is None:
        print(f'Warning: there is no "{config.VALUE_CLASS_NAME}" class in annotation file {image.json_file}')
        return

    def reorder_along_line(shape: dict, line):
        """
        Changes order of points in quadrangle (4 point polygon) so that it is clockwise, and starts from closest to
        input line edge
        :param shape: input shape to transform
        :param line: line in form of two points iterable
        """
        points_num = len(shape[config.POINTS_TAG])
        if points_num != 4:
            print(f'Warning: still {len(shape[config.POINTS_TAG])} points in "{shape[config.SHAPE_TYPE_TAG]}" '
                  f'(annotation: {image.json_file})')
            return

        def get_poly_edge(index: int):
            index = index % points_num
            index2 = (index + 1) % points_num
            return [shape[config.POINTS_TAG][index], shape[config.POINTS_TAG][index2]]

        line = np.array(line)
        first_edge = np.array(get_poly_edge(0))
        second_edge = np.array(get_poly_edge(1))

        cross = np.cross(first_edge[1] - first_edge[0], second_edge[1] - second_edge[0])
        if cross < 0:
            print(f'Warning: points order is not clockwise in "{shape[config.LABEL_TAG]}" '
                  f'(annotation: {image.json_file})')
            shape[config.POINTS_TAG].reverse()
            reversed_points = shape[config.POINTS_TAG]
            shape[config.POINTS_TAG] = reversed_points[3:] + reversed_points[:3]

        first_edge = np.array(get_poly_edge(0))
        second_edge = np.array(get_poly_edge(1))

        def projection(line_dir: np.array, vector: np.array):
            return np.dot(line_dir[1] - line_dir[0], vector[1] - vector[0])

        if abs(projection(line, first_edge)) > abs(projection(line, second_edge)):
            start_index = 0
        else:
            start_index = 1
            first_edge = second_edge
        second_edge = get_poly_edge(start_index + 2)

        distance_to_line1 = distance_to_line(line, first_edge[0]) + distance_to_line(line, first_edge[1])
        distance_to_line2 = distance_to_line(line, second_edge[0]) + distance_to_line(line, second_edge[1])
        if distance_to_line1 > distance_to_line2:
            start_index = start_index + 2
        if start_index != 0:
            print(f'Warning: reordering polygon "{shape[config.LABEL_TAG]}" (annotation {image.json_file})"')
            shape[config.POINTS_TAG] = shape[config.POINTS_TAG][start_index:] + shape[config.POINTS_TAG][:start_index]

    reorder_along_line(value_shape, [[0., 0.], [1., 0.]])
    base_line = np.array([value_shape[config.POINTS_TAG][0], value_shape[config.POINTS_TAG][1]])
    base_line_down = np.array([value_shape[config.POINTS_TAG][3], value_shape[config.POINTS_TAG][2]])
    for shape in image.shapes:
        if shape[config.LABEL_TAG] in [config.VALUE_CLASS_NAME, config.METER_CLASS_NAME] \
                or len(shape[config.POINTS_TAG]) != 4:
            continue
        reorder_along_line(shape, base_line)
        if distance_to_line(base_line_down, shape[config.POINTS_TAG][2]) > config.BASE_LINE_DISTANCE_THRESHOLD:
            shape[config.POINTS_TAG][2] = line_intersection(base_line_down,
                                                            [shape[config.POINTS_TAG][1], shape[config.POINTS_TAG][2]])
        if distance_to_line(base_line_down, shape[config.POINTS_TAG][3]) > config.BASE_LINE_DISTANCE_THRESHOLD:
            shape[config.POINTS_TAG][3] = line_intersection(base_line_down,
                                                            [shape[config.POINTS_TAG][3], shape[config.POINTS_TAG][0]])


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
                    circle_points = shape[config.POINTS_TAG].copy()
                    circle_bounding_rect = circle_to_rect(circle_points)
                    jpg_path = str(pathlib.Path(file_path).with_suffix(".jpg"))
                    meter_whole_image = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
                    clip_rect_by_image_shape(circle_bounding_rect, meter_whole_image.shape)
                    meter_image = meter_whole_image[circle_bounding_rect[0][1]:circle_bounding_rect[1][1],
                                                    circle_bounding_rect[0][0]:circle_bounding_rect[1][0]]

                    shapes = select_inside_circle(json_content[config.SHAPE_TAG], circle_points)
                    shift_shape(circle_points, circle_bounding_rect[0])
                    image_with_info = ImageWithAnnotation(meter_image, circle_points, shapes,
                                                          water_meter_class, file_path)
                    shift_shapes(image_with_info.shapes, circle_bounding_rect[0])
                    normalize_annotation(image_with_info)
                    image_info_list.append(image_with_info)


def is_shape_digit(label):
    """
    Checks if shape label is digit (character)
    :param label: shape label
    :return: True if label is not "meter" or "value" (or may be something else in future)
    """
    return label not in [config.METER_CLASS_NAME, config.VALUE_CLASS_NAME]


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
            if is_shape_digit(label) and '.' not in label:
                if image.class_name not in char_map.keys():
                    char_map[image.class_name] = dict()
                if not config.USE_R_DIGIT_PREFIX:
                    if label.startswith("r"):
                        label = label[1:]
                        shape[config.LABEL_TAG] = label
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
        key = cv2.waitKey(20)
        if key == 27 or key == 32 or key == 13:
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
    """
    Calculates perspective transform matrix for water meter image projection
    :param alpha: rotation angle (correspond Oz axis)
    :param beta: tilt angle (correspond Oy axis)
    :param gamma: tilt angle (correspond Oz axis)
    :param x_displacement: fraction of image width, where to place new center of water meter image
    :param y_displacement: fraction of image height, where to place new center of water meter image
    :param x_c: x coordinate of source circle
    :param y_c: y coordinate of source circle
    :param R: radius of source circle
    :param size: destination image size
    :return: tuple: (Transformation matrix, 4 new bounding box coordinates)
    """
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
    return M, (x_left, x_right, y_top, y_bottom)


def imprint_random_digit(image, meter_class, shape, character_map):
    """
    Places random digit image from character_map to the place described by shape
    :param image: image to modify
    :param shape: description of position where to imprint random digit (shape description will be changed too)
    :param meter_class: class of water meter, to select proper variants from character_map
    :param character_map:  mapping of class and character labels to images and shape descriptions with polygons
    """
    if meter_class not in character_map.keys():
        return
    label = shape[config.LABEL_TAG]
    if label.startswith('r'):
        digits_to_select = [l for l in character_map[meter_class].keys() if l.startswith('r')]
    else:
        digits_to_select = [l for l in character_map[meter_class].keys() if not l.startswith('r')]
    if len(digits_to_select) == 0:
        return
    source_label = random.choice(digits_to_select)
    source_image, source_shape = random.choice(character_map[meter_class][source_label])
    points_source = np.array(source_shape[config.POINTS_TAG])
    points_dest = np.array(shape[config.POINTS_TAG])
    if points_source.shape[0] != 4 or points_dest.shape[0] != 4:
        return  # incorrect polygons in annotation
    size = np.array([image.shape[1], image.shape[0]])
    digit_min_bounds = points_dest.min(axis=0)
    digit_max_bounds = points_dest.max(axis=0)
    crop_image_min = np.clip(digit_min_bounds - config.DIGIT_IMPRINT_SMOOTH_RADIUS, [0, 0], size).astype(int)
    crop_image_max = np.clip(digit_max_bounds + config.DIGIT_IMPRINT_SMOOTH_RADIUS, [0, 0], size).astype(int)
    if crop_image_min[0] == crop_image_max[0] or crop_image_min[1] == crop_image_max[1]:
        return  # digit is out of image
    sub_image = image[crop_image_min[1]:crop_image_max[1], crop_image_min[0]:crop_image_max[0], :]
    temp_size = (sub_image.shape[1], sub_image.shape[0])
    points_dest = points_dest - np.array([crop_image_min[0], crop_image_min[1]])
    M = cv2.getPerspectiveTransform(points_source.astype("float32"), points_dest.astype("float32"))
    sub_image_temp = cv2.warpPerspective(source_image, M, temp_size,
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0, 0))
    mask = np.zeros((sub_image.shape[0], sub_image.shape[1], 1), dtype=np.uint8)
    cv2.fillPoly(mask, points_dest.astype(int).reshape(1, -1, 2), 255)
    mask = cv2.blur(mask, (config.DIGIT_IMPRINT_SMOOTH_RADIUS // 2, config.DIGIT_IMPRINT_SMOOTH_RADIUS // 2))
    for i in range(3):
        sub_image[:, :, i] = ((np.multiply(sub_image[:, :, i].astype(np.uint16), 255 - mask) +
                               np.multiply(sub_image_temp[:, :, i].astype(np.uint16), mask)) // 255
                              ).astype(np.uint8)
    shape[config.LABEL_TAG] = source_label


def digit_shape_inside(shape: list, size) -> bool:
    """
    Check if shape with digit label is inside image with input size
    :param shape: list of points
    :param size: width, height - list or tuple
    :return: True, if every point of shape inside image
    """
    if is_shape_digit(shape[config.LABEL_TAG]):
        for point in shape[config.POINTS_TAG]:
            if point[0] < 0 or point[0] >= size[0] or point[1] < 0 or point[1] >= size[1]:
                return False
    return True


def get_transformed_image(background: np.array, image: np.array, x_c: float, y_c: float, R: float,
                          M: np.array, rectangle: tuple, character_map: dict) -> ImageWithAnnotation:
    """
    Places image.image to background using perspective transformation matrix M, and transformes all shapes coordinates
    :param background: background image
    :param image: ImageWithAnnotation class, describing water meter image
    :param x_c: x coordinate of water meter circle
    :param y_c: y coordinate of water meter circle
    :param R: radius of water meter circle
    :param M: perspective transformation matrix
    :param rectangle: bounding box (in form of tuple) for transformed circle after M perspective transform
    :param character_map: container with digits description on other water meters for random imprinting procedure
    :return: new ImageWithAnnotation with imprinted water meter and randomly swapped digits. shapes coordinates are
    properly transformed
    """
    size = (background.shape[1], background.shape[0])
    meter_image = cv2.cvtColor(image.image, cv2.COLOR_BGR2BGRA)
    mask = np.zeros(image.image.shape[:2], np.uint8)
    cv2.circle(mask, (int(x_c), int(y_c)), int(R), 255, -1)
    meter_image[:, :, 3] = mask
    generated_image = cv2.warpPerspective(meter_image, M, size,
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0, 0))
    mask = generated_image[:, :, 3]
    mask = cv2.blur(mask, (10, 10))
    for i in range(3):
        generated_image[:, :, i] = ((np.multiply(background[:, :, i].astype(np.uint16), 255 - mask) +
                                     np.multiply(generated_image[:, :, i].astype(np.uint16), mask)) / 255
                                    ).astype(np.uint8)
    generated_image_with_annotations = ImageWithAnnotation(generated_image, image.circle_points, image.shapes,
                                                           image.class_name, image.json_file)
    for shape in generated_image_with_annotations.shapes:
        points_in = np.array(shape[config.POINTS_TAG]).reshape(-1, 1, 2)
        points_out = cv2.perspectiveTransform(points_in, M)
        shape[config.POINTS_TAG] = points_out.reshape(-1, 2).tolist()
        if is_shape_digit(shape[config.LABEL_TAG]):
            imprint_random_digit(generated_image_with_annotations.image, image.class_name, shape, character_map)

    generated_image_with_annotations.shapes = \
        [shape for shape in generated_image_with_annotations.shapes if digit_shape_inside(shape, size)]
    generated_image_with_annotations.set_meter_rectangle(*rectangle)
    return generated_image_with_annotations


def get_iou(bb1: tuple, bb2: tuple) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Copied (with modifications) from
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    :param bb1: (x1_left, x1_right, y1_top, y1_bottom)
    :param bb2: (x2_left, x2_right, y2_top, y2_bottom)

    :return:  intersection over union in [0, 1]
    """
    x1_left = min(bb1[0], bb1[1])
    x1_right = max(bb1[0], bb1[1])
    y1_top = min(bb1[2], bb1[3])
    y1_bottom = max(bb1[2], bb1[3])
    x2_left = min(bb2[0], bb2[1])
    x2_right = max(bb2[0], bb2[1])
    y2_top = min(bb2[2], bb2[3])
    y2_bottom = max(bb2[2], bb2[3])

    # determine the coordinates of the intersection rectangle
    x_left = max(x1_left, x2_left)
    y_top = max(y1_top, y2_top)
    x_right = min(x1_right, x2_right)
    y_bottom = min(y1_bottom, y2_bottom)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (x1_right - x1_left + 1) * (y1_bottom - y1_top + 1)
    bb2_area = (x2_right - x2_left + 1) * (y2_bottom - y2_top + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def check_allowed_overlapping(new_box: tuple, boxes: list) -> bool:
    """
    Checks if any of bounding box in the list overlaps with newly generated bounding box
    :param new_box: (x1_left, x1_right, y1_top, y1_bottom) bounding box tuple
    :param boxes: list of bounding box tuples of same format as new_box
    :return: True if no overlaps above iou config.MAX_OCCLUSION_IOU threshold
    """
    for box in boxes:
        if get_iou(box, new_box) > config.MAX_OCCLUSION_IOU:
            return False
    return True


def generate_random_image(meter_images: list, background_images: list, character_map: dict,
                          generator_config: dict = config, fixed_config: FixedGeneratorConfig = None):
    """
    Generates random image with water meter
    :param meter_images: list of water meter images with annotations for random selection to imprint
    :param background_images: list of backgrounds for random choice
    :param character_map: mapping of class and character labels to images and shape descriptions with polygons
    :param generator_config: various parameters with limits for random selection of rotations, displacement, zoom and
    other options of imprint process
    :param fixed_config: FixedGeneratorConfig class, overrides random selection if set
    :return: image with randomly selected background and several randomly placed water meters, and its annotation
    """
    size = tuple(generator_config.GENERATOR_IMAGE_SIZE)
    background_source = random.choice(background_images)
    bg_width, bg_height = background_source.shape[1], background_source.shape[0]
    x_left_field = int(random.uniform(0, config.BACKGROUND_RANDOM_CROP_FIELD) * bg_width)
    x_right_field = bg_width - int(random.uniform(0, config.BACKGROUND_RANDOM_CROP_FIELD) * bg_width)
    y_top_field = int(random.uniform(0, config.BACKGROUND_RANDOM_CROP_FIELD) * bg_height)
    y_bottom_field = bg_height - int(random.uniform(0, config.BACKGROUND_RANDOM_CROP_FIELD) * bg_height)
    background_source = background_source[x_left_field:x_right_field, y_top_field: y_bottom_field]
    background = cv2.resize(background_source, size, cv2.INTER_CUBIC)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

    image_num = random.choice(range(config.MAX_METERS_IN_IMAGE + 1))
    if image_num == 0 and random.uniform(0.0, 1.0) > 0.2:
        image_num += 1
    all_shapes = []
    all_boxes = []
    for i in range(image_num):
        image = random.choice(meter_images)

        if fixed_config is None or i != 0:
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

        x_c = image.circle_points[0][0]
        y_c = image.circle_points[0][1]
        R = math.dist(image.circle_points[0], image.circle_points[1])
        M, rectangle\
            = calculate_transform_parameters(alpha, beta, gamma, x_displacement, y_displacement, zoom_coefficient,
                                             x_c, y_c, R, size)
        if check_allowed_overlapping(rectangle, all_boxes):
            generated_image_with_annotations = get_transformed_image(background, image, x_c, y_c, R,
                                                                     M, rectangle, character_map)
            background = generated_image_with_annotations.image
            all_shapes.extend(generated_image_with_annotations.shapes)
            all_boxes.append(rectangle)

    return ImageWithAnnotation(background[:, :, :3], [], all_shapes, "", "")


def main():
    """
    Function to hide variables from global module space
    """
    global config

    parser = argparse.ArgumentParser(description='Takes images from dateset folder, '
                                                 'extracts water meter images from them '
                                                 'using given annotations, places it randomly on '
                                                 'background image, and, finally, stores result into output folder.')
    parser.add_argument('--config', metavar='<path_to_configfiles>', type=str,
                        default=None,
                        help='Path to config file')
    parser.add_argument('--backgrounds_path', metavar='<path_to_backgrounds>', type=str,
                        default=config.BACKGROUNDS_PATH,
                        help='Path to backgrounds folder')
    parser.add_argument('--dataset_path', metavar='<path_to_dataset>', type=str,
                        default=config.DATASET_PATH,
                        help='Path to dataset folder')
    parser.add_argument('--output_path', metavar='<output_path>', type=str,
                        default=config.OUTPUT_PATH,
                        help='Path to output folder')
    parser.add_argument('--visualize', type=str,
                        help='Show opencv demo window')
    args = parser.parse_args()

    if args.config:
        config = read_config_file(args.config)
    else:
        config.BACKGROUNDS_PATH = args.backgrounds_path
        config.DATASET_PATH = args.dataset_path
        config.OUTPUT_PATH = args.output_path
    print(f"Dataset folder: {config.DATASET_PATH}\n"
          f"Background folder: {config.BACKGROUNDS_PATH}\n"
          f"Output folder: {config.OUTPUT_PATH}")
    dataset_jsons = get_file_list(config.DATASET_PATH, "*.json")
    dataset_jsons.sort()
    background_files = get_file_list(config.BACKGROUNDS_PATH, "*.jpg")
    background_files.sort()
    print("Dataset files: ", dataset_jsons)
    print("Background files: ", background_files)
    backgrounds = []
    meter_images = []
    for file in tqdm(background_files, "Reading backgrounds"):
        bg_image = cv2.imread(file, cv2.IMREAD_COLOR)
        backgrounds.append(bg_image)
    for file in tqdm(dataset_jsons, "Reading labelme annotations"):
        append_meters_image_and_info(config.DATASET_PATH, file, meter_images)
    characters_map = collect_chars_map(meter_images)
    if args.visualize:
        show_demo_window(meter_images, backgrounds, characters_map, config)

    file_data_template = {
      "version": "5.1.1",
      "flags": {},
      "shapes": [],
      "imagePath": None,
      "imageData": None,
      "imageHeight": -1,
      "imageWidth": -1
    }
    output_path = pathlib.Path(config.OUTPUT_PATH)
    for index in tqdm(range(config.NUM_OF_GENERATED_IMAGES), "Saving files"):
        image_file_name = f'{index:06d}.jpg'
        json_file_name = f'{index:06d}.json'
        image = generate_random_image(meter_images, backgrounds, characters_map, config)
        cv2.imwrite(str(output_path/image_file_name), image.image)
        file_data_template["imagePath"] = image_file_name
        file_data_template["imageHeight"] = image.image.shape[0]
        file_data_template["imageWidth"] = image.image.shape[1]
        file_data_template["shapes"] = image.shapes
        with open(output_path/json_file_name, "w") as f:
            json.dump(file_data_template, f, indent=2)


if __name__ == '__main__':
    main()


