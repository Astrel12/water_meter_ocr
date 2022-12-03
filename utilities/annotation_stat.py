"""
This script scans folder with labelme json annotations and outputs number of annotated classes on images

Auther: Anatoly Khamukhin
"""

import argparse
import os
import glob
import json


def calculate_stat(json_file: str, class_stat: dict):
    """
    Function processes one json file and stores numbers of clases to dict
    """
    with open(json_file) as f:
        json_content = json.load(f)
        if 'shapes' in json_content:
            for shape in json_content['shapes']:
                if shape['label'] in class_stat:
                    class_stat[shape['label']] = class_stat.get(shape['label'], 0) + 1
                else:
                    class_stat[shape['label']] = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates number of different classes in labelme annotation '
                                                 'for given folder')
    parser.add_argument('--path', metavar='<path_to_annotation_folder>', type=str,
                        help='Path to dataset folder')

    args = parser.parse_args()

    print("Processing folder: ", args.path)

    class_stat = dict()
    for path, dirs, files in os.walk(args.path):
        for file in glob.glob(path+os.sep+"*.json"):
            calculate_stat(file, class_stat)

    for key in sorted(class_stat):
        print(f"{key} : {class_stat[key]}")
