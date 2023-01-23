
import cv2
import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + "squeezedet_keras")
from squeezedet_keras.main.model.squeezeDet import SqueezeDet
from squeezedet_keras.main.config.create_config import load_dict
from squeezedet_keras.main.model.evaluation import filter_batch
from squeezedet_keras.main.model.visualization import bbox_transform_single_box


class SqueezeDetInference:

    def __init__(self, checkpoint, config):
        self.config = load_dict(config)
        self.config.BATCH_SIZE = 1
        self.squeeze = SqueezeDet(self.config)
        self.squeeze.model.load_weights(checkpoint)

    def predict(self, image):
        resized_image = cv2.resize(image, (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT), cv2.INTER_CUBIC)
        resized_image = resized_image.astype(np.float32)
        normalized_image = (resized_image - np.mean(resized_image)) / np.std(resized_image)
        img_array = tf.keras.utils.img_to_array(normalized_image)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        y_pred = self.squeeze.model.predict(img_array)
        boxes, classes, scores = filter_batch(y_pred, self.config)
        detected_boxes = []
        for b in boxes[0]:
            detected_box = bbox_transform_single_box(b)
            detected_box[0] = detected_box[0] * image.shape[1] / self.config.IMAGE_WIDTH
            detected_box[1] = detected_box[1] * image.shape[0] / self.config.IMAGE_HEIGHT
            detected_box[2] = detected_box[2] * image.shape[1] / self.config.IMAGE_WIDTH
            detected_box[3] = detected_box[3] * image.shape[0] / self.config.IMAGE_HEIGHT
            detected_boxes.append(detected_box)
        class_names = [self.config.CLASS_NAMES[c] for c in classes[0]]
        return detected_boxes, class_names, scores[0]


class TwoStageInference:

    def __init__(self, config):
        self.stage1 = SqueezeDetInference(config.STAGE1_CHECKPOINT, config.STAGE1_CONFIG)
        self.stage2 = SqueezeDetInference(config.STAGE2_CHECKPOINT, config.STAGE2_CONFIG)
        self.last_boxes = []
        self.last_classes = []
        self.last_scores = []
        self.last_digit_boxes = []
        self.last_digits = []
        self.last_digit_scores = []
        self.ocr_boxes = []
        self.ocr_results = []
        self.ocr_scores = []

    def predict(self, image):
        self.last_boxes, self.last_classes, self.last_scores = self.stage1.predict(image)
        self.ocr_boxes, self.ocr_results, self.ocr_scores = [], [], []
        self.last_digit_boxes, self.last_digits, self.last_digit_scores = [], [], []

        for b in self.last_boxes:
            rect = b.copy()
            rect[0] = np.clip(int(rect[0]), 0, image.shape[1] - 1)
            rect[1] = np.clip(int(rect[1]), 0, image.shape[0] - 1)
            rect[2] = np.clip(int(rect[2]), 0, image.shape[1] - 1)
            rect[3] = np.clip(int((rect[3] + rect[1])/2), 0, image.shape[0] - 1)
            meter_image = image[rect[1]:rect[3], rect[0]:rect[2], :]
            digit_boxes, digits, digit_scores = self.stage2.predict(meter_image)
            self.last_digit_boxes.extend(digit_boxes)
            self.last_digits.extend(digits)
            self.last_digit_scores.extend(digit_scores)
            for db in digit_boxes:
                db[0] = db[0] + rect[0]
                db[1] = db[1] + rect[1]
                db[2] = db[2] + rect[0]
                db[3] = db[3] + rect[1]
            score = np.min(digit_scores) if len(digit_scores) > 0 else 0.0
            if len(digits) >= 7 and score > 0.6:
                self.ocr_boxes.append(b)
                self.ocr_scores.append(score)
                list_of_pairs = sorted([((b[0] + b[2]) / 2, c) for b, c in zip(digit_boxes, digits)])
                _, digits_sorted_by_x = zip(*list_of_pairs)
                self.ocr_results.append(''.join(digits_sorted_by_x))

        return self.ocr_boxes, self.ocr_results, self.ocr_scores
