"""
Standard YOLO object detection algorithm to detect objects present in COCO dataset

Partly sourced from : https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html

@Co-Authors : Sujith Umapathy, Reshmi
"""

import time

import cv2
import numpy as np
from configuration import Config as conf
import matplotlib.pyplot as plt
import pygame


def transform_image(image, angle):
    """
    Since the image sent from our scanner is rotated and flipped,
    we rotate the image anti-clockwise by 90 degree, and flip it along y axis
    This produces an image as seen by our scanner
    :param image: image to transform
    :param angle: angle of rotation
    :return: transformed image
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    result = cv2.flip(result, 1)
    return result


class Detector:
    def __init__(self):
        np.random.seed(42)

        self.root = f'{conf.object_detection_path}'

        self.net = cv2.dnn.readNetFromDarknet(
            f'{self.root}yolov3.cfg',
            f'{self.root}/weights/yolov3.weights')

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        self.classes = open(f'{self.root}coco.names').read().strip().split('\n')
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_object(self, image_to_detect):
        """
        Uses the image slice and detects if an Object of Interest is available
        :param image_to_detect: Image to check
        :return: dictionary consisting of obstacles and important objects
        """
        image_dict = {}
        standard_yolo_ip_size = (224, 224)
        scale_factor = 255.0

        image_to_detect = transform_image(image_to_detect, -90)
        copy = image_to_detect.copy()

        blob = cv2.dnn.blobFromImage(copy, 1 / scale_factor, standard_yolo_ip_size, swapRB=True, crop=False)

        self.net.setInput(blob)
        t0 = time.time()
        outputs = self.net.forward(self.ln)

        outputs = np.vstack(outputs)
        self.post_process(copy, outputs, 0.5, image_dict)

        plt.imshow(copy)
        plt.show()
        return image_dict

    def post_process(self, img, outputs, conf, image_dict):
        H, W = img.shape[:2]
        boxes = []
        confidences = []
        classIDs = []
        # Classes we classify as obstacles,
        # Issue : Dog is classified as cat or bird sometimes, hence the addition of different classes
        obstacle_class = ['bird', 'cat', 'dog']
        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w // 2), int(y - h // 2)
                p1 = int(x + w // 2), int(y + h // 2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf - 0.1)

        obstacle_list = []
        important_list = []

        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                print(f'Identified Class is : {self.classes[classIDs[i]]}')

                if self.classes[classIDs[i]] in obstacle_class:
                    obstacle_list.append(pygame.rect.Rect(x, y, w, h))
                elif self.classes[classIDs[i]] == 'bottle':
                    important_list.append(pygame.rect.Rect(x, y, w, h))

            if len(obstacle_list) > 0:
                image_dict['obstacles'] = obstacle_list
            if len(important_list) > 0:
                image_dict['important'] = important_list
