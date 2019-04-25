import cv2
import numpy as np
import time

class WeatheringDegreeMap:
    def __init__(self, input_img, most_weathered_pixel, least_weathered_pixel):
        self.input_image = input_img