import cv2
import numpy as np
import numexpr as ne
import scipy.optimize as sp
import time
import math
from PixelFeatureVector import PixelFeatureVector

# 経年変化度合い [-1.0, 1.0]
class WeatheringDegreeMap:
    def __init__(self, input_img, most_weathered_pixel, least_weathered_pixel):
        self.input_image = input_img
        self.most_weathered_pixel = most_weathered_pixel
        self.least_weathered_pixel = least_weathered_pixel
        self.weathering_degree = np.zeros_like(input_img)
        self.pixel_future_vector = PixelFeatureVector(input_img)

    # E(α) = sum_{i∈Ω}(d_i - sum_{j∈Ω}(α_jφ_ij))^2をAα = bの形に変形して
    # non negative least squares(非負最小二乗法)を使って解くことでαを求める．
    def compute_weathering_degree(self):
        matrix_A = compute_matrix_A()
        matrix_b = compute_matrix_b()
        coefficient_alpha = sp.nnls(A, b)[0]

    def compute_matrix_A(self):
        SUM_WEATHERED_PIXEL = len(self.most_weathered_pixel)+len(self.least_weathered_pixel)
        matrix_A = np.zeros((SUM_WEATHERED_PIXEL, SUM_WEATHERED_PIXEL))

        
