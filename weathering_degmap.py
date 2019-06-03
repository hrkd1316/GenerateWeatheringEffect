# coding=utf-8
import cv2
import numpy as np
import scipy.optimize as sp
import time

from pixel_feature import PixelFeatureVector
import extract_foreground_area as extract
from graph_cut import graph_cut 


# 経年変化度合い [0.0, 1.0]
class WeatheringDegreeMap:
    def __init__(self, input_img, most_weathered_pixel, least_weathered_pixel):
        self.input_img = input_img
        self.pixel_future_vector = PixelFeatureVector(input_img)

        self.most_weathered_pixel = most_weathered_pixel
        self.least_weathered_pixel = least_weathered_pixel
        self.SUM_WEATHERED_PIXEL_NUM = len(most_weathered_pixel) + len(least_weathered_pixel)
        self.merge_weathered_pixel = most_weathered_pixel + least_weathered_pixel

        self.weathering_degree_map = np.zeros(input_img.shape[:2])

    # "Single Image Weathering via Exemplar Propagation"のargmin(E(α))を
    # non negative least squares(非負最小二乗法)を使って解くことでαを求める．
    def compute_coefficient_alpha(self):
        # 最小二乗法を使うとargmin(E(α))を
        # 正規方程式 (G^T)Gα = (G^T)y と表せる
        # 論文に合わせるとG = Φ, y = d
        matrix_phi = np.array([[self.pixel_future_vector.compute_RBF(i[0], i[1], j[0], j[1])
                                for j in self.merge_weathered_pixel] for i in self.merge_weathered_pixel])

        # A = (φ^T)φα
        matrix_A = np.dot(matrix_phi.T, matrix_phi)

        # b = (φ^T)d
        # most_weathered_pixelにはd = 1, least_weathered_pixelにはd=0.01をセット
        matrix_d = np.array([1 if i < len(self.most_weathered_pixel) else 0.01
                             for i in range(self.SUM_WEATHERED_PIXEL_NUM)])
        matrix_b = np.dot(matrix_phi.T, matrix_d)

        # nnls(ndarray A, ndarray b)
        # 返り値: ndarray α (解)
        # argmin(α) ||Aα = b|| α >= 0を解く
        return sp.nnls(matrix_A, matrix_b)[0]

    # 各ピクセルpの経年変化度D_pを求める
    def compute_weathering_degree_map(self):
        print("compute_weathering_start")
        coefficient_alpha = self.compute_coefficient_alpha()

        self.weathering_degree_map = np.array(
            [[sum([coefficient_alpha[i] *
                   self.pixel_future_vector.compute_RBF(xp,
                                                        yp,
                                                        self.merge_weathered_pixel[i][0],
                                                        self.merge_weathered_pixel[i][1])
                   for i in range(self.SUM_WEATHERED_PIXEL_NUM)])
              if self.pixel_future_vector.luminance[yp, xp] != 0 else 0
              for xp in range(self.pixel_future_vector.width)]
             for yp in range(self.pixel_future_vector.height)])

        # [0.0, 1.0]に正規化
        self.weathering_degree_map = self.weathering_degree_map / np.max(self.weathering_degree_map)
        return self.weathering_degree_map

    # 経年変化度マップの全ピクセルを風化/非風化の二値にラベリングする
    def labeling_weathering_degree_map(self):
        print("labeling_start")

        labeled_map = graph_cut(self.weathering_degree_map, self.most_weathered_pixel, self.least_weathered_pixel)

        return labeled_map


if __name__ == "__main__":
    test_weathering_degree_map = cv2.imread("./OutputImageSet/rust2_large_weathering_degree_map.png",
                                            cv2.IMREAD_GRAYSCALE) / 255.0
    binarize_weathering_degree_map(test_weathering_degree_map)
