import cv2
import numpy as np
import math

from patch_match import NNF

class WeatheringExampler:
    def __init__(self, input_img, labeled_img, EXAMPLER_SIZE):
        self.input_img = input_img
        self.labeled_img = labeled_img
        self.EXAMPLER_SIZE = EXAMPLER_SIZE
    
    def create_weathering_exampler(self):
        most_weathered_rect, only_weathered_rect = self.find_rectangle_with_sat()
        cv2.imshow("only_weathered_rect", only_weathered_rect)
        weathering_exampler = self.fill_hole(only_weathered_rect, most_weathered_rect)

        return most_weathered_rect, only_weathered_rect, weathering_exampler
    
    # SAT: Summed Area Tables
    # 最も風化ラベルの多い矩形領域(EXAMPLE_SIZE × EXAMPLE_SIZE)を探す
    def find_rectangle_with_sat(self):
        print("find_rectangle_start")
        max_in_all_sat = 0
        sat = np.zeros((self.EXAMPLER_SIZE, self.EXAMPLER_SIZE))
        rect_point = [0, 0] #最も風化ラベルの多い矩形領域の開始座標

        sat = np.array([[np.cumsum(self.labeled_img[y:y+self.EXAMPLER_SIZE, x:x+self.EXAMPLER_SIZE])[-1]
                         for x in range(self.labeled_img.shape[1] - self.EXAMPLER_SIZE)]
                        for y in range(self.labeled_img.shape[0] - self.EXAMPLER_SIZE)])
        
        rect_point = list(nd[0] for nd  in np.where(sat == sat.max()))
        rect_height = rect_point[0] + self.EXAMPLER_SIZE
        rect_width = rect_point[1] + self.EXAMPLER_SIZE

        #
        most_weathered_rect =  self.input_img[rect_point[0]:rect_height, rect_point[1]:rect_width]
        # 風化ラベルのピクセルのみを残す
        only_weathered_rect = np.where(
            self.labeled_img[rect_point[0]:rect_height, rect_point[1]:rect_width, np.newaxis] == 255,
        self.input_img[rect_point[0]:rect_height, rect_point[1]:rect_width], 0)
        
        return most_weathered_rect, only_weathered_rect

    # 非風化ラベルのピクセル部分に穴が開くのでpatch matchを使って穴埋めを行う
    # ref_img: 穴が開く前の画像 (most_weathered_rect)
    # tgt_img: 穴が開いた画像  (only_weathered_rect)
    def fill_hole(self, ref_img, tgt_img):
        print("fill_hole_start")
        NNF_ = NNF(ref_img, tgt_img, 6)
        return NNF_.reconstruct_img()