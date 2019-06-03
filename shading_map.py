import cv2
import numpy as np

from pixel_feature import PixelFeatureVector

class ShadingMap:
    def __init__(self, input_img, weathering_degree_map, labeled_map):
        self.input_img = input_img
        self.labeld_map = labeled_map
        self.alter_weathring_map = np.where(labeled_map == 0, weathering_degree_map, -1)
        self.pixel_feature = PixelFeatureVector(input_img)
        
    """
    def calc_section_ave_lum(self):
        BIN_NUM = 20

        # 非風化領域の風化度を20区間に等分
        bins = np.linspace(0, np.amax(self.alter_weathring_map), BIN_NUM)
        # 各ピクセルがどの区間に当たるか
        bin_index = np.digitize(self.alter_weathring_map, bins, right = True)
        #各区間の平均値を求める
        section_lum = [0] * (BIN_NUM+1)
        [[section_lum[bin_index[y, x]].append(self.pixel_feature.luminance)
          for x in range(self.input_img.shape[1])]
         for y in range(self.input_img.shape[0])]

        section_ave_lum = np.average(np.array(section_lum), axis=1)

        return bin_index, section_ave_lum
    
    def compute_shading_map(self):
        bin_index, section_ave_lum = self.calc_section_ave_lum()
        shading_map = [[self.pixel_feature.luminance[y, x] / section_ave_lum[bin_index[y, x]]
                        if self.alter_weathring_map[y, x] != -1 else self.pixel_feature.luminance[y, x]
                        for x in range(self.input_img.shape[1])]
                       for y in range(self.input_img.shape[0])]
        
        return shading_map
    """

    def compute_shading_map(self):
        shading_map = np.ones(self.input_img.shape[:2], dtype=np.float64)

        shading_map = np.where(self.pixel_feature.luminance != 0, self.pixel_feature.luminance, 1.0)
        return shading_map