import cv2
import numpy as np

from gen_texture import TextureFromPatch

class WeatheringTexture(TextureFromPatch):

    def __init__(self, input_img, weathering_degree_map, weathering_exampler, labeled_map):
        super(WeatheringTexture, self).__init__(input_img.shape, weathering_exampler, PATCH_SIZE=60)
        self.weathering_degree_map_Dd = weathering_degree_map
        self.labeld_map_Ld = labeled_map
            
    def generate_weathering_texture(self):
        self.quilt_patch_random()
        self.quilt_patch_optimization()
        print("quilt_optimization_60")
        self.PATCH_SIZE = 20
        self.quilt_patch_optimization()
        print("quilt_optimization_20")
        return self.texture_img