import cv2
import numpy as np
from pixel_feature import PixelFeatureVector

def render_weathering_effect(input_img, weahtering_deg_map, shading_map, weathering_texture):
    weahtering_deg_map_update = np.where(weahtering_deg_map < 0.7, 0, weahtering_deg_map)

    input_pixel_feature_ = PixelFeatureVector(input_img)
    texture_pixel_feature_ = PixelFeatureVector(weathering_texture)
    output_img = np.zeros_like(input_img, dtype='float32')
    output_img = output_img.transpose((2, 0, 1))
    S_update = np.where(weahtering_deg_map < 0.5, 1, input_pixel_feature_.luminance)
    output_img[0] = (weahtering_deg_map_update*texture_pixel_feature_.luminance
                     + (1-weahtering_deg_map_update)*input_pixel_feature_.luminance) \
                     * S_update
    output_img[1] = (weahtering_deg_map_update*texture_pixel_feature_.chroma_a
                     + (1-weahtering_deg_map_update)*input_pixel_feature_.chroma_a) 
    output_img[2] = (weahtering_deg_map_update*texture_pixel_feature_.chroma_b
                     + (1-weahtering_deg_map_update)*input_pixel_feature_.chroma_b)
    output_img = output_img.transpose((1, 2, 0))
    output_img = cv2.cvtColor(np.uint8(output_img*255), cv2.COLOR_Lab2BGR)
    
    return output_img