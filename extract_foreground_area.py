import cv2
import numpy as np

# 参照:http://urx2.nu/ZHbM (OpenCV Tutorials)
def extract_foreground_area(input_img, rect_cover_foreground):
    mask_img = np.zeros(input_img.shape[:2], np.uint8)
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode)
    cv2.grabCut(input_img, mask_img, rect_cover_foreground, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask_img2 = np.where((mask_img == 2) | (mask_img == 0), 0, 1).astype('uint8')
    output_img = input_img * mask_img2[:, :, np.newaxis]
    return output_img
