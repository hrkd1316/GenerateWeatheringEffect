import cv2
import numpy as np

def save_display_img(input_img, input_img_name, state_text):
    cv2.imshow(state_text, input_img)
    cv2.imwrite("./OutputImageSet/" + input_img_name + "_" + state_text + ".png", input_img)