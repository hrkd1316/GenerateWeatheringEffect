import cv2
import numpy as np 

def generate_weathering_degree(input_img, through_pixel):
    gray_input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray_input_img)
    # ユーザ選択領域を二値画像で表す
    passage_route_img = np.zeros(gray_input_img.shape, dtype = np.uint8)
    [cv2.rectangle(passage_route_img,
                   (pixel[1] - 40, pixel[0] - 40), (pixel[1] + 40, pixel[0] + 40),
                   (255, 255, 255), -1)
     for pixel in through_pixel]

    ADJUSTMENT_PARAM = 1.5
    dist_map = cv2.distanceTransform(passage_route_img, cv2.DIST_L2, 5)
    #dist_map = cv2.distanceTransform(passage_route_on_obj_img, cv2.DIST_L2, 5)
    # 中心1点じゃなくて中心らへんの領域を最大値にしたいので全体的に底上げ
    dist_map = dist_map / np.amax(dist_map)  * ADJUSTMENT_PARAM
    dist_map[dist_map > 1.0] = 1.0
    # 元の画像で画素値0の部分を除いておく
    passage_route_on_obj_img = np.where(gray_input_img == 0, 0, passage_route_img)
    dist_map_on_obj_img = np.where(gray_input_img == 0, 0, dist_map)
    return passage_route_on_obj_img, dist_map_on_obj_img.astype(np.float16)

