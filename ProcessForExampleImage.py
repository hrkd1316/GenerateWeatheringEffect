import cv2
import time
import numpy as np
from ProcessOnGUI import ProcessOnGUI
from WeatheringDegreeMap import WeatheringDegreeMap


class ProcessForExampleImage():
    def __init__(self):
        self.example_img_name = "rust2_large"
        self.example_img = cv2.imread("./InputImageSet/" + self.example_img_name + ".jpg")

    def run_processes(self):
        # GUI操作
        process_on_gui_ = ProcessOnGUI(self.example_img)
        process_on_gui_.supervise_GUI_for_example_image()

        # 経年変化度D計算
        start_time = time.time()
        weathering_degree_map_ = WeatheringDegreeMap(process_on_gui_.input_img_foreground,
                                                     process_on_gui_.most_weathered_pixel,
                                                     process_on_gui_.least_weathered_pixel)
        weathering_degree_D = weathering_degree_map_.compute_weathering_degree_map()
        print("compute_weathering_degree_time:{0}".format(time.time() - start_time) + "[sec]")
        self.save_display_img(np.uint8(weathering_degree_D*255), "weathering_degree_map")
        #cv2.imshow("weathering_degree_map", weathering_degree_D)
        #cv2.imwrite("./OutputImageSet/" + self.example_img_name + "_weathering_degree.png", weathering_degree_D)

        # Dを風化/非風化に二値化したL
        start_time = time.time()
        binarize_map_L = weathering_degree_map_.binarize_weathering_degree_map()
        print("binarize_time:{0}".format(time.time() - start_time) + "[sec]")
        self.save_display_img(binarize_map_L, "binarize_map")
        #cv2.imshow("binarize_map", binarize_map_L)
        #cv2.imwrite("./OutputImageSet/" + self.example_img_name + "_binarize_map.png", binarize_map_L)

        
        
        
        
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    def save_display_img(self, input_img, img_name):
        cv2.imshow(img_name, input_img)
        cv2.imwrite("./OutputImageSet/" + self.example_img_name + "_" + img_name + ".png", input_img)

if __name__ == "__main__":
    process_for_example_image = ProcessForExampleImage()
    process_for_example_image.run_processes()

