import cv2
import time
import numpy as np
from ProcessOnGUI import ProcessOnGUI
from WeatheringDegreeMap import WeatheringDegreeMap
from WeatheringExampler import WeatheringExampler


class ProcessForExampleImage():
    def __init__(self):
        self.example_img_name = "rust2_large"
        self.example_img = cv2.imread("./InputImageSet/" + self.example_img_name + ".jpg")

    def run_processes(self):
        total_process_time = 0
        # GUI操作
        process_on_gui_ = ProcessOnGUI(self.example_img)
        process_on_gui_.supervise_GUI_for_example_image()

        # ピクセルpの経年変化度Dp[0.0, 1.0]を計算
        start_time = time.time()
        weathering_degree_map_ = WeatheringDegreeMap(process_on_gui_.input_img_foreground,
                                                     process_on_gui_.most_weathered_pixel,
                                                     process_on_gui_.least_weathered_pixel)
        weathering_degree_D = weathering_degree_map_.compute_weathering_degree_map()
        end_time = time.time() - start_time
        print("compute_weathering_degree_time:{0}".format(end_time) + "[sec]")
        total_process_time += end_time
        self.save_display_img(np.uint8(weathering_degree_D*255), "weathering_degree_map")


        # Dに風化/非風化の二値をラベリングしたL[0 or 1]
        start_time = time.time()
        labeled_map_L = weathering_degree_map_.labeling_weathering_degree_map()
        end_time = time.time() - start_time
        print("labeling_time:{0}".format(end_time) + "[sec]")
        total_process_time += end_time
        self.save_display_img(labeled_map_L, "binarize_map")

        # 最も風化ラベルを持った矩形領域から生成した風化モデルT[
        start_time = time.time()
        weathring_exampler_ = WeatheringExampler(self.example_img, labeled_map_L, EXAMPLER_SIZE = 150)
        most_weathered_rect, only_weathered_rect, weathring_exampler_T = \
            weathring_exampler_.create_weathering_exampler()
        end_time = time.time() - start_time
        print("exampler_time:{0}".format(end_time) + "[sec]")
        total_process_time += end_time
        self.save_display_img(most_weathered_rect, "most_weathered_rect")
        self.save_display_img(only_weathered_rect, "only_weathered_rect")
        self.save_display_img(weathring_exampler_T, "weathering_exampler")


        print("total_time:{0}".format(total_process_time) + "[sec]")
        
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

