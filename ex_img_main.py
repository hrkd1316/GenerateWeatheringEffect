import cv2
import time
import numpy as np
from weathering_gui import ProcessOnGUI
from weathering_degmap import WeatheringDegreeMap
from weathering_exampler import WeatheringExampler
from sv_disp_img import save_display_img


class ProcessForExampleImage():
    def __init__(self):
        self.example_img_name = "rust2_large"
        self.example_img = cv2.imread("./InputImageSet/" + self.example_img_name + ".jpg")

    def run_processes(self):
        total_process_time = 0
        # GUI操作
        process_on_gui_ = ProcessOnGUI(self.example_img)
        process_on_gui_.launch_gui_for_example_img()

        # ピクセルpの経年変化度Dp[0.0, 1.0]を計算
        start_time = time.time()
        weathering_degree_map_ = WeatheringDegreeMap(process_on_gui_.input_img_foreground,
                                                     process_on_gui_.most_weathered_pixel,
                                                     process_on_gui_.least_weathered_pixel)
        weathering_degree_D = weathering_degree_map_.compute_weathering_degree_map()
        end_time = time.time() - start_time
        print("compute_weathering_degree_time:{0}".format(end_time) + "[sec]")
        total_process_time += end_time
        save_display_img(np.uint8(weathering_degree_D*255), self.example_img_name, "weathering_degree_map")


        # Dに風化/非風化の二値をラベリングしたL[0 or 1]を計算
        start_time = time.time()
        labeled_map_L = weathering_degree_map_.labeling_weathering_degree_map()
        end_time = time.time() - start_time
        print("labeling_time:{0}".format(end_time) + "[sec]")
        total_process_time += end_time
        save_display_img(labeled_map_L, self.example_img_name, "binarize_map")

        # 最も風化ラベルが集まった矩形領域から風化モデルTを生成
        start_time = time.time()
        weathring_exampler_ = WeatheringExampler(self.example_img, labeled_map_L, EXAMPLER_SIZE = 150)
        most_weathered_rect, only_weathered_rect, weathring_exampler_T = \
            weathring_exampler_.create_weathering_exampler()
        end_time = time.time() - start_time
        print("exampler_time:{0}".format(end_time) + "[sec]")
        total_process_time += end_time
        save_display_img(most_weathered_rect, self.example_img_name, "most_weathered_rect")
        save_display_img(only_weathered_rect, self.example_img_name, "only_weathered_rect")
        save_display_img(weathring_exampler_T, self.example_img_name, "weathering_exampler")


        print("total_time:{0}".format(total_process_time) + "[sec]")
        
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_for_example_img_ = ProcessForExampleImage()
    process_for_example_img_.run_processes()

