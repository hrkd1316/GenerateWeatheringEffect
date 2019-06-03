import cv2
import time
import numpy as np
from weathering_gui import ProcessOnGUI
from gen_weathering_deg import generate_weathering_degree
from shading_map import ShadingMap
from weathering_texture import WeatheringTexture
import render_effect as render
import sv_disp_img as sv

class ProcessForTargetImage():
    def __init__(self):
        self.target_img_name = "tgt_img1_large"
        self.target_img = cv2.imread("./InputImageSet/" + self.target_img_name + ".jpg")

    def run_processes(self):
        total_process_time = 0

        # GUIの起動
        process_on_gui_ = ProcessOnGUI(self.target_img)
        process_on_gui_.launch_gui_for_target_img()
        self.target_img_fgnd = process_on_gui_.input_img_foreground

        # ユーザ入力パスを基に風化度D', ラベリングしたL'を計算
        start_time = time.time()
        labeled_map_Ld, weathering_degree_map_Dd = \
            generate_weathering_degree(self.target_img_fgnd, process_on_gui_.through_pixel)
        end_time = time.time() - start_time
        print("generate_weathering_degree_time:{0}".format(end_time) + "[sec]")
        total_process_time += end_time
        sv.save_display_img(labeled_map_Ld, self.target_img_name, "labeled_map")
        sv.save_display_img(np.float64(weathering_degree_map_Dd), self.target_img_name, "generated_weathering_degree")

        # シェーディングマップS[0.0, 1.0]を計算
        start_time = time.time()
        shading_map_ = ShadingMap(self.target_img_fgnd, weathering_degree_map_Dd, labeled_map_Ld)
        shading_map_S = shading_map_.compute_shading_map()
        end_time = time.time() - start_time
        print("shading_map_time:{0}".format(end_time) + "[sec]")
        total_process_time += end_time
        sv.save_display_img(shading_map_S, self.target_img_name, "shading_map")

        """
        # 風化テクスチャZ[0, 255]を計算
        example_img_name = "rust2_large"
        weathering_exampler_T = cv2.imread("./OutputImageSet/" + example_img_name + "_weathering_exampler.png")
        start_time = time.time()
        weathering_texture_ = \
            WeatheringTexture(self.target_img_fgnd, weathering_degree_map_Dd, weathering_exampler_T, labeled_map_Ld)
        weathering_texture_Z = weathering_texture_.generate_weathering_texture()
        end_time = time.time() - start_time
        print("weathering_texture_time:{0}".format(end_time) + "[sec]")
        total_process_time += end_time
        sv.save_display_img(weathering_texture_Z, self.target_img_name, "weathering_texture")
        """
        texture_img_name = "tgt_img1_large"
        weathering_texture_Z = cv2.imread("./OutputImageSet/" + texture_img_name + "_weathering_texture.png")
        # 風化効果を描画
        output_img = render.render_weathering_effect(self.target_img, weathering_degree_map_Dd,
                                                     shading_map_S, weathering_texture_Z)
        sv.save_display_img(output_img, self.target_img_name, "output.png")
        print("total_time:{0}".format(total_process_time) + "[sec]")
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_for_target_img_ = ProcessForTargetImage()
    process_for_target_img_.run_processes()