# coding=utf-8
import cv2
import time
import extract_foreground_area as extract


class ProcessOnGUI():
    def __init__(self, input_img):
        self.input_img = input_img.copy()
        self.img_for_display = input_img.copy()

        # 前景抽出
        self.input_img_foreground = input_img.copy()
        self.img_foreground_for_display = input_img.copy()
        self._is_selected_rect = True
        self._rect_cover_foreground = [-1, -1, -1, -1]

        # 最も風化している/風化していないピクセルの保存
        self.most_weathered_pixel = []
        self.least_weathered_pixel = []

        # 風化したい領域の塗りつぶし，通過ピクセルの保存
        self._can_draw = False
        self.through_pixel = []

    def launch_gui_for_example_img(self):
        window_name_input = "input image"
        window_name_foreground = "input image foreground"

        cv2.namedWindow(window_name_input, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(window_name_foreground, cv2.WINDOW_AUTOSIZE)

        # それぞれのウィンドウにおけるマウスが行う動作を設定する
        # input image：前景を囲む矩形を選択すると前景を抽出する
        # input image foreground：最も風化している点/風化していない点を選択し，座標を保存する
        cv2.setMouseCallback(window_name_input, self.mouse_select_rect_cover_foreground)
        cv2.setMouseCallback(window_name_foreground, self.mouse_select_most_least_weathered_pixel)

        while True:
            cv2.imshow(window_name_input, self.img_for_display)
            cv2.imshow(window_name_foreground, self.img_foreground_for_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def launch_gui_for_target_img(self):
        window_name_input = "input image"
        window_name_foreground = "input image foreground"

        cv2.namedWindow(window_name_input, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(window_name_foreground, cv2.WINDOW_AUTOSIZE)

        # それぞれのウィンドウにおけるマウスが行う動作を設定する
        # input image：前景を囲む矩形を選択すると前景を抽出する
        # input image foreground：風化させたい領域をドラッグ操作すると塗りつぶされ、通った座標が保存される
        cv2.setMouseCallback(window_name_input, self.mouse_select_rect_cover_foreground)
        cv2.setMouseCallback(window_name_foreground, self.mouse_select_weathering_region)

        while True:
            [cv2.rectangle(self.img_foreground_for_display,
                           (pixel[1] - 15, pixel[0] - 15), (pixel[1] + 15, pixel[0] + 15),
                           (255, 0, 0), -1)
             for pixel in self.through_pixel]
            cv2.imshow(window_name_input, self.img_for_display)
            cv2.imshow(window_name_foreground, self.img_foreground_for_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            cv2.imwrite("./OutputImageSet/input_img_forground.png", self.img_foreground_for_display)

    # 2つの座標(矩形の左上と右下)から[x, y, width, height]のrect情報を作成
    # GrabCut()にrect情報を渡すと前景を抽出した画像を返す
    # todo: エラー処理をしよう
    def mouse_select_rect_cover_foreground(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self._is_selected_rect:
            cv2.circle(self.img_for_display, (x, y), 2, (0, 0, 255), -1)
            # 一度目の選択(矩形の左上)
            if self._rect_cover_foreground[0] == -1:
                self._rect_cover_foreground[0] = x
                self._rect_cover_foreground[1] = y
            # 二度目の選択(矩形の右下)
            else:
                self._rect_cover_foreground[2] = x - self._rect_cover_foreground[0]
                self._rect_cover_foreground[3] = y - self._rect_cover_foreground[1]
                self._is_selected_rect = False

                # GraphCutを用いて前景抽出
                start_time = time.time()
                self.input_img_foreground = \
                    extract.extract_foreground_area(self.input_img, tuple(self._rect_cover_foreground))
                self.img_foreground_for_display = self.input_img_foreground.copy()
                print("extract_foreground_time:{0}".format(time.time() - start_time) + "[sec]")

    # 与えられる'x, y'を保存
    # 左クリック：最も風化したピクセル
    # 右クリック：最も風化していないピクセル
    def mouse_select_most_least_weathered_pixel(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.img_foreground_for_display, (x, y), 2, (0, 0, 0), -1)
            self.most_weathered_pixel.append([x, y])

        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(self.img_foreground_for_display, (x, y), 2, (255, 255, 255), -1)
            self.least_weathered_pixel.append([x, y])

    def mouse_select_weathering_region(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self._can_draw = True
            if [y, x] not in self.through_pixel:
                self.through_pixel.append([y, x])

        if event == cv2.EVENT_LBUTTONUP:
            self._can_draw = False

        if event == cv2.EVENT_MOUSEMOVE and self._can_draw:
            if self._can_draw and [y, x] not in self.through_pixel:
                self.through_pixel.append([y, x])


