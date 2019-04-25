import numpy as np
import cv2

class PixelFeatureVector:
    def __init__(self, input_img):

        # OpenCVはデフォルトではBGR色空間, 今後の処理の為にLab色空間に変えておく
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2Lab)
        self.luminance, self.chroma_a, self.chroma_b = cv2.split(input_img)

        # デフォルトでは[0, 255], [0.0, 1.0]へと正規化
        self.luminance = self.luminance / 255.0
        self.chroma_a = self.chroma_a / 255.0
        self.chroma_b = self.chroma_b / 255.0

        # 画素iの特徴ベクトルを計算する
        # feature_vector_i = [L_i/da, a_i/da, b_i/da, x_i/ds, y_i/ds]
        # L_i, a_i,　b_iはLab色空間の値
        # x_i, y_iは正規化画素座標 [-1.0, 1.0]
        # da=delta_a, ds=delta_sはappearance, spatialな局所性の影響を制御するパラメタ
        # 値は先行研究の観察に基づいて決めました
        self.height, self.width = input_img.shape[:2]
        self.feature_vector = np.zeros((5, self.height, self.width))
        delta_a = 0.2
        delta_s = 50
        self.feature_vector[0] = np.array([Li for Li in self.luminance]) / delta_a
        self.feature_vector[1] = np.array([ai for ai in self.chroma_a]) / delta_a
        self.feature_vector[2] = np.array([bi for bi in self.chroma_b]) / delta_a

        # 座標はデフォルトでは左上原点なので画像中央を原点にする
        half_width = self.width / 2
        half_height = self.height / 2
        normalize_x = (np.array([xi for xi in range(self.width)]) - half_width) / half_width
        normalize_y = (np.array([yi for yi in range(self.height)]) - half_height) / half_height
        self.feature_vector[3] = normalize_x[np.newaxis, :] / delta_s
        self.feature_vector[4] = normalize_y[:, np.newaxis] / delta_s

    # RadialBasisFunction(RBF)を計算する
    def computeRBF(self, xi, yi, xj, yj):
        pass


if __name__ == "__main__":
    test_img = cv2.imread("./InputImageSet/rust2_large.jpg")
    pixel_feature_vector = PixelFeatureVector(test_img)

