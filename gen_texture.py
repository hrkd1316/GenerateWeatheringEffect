import cv2
import numpy as np
from random import randint

from patch_match import NNF

class TextureFromPatch(NNF):
    def __init__(self, texture_shape, basis_img, PATCH_SIZE):
        self.texture_img = np.zeros(texture_shape, np.uint8)
        self.basis_img = basis_img
        self.PATCH_SIZE = PATCH_SIZE
        self.OVERLAP_WIDTH = PATCH_SIZE // 4
        self.PATCH_INTERVAL = PATCH_SIZE - self.OVERLAP_WIDTH

        self.texture_height, self.texture_width = self.texture_img.shape[:2]
        super(TextureFromPatch, self).__init__(self.basis_img, self.texture_img, 60)

    def quilt_patch_random(self):
        print("random_quilt_start")
        prev_patch = self.choose_random_patch()
        self.texture_img[0:self.PATCH_SIZE, 0:self.PATCH_SIZE] = prev_patch
        cur_patch = prev_patch
        for y in range(0, self.texture_height-1, self.PATCH_INTERVAL):
            for x in range(0, self.texture_width-1, self.PATCH_INTERVAL):
                cur_patch = self.choose_random_patch()
                if y == 0:
                    if x != 0:
                        self.stitch_patch(y, x, cur_patch, prev_patch, True, False)
                else:
                    if x == 0:
                        prev_patch = self.stitch_patch(y, x, cur_patch, None, False, True)
                    else:
                        prev_patch = self.stitch_patch(y, x, cur_patch ,prev_patch, True, True)
                prev_patch = cur_patch
        cv2.imshow("randomquilt", self.texture_img)

    def quilt_patch_optimization(self):
        print("optimization_quilt_start")
        self.tgt_img = self.texture_img
        self.improve_nnf()
        print("improve_nnf_end")
        prev_patch = self.choose_nnf_patch(0, 0)
        for y in range(0, self.texture_height, self.PATCH_INTERVAL):
            for x in range(0, self.texture_width, self.PATCH_INTERVAL):
                cur_patch = self.choose_nnf_patch(y, x)
                if y == 0:
                    if x != 0:
                        self.stitch_patch(y, x, cur_patch, prev_patch, True, False)
                else:
                    if x == 0:
                        prev_patch = self.stitch_patch(y, x, cur_patch, None, False, True)
                    else:
                        prev_patch = self.stitch_patch(y, x, cur_patch ,prev_patch, True, True)
                prev_patch = cur_patch

    def stitch_patch(self, y, x, cur_patch, prev_patch, need_v_stitch, need_h_stitch):
        if need_v_stitch:
            ovlap_band_diff = self.calc_diff_vertical(prev_patch, cur_patch)
            ovlap_band_diff = cv2.cvtColor(ovlap_band_diff, cv2.COLOR_BGR2GRAY)
            seam_v = self.find_min_path_in_ovlap(ovlap_band_diff)
            for i in range(cur_patch.shape[0]):
                for j in range(int(seam_v[i]), cur_patch.shape[1]):
                    paste_tgt_y = y + i
                    paste_tgt_x = x + j
                    if paste_tgt_y < self.texture_height and paste_tgt_x < self.texture_img.shape[1]:
                        self.texture_img[paste_tgt_y, paste_tgt_x] = cur_patch[i, j]
            ovlap_band_diff = np.delete(np.delete(ovlap_band_diff, np.s_[:], 1), np.s_[:], 0)

        if need_h_stitch:
            upper_patch = self.texture_img[y - self.PATCH_INTERVAL:y + self.OVERLAP_WIDTH, x:x + self.PATCH_SIZE]
            ovlap_band_diff = self.calc_diff_horizontal(upper_patch, cur_patch)
            ovlap_band_diff = cv2.cvtColor(ovlap_band_diff, cv2.COLOR_BGR2GRAY)
            seam_h =self.find_min_path_in_ovlap(ovlap_band_diff.T)
            for j in range(upper_patch.shape[1]):
                for i in range(int(seam_h[j]), cur_patch.shape[0]):
                    paste_tgt_y = y + i
                    paste_tgt_x = x + j
                    if paste_tgt_y < self.texture_img.shape[0] and paste_tgt_x < self.texture_width:
                        self.texture_img[paste_tgt_y, paste_tgt_x] = cur_patch[i, j]

    def choose_random_patch(self):
        rand_patch_y = randint(0, self.basis_img.shape[0] - self.PATCH_SIZE)
        rand_patch_x = randint(0, self.basis_img.shape[1] - self.PATCH_SIZE)
        return self.basis_img[rand_patch_y:rand_patch_y+self.PATCH_SIZE, rand_patch_x:rand_patch_x+self.PATCH_SIZE, :]

    def choose_nnf_patch(self, y, x):
        nnf_y, nnf_x = self.nnf[y, x]
        rect_top = rect_left = 0
        if nnf_y + self.PATCH_SIZE > self.basis_img.shape[0]:
            rect_top = self.PATCH_SIZE - (self.basis_img.shape[0] - nnf_y)
        if nnf_x + self.PATCH_SIZE > self.basis_img.shape[1]:
            rect_left = self.PATCH_SIZE - (self.basis_img.shape[1] - nnf_x)
        print("y, x=", y, x)
        print("nnf_y, nnf_x=", nnf_y, nnf_x)
        print("shape", self.basis_img[nnf_y-rect_top:nnf_y-rect_top+self.PATCH_SIZE,
                              nnf_x-rect_left:nnf_x-rect_left+self.PATCH_SIZE].shape)
        return self.basis_img[nnf_y-rect_top:nnf_y-rect_top+self.PATCH_SIZE,
                              nnf_x-rect_left:nnf_x-rect_left+self.PATCH_SIZE]

    def calc_diff_vertical(self, prev_patch, cur_patch):
        prev_patch = np.array(prev_patch).astype(np.uint8)
        cur_patch = np.array(cur_patch).astype(np.uint8)
        # オーバーラップ部分の差分を計算
        v_diff = (prev_patch[:, -self.OVERLAP_WIDTH:] - cur_patch[:, :self.OVERLAP_WIDTH]) ** 2
        return v_diff

    def calc_diff_horizontal(self, upper_patch, cur_patch):
        upper_patch = np.array(upper_patch).astype(np.uint8)
        cur_patch = np.array(cur_patch).astype(np.uint8)
        # オーバーラップ部分の差分を計算
        h_diff = (upper_patch[-self.OVERLAP_WIDTH:, :] - cur_patch[:self.OVERLAP_WIDTH, :upper_patch.shape[1]]) ** 2
        return h_diff

    def find_min_path_in_ovlap(self, ovlap_band_diff):
        prev_cost, min_path = self.find_min_path_from_x(ovlap_band_diff, 0)
        for x in range(1, len(ovlap_band_diff[0])):
            cur_cost, temp_path = self.find_min_path_from_x(ovlap_band_diff, x)
            if cur_cost < prev_cost:
                min_path = temp_path
                prev_cost = cur_cost

        return min_path

    # x列目から始めるパスの中で最小コストとなるものを返す
    def find_min_path_from_x(self, ovlap_band_diff, x):
        path = np.zeros(len(ovlap_band_diff))
        path[0] = x
        cur_cost = ovlap_band_diff[0][x]
        for y in range(1, len(ovlap_band_diff)):
            x = int(path[y - 1])

            if x == 0:
                cur_cost = cur_cost + min(ovlap_band_diff[y][x], ovlap_band_diff[y][x + 1])
                if ovlap_band_diff[y][x] < ovlap_band_diff[y][x + 1]:
                    path[y] = x
                else:
                    path[y] = x +1
            elif x == ovlap_band_diff.shape[1] - 1:
                cur_cost = cur_cost + min(ovlap_band_diff[y][x], ovlap_band_diff[y][x - 1])
                if ovlap_band_diff[y][x] < ovlap_band_diff[y][x - 1]:
                    path[y] = x
                else:
                    path[y] = x - 1
            else:
                cur_cost = cur_cost + min(ovlap_band_diff[y][x - 1], ovlap_band_diff[y][x + 1])
                if ovlap_band_diff[y][x] <= ovlap_band_diff[y][x - 1]:
                    if ovlap_band_diff[y][x] <= ovlap_band_diff[y][x + 1]:
                        path[y] = x
                    else:
                        path[y] = x + 1
                else:
                    path[y] = x - 1
        return cur_cost, path


if __name__ == '__main__':
    input = cv2.imread("./InputImageSet/tgt_img1_large.jpg")
    exampler = cv2.imread("./OutputImageSet/rust2_large_weathering_exampler.png")

    gen_texture_ = TextureFromPatch(input.shape, exampler, 60)
    gen_texture_.quilt_patch_random()
    gen_texture_.quilt_patch_optimization()
    gen_texture_.PATCH_SIZE = 20
    gen_texture_.OVERLAP_WIDTH = gen_texture_.PATCH_SIZE // 4
    gen_texture_.PATCH_INTERVAL = gen_texture_.PATCH_SIZE - gen_texture_.OVERLAP_WIDTH
    gen_texture_.quilt_patch_optimization()
    while (True):
        cv2.imshow("optimization", gen_texture_.texture_img)
        key = cv2.waitKey(1) & 0xFF
        # quit
        if key == ord("q"):
            break
    cv2.destroyAllWindows()