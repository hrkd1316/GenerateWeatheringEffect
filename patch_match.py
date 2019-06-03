import cv2
import numpy as np

# NNF: Nearest Neighbor Field(最近傍フィールド)
class NNF:
    def __init__(self, src_img, tgt_img, BOXSIZE):
        self.src_img = src_img
        self.tgt_img = tgt_img
        self.BOXSIZE = BOXSIZE // 2

        # 3つのチャンネルでNNFを構成する
        self.nnf = np.zeros((2, self.tgt_img.shape[0], self.tgt_img.shape[1])).astype(np.int)
        self.nnf_dist = np.zeros(self.tgt_img.shape[:2])
        self.initialize_nnf()

    def initialize_nnf(self):
        self.nnf[0] = np.random.randint(self.src_img.shape[0], size = self.tgt_img.shape[:2])
        self.nnf[1] = np.random.randint(self.src_img.shape[1], size = self.tgt_img.shape[:2])
        self.nnf = self.nnf.transpose((1, 2, 0))
        self.nnf_dist = np.array([[self.calc_dist(y, x, self.nnf[y, x, 0], self.nnf[y, x, 1])
                                   for x in range(self.tgt_img.shape[1])]
                                  for y in range(self.tgt_img.shape[0])])

    def calc_dist(self, tgt_y, tgt_x, src_y, src_x):
        # 計算する矩形領域の左上の点と右上の点を決定する
        rect_top = rect_left = self.BOXSIZE // 2
        rect_bottom = rect_right = self.BOXSIZE // 2 + 1

        rect_top = min(tgt_y, src_y, rect_top)
        rect_left = min(tgt_x, src_x, rect_left)
        rect_bottom = min(self.src_img.shape[0]-src_y, self.tgt_img.shape[0]-tgt_y, rect_bottom)
        rect_right = min(self.src_img.shape[1]-src_x, self.tgt_img.shape[1]-tgt_x, rect_right)

        return np.sum((self.src_img[src_y-rect_top:src_y+rect_bottom, src_x-rect_left:src_x+rect_right]
                       - self.tgt_img[tgt_y-rect_top:tgt_y+rect_bottom, tgt_x-rect_left:tgt_x+rect_right])**2) \
               / (rect_left + rect_right) / (rect_top + rect_bottom)

    def improve_nnf(self, total_iter = 3):
        [[[self.find_best_match(y, x) for x in range(self.tgt_img.shape[1])]
                                      for y in range(self.tgt_img.shape[0])]
                                      for iter in range(total_iter)]

    def find_best_match(self, y, x):
        best_y, best_x, best_dist = self.nnf[y, x, 0], self.nnf[y, x, 1], self.nnf_dist[y, x]
        best_y, best_x, best_dist = self.propagation(y, x, best_y, best_x, best_dist)
        best_y, best_x, best_dist = self.random_search(y, x, best_y, best_x, best_dist)
        self.nnf[y, x] = [best_y, best_x]
        self.nnf_dist[y, x] = best_dist
        print("y, x", y, x)

    # todo: 何かもうちょっと効率的にできないものか
    # (y,x)からshift_pixelだけずらした点でのnnfを見る
    # (y,x)+shift_pixelの評価値を見て新たな点の方がよければ(y,x)のnnfを変更する
    def propagation(self, y, x, best_y, best_x, best_dist):

        for i in reversed((range(4))):
            shift_pixel = 2 ** i
            if y - shift_pixel >= 0:
                candidate_y, candidate_x = self.nnf[y-shift_pixel, x, 0]+shift_pixel, self.nnf[y-shift_pixel, x, 1]
                if candidate_y < self.src_img.shape[0]:
                    candidate_dist = self.calc_dist(y, x, candidate_y, candidate_x)
                    if candidate_dist < best_dist:
                        best_y, best_x, best_dist = candidate_y, candidate_x, candidate_dist
            
            if x - shift_pixel >= 0:
                candidate_y, candidate_x = self.nnf[y, x-shift_pixel, 0], self.nnf[y, x-shift_pixel, 1]+shift_pixel
                if candidate_x < self.src_img.shape[1]:
                    candidate_dist = self.calc_dist(y, x, candidate_y, candidate_x)
                    if candidate_dist < best_dist:
                        best_y, best_x, best_dist = candidate_y, candidate_x, candidate_dist

            if y + shift_pixel < self.src_img.shape[0]:
                candidate_y, candidate_x = self.nnf[y+shift_pixel, x, 0]-shift_pixel, self.nnf[y+shift_pixel, x, 1]
                if candidate_y >= 0:
                    candidate_dist = self.calc_dist(y, x, candidate_y, candidate_x)
                    if candidate_dist < best_dist:
                        best_y, best_x, best_dist = candidate_y, candidate_x, candidate_dist

            if x + shift_pixel < self.src_img.shape[1]:
                candidate_y, candidate_x = self.nnf[y, x+shift_pixel, 0], self.nnf[y, x+shift_pixel, 1]-shift_pixel
                if candidate_x >= 0:
                    candidate_dist = self.calc_dist(y, x, candidate_y, candidate_x)
                    if candidate_dist < best_dist:
                        best_y, best_x, best_dist = candidate_y, candidate_x, candidate_dist

        return best_y, best_x, best_dist

    def random_search(self, y, x, best_y, best_x, best_dist):
        rand_shift = min(self.src_img.shape[0]//2, self.src_img.shape[1]//2)
        while rand_shift > 0:
            try:
                min_y = max(best_y - rand_shift, 0)
                max_y = min(best_y + rand_shift, self.src_img.shape[0])
                min_x = max(best_x - rand_shift, 0)
                max_x = min(best_x + rand_shift, self.src_img.shape[1])
                candidate_y = np.random.randint(min_y, max_y)
                candidate_x = np.random.randint(min_x, max_x)
                candidate_dist = self.calc_dist(y, x, candidate_y, candidate_x)
                if candidate_dist < best_dist:
                    best_y, best_x, best_dist = candidate_y, candidate_x, candidate_x

            except:
                print("=============Exception===============")
                print("y, x=", y, x)
                print("rand_d",rand_shift)
                print("miny, maxy", min_y, max_y)
                print("minx, maxx", min_x, max_x)
                print("besty, bestx", best_y, best_x)
                print("bestd", best_dist)

            rand_shift = rand_shift // 2

        return best_y, best_x, best_dist

    def reconstruct_img(self):
        self.improve_nnf(total_iter=5)

        output_img = np.array([[self.src_img[self.nnf[y, x, 0], self.nnf[y, x, 1]]
                                for x in range(self.tgt_img.shape[1])]
                               for y in range(self.tgt_img.shape[0])])

        return output_img

if __name__ == '__main__':
    patch = cv2.imread("./OutputImageSet/rust2_large_most_weathered_rect.png")
    exampler = cv2.imread("./OutputImageSet/testimg2.png")

    patch_match = NNF(patch, exampler, 5)
    T = patch_match.reconstruct_img()
    while(True):
        cv2.imshow("tgt", patch)
        cv2.imshow("ref", exampler)
        cv2.imshow("fill", T)
        key = cv2.waitKey(1) & 0xFF
        # quit
        if key == ord("q"):
          break
    cv2.destroyAllWindows()