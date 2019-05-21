import cv2
import numpy as np

# NNF: Nearest Neighbor Field(最近傍フィールド)
class NNF:
    def __init__(self, ref_img, tgt_img, BOXSIZE):
        self.ref_img = ref_img
        self.tgt_img = tgt_img
        self.BOXSIZE = BOXSIZE // 2

        # 3つのチャンネルでNNFを構成する
        self.nnf = np.zeros((2, self.ref_img.shape[0], self.ref_img.shape[1])).astype(np.int)
        self.nnf_dist = np.zeros(self.ref_img.shape[:2])
        self.initialize_nnf()

    def initialize_nnf(self):
        self.nnf[0] = np.random.randint(self.tgt_img.shape[0], size = self.ref_img.shape[:2])
        self.nnf[1] = np.random.randint(self.tgt_img.shape[1], size = self.ref_img.shape[:2])
        self.nnf = self.nnf.transpose((1, 2, 0))
        self.nnf_dist = np.array([[self.calc_dist(y, x, self.nnf[y, x, 0], self.nnf[y, x, 1])
                                   for x in range(self.ref_img.shape[1])]
                                  for y in range(self.ref_img.shape[0])])

    def calc_dist(self, yi, xi, yj, xj):
        # 計算する矩形領域の左上の点と右上の点を決定する
        rect_top = rect_left = self.BOXSIZE // 2
        rect_bottom = rect_right = self.BOXSIZE // 2 + 1

        rect_top = min(yi, yj, rect_top)
        rect_left = min(xi, xj, rect_left)
        rect_bottom = min(self.ref_img.shape[0]-yi, self.tgt_img.shape[0]-yj, rect_bottom)
        rect_right = min(self.ref_img.shape[1]-xi, self.tgt_img.shape[1]-xj, rect_right)
        #print("-----------------------------------")
        #print("yi, yj, rect_top=", yi, yj, rect_top)
        #print("xi, xj, rect_left=", xi, xj, rect_left)
        #print("yi-, yj-, rect_bottom=", self.ref_img.shape[0]-yi, self.tgt_img.shape[0]-yj, rect_bottom)
        #print("xi-, xj-, rect_right=", self.ref_img.shape[1]-xi, self.tgt_img.shape[1]-xj, rect_right)

        return np.sum((self.ref_img[yi-rect_top:yi+rect_bottom, xi-rect_left:xi+rect_right]
                       - self.tgt_img[yj-rect_top:yj+rect_bottom, xj-rect_left:xj+rect_right])**2) \
               / (rect_left + rect_right) / (rect_top + rect_bottom)

    def improve_nnf(self, total_iter):
        for iter in range(total_iter):
            for y in range(self.ref_img.shape[0]):
                for x in range(self.ref_img.shape[1]):
                    best_y, best_x, best_dist = self.nnf[y, x, 0], self.nnf[y, x, 1], self.nnf_dist[y, x]
                    print("------------initial------------")
                    print("y, x", y, x)
                    print("besty_nnf, bestx_nnf", self.nnf[best_y, best_x, 0], self.nnf[best_y, best_x, 1])
                    best_y, best_x, best_dist = self.propagation(y, x, best_y, best_x, best_dist)
                    best_y, best_x, best_dist = self.random_search(y, x, best_y, best_x, best_dist)
                    self.nnf[y, x] = [best_y, best_x]
                    self.nnf_dist[y, x] = best_dist
                    print("-----------after---------------")
                    print("besty_nnf, bestx_nnf", self.nnf[best_y, best_x, 0], self.nnf[best_y, best_x, 1])
    # todo: 何かもうちょっと効率的にできないものか
    # (y,x)からshift_pixelだけずらした点でのnnfを見る
    # (y,x)+shift_pixelの評価値を見て新たな点の方がよければ(y,x)のnnfを変更する
    def propagation(self, y, x, best_y, best_x, best_dist):
        print("-------propagation--------")
        print("before best_y, best_x, best_dist", best_y, best_x, best_dist)
        for i in reversed((range(4))):
            shift_pixel = 2 ** i
            if y - shift_pixel >= 0:
                candidate_y, candidate_x = self.nnf[y-shift_pixel, x, 0]+shift_pixel, self.nnf[y-shift_pixel, x, 1]
                if candidate_y < self.tgt_img.shape[0]:
                    candidate_dist = self.calc_dist(y, x, candidate_y, candidate_x)
                    if candidate_dist < best_dist:
                        print("change y-shift")
                        best_y, best_x, best_dist = candidate_y, candidate_x, candidate_dist
            
            if x - shift_pixel >= 0:
                candidate_y, candidate_x = self.nnf[y, x-shift_pixel, 0], self.nnf[y, x-shift_pixel, 1]+shift_pixel
                if candidate_x < self.tgt_img.shape[1]:
                    candidate_dist = self.calc_dist(y, x, candidate_y, candidate_x)
                    if candidate_dist < best_dist:
                        print("change x-shift")
                        best_y, best_x, best_dist = candidate_y, candidate_x, candidate_dist

            if y + shift_pixel < self.ref_img.shape[0]:
                candidate_y, candidate_x = self.nnf[y+shift_pixel, x, 0]-shift_pixel, self.nnf[y+shift_pixel, x, 1]
                if candidate_y >= 0:
                    candidate_dist = self.calc_dist(y, x, candidate_y, candidate_x)
                    if candidate_dist < best_dist:
                        print("change y+shift")
                        best_y, best_x, best_dist = candidate_y, candidate_x, candidate_dist

            if x + shift_pixel < self.ref_img.shape[1]:
                candidate_y, candidate_x = self.nnf[y, x+shift_pixel, 0], self.nnf[y, x+shift_pixel, 1]-shift_pixel
                if candidate_x >= 0:
                    candidate_dist = self.calc_dist(y, x, candidate_y, candidate_x)
                    if candidate_dist < best_dist:
                        print("change x+shift")
                        best_y, best_x, best_dist = candidate_y, candidate_x, candidate_dist
        if best_dist != 0:
            print("after best_y, best_x, best_dist", best_y, best_x, best_dist)
            print("besty_nnf, bestx_nnf", self.nnf[best_y, best_x, 0], self.nnf[best_y, best_x, 1])
        return best_y, best_x, best_dist

    def random_search(self, y, x, best_y, best_x, best_dist):
        rand_shift = min(self.tgt_img.shape[0]//2, self.tgt_img.shape[1]//2)
        while rand_shift > 0:
            try:
                min_y = max(best_y - rand_shift, 0)
                max_y = min(best_y + rand_shift, self.tgt_img.shape[0])
                min_x = max(best_x - rand_shift, 0)
                max_x = min(best_x + rand_shift, self.tgt_img.shape[1])

                candidate_y = np.random.randint(min_y, max_y)
                candidate_x = np.random.randint(min_x, max_x)
                candidate_dist = self.calc_dist(y, x, candidate_y, candidate_x)
                if candidate_dist < best_dist:
                    best_y, best_x, best_dist = candidate_y, candidate_x, candidate_x
                    print("------------random search----------")
                    print("besty, bestx, bestd", best_y, best_x, best_dist)

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

        result_img = np.array([[self.tgt_img[self.nnf[y, x, 0], self.nnf[y, x, 1]]
                                for x in range(self.ref_img.shape[1])]
                               for y in range(self.ref_img.shape[0])])

        return result_img

if __name__ == '__main__':
    patch = cv2.imread("./OutputImageSet/rust2_large_most_weathered_rect.png")
    exampler = cv2.imread("./OutputImageSet/testimg1.png")

    patch_match = NNF(patch, exampler, 7)
    T = patch_match.reconstruct_img()
    while(True):
        cv2.imshow("fill", T)
        key = cv2.waitKey(1) & 0xFF
        # quit
        if key == ord("q"):
          break
    cv2.destroyAllWindows()