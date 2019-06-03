import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import maxflow
import cv2

# グレースケールの画像をある基準をもって二値化する
# 基準はsource_pixelとsink_pixelによって与えられる
# ex.)明確に風化したピクセル群と非風化のピクセル群を与えて全ピクセルを風化/非風化の二値にわける
def graph_cut(input_img, source_pixel, sink_pixel):
    # カラー画像が与えられた際はグレースケールに変換する
    if(np.array(input_img).ndim == 3):
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)


    graph, node_id = create_graph(input_img, source_pixel, sink_pixel)
    graph.maxflow()
    # 各ノードがどちらのセグメントに属しているかnode_idと同じシェイプの配列で返される
    which_segment = graph.get_grid_segments(node_id)
    # sourceに属しているノードのインデックスにはFalseが格納されているのでNOTをとる
    labeled_img = np.int_(np.logical_not(which_segment))

    return labeled_img.astype('uint8') * 255

# 画像の各ピクセルをノードとしたHxWの有向グラフを作成 H: 入力画像の縦の長さ, W: 横の長さ
# 各エッジの重みは頂点同士の画素値の差に依る
def create_graph(input_img, source_pixel, sink_pixel):

    graph = maxflow.Graph[float]()
    node_id = graph.add_grid_nodes(input_img.shape[:2])
    graph = add_nearest_neighbor_edge(input_img, graph, node_id, "vertical")
    graph = add_nearest_neighbor_edge(input_img, graph, node_id, "horizontal")

    graph = add_st_edge(input_img, graph, source_pixel, sink_pixel)
    return graph, node_id

# 4近傍とのエッジを追加する
def add_nearest_neighbor_edge(input_img, graph, node_id, edge_direction):

    pad_direction = np.ones((2, 2), dtype="int8") # 画像端を計算する際の調整用にパディング
    x_diff = y_diff = 0 # どの方向の隣接画素と計算するか決める
    structure = np.zeros((3, 3)) # どの方向にエッジを伸ばすか

    if edge_direction == "horizontal":
        # x方向にのみパディングする
        #pad_direction = [[0,0], [1,1]]
        x_diff = 1
        y_diff = 0
        structure = [[0, 0, 0],
                     [0, 0, 1],
                     [0, 0, 0]]
    elif edge_direction == "vertical":
        # y方向のみにパディングする
        #pad_direction = [[1,1], [0,0]]
        x_diff = 0
        y_diff = 1
        structure = [[0, 0, 0,
                      0, 0, 0,
                      0, 1, 0]]

    # 隣接画素との差分を用いてエッジの容量計算を行うため外枠を0で埋めた画像を生成
    pad_img = np.pad(input_img, pad_direction, 'constant', constant_values = 0)

    # パラメータ係数BETAのとLAMBDA決定
    diff_neighboring = np.array([[(pad_img[y, x] - pad_img[y+y_diff, x+x_diff])**2
                               for x in range(input_img.shape[1])]
                              for y in range(input_img.shape[0])])
    BETA = (2 * np.average(diff_neighboring)) ** -1
    LAMBDA = 1


    weigths = np.zeros((input_img.shape))
    weights = [[LAMBDA * np.exp((-BETA) * (pad_img[y, x] - pad_img[y+y_diff, x+x_diff])**2)
                for x in range(input_img.shape[1])]
               for y in range(input_img.shape[0])]

    graph.add_grid_edges(node_id, structure = structure, weights = weights, symmetric = True)

    return graph

# 2つのターミナル頂点とのエッジを追加する s: source, t: sink
def add_st_edge(input_img, graph, source_node, sink_node):
    # 各ノードとsourceノード, sinkノード間のエッジを作る
    for y in range(input_img.shape[0]):
        for x in range(input_img.shape[1]):
            node_id = y * input_img.shape[1] + x
            s_capacity = np.inf
            t_capacity = np.inf
            if input_img[y, x] != 0.0:
                t_capacity = -np.log(input_img[y, x])
            if input_img[y, x] != 1.0:
                s_capacity = -np.log(1 - input_img[y, x])
            #print(s_capacity, t_capacity)

            graph.add_tedge(node_id, s_capacity, t_capacity)

    # 既にsource, sinkラベルが与えられたものとは無限大容量のエッジを作る
    for x, y in source_node:
        node_id = y * input_img.shape[1] + x
        graph.add_tedge(node_id, np.inf, 0)

    for x, y in sink_node:
        node_id = y * input_img.shape[1] + x
        graph.add_tedge(node_id, 0, np.inf)

    return graph