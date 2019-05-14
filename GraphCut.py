import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import maxflow
import cv2

# グレースケールの画像をある基準をもって二値化する
# 基準はsource_pixelとsink_pixelによって与えられる
# ex.)明確に風化したピクセルと非風化のピクセルを与えて全ピクセルを風化/非風化の二値にわける
def graph_cut(input_img, source_pixel, sink_pixel):
    # カラー画像が与えられた際はグレースケールに変換する
    if(np.array(input_img).ndim == 3):
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    graph, node_id = create_graph(input_img, source_pixel, sink_pixel)
    graph.maxflow()
    """
    nx_graph = graph.get_nx_graph()
    print("get_nxgraph done")
    plot_graph(nx_graph)
    print("plot done")
    """
    which_segment = graph.get_grid_segments(node_id)
    binary_img = np.int_(np.logical_not(which_segment))

    return binary_img.astype('uint8') * 255
    
# 画像の各ピクセルをノードとしたHxWの有向グラフを作成 H: 入力画像の縦の長さ, W: 横の長さ
# 各エッジの重みは頂点同士の画素値の差に依る
def create_graph(input_img, source_pixel, sink_pixel):
    graph = maxflow.Graph[float]()
    node_id = graph.add_grid_nodes((input_img.shape[:2]))
    graph = add_nearest_neighbor_edge(input_img, graph, node_id, "horizontal")
    graph = add_nearest_neighbor_edge(input_img, graph, node_id, "vertical")
    graph = add_st_edge(input_img, graph, source_pixel, sink_pixel)

    return graph, node_id

# 4近傍とのエッジを追加する
def add_nearest_neighbor_edge(input_img, graph, node_id, edge_direction):
    # パラメータ係数
    LAMBDA = 1
    COEFFICIENT_K = 1

    pad_direction = np.zeros((2, 2)) # 画像端を計算する際の調整用にパディング
    x_difference = y_difference = 0 # どの方向の隣接画素と計算するか決める
    structure = np.zeros((3, 3)) # どの方向にエッジを伸ばすか

    if edge_direction == "horizontal":
        # x方向にのみパディングする
        pad_direction = [[0,0], [1,1]]
        x_difference = 1
        y_difference = 0
        structure = [[0, 0, 0],
                     [0, 0, 1],
                     [0, 0, 0]]
    elif edge_direction == "vertical":
        # y方向のみにパディングする
        pad_direction = [[1,1], [0,0]]
        x_difference = 0
        y_difference = 1
        structure = [[0, 0, 0,
                      0, 0, 0,
                      0, 1, 0]]

    pad_img = np.pad(input_img, pad_direction, 'constant', constant_values = 0)
    weigths = np.zeros((input_img.shape))
    weights = [[LAMBDA * np.exp((-COEFFICIENT_K) * abs(pad_img[y, x] - pad_img[y+y_difference, x+x_difference]))
                for x in range(input_img.shape[1])]
               for y in range(input_img.shape[0])]
    np.set_printoptions(threshold=np.inf)
    print(np.array(weights).shape)
    graph.add_grid_edges(node_id, structure = structure, weights = weights, symmetric = True)
    return graph

# 2つのターミナル頂点とのエッジを追加する s: source, t: sink
def add_st_edge(input_img, graph, source_node, sink_node):
    for x, y in source_node:
        node_id_connect_source = y * input_img.shape[1] + x
        graph.add_tedge(node_id_connect_source, np.inf, 0)

    for x, y in sink_node:
        node_id_connect_sink = y * input_img.shape[0] + x
        graph.add_tedge(node_id_connect_sink, 0, np.inf)

    return graph

def plot_graph(nx_graph):
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(nx_graph)
    nx.draw_networkx(nx_graph, pos)

    plt.axis("off")
    plt.savefig("./OutputImageSet/test.png")
    plt.show()

