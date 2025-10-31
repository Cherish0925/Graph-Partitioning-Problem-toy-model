import numpy as np
import pymetis
import networkx as nx
import matplotlib.pyplot as plt

# 无向图：图的邻接列表，它是一个列表的列表，其中每个元素都是一个数组，代表图中一个顶点的邻接顶点
adjacency_list = [np.array([4, 2, 1]), # 顶点 0 与顶点 1, 2, 4 相连
                  np.array([0, 2, 3]), # 顶点 1 与顶点 0, 2, 3 相连
                  np.array([4, 3, 1, 0]), # 顶点 2 与顶点 0, 1, 3, 4 相连
                  np.array([1, 2, 5, 6]), # 顶点 3 与顶点 1, 2, 5, 6 相连
                  np.array([0, 2, 5]), # 顶点 4 与顶点 0, 2, 5 相连
                  np.array([4, 3, 6]), # 顶点 5 与顶点 4, 3, 6 相连
                  np.array([5, 3])] # 顶点 6 与顶点 3, 5 相连


# 可视化图划分结果
# 创建图
G = nx.Graph()
# 先添加所有节点
for i in range(len(adjacency_list)):
    G.add_node(i)

for i in range(len(adjacency_list)):
    for j in adjacency_list[i]:
        print(f"G.add_edges_from({[(i, j)]})")
        G.add_edges_from([(i, j)]) #  2-tuples (u, v) or 3-tuples (u, v, d)

# 可视化原始图
plt.figure(figsize=(8, 6))
plt.subplot(121)
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, alpha=0.7)
plt.title("Original Unweighted Undirected Graph")
plt.axis('off')


n_cuts, membership = pymetis.part_graph(2, adjacency=adjacency_list)
# n_cuts = 3
# membership = [1, 1, 1, 0, 1, 0, 0]
print(f"n_cuts={n_cuts}")
print(f"membership={membership}")

nodes_part_0 = np.argwhere(np.array(membership) == 0).ravel() # [3, 5, 6]
nodes_part_1 = np.argwhere(np.array(membership) == 1).ravel() # [0, 1, 2, 4]
print(f"nodes_part_0={nodes_part_0}")
print(f"nodes_part_1={nodes_part_1}")

# 可视化图划分结果
colors = ['red' if member == 0 else 'blue' for member in membership]
print(f"colors={colors}")
# 打印图的节点索引
print("Node indices in the graph:", list(G.nodes()))
plt.subplot(122)
nx.draw(G, node_color=colors, with_labels=True, edge_color='gray', node_size=500, alpha=0.7)
plt.title("After Graph Partitioning")
plt.axis('off')

plt.tight_layout()
plt.show()


