"""
    参数为三个一维 list : xadj adjncy eweights
    而 xadj 列表则用于指示每个顶点对应的邻接顶点列表在 adjncy 中的起始和结束位置。
    adjncy 列表存储的是所有顶点的邻接顶点的索引，
    eweights 列表与 adjncy 列表一一对应，表示每条边的权重。
"""


import numpy as np
import pymetis
import networkx as nx
import matplotlib.pyplot as plt
# 定义图的邻接结构和边权重
# xadj 表示每个顶点的起始索引和结束索引
# xadj = np.array([0, 3, 6, 10, 11, 14, 16], dtype=np.int32)
# xadj=[0,2,4,6,8]
# adjncy 表示所有顶点的邻接顶点的索引
# adjncy = np.array([1, 2, 4,0, 2, 3,0, 1, 3, 5, 6, 2, 4, 3, 5, 6], dtype=np.int32)
# adjncy=[1,2,0,3,0,3,1,2]
# 定义边权重
# eweights = np.array([3, 2, 1, 3, 4, 5, 2, 4, 7, 6, 5, 7, 8, 9, 10, 11], dtype=np.int32)
# eweights=[1,5,1,5,5,1,5,1]
# 设置随机种子以确保结果可复现
# np.random.seed(42)

num_vertices = 100

# 初始化邻接矩阵为全零矩阵
adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

# 计算右上三角部分的索引
mask = np.triu(np.ones((num_vertices, num_vertices), dtype=bool), k=1)

# 计算右上三角部分的总元素数
total_upper_triangle = np.count_nonzero(mask)

# 计算需要设置为1的元素数量（稀疏程度为0.05）
num_ones = int(total_upper_triangle * 0.05)+3
print(f"num_ones={num_ones}")
# 随机选择要设置为1的索引
upper_triangle_indices = np.argwhere(mask)
random_indices = np.random.choice(total_upper_triangle, num_ones, replace=False)
# 随机生成权重（均匀分布在1到100之间）
wmax = 100
random_weights = np.random.randint(1, wmax + 1, size=num_ones)
# 设置右上三角部分的元素为1
# 设置右上三角部分的元素为随机权重
for idx, weight in zip(random_indices, random_weights):
    row, col = upper_triangle_indices[idx]
    adj_matrix[row, col] = weight

# 使矩阵对称（左下三角部分复制右上三角部分）
adj_matrix = adj_matrix + adj_matrix.T

# 将对角线设置为0
np.fill_diagonal(adj_matrix, 0)
print(f"np.mean(adj_matrix)={np.mean(adj_matrix)},np.sum(adj_matrix)={np.sum(adj_matrix)}")
# 可视化邻接矩阵
# plt.figure(figsize=(10, 8))
cax = plt.matshow(adj_matrix, cmap='Blues')  # 使用Blues颜色映射，0为白色，10为深蓝色
plt.colorbar(cax, fraction=0.046, pad=0.04)  # 添加颜色条
plt.title("Adjacency Matrix", fontsize=15)
plt.xlabel("Node Index")
plt.ylabel("Node Index")

adjacency=adj_matrix
adj_list = []
for i in range(num_vertices):
    neighbors = np.where(adjacency[i] == 1)[0].tolist()
    adj_list.append(neighbors)

# 打印邻接列表
print("Adjacency List:")
for i, neighbors in enumerate(adj_list):
    print(f"{i}: {neighbors}")
# 使用 nested_dissection 函数计算填充减少排序 填充减少排序

perm = pymetis.nested_dissection(adj_list)
dissection=perm[0]
adj_permuted = adjacency.copy()
# 根据 perm 数组重新排列邻接矩阵
# 根据顶点的新排列顺序更新邻接矩阵
for i in range(num_vertices):
    for j in range(num_vertices):
        # 找到原始顶点索引
        original_i = dissection.index(i)
        original_j = dissection.index(j)
        # 根据原始顶点索引更新新邻接矩阵
        adj_permuted[i, j] = adjacency[original_i, original_j]

cax = plt.matshow(adj_permuted, cmap='Blues')  # 使用Blues颜色映射，0为白色，10为深蓝色
plt.colorbar(cax, fraction=0.046, pad=0.04)  # 添加颜色条
plt.title("Permuted Adjacency Matrix", fontsize=15)
plt.xlabel("Node Index")
plt.ylabel("Node Index")
# adj_matrix=adj_permuted

# 将邻接矩阵转换为xadj, adjncy, eweights格式
adjncy = []
eweights = []
for i in range(num_vertices):
    for j in range(num_vertices):
        if adj_matrix[i][j] != 0:
            adjncy.append(j)
            eweights.append(adj_matrix[i][j])

# 计算xadj
xadj = [0]
count = 0
for i in range(num_vertices):
    count += np.count_nonzero(adj_matrix[i])
    xadj.append(count)
xadj = np.array(xadj, dtype=np.int32)
adjncy = np.array(adjncy, dtype=np.int32)
eweights = np.array(eweights, dtype=np.int32)

for i in range(0,len(xadj)-1):  # 这个是节点总数目
    for j in range(0,xadj[i+1]-xadj[i]):
        print(f"从节点{i}到节点{adjncy[xadj[i]+j]}的边权重为{eweights[xadj[i]+j]}")



# 定义顶点权重（如果需要的话）
vweights = None  # 无顶点权重，如果需要可以定义为一个数组
# 划分图
nparts = 2  # 将图划分为两个分区
n_cuts, membership = pymetis.part_graph(
    nparts, xadj=xadj, adjncy=adjncy, eweights=eweights)

print("n_cuts:", n_cuts)
print("membership:", membership)

# 创建图
G = nx.Graph()

# 添加节点
num_vertices = len(xadj) - 1
for i in range(num_vertices):
    G.add_node(i)

# 添加边及其权重
for i in range(num_vertices):
    start = xadj[i]
    end = xadj[i + 1]
    for j in range(start, end):
        neighbor = adjncy[j]
        weight = eweights[j]
        G.add_edge(i, neighbor, weight=weight)

# 设置边的粗细（基于权重）
edge_widths = [G[u][v]['weight'] *2 for u, v in G.edges()]
# 根据分区结果设置节点颜色
partition_colors = ['red' if part == 0 else 'blue' for part in membership]
# 可视化图
pos = nx.shell_layout(G)  # 使用 shell 布局
plt.figure(figsize=(10, 8))

# # 绘制节点
# nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, alpha=0.7)
# 绘制节点，应用分区颜色
nx.draw_networkx_nodes(G, pos, node_color=partition_colors, node_size=800, alpha=0.7)

# 绘制边
edge_colors = []
for u, v in G.edges():
    if membership[u] == membership[v]:
        # 同一类别内部的边颜色
        if membership[u] == 0:
            edge_colors.append('red')
        else:
            edge_colors.append('blue')
    else:
        # 不同类别之间的边颜色
        edge_colors.append('green')

nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)

# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# 绘制权重标签
edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='red')

plt.title("Weighted Undiredted Graph after Partitioning", fontsize=15)
plt.axis('off')
plt.show()

