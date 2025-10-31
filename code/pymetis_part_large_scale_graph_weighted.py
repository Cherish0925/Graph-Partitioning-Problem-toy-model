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

# 生成20x20的随机对称邻接矩阵
# np.random.seed(42)  # 设置随机种子以确保结果可复现
num_vertices = 12
adj_matrix = np.random.randint(0, 11, size=(num_vertices, num_vertices))  # 随机生成0到10之间的整数
# 使矩阵对称
adj_matrix = np.tril(adj_matrix) + np.tril(adj_matrix, -1).T
# 将对角线设置为0
np.fill_diagonal(adj_matrix, 0)
# 可视化邻接矩阵
# plt.figure(figsize=(10, 8))
cax = plt.matshow(adj_matrix, cmap='Blues')  # 使用Blues颜色映射，0为白色，10为深蓝色
plt.colorbar(cax, fraction=0.046, pad=0.04)  # 添加颜色条
plt.title("Adjacency Matrix", fontsize=15)
plt.xlabel("Node Index")
plt.ylabel("Node Index")





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
nparts = 3  # 将图划分为两个分区
n_cuts, membership = pymetis.part_graph(
    nparts, xadj=xadj, adjncy=adjncy, eweights=eweights)



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
edge_widths = [G[u][v]['weight'] * 0.55 for u, v in G.edges()]

# 根据分区结果设置节点颜色
partition_colors = []
for part in membership:
    if part == 0:
        partition_colors.append('red')
    elif part == 1:
        partition_colors.append('blue')
    else:
        partition_colors.append('green')
# 可视化图
pos = nx.shell_layout(G)  # 使用 shell 布局
plt.figure(figsize=(10, 8))

# # 绘制节点
# nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, alpha=0.7)
# 绘制节点，应用分区颜色
nx.draw_networkx_nodes(G, pos, node_color=partition_colors, node_size=800, alpha=0.7)
# 绘制边
edge_colors = []
internal_weights = {0: 0, 1: 0, 2: 0}  # 用于存储每个分区内部边的权重总和

for u, v in G.edges():
    if membership[u] == membership[v]:
        part = membership[u]
        # 同一类别内部的边颜色
        if part == 0:
            edge_colors.append('red')
        elif part == 1:
            edge_colors.append('blue')
        else:
            edge_colors.append('green')
    else:
        # 不同类别之间的边颜色
        edge_colors.append('black')

# 使用集合避免重复计数边
edge_set = set()
internal_edges = {0: 0, 1: 0, 2: 0}    # 用于存储每个分区内部的边数
between_edges = 0                        # 用于存储不同分区之间的边数
for u, v in G.edges():
    # 确保每条边只处理一次
    if (u, v) not in edge_set and (v, u) not in edge_set:
        edge_set.add((u, v))
        if membership[u] == membership[v]:
            part = membership[u]
            print(f"G[u][v]['weight']={G[u][v]['weight']}")
            internal_weights[part] += G[u][v]['weight']
            internal_edges[part] += 1
        else:
            between_edges += 1

print(f"一共{len(xadj)-1}个节点，{int(len(adjncy)/2)}条边，总权重为{int(np.sum(eweights)/2)}")
print(f"一共{len(xadj)-1}个节点，分类结果为membership:", membership)
print(f"划分 0 中所有边的内部权重为: {internal_weights[0]}，边数为: {internal_edges[0]}")
print(f"划分 1 中所有边的内部权重为: {internal_weights[1]}，边数为: {internal_edges[1]}")
print(f"划分 2 中所有边的内部权重为: {internal_weights[2]}，边数为: {internal_edges[2]}")
print(f"不同分区之间被切割掉的边的总权重为: {n_cuts}， 边数为: {between_edges}")

# 绘制边
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)

# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

# 绘制权重标签
# edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='red')

plt.title("Weighted Undiredted Graph after Partitioning", fontsize=15)
plt.axis('off')
plt.show()

