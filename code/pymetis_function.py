import networkx as nx
import matplotlib.pyplot as plt
import pymetis
print(dir(pymetis)) # nested_dissection part_graph part_mesh verify_nd
# 创建一个图
G = nx.Graph()
edges = [
    (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 6),(6,7)
]
G.add_edges_from(edges)

# 获取图的节点数
num_vertices = len(G.nodes())
print(f"num_vertices={num_vertices}")
# 将图的邻接字典转换为邻接列表和邻接索引列表
adjacency = [list(G.neighbors(n)) for n in range(1, num_vertices + 1)]
print(f"adjacency={adjacency}")
xadj = list(range(1, num_vertices + 1))

# 使用 pymetis 进行图划分
n_cuts, membership = pymetis.part_graph(2, xadj=xadj, adjncy=adjacency)
print(f"n_cuts={n_cuts},membership={membership}")
# 可视化图划分结果
pos = nx.spring_layout(G)  # 使用spring布局并设置随机种子以确保可重复性
colors = ['red' if member == 0 else 'blue' for member in membership]
nx.draw(G, pos, node_color=colors, with_labels=True, edge_color='gray', node_size=500, alpha=0.7)
plt.title("Graph Partitioning")
plt.axis('off')
plt.show()

# 打印最大割的两个子集
print("Set A:", [i for i, member in enumerate(membership) if member == 0])
print("Set B:", [i for i, member in enumerate(membership) if member == 1])