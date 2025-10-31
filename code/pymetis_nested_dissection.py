import numpy as np
import matplotlib.pyplot as plt
import pymetis

# 创建一个随机的邻接矩阵
num_vertices = 20

# 生成一个随机的邻接矩阵
adjacency = np.random.randint(0, 2, size=(num_vertices, num_vertices))

# 确保邻接矩阵是对称的
adjacency = (adjacency + adjacency.T) // 2  # 将矩阵和其转置相加，然后除以2，结果向下取整

# 将对角线元素设置为0，因为对角线不表示边
np.fill_diagonal(adjacency, 0)

# adjacency 是一个二维 NumPy 数组，其中 adjacency[i][j] 为 1 表示顶点 i 和顶点 j 之间存在边，为 0 表示不存在边。
# 将邻接矩阵转换为Metis需要的格式
xadj = np.zeros(num_vertices + 1, dtype=int)

adjncy = []
for i in range(num_vertices):
    adjncy.extend(np.where(adjacency[i] == 1)[0].tolist())
    xadj[i + 1] = len(adjncy)

print(f"adjacency={adjacency}")
# 将邻接矩阵转换为邻接列表
num_vertices = adjacency.shape[0]
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
# 函数的输入是一个邻接矩阵，可以是 Python 风格的邻接列表或者 C 风格的 xadj 和 adjncy 数组。输出是一个排列数组 perm，它表示顶点的新顺序。
print(f"perm={perm}")
dissection=perm[0]
print(dissection)
# 第一个数组（顶点排列数组）：
# 这个数组表示图的顶点经过填充减少排序（fill-reducing ordering）后的新顺序。

# 可视化原始邻接矩阵
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(adjacency, cmap='binary', interpolation='none')
plt.title('Original Adjacency Matrix')
plt.colorbar()
plt.xticks([])
plt.yticks([])
# plt.tight_layout()
# plt.show()
#
# # 可视化排序后的邻接矩阵
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


# 可视化排序后的邻接矩阵
plt.subplot(122)
plt.imshow(adj_permuted, cmap='binary', interpolation='none')
plt.title('Permuted Adjacency Matrix')
plt.colorbar()
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
