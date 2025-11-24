inf = float('inf')


def dijkstra(matrix, source, destination,id):
    Min = 1E100
    n = len(matrix)
    m = len(matrix[0])
    if source >= n or n != m:
        print('Error!')
        return
    found = [source]  # 已找到最短路径的节点
    cost = [Min] * n  # source到已找到最短路径的节点的最短距离
    cost[source] = 0
    path = [[]] * n  # source到其他节点的最短路径
    path[source] = [source]
    while len(found) < n:  # 当已找到最短路径的节点小于n时
        min_value = Min + 1
        col = -1
        row = -1
        for f in found:  # 以已找到最短路径的节点所在行为搜索对象
            for i in [x for x in range(n) if x not in found]:  # 只搜索没找出最短路径的列
                if matrix[f][i] + cost[f] < min_value:  # 找出最小值
                    min_value = matrix[f][i] + cost[f]  # 在某行找到最小值要加上source到该行的最短路径
                    row = f  # 记录所在行列
                    col = i
        if col == -1 or row == -1:  # 若没找出最小值且节点还未找完，说明图中存在不连通的节点
            break
        found.append(col)  # 在found中添加已找到的节点
        cost[col] = min_value  # source到该节点的最短距离即为min_value
        path[col] = path[row][:]  # 复制source到已找到节点的上一节点的路径
        path[col].append(col)  # 再其后添加已找到节点即为source到该节点的最短路径
    # print("id", id, "路径", path[destination])
    return path[destination]


if __name__ == '__main__':
    graph = [[0.00000, inf, inf, 5.00000, inf, inf, inf, inf, inf, inf, inf, inf],
             [0.00000, 0.00000, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
             [50.00000, inf, 0.00000, inf, inf, 0.00000, inf, inf, inf, inf, inf, inf],
             [inf, inf, inf, 0.00000, 0.00000, 50.00000, inf, inf, inf, inf, inf, inf],
             [inf, inf, inf, inf, 0.00000, inf, inf, inf, inf, inf, inf, inf],
             [inf, inf, inf, inf, inf, 0.00000, inf, inf, 1000.00000, inf, inf, inf],
             [inf, inf, inf, inf, inf, inf, 0.00000, inf, inf, 5.00000, inf, inf],
             [inf, inf, inf, inf, inf, inf, 0.00000, 0.00000, inf, inf, inf, inf],
             [inf, inf, inf, inf, inf, inf, 50.00000, inf, 0.00000, inf, inf, 0.00000],
             [inf, inf, inf, inf, inf, inf, inf, inf, inf, 0.00000, 0.00000, 50.00000],
             [inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 0.00000, inf],
             [inf, inf, 1000.00000, inf, inf, inf, inf, inf, inf, inf, inf, 0.00000]]
    path = dijkstra(graph, 9, 1)
    print(path)
