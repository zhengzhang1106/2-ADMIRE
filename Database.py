import numpy as np

# 网络和业务信息
job_number = 200  # 业务数量
wavelength_number = 3  # 波长数量
wavelength_capacity = 10  # 单波长容量
node_number = 9  # 节点个数
link_number = 12  # 网络中总的链路数
wavelength_number_all = link_number * wavelength_number * 2  # 72

time = 24  # 所有时刻数量，目前认为是24小时

# data_path = 'traffic/Traffic0717.csv'   # 只有一个时刻
# data_path = 'traffic/Traffic0822.csv'     # 30个训练集（不再使用）

data_path0 = 'traffic/traffic_2020-09-14.csv'
data_path1 = 'traffic/traffic_2020-09-16.csv'
data_path2 = 'traffic/traffic_2020-09-17.csv'
data_path3 = 'traffic/traffic_2020-09-18.csv'
data_path4 = 'traffic/traffic_2020-09-20.csv'
data_path5 = 'traffic/traffic_2020-09-21.csv'
data_path6 = 'traffic/traffic_2020-09-22.csv'
data_path7 = 'traffic/traffic_2020-09-23.csv'
data_path8 = 'traffic/traffic_2020-09-24.csv'
data_path9 = 'traffic/traffic_2020-09-25.csv'
data_path10 = 'traffic/traffic_2020-09-26.csv'
data_path11 = 'traffic/traffic_2020-09-27.csv'
data_path12 = 'traffic/traffic_2020-09-29.csv'
data_path13 = 'traffic/traffic_2020-10-01.csv'
data_path14 = 'traffic/traffic_2020-10-02.csv'
data_path15 = 'traffic/traffic_2020-10-03.csv'
data_path16 = 'traffic/traffic_2020-10-05.csv'
data_path17 = 'traffic/traffic_2020-10-08.csv'
data_path18 = 'traffic/traffic_2020-10-10.csv'
data_path19 = 'traffic/traffic_2020-10-12.csv'
data_path20 = 'traffic/traffic_2020-10-13.csv'
data_path21 = 'traffic/Traffic0704.csv'   # 100个测试集


# data_path = 'traffic/traffic_2020-10-14.csv'
# data_path = 'traffic/Traffic0704_double.csv'
# data_path = 'traffic/traffic_2020-09-14_double.csv'
# data_path = 'traffic/traffic_2020-10-12_double.csv'
# data_path = 'traffic/traffic_2020-10-13_double.csv'

data_path_list = [data_path0,data_path1,data_path2,data_path3,data_path4,data_path5,data_path6,data_path7,
                  data_path8,data_path9,data_path10,data_path11,data_path12,data_path13, data_path14,
                  data_path15,data_path16,data_path17,data_path18,data_path19,data_path20,data_path21]

# 随机产生的业务源（测试集）
data_path_random6 = 'random/traffic_random6.csv'
data_path_random26 = 'random/traffic_random26.csv'


data_path_random_list = {6: data_path_random6, 26: data_path_random26,}


# 最小化新建光路数(MinLP):减少新建光路数量（光路数量最少）
GrmE_weight = 20
LPE_weight = 1
TxE_weight = 200
RxE_weight = 200
WLE_weight = 10
MuxE_weight = 0
DeMuxE_weight = 0
WBE_weight = 0


# 让DRL的动作空间在[-1,1]范围内
weight_max = 1  # 强化学习辅助图权重上限
weight_min = -1  # 强化学习辅助图权重上限
# increase = 5    # 权重增长步长

# 物理拓扑连接关系
graph_connect = np.array([[0, 1, 1, 0, 1, 0, 0, 0, 0],
                          [1, 0, 0, 1, 0, 0, 0, 0, 0],
                          [1, 0, 0, 1, 0, 1, 0, 0, 0],
                          [0, 1, 1, 0, 0, 0, 1, 0, 0],
                          [1, 0, 0, 0, 0, 1, 0, 1, 0],
                          [0, 0, 1, 0, 1, 0, 0, 0, 1],
                          [0, 0, 0, 1, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 1, 1, 1, 0]], dtype=int)

# 物理链路及波长，考虑24小时，所以是72*9*9
links_physical = np.zeros((time*wavelength_number, node_number, node_number))


# wave = links_physical.shape[0]  # 维度
# row = links_physical.shape[1]  # 行
# col = links_physical.shape[2]  # 列
wave, row, col = links_physical.shape
# print('wave',wave)
for i in range(row):
    for j in range(col):
        if graph_connect[i][j] == 1:
            for k in range(wave):
                links_physical[k][i][j] = wavelength_capacity
        else:
            for k in range(wave):
                links_physical[k][i][j] = -1.


# print(links_physical)


def clear(links):
    for i in range(row):
        for j in range(col):
            if graph_connect[i][j] == 1:
                for k in range(wave):
                    links[k][i][j] = wavelength_capacity
            else:
                for k in range(wave):
                    links[k][i][j] = -1.
    # print(links)
    return links


if __name__ == '__main__':
    clear(links_physical)
    print(wavelength_number_all)
    print(links_physical)
