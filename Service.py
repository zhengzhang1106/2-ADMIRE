import pandas as pd
import copy
import Database
import random

path1 = 'traffic_0916+0704_1.5.csv'
data = pd.read_csv(path1, engine='python')
service_src_all = list(data['src'])  # 所有源节点
service_dst_all = list(data['dst'])  # 所有目的节点

# 要获取不同时刻的流量，改usercols即可
service_traffic_all1 = pd.read_csv(path1, engine='python',
                                   usecols=list(range(3, Database.time + 3)))  # 记录所有业务的流量（多个时刻）
service_traffic_all1 = service_traffic_all1.values.tolist()

service_list = []  # 业务信息集合

service = {}  # 存储当前业务，字典，存储id，src，dest，traffic


# traffic_min_list = []  # 最小流量
# traffic_max_list = []  # 最大流量
# for i in range(len(service_traffic_all)):
#     traffic_min_list.append(min(service_traffic_all[i]))
#     traffic_max_list.append(max(service_traffic_all[i]))
# traffic_min = min(traffic_min_list)
# traffic_max = max(traffic_max_list)
# print(traffic_max)
# service示例
# service = {
#     'id': 1,
#     'src': 5,
#     'dest': 4,
#     'traffic': 0.5
# }
# 产生业务的时候就限制长度了，不需要time_update


def generate_service(start, end):
    service_list.clear()
    # ran_num = random.randint(0,1)
    # print("随机数",ran_num)
    # if ran_num == 0:
    traffic_list = service_traffic_all1
    # else:
    #     traffic_list = service_traffic_all2

    for i in range(Database.job_number):
        service['id'] = i
        service['src'] = service_src_all[i]
        service['dest'] = service_dst_all[i]
        service['traffic'] = traffic_list[i][start:end]  # 流量是一个1*24的向量
        service_list.append(service.copy())
        service.clear()
    # print(service_list[0])


if __name__ == '__main__':
    for i in range(10):
        generate_service(0, 24)
