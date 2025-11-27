import gym
from gym.spaces import Box
from gym.spaces import Dict
import numpy as np
import Database
import RWA
import Service
import AuGraph


class AuGraphEnv(gym.Env):
    """
        自定义环境必须继承自gym.Env，并实现reset和step方法
        在方法__init__中，必须带第二个参数，用于传递envconfig
    """
    lightpath_cumulate = 0  # 当前光路数
    au_edge_list = []   # 存储RWA结果

    # 初始化
    def __init__(self, env_config):
        self.action_space = Box(low=np.zeros(5), high=Database.weight_max * np.ones(5), dtype=np.int32)  # 5个连续动作空间，0-255，以5增长
        self.observation_space = Dict({
            'phylink': Box(low=-1*np.ones([Database.wavelength_number * Database.time, Database.node_number, Database.node_number]),
                           high=np.array(Database.clear(Database.links_physical)), dtype=np.float32),  # 物理链路9*9矩阵

            # 业务id,[0]
            'request_index': Box(low=np.array([0]), high=np.array([Database.job_number-1]), shape=(1,), dtype=np.int32),
            # 业务源、目的节点
            'request_src': Box(low=np.array([0]), high=np.array([Database.node_number-1]), shape=(1,), dtype=np.int32),
            'request_dest': Box(low=np.array([0]), high=np.array([Database.node_number-1]), shape=(1,), dtype=np.int32),
            # 业务流量
            'request_traffic': Box(low=np.zeros(Database.time), high=Database.wavelength_capacity * np.ones(Database.time), dtype=np.float32)
        })
        # self.init()
        self.reset()

    # 还原环境
    def reset(self):
        """
        每完成一个episode,环境重新初始化
        :return: 返回环境初始化状态
        """
        print("reset")
        Service.generate_service(0, Database.time)  # 产生业务
        links = Database.clear(Database.links_physical)  # 清空物理链路
        AuGraph.links_virtual_list.clear()      # 清空虚拟链路
        index = 0
        src, dest, traffic = self.find_req_info(index)
        self.observation = {
            'phylink': np.array(links, dtype=np.float32),
            'request_index': np.array([index], dtype=np.int32),
            'request_src': np.array([src], dtype=np.int32),
            'request_dest': np.array([dest], dtype=np.int32),
            'request_traffic': np.array(traffic, dtype=np.float32)
        }
        self.done = False
        self.step_num = 0
        AuGraphEnv.lightpath_cumulate = 0
        AuGraphEnv.job_success = 0
        AuGraphEnv.au_edge_list.clear()
        return self.observation


    # 步骤，更新环境，设置奖励
    # 所选动作作用于环境后环境返回的奖励和下一步状态，并判断是否达到终止状态
    # :param action:输入是动作的序号
    # :return:输出是：下一步状态，立即回报，是否终止，调试项
    def step(self, action) -> tuple:
        self.step_num += 1  # step数量加1
        # print('step_num', self.step_num)
        # print("action", action)
        action_t = action * 1000     # 对输出的动作取绝对值
        # print("action_new", action_t)
        request_index_current = self.observation['request_index'][0]  # 当前业务索引
        # print("index",request_index_current)
        # 需要将动作参数（辅助图权重）传入辅助图初始化，然后进行路由，计算物理链路剩余带宽，作为奖励
        if request_index_current == 0:
            AuGraph.au_graph_init(action_t)  # 初始化权重
            flag, lightpath_num, au_edge_collection = RWA.route_wave_assign(action_t,request_index_current)  # flag表示是否选路成功，wave_used表示使用的波长（用于计算奖励）
        else:
            AuGraph.update_au_graph_weight(action_t)  # 用动作更新权重
            flag, lightpath_num, au_edge_collection = RWA.route_wave_assign(action_t, request_index_current)

        if flag:
            if request_index_current == Database.job_number - 1:  # 所有业务都部署完成，结束此次迭代
                self.done = True
                request_index = request_index_current
            else:
                request_index = request_index_current + 1  # 业务往后一个

            request_src, request_dest, request_traffic = self.find_req_info(request_index)
            self.observation = {
                'phylink': np.array(Database.links_physical, dtype=np.float32),
                'request_index': np.array([request_index], dtype=np.int32),
                'request_src': np.array([request_src], dtype=np.int32),
                'request_dest': np.array([request_dest], dtype=np.int32),
                'request_traffic': np.array(request_traffic, dtype=np.float32)
            }
            reward = lightpath_num * (-1)
            AuGraphEnv.lightpath_cumulate += lightpath_num
            print('id', request_index_current, 'weight', action_t, "lightpath_cum", AuGraphEnv.lightpath_cumulate, "lightpath_cur", lightpath_num, "reward", reward)
        else:  # 选路失败不更新物理网络状态和业务
            if request_index_current == Database.job_number - 1:  # 所有业务都部署完成，结束此次迭代
                self.done = True
                request_index = request_index_current
            else:
                request_index = request_index_current + 1

            request_src, request_dest, request_traffic = self.find_req_info(request_index)
            self.observation = {
                'phylink': np.array(Database.links_physical, dtype=np.float32),
                'request_index': np.array([request_index], dtype=np.int32),
                'request_src': np.array([request_src], dtype=np.int32),
                'request_dest': np.array([request_dest], dtype=np.int32),
                'request_traffic': np.array(request_traffic, dtype=np.float32)
            }
            reward = -10
            print('id', request_index_current, 'weight', action_t, "lightpath_cum", AuGraphEnv.lightpath_cumulate, "lightpath_cur", lightpath_num, "reward", reward)

        AuGraphEnv.au_edge_list = au_edge_collection
        return self.observation, reward, self.done, {}  # 最后的return全返回了，如果没到最左边或者最右边，self.done=False

    def render(self, mode='human'):
        pass

    def find_req_info(self, index):
        src = Service.service_list[index]['src']
        dest = Service.service_list[index]['dest']
        traffic = Service.service_list[index]['traffic']
        # print('id',Service.service_list[index]['id'],' src',src,' ,dest',dest,' traffic',traffic[0])
        return src, dest, traffic

