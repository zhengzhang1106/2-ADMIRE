import numpy as np
import Database

Inf = float('inf')  # 无穷大
node = Database.node_number
layer = Database.wavelength_number + 2  # 辅助图层数,5
au_size = node * layer * 2  # 矩阵大小,90

lightpath_new = 1  # 代表新建光路
lightpath_old = 2  # 代表已建光路

au_graph_weight = Inf * np.ones((au_size, au_size), dtype=int)  # 辅助图权重,90行90列
used_count = np.zeros((au_size, au_size), dtype=int)  # 记录发射器边和接收器边的使用次数（波长层节点下标）
au_node_degree = np.zeros(node, dtype=int)  # 用于记录节点的度

save_addEdge = []  # 记录对于当前请求，添加的辅助图的边的集合，每次用完记得清空
links_virtual_list = []  # 虚拟链路集合，虚拟链路是一个字典，存储src，dest，route，capacity，wavelength_id
# links_virtua示例
# links_virtual = {
#     'src': 5,
#     'dest': 4,
#     'route':[5,4],
#     'capacity':[5.5,4.5],
#     'wavelength_id':1
# }

# 各边的权重：0、疏导边；1、光路边；2、发射机边；3、接收机边；4、波长链路边；5、复用器边；6、解复用器边；7、波长旁路边
# edge_weight = [Database.GrmE_weight, Database.LPE_weight, Database.TxE_weight, Database.RxE_weight, Database.WLE_weight, Database.MuxE_weight,Database.DeMuxE_weight]

# 让发射机边和接收机边权重相同，复用器边和接收器边相同
# 各边的权重：0、疏导边；1、光路边；2、发射机边/接收机边；3、波长链路边；4、复用器边/解复用器边；5、波长旁路边；
edge_weight = [Database.GrmE_weight, Database.LPE_weight, Database.TxE_weight, Database.WLE_weight, Database.MuxE_weight]


# 辅助图权重初始化
def au_graph_init(weight):
    # 对角线元素为0
    for i in range(au_size):
        for j in range(au_size):
            if i == j:
                au_graph_weight[i][j] = 0

    row, col = Database.graph_connect.shape
    single_num = layer * 2  # 一个节点的辅助图节点数,5*2=10
    # 初始化疏导边、复用器边、解复用器边、发射机边、接收机边权重
    for i in range(row):  # i表示节点
        au_graph_weight[i * single_num + 1][i * single_num] = weight[4]  # 解复用器边,4,6
        au_graph_weight[i * single_num + layer][i * single_num + layer + 1] = weight[4]  # 复用器边,4,5
        au_graph_weight[i * single_num][i * single_num + layer] = weight[0]  # 疏导边,0,0
        # au_graph_weight[i * single_num + 1][i * single_num] = Database.DeMuxE_weight  # 解复用器边,4,6
        # au_graph_weight[i * single_num + layer][i * single_num + layer + 1] = Database.MuxE_weight  # 复用器边,4,5

        count = 2
        while count < layer:
            au_graph_weight[i * single_num + count][i * single_num + count + layer] = Database.WBE_weight  # 波长旁路边,5,7
            au_graph_weight[i * single_num + count][i * single_num] = weight[2]  # 接收机边,2,3
            au_graph_weight[i * single_num + layer][i * single_num + layer + count] = weight[2]  # 发射机边,2,2
            count += 1

    # 初始化波长链路边权重
    # in：i*layer*2+count out:i*layer*2+layer+count
    for i in range(row):
        for j in range(col):
            if Database.graph_connect[i][j] == 1:
                au_node_degree[i] += 1

                count = 2
                while count < layer:
                    # 智能
                    au_graph_weight[i * single_num + layer + count][j * single_num + count] = weight[3] + count * 200  # 波长链路边,3,4,i=0,j=1,7->12,
                    # 传统
                    # au_graph_weight[i * single_num + layer + count][j * single_num + count] = weight[3]  # 波长链路边,3,i=0,j=1,7->12,
                    count += 1
    return au_graph_weight

# 更新辅助图权重
# 因为使用的波长链路边的权重会变为无穷
def update_au_graph_weight(weight):
    row, col = Database.graph_connect.shape
    single_num = layer * 2  # 一个节点的辅助图节点数,5*2=10
    # 初始化疏导边、复用器边、解复用器边、发射机边、接收机边权重
    for i in range(row):  # i表示节点
        au_graph_weight[i * single_num][i * single_num + layer] = weight[0]  # 疏导边,0,0
        au_graph_weight[i * single_num + 1][i * single_num] = weight[4]  # 解复用器边,4,6
        au_graph_weight[i * single_num + layer][i * single_num + layer + 1] = weight[4]  # 复用器边,4,5
        # au_graph_weight[i * single_num + 1][i * single_num] = Database.DeMuxE_weight  # 解复用器边,4,6
        # au_graph_weight[i * single_num + layer][i * single_num + layer + 1] = Database.MuxE_weight  # 复用器边,4,5

        count = 2
        while count < layer:
            au_graph_weight[i * single_num + count][i * single_num + count + layer] = Database.WBE_weight  # 波长旁路边,5,7
            if au_graph_weight[i * single_num + count][i * single_num] != Inf:
                au_graph_weight[i * single_num + count][i * single_num] = weight[2]  # 接收机边,2,3
            if au_graph_weight[i * single_num + layer][i * single_num + layer + count] != Inf:
                au_graph_weight[i * single_num + layer][i * single_num + layer + count] = weight[2]  # 发射机边,2,2
            count += 1

    # 初始化波长链路边权重
    # in：i*layer*2+count out:i*layer*2+layer+count
    for i in range(row):
        for j in range(col):
            if Database.graph_connect[i][j] == 1:
                count = 2
                while count < layer:
                    # 波长链路边,3,4,i=0,j=1,7->12,
                    if au_graph_weight[i * single_num + layer + count][j * single_num + count] != Inf:
                        # 智能
                        au_graph_weight[i * single_num + layer + count][j * single_num + count] = weight[3] + count * 200
                        # 传统
                        # au_graph_weight[i * single_num + layer + count][j * single_num + count] = weight[3]
                    count += 1
    return au_graph_weight


# 若虚拟链路容量满足，则在辅助图光路层上添加对应的光路边
# ser_traffic和virtual_links的capacity都是数组
def add_edge(virtual_links, ser_traffic, weight):
    if virtual_links:  # 若虚拟链路不为空
        for cur_vir_link in virtual_links:
            flag = True
            for index in range(len(ser_traffic)):
                if cur_vir_link['capacity'][index] < ser_traffic[index]:
                    flag = False
                    break

            if flag:
                src = cur_vir_link['src']  # 物理源节点
                dest = cur_vir_link['dest']  # 物理目的节点
                # dest_in：dest*layer*2+1 src_out:src*layer*2+layer+1
                au_graph_weight[src * layer * 2 + layer + 1][dest * layer * 2 + 1] = weight[1]  # 设置光路边权重，源节点的out到目的节点的in
                save_addEdge.append([src * layer * 2 + layer + 1, dest * layer * 2 + 1])
        # print("添加边：", save_addEdge)


# 选路完成后删除添加的光路边，相当于每次添加光路边之前，原始的辅助图上都没有光路边
def delete_edge():
    if save_addEdge:
        for edge in save_addEdge:
            src = edge[0]
            dest = edge[1]
            au_graph_weight[src][dest] = Inf
        save_addEdge.clear()


# 更新辅助图
# service是字典
def update_au_graph(au_edge_list, service):
    continue_index = -1
    for i in range(len(au_edge_list)):
        if i > continue_index:
            cur_au_edge = au_edge_list[i]  # 当前边的属性，字典
            cur_edge_att = cur_au_edge['attribute']
            # 疏导边、复用器边、解复用器边、波长旁路边
            if cur_edge_att == 0 or cur_edge_att == 5 or cur_edge_att == 6 or cur_edge_att == 7:
                continue

            elif cur_edge_att == 2 or cur_edge_att == 3:  # 发射机边、接收机边
                used_count[cur_au_edge['src_vir']][cur_au_edge['dest_vir']] += 1  # 边使用次数+1
                # cur_au_edge['used_count'] = used_count[cur_au_edge['src_vir']][cur_au_edge['dest_vir']]
                if used_count[cur_au_edge['src_vir']][cur_au_edge['dest_vir']] == au_node_degree[
                    cur_au_edge['dest_phy']]:  # 如果边使用次数到达上限（节点的度），则删除边
                    au_graph_weight[cur_au_edge['src_vir']][cur_au_edge['dest_vir']] = Inf

            elif cur_edge_att == 1:  # 光路边
                src = cur_au_edge['src_phy']
                dest = cur_au_edge['dest_phy']
                cur_au_edge['lightpath_attribute'] = lightpath_old
                cur_wavelength_id = -1;  # 波长从0开始，若波长为-1说明出错
                cur_route = []  # 路由

                # 找第一条可用波长（第一条可用的寻找顺序是按照光路的建立顺序进行寻找）
                for j in range(len(links_virtual_list)):
                    cur_vir_link = links_virtual_list[j]  # 获取当前虚拟链路
                    flag = True
                    if cur_vir_link['src'] == src and cur_vir_link['dest'] == dest:
                        for index in range(len(service['traffic'])):
                            if cur_vir_link['capacity'][index] < service['traffic'][index]:
                                flag = False
                                break
                        if flag:  # 找到第一个容量满足的虚拟链路
                            for index in range(len(service['traffic'])):
                                cur_vir_link['capacity'][index] -= service['traffic'][index]
                            cur_route = cur_vir_link['route']
                            cur_wavelength_id = cur_vir_link['wavelength_id']
                            break
                cur_au_edge['wavelength_id'] = cur_wavelength_id
                if len(cur_route) == 0:
                    print(service['id'], "没找到波长！")

                # 更新物理链路容量（多个时刻）
                for j in range(len(cur_route) - 1):
                    for index in range(len(service['traffic'])):
                        Database.links_physical[cur_wavelength_id + index * Database.wavelength_number][cur_route[j]][
                            cur_route[j + 1]] -= service['traffic'][index]

            elif cur_edge_att == 4:  # 波长链路边

                # 使用单条波长链路边（下一条边不是波长链路边），新建光路需要以下几步：
                # 1: 判断是否需要新建光路（这一步暂时可以不做，直接覆盖之前的值，因为目前权重认为权重不变，容量是剩余中最大的）
                # 2：更新虚拟链路links_virtual的容量
                # 3：更新物理链路links_physical的容量（矩阵）
                # 4: 删除使用的波长链路边（辅助图权重设为inf，容量设为0）
                # 5：记录AuEgde使用的波长id
                # 6：设置光路边类型为新建光路
                if au_edge_list[i + 1]['attribute'] == 3:
                    src_phy = cur_au_edge['src_phy']
                    dest_phy = cur_au_edge['dest_phy']
                    src_vir = cur_au_edge['src_vir']
                    dest_vir = cur_au_edge['dest_vir']
                    wavelength_id = src_vir % layer - 2  # 波长从0开始

                    # 步骤2
                    # 记录剩余波长
                    wave_cap_left = []
                    for index in range(len(service['traffic'])):
                        wave_cap_left.append(Database.wavelength_capacity - service['traffic'][index])
                    vir_link_tmp = {
                        'src': src_phy,
                        'dest': dest_phy,
                        'route': [src_phy, dest_phy],
                        'capacity': wave_cap_left,
                        'wavelength_id': wavelength_id
                    }
                    links_virtual_list.append(vir_link_tmp.copy())
                    vir_link_tmp.clear()
                    # 步骤3
                    for index in range(len(service['traffic'])):
                        Database.links_physical[wavelength_id + index * Database.wavelength_number][src_phy][
                            dest_phy] -= service['traffic'][index]
                    # 步骤4
                    au_graph_weight[src_vir][dest_vir] = Inf
                    # 步骤5
                    cur_au_edge['wavelength_id'] = wavelength_id
                    # 步骤6
                    cur_au_edge['lightpath_attribute'] = lightpath_new  # 新建光路

                # 使用多条波长链路边（下一条边还是波长链路边）新建光路需要以下几步：
                # 1: 找到连续的波长链路边集合
                # 2: 记录新建光路边的物理 / 虚拟源目的节点及波长
                # 3: 判断是否需要新建光路（这一步暂时可以不做，直接覆盖之前的值，因为目前权重认为权重不变，容量是剩余中最大的）
                # 4：更新虚拟链路links_virtual的容量
                # for循环
                # 5：更新物理链路links_physical的容量（矩阵）
                # 6: 删除使用的所有的波长链路边（辅助图权重设为inf，容量设为0）
                # 7：记录AuEgde使用的波长id
                # Tips: 在这个过程中可以设置新建光路和已经光路类型!!
                elif au_edge_list[i + 1]['attribute'] == 7:
                    # 步骤1
                    WLE_list = []  # 存储波长链路边集合
                    for j in range(i, len(au_edge_list)):
                        if au_edge_list[j]['attribute'] == 4:
                            WLE_list.append(au_edge_list[j])
                        elif au_edge_list[j]['attribute'] == 7:
                            continue
                        else:
                            continue_index = j - 1  # 从最后一个波长链路边的下一条边开始遍历
                            break  # 跳出for循环
                    # 步骤2
                    src_lpe_phy = WLE_list[0]['src_phy']
                    dest_lpe_phy = WLE_list[len(WLE_list) - 1]['dest_phy']
                    wavelength_id = (WLE_list[0]['src_vir'] % layer) - 2

                    # 步骤4
                    # 记录物理节点上的路由
                    cur_route_lpe = [src_lpe_phy]
                    for j in range(len(WLE_list)):
                        cur_route_lpe.append(WLE_list[j]['dest_phy'])
                    # 记录链路剩余容量
                    wave_cap_left = []
                    for index in range(len(service['traffic'])):
                        wave_cap_left.append(Database.wavelength_capacity - service['traffic'][index])
                    vir_link_tmp_lpe = {
                        'src': src_lpe_phy,
                        'dest': dest_lpe_phy,
                        'route': cur_route_lpe,
                        'capacity': wave_cap_left,
                        'wavelength_id': wavelength_id
                    }
                    links_virtual_list.append(vir_link_tmp_lpe.copy())
                    vir_link_tmp_lpe.clear()

                    # 步骤5、6、7
                    for j in range(len(WLE_list)):
                        src_vir_tmp = WLE_list[j]['src_vir']
                        dest_vir_tmp = WLE_list[j]['dest_vir']
                        src_phy_tmp = WLE_list[j]['src_phy']
                        dest_phy_tmp = WLE_list[j]['dest_phy']

                        # 步骤5
                        for index in range(len(service['traffic'])):
                            Database.links_physical[wavelength_id + index * Database.wavelength_number][src_phy_tmp][
                                dest_phy_tmp] -= service['traffic'][index]

                        # 步骤6
                        au_graph_weight[src_vir_tmp][dest_vir_tmp] = Inf

                        # 步骤7
                        WLE_list[j]['wavelength_id'] = wavelength_id
                        WLE_list[j]['lightpath_attribute'] = lightpath_new

            else:
                print("边类型出错")
            # i = len(WLE_list) * 2 - 1

    return au_edge_list





# 根据路由得到边类型
# au_edge示例
# au_edge = {
#     'src_vir': 10,
#     'dest_vir': 15,
#     'src_phy': 2,
#     'dest_phy': 2,
#     'attribute': 0,
#     'lightpath_attribute': 1,
#     'wavelength_id': 1,
#     'used_count': 0   # 可以不要
# }
def edge_convert(route_path):
    au_edge_list = []  # 当前路径边属性的集合
    for i in range(len(route_path) - 1):
        au_edge = {}
        au_edge['src_vir'] = route_path[i]  # 第i条边的虚拟源节点
        au_edge['dest_vir'] = route_path[i + 1]  # 第i条边的虚拟目的节点
        au_edge['src_phy'] = int(route_path[i] / (layer * 2))  # 第i条边的物理源节点)
        au_edge['dest_phy'] = int(route_path[i + 1] / (layer * 2))  # 第i条边的物理目的节点
        au_edge['attribute'] = compute_attribute(au_edge)  # 边属性
        au_edge_list.append(au_edge.copy())
        au_edge.clear()
    return au_edge_list


# 计算边的属性
def compute_attribute(au_edge):
    attribute = -1
    # 对于一条边，前一个节点是a，后一个节点是b
    src_vir = au_edge['src_vir']  # a的虚拟节点
    dest_vir = au_edge['dest_vir']  # b的虚拟节点
    src_phy = au_edge['src_phy']  # a的物理节点
    dest_phy = au_edge['dest_phy']  # b的物理节点
    src_layer = src_vir % layer  # a在辅助图上的层数
    dest_layer = dest_vir % layer  # b在辅助图上的层数

    # 各边的权重：0、疏导边；1、光路边；2、发射机边；3、接收机边；
    # 4、波长链路边；5、复用器边；6、解复用器边；7、波长旁路边
    if dest_vir - src_vir == 1:
        attribute = 5  # 复用器边
    elif src_vir - dest_vir == 1:
        attribute = 6  # 解复用器边
    elif abs(src_vir - dest_vir) > 1:
        if src_layer == dest_layer:
            if src_layer == 0:
                attribute = 0  # 疏导边
            elif src_layer == 1:
                attribute = 1  # 光路边
            elif src_layer > 1:
                if src_phy == dest_phy:
                    attribute = 7  # 波长旁路边
                else:
                    attribute = 4  # 波长链路边
        else:
            if src_phy == dest_phy:
                if src_layer == 0 and dest_layer > 1:
                    attribute = 2  # 发射机边
                elif dest_layer == 0 and src_layer > 1:
                    attribute = 3  # 接收机边
    return attribute

# if __name__ == '__main__':
# au_graph_init(edge_weight)
# print(au_graph_weight)
# add_edge()

# route = [5,8,13,18,23,20]
# result = edge_convert(route)
# for i in range(len(result)):
#     print(result[i])
