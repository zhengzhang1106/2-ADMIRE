import RWA
import Service
import AuGraph
import numpy as np
import Database

result_rwa_phy = []  # 记录各业务物理网络RWA结果
result_rwa_vir = []  # 记录各业务虚拟网络RWA结果
result_odl = []  # 记录各业务ODL结果

light_node_config = 1  # 只配光节点
electric_node_config = 2  # 只配电节点
li_ele_node_config = 3  # 配光+电节点
none_node_config = 4  # 无需配置


def odl_result(au_edges_list):
    result_rwa_phy.clear()
    result_rwa_vir.clear()
    result_odl.clear()
    # au_edges_list = RWA.route_wave_assign(AuGraph.edge_weight)
    for i in range(len(au_edges_list)):
        cur_traffic_res = au_edges_list[i]
        cur_service = Service.service_list[i]
        if len(cur_traffic_res) != 0:
            # ----------物理拓扑RWA结果----------
            res_tmp_rwa_phy = []
            res_tmp_rwa_phy.append(cur_service['id'])
            for j in range(len(cur_traffic_res)):
                cur_traffic_au_edge = cur_traffic_res[j]
                if cur_traffic_au_edge['attribute'] == 4:
                    res_tmp_rwa_phy.append(cur_traffic_au_edge['src_phy'])
                    res_tmp_rwa_phy.append(cur_traffic_au_edge['wavelength_id'])
                if cur_traffic_au_edge['attribute'] == 1:
                    for vir_link in AuGraph.links_virtual_list:
                        if cur_traffic_au_edge['src_phy'] == vir_link['src'] and \
                                cur_traffic_au_edge['dest_phy'] == vir_link['dest'] and \
                                cur_traffic_au_edge['wavelength_id'] == vir_link['wavelength_id']:
                            for r in range(len(vir_link['route']) - 1):
                                res_tmp_rwa_phy.append(vir_link['route'][r])
                                res_tmp_rwa_phy.append(vir_link['wavelength_id'])
            res_tmp_rwa_phy.append(cur_service['dest'])  # 添加目的节点
            result_rwa_phy.append(res_tmp_rwa_phy.copy())

            # ----------虚拟拓扑RWA结果----------
            res_tmp_rwa_vir = []
            res_tmp_rwa_vir.append(cur_service['id'])
            continue_id = -1
            for j in range(0, len(cur_traffic_res)):
                if j > continue_id:
                    cur_traffic_au_edge = cur_traffic_res[j]
                    if cur_traffic_au_edge['attribute'] == 4:
                        if cur_traffic_res[j + 1]['attribute'] == 3:
                            res_tmp_rwa_vir.append(cur_traffic_au_edge['src_phy'])
                            res_tmp_rwa_vir.append(cur_traffic_au_edge['wavelength_id'])
                        elif cur_traffic_res[j + 1]['attribute'] == 7:
                            wle_list = []
                            for m in range(j, len(cur_traffic_res)):
                                if cur_traffic_res[m]['attribute'] == 4:
                                    wle_list.append(cur_traffic_res[m])
                                elif cur_traffic_res[m]['attribute'] == 7:
                                    continue
                                else:
                                    continue_id = m - 1  # 从最后一个波长链路边的下一条边开始遍历
                                    break
                            res_tmp_rwa_vir.append(wle_list[0]['src_phy'])
                            res_tmp_rwa_vir.append(wle_list[0]['wavelength_id'])

                    if cur_traffic_au_edge['attribute'] == 1:
                        res_tmp_rwa_vir.append(cur_traffic_au_edge['src_phy'])
                        res_tmp_rwa_vir.append(cur_traffic_au_edge['wavelength_id'])

            res_tmp_rwa_vir.append(cur_service['dest'])
            result_rwa_vir.append(res_tmp_rwa_vir.copy())

            # ----------ODL结果----------
            node_config = lightpathConvertNode(cur_traffic_res)
            # print("节点配置",node_config)
            res_tmp_odl = []
            res_tmp_odl.append(cur_service['id'])  # 业务id
            for j in range(len(cur_traffic_res)):
                cur_traffic_au_edge = cur_traffic_res[j]
                if cur_traffic_au_edge['attribute'] == 4:  # 波长链路边
                    res_tmp_odl.append(cur_traffic_au_edge['src_phy'])  # auEdge的源节点
                    res_tmp_odl.append(node_config[cur_traffic_au_edge['src_phy']])  # 节点配置
                    res_tmp_odl.append(cur_traffic_au_edge['lightpath_attribute'])  # 光路类型
                    res_tmp_odl.append(cur_traffic_au_edge['wavelength_id'])  # 波长序号
                    # if node_config[cur_traffic_au_edge['src_phy']] == 0:
                        # print(cur_service['id'],"节点配置出错")
                    if cur_traffic_au_edge['wavelength_id'] == -1:
                        print("波长出错")
                if cur_traffic_au_edge['attribute'] == 1:  # 光路边
                    for vir_link in AuGraph.links_virtual_list:
                        if cur_traffic_au_edge['src_phy'] == vir_link['src'] and \
                                cur_traffic_au_edge['dest_phy'] == vir_link['dest'] and \
                                cur_traffic_au_edge['wavelength_id'] == vir_link['wavelength_id']:
                            for r in range(len(vir_link['route']) - 1):
                                if r == 0:
                                    res_tmp_odl.append(vir_link['route'][r])  # 节点
                                    res_tmp_odl.append(node_config[vir_link['route'][r]])  # 节点配置
                                    res_tmp_odl.append(cur_traffic_au_edge['lightpath_attribute'])  # 光路类型
                                    res_tmp_odl.append(vir_link['wavelength_id'])  # 波长
                                else:
                                    res_tmp_odl.append(vir_link['route'][r])  # 节点
                                    res_tmp_odl.append(none_node_config)  # 节点配置
                                    res_tmp_odl.append(cur_traffic_au_edge['lightpath_attribute'])  # 光路类型
                                    res_tmp_odl.append(vir_link['wavelength_id'])  # 波长

                                # if node_config[vir_link['route'][r]] == 0:
                                #     print("业务",cur_service['id'],"节点",vir_link['route'][r],"节点配置出错")
                                if cur_traffic_au_edge['wavelength_id'] == -1:
                                    print("波长出错")

            res_tmp_odl.append(cur_service['dest'])  # 添加目的节点
            res_tmp_odl.append(node_config[cur_service['dest']])  # 节点配置
            result_odl.append(res_tmp_odl.copy())  # 加入当前业务的odl结果

        else:
            # 部署失败
            result_rwa_phy.append([cur_service['id']], -1)
            result_rwa_vir.append([cur_service['id']], -1)
            result_odl.append([cur_service['id']], -1)


# 根据新建和已建光路计算节点配置类型
# 1：光；2：电；3：光+电；3有最大优先级；4：中间节点
# 输入是某个业务的au_edges

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

# 各边的权重：0、疏导边；1、光路边；2、发射机边；3、接收机边；
# 4、波长链路边；5、复用器边；6、解复用器边；7、波长旁路边
def lightpathConvertNode(au_edges):
    node_conf = np.zeros(Database.node_number)
    for i in range(len(au_edges)):
        cur_au_edge = au_edges[i]  # 当前辅助图边
        cur_au_edge_att = cur_au_edge['attribute']
        if cur_au_edge_att == 5 or cur_au_edge_att == 6:  # 复用器边和解复用器边
            # 已建的源目的都是只配电:2
            node_conf[cur_au_edge['src_phy']] = int(max(node_conf[cur_au_edge['src_phy']], electric_node_config))
        elif cur_au_edge_att == 1:  # 光路边
            # 光路边源目的节点都是只配电
            node_conf[cur_au_edge['src_phy']] = int(max(node_conf[cur_au_edge['src_phy']], electric_node_config))
            node_conf[cur_au_edge['dest_phy']] = int(max(node_conf[cur_au_edge['dest_phy']], electric_node_config))
        elif cur_au_edge_att == 2 or cur_au_edge_att == 3:  # 发射器边和接收器边
            # 新建光路的源和目的都是光 + 电：3
            node_conf[cur_au_edge['src_phy']] = int(max(node_conf[cur_au_edge['src_phy']], li_ele_node_config))
        elif cur_au_edge_att == 4:  # 波长链路边
            node_conf[cur_au_edge['src_phy']] = int(max(node_conf[cur_au_edge['src_phy']], light_node_config))
            node_conf[cur_au_edge['dest_phy']] = int(max(node_conf[cur_au_edge['dest_phy']], light_node_config))
    return node_conf
