import AuGraph
import Database
import Service
import Dijkstra
import Compute

ser_route_list = []  # 存储所有业务的路由结果
au_edge_collection = []  # 存储所有业务的au_edge集合


def route_wave_assign(weight, index):
    job_fail = 0  # 失败业务
    # AuGraph.au_graph_init(weight)
    # Service.generate_service()  # 产生业务

    service = Service.service_list[index]
    # 业务信息
    ser_id = service['id']
    ser_src = service['src']
    ser_dest = service['dest']
    ser_traffic = service['traffic']  # 是数组
    # print('ser_traffic', ser_traffic)

    # 将源目节点的编号转换为辅助图上的节点的编号
    au_src = ser_src * AuGraph.layer * 2 + AuGraph.layer
    au_dest = ser_dest * AuGraph.layer * 2

    # 添加满足容量需求的光路边
    AuGraph.add_edge(AuGraph.links_virtual_list, ser_traffic, weight)

    # 计算最短路，并添加到路由集合中
    route_tmp = Dijkstra.dijkstra(AuGraph.au_graph_weight, au_src, au_dest, ser_id)

    # 删除添加的光路边
    AuGraph.delete_edge()

    if route_tmp:
        ser_route_list.append(route_tmp)
        au_edges = AuGraph.edge_convert(route_tmp)
        au_edges = AuGraph.update_au_graph(au_edges, service)
        au_edge_collection.append(au_edges)
        service['condition'] = True
        lightpath_num = Compute.compute_lightpath(au_edges)     # 已建光路数量
        return True, lightpath_num, au_edge_collection
    else:
        service['condition'] = False
        job_fail += 1
        au_edge_collection.append([])
        ser_route_list.append([])
        return False, 0, au_edge_collection
