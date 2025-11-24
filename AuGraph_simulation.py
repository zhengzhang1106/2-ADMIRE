import AuOdlConvert
import Compute
import Service
import Database
import AuGraph
import RWA
import numpy as np

if __name__ == '__main__':
    lightpath_cumulate = 0  # 累计光路数量
    for count in range(1):
        Service.generate_service(0, Database.time)
        Database.clear(Database.links_physical)
        AuGraph.links_virtual_list.clear()  # 清空虚拟链路
        # AuGraph.links_virtual.clear()
        for i in range(Database.job_number):
            if i == 0:
                AuGraph.au_graph_init(AuGraph.edge_weight)
                flag, lightpath_num, au_edges_list = RWA.route_wave_assign(AuGraph.edge_weight, i)
                lightpath_cumulate += lightpath_num
                print(i, "光路数量", lightpath_num, "累计光路数量", lightpath_cumulate)
                # print("-------------------------------------------------------")
                # lp_num = len(AuGraph.links_virtual_list)
                # print("一样吗", lp_num - lightpath_cumulate)
            else:
                AuGraph.update_au_graph_weight(AuGraph.edge_weight)
                flag, lightpath_num, au_edges_list = RWA.route_wave_assign(AuGraph.edge_weight, i)
                lightpath_cumulate += lightpath_num
                print(i, "光路数量", lightpath_num, "累计光路数量", lightpath_cumulate)
                # print("-------------------------------------------------------")
                # lp_num = len(AuGraph.links_virtual_list)
                # print("一样吗", lp_num - lightpath_cumulate)


    # print('路由结果', RWA.ser_route_list)
    AuOdlConvert.odl_result(au_edges_list)
    # print("RWA", AuOdlConvert.result_rwa_phy)
    # count = 0
    # path_length = []
    # while count < 100:
    #     length = int((len(AuOdlConvert.result_rwa_phy[count])-1)/2) + int((len(AuOdlConvert.result_rwa_phy[count+1])-1)/2)
    #     path_length.append(length)
    #     count += 2
    # print("路径长度", len(path_length))
    # print("MinWL")
    # for pathi in path_length:
    #     print(pathi)

    # print("虚拟拓扑跳数50")
    # print("MinWL")
    # index = 1
    # while index < len(virtual_hop_cumulate_list):
    #     virtual_hop_cumulate_list[index] += virtual_hop_cumulate_list[index - 1]
    #     index += 2
    #
    # index = 1
    # while index < len(virtual_hop_cumulate_list):
    #     print(virtual_hop_cumulate_list[index])
    #     index += 2
    # print(Service.path1)
    #
    # with open('ser_route_list', 'w') as f:
    #     for i in range(len(RWA.ser_route_list)):
    #         f.write(str(RWA.ser_route_list[i]))
    #         f.write('\n')
    #     f.close()
    #
    # with open('rwa_phy.txt', 'w') as f:
    #     for i in range(len(AuOdlConvert.result_rwa_phy)):
    #         f.write(str(AuOdlConvert.result_rwa_phy[i]))
    #         f.write('\n')
    #     f.close()
    #
    # with open('rwa_vir.txt', 'w') as file:
    #     for i in range(len(AuOdlConvert.result_rwa_vir)):
    #         file.write(str(AuOdlConvert.result_rwa_vir[i]))
    #         file.write('\n')
    #     file.close()
    #
    # with open('phyLinks.txt', 'w') as file:
    #     for data in Database.links_physical:
    #         np.savetxt(file, data, fmt='%.3f', delimiter='\t')
    #         # file.write('\n')
    #     file.close()
    #
    # with open('link_vir.txt', 'w') as file:
    #     for k in range(len(AuGraph.links_virtual_list)):
    #         file.write(str(AuGraph.links_virtual_list[k]))
    #         file.write('\n')
    #     file.close()
    #
    # with open('odl_minwl.txt', 'w') as file:
    #     for i in range(len(AuOdlConvert.result_odl)):
    #         file.write(str(AuOdlConvert.result_odl[i]))
    #         file.write('\n')
    #     file.close()
