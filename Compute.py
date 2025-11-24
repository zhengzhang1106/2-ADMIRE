
def compute_lightpath(au_edge_list):  # 计算新建光路数量（一对Tx+Rx是一条新建光路，数Tx边的数量）
    lightpath_num = 0
    for i in range(len(au_edge_list)):
        cur_au_edge = au_edge_list[i]
        cur_edge_att = cur_au_edge['attribute']  # 当前边的属性
        if cur_edge_att == 2:  # 如果是发射机边，则新建光路数量+1
            lightpath_num += 1
    # print("光路数量：",lightpath_num)

    return lightpath_num


