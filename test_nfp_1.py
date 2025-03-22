# -*- coding: utf-8 -*-
from nfp_function import Nester, content_loop_rate, set_target_loop, draw_result
from tools import input_utls
from settings import BIN_WIDTH, BIN_NORMAL, BIN_CUT_BIG
import time
import logging


if __name__ == '__main__':
    start=time.time()
    n = Nester()

    s=input_utls.parse_csv_to_list('/Users/louliyuan/Desktop/no_fit_polygon_py3-master/测试数据1-1/convex_hulls004.csv')
    # s = input_utls.input_polygon('dxf_file/E6.dxf')
    if s is None:
        print("Error: The parsed data is None.")
    else:
        # print("Parsed data length:", len(s))
        # print("Sample data:", s[:5])  # 打印前5个元素以确认格式
        input_utls.batch_process(n, s, batch_size=100)
    # n.add_objects(s)


    # if n.shapes_max_length > BIN_WIDTH:
    #     BIN_NORMAL[2][0] = n.shapes_max_length
    #     BIN_NORMAL[3][0] = n.shapes_max_length
    #
    #     #新增代码
    #     BIN_NORMAL[1][1] = n.shapes_max_height
    #     BIN_NORMAL[2][1] = n.shapes_max_height

    # 选择面布
    n.add_container()
    # n.add_container(BIN_NORMAL)
    # 运行计算
    n.run()
    middle1=time.time()
    logging.info(f'循环一次的时间:{middle1-start}')
    # 设计退出条件
    res_list = list()
    best = n.best
    # 放置在一个容器里面
    # set_target_loop(best, n)    # T6

    # 循环特定次数
    #content_loop_rate(best, n, loop_time=1)  # T7 , T4
    # draw_result(best['placements'], n.shapes, n.container, n.container_bounds)
    end=time.time()
    logging.info(f'time:{end-start}')