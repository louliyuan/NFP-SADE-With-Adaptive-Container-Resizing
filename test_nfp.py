# -*- coding: utf-8 -*-
from nfp_function import Nester, content_loop_rate, set_target_loop, draw_and_save_shapes
from tools import input_utls
from settings import BIN_WIDTH, BIN_NORMAL, BIN_CUT_BIG, file_name
import time
import logging
import matplotlib.font_manager as fm
if __name__ == '__main__':
    start=time.time()
    n = Nester()
    # s=input_utls.parse_csv_to_list('C:/Users\Zhan\Downloads/no_fit_polygon_py3-master/no_fit_polygon_py3-master/test_data/convex_hulls.csv')
    s = input_utls.parse_csv_to_list('/Users/louliyuan/Desktop/纹理重组/no_fit_polygon_py3-master/测试数据1-1/convex_hulls008.csv')
    # s = input_utls.input_polygon('dxf_file/E6.dxf')
    if s is None:
        print("Error: The parsed data is None.")
    else:
        input_utls.batch_process(n, s, batch_size=100)

    # """"解决Mac字体问题"""
    # # 获取所有文字的路径
    # fonts = fm.findSystemFonts()
    #
    # # 筛选出中文字体
    # chinese_fonts = []
    # for font in fonts:
    #     try:
    #         font_prop = fm.FontProperties(fname=font)
    #         if 'Sim' in font_prop.get_name() or 'Hei' in font_prop.get_name() or 'Kai' in font_prop.get_name():
    #             chinese_fonts.append(font)
    #     except Exception as e:
    #         continue
    # # 打印中文字体路径
    # for font in chinese_fonts:
    #     print(font)

    # 选择面布
    n.add_container()
    # n.add_container(BIN_NORMAL)
    # 运行计算
    n.run()
    # output_folder = '/yujingle/no_fit_polygon/shapes_polygon'
    # draw_and_save_shapes(n, output_folder)
    middle1 = time.time()
    logging.info(f'循环一次的时间:{middle1-start}')
    # 设计退出条件
    res_list = list()
    best = n.best
    # 放置在一个容器里面
    # set_target_loop(best, n)    # T6

    # 循环特定次数
    content_loop_rate(best, n, file_name, loop_time=1)  # T7 , T4
    rest_paths = n.rest_paths

    # 如果有剩余路径，则创建新的 Nester 对象进行处理
    if rest_paths:
        formatted_rest_paths = input_utls. convert_rest_paths_to_add_objects_format(n.rest_paths)

        # 创建第二个 Nester 对象
        n2 = Nester()
        input_utls.process_data(n2, rest_paths)
        # input_utls. process_all(n2, formatted_rest_paths)
        #初始化新的容器
        n2.add_container()

        # 运行第二个 Nester 对象的排样计算
        n2.run()
        middle2 = time.time()
        logging.info(f'第二次运行的时间:{middle2 - middle1}')

        # 保存第二次运行的最佳结果
        best2 = n2.best
        file_name=file_name+1
        # 循环排样优化
        content_loop_rate(best2, n2, file_name, loop_time=1)

    end = time.time()
    logging.info(f'time:{end-start}')