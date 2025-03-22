# -*- coding: utf-8 -*-

POPULATION_SIZE = 3  # 基因组数 10
MUTA_RATE = 20  # 变异概率
SADE_MUTATION_RATE = 0.5 # SADE 算法变异概率
mutagen = 1.0 #
radioactivity = 0.05
cross_rate = 0.1
mutation_rate = 0.5
ROTATIONS = 1  # 旋转选择， 1： 不能旋转
# 单位都是MM(毫米)
SPACING = 2  # 图形间隔空间
# 不同面料尺寸
BIN_HEIGHT = 2048 #1400
BIN_WIDTH = 2048 #3000
BIN_NORMAL = [
    [0, 0],
    [0, BIN_HEIGHT],
    [BIN_WIDTH, BIN_HEIGHT],
    [BIN_WIDTH, 0],
]  # 一般布是无限长
BIN_CUT_BIG = [[0, 0], [0, 1570], [2500, 1570], [2500, 0]]  # 切割机尺寸 1
BIN_CUT_SMALL = [[0, 0], [0, 1200], [1500, 1200], [1500, 0]]  # # 切割机尺寸 2
file_name=7