import os
import time
import math
import json
import random
import copy
import logging
import multiprocessing
from Polygon import Polygon
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
import pyclipper
from concurrent.futures import ThreadPoolExecutor, as_completed
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
from tools import placement_worker, nfp_utls
from settings import SPACING, ROTATIONS, BIN_HEIGHT, POPULATION_SIZE, MUTA_RATE, SADE_MUTATION_RATE, radioactivity, \
    mutagen, cross_rate, mutation_rate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 设置日志
logger = logging.getLogger(__name__)

# 添加文件处理器以将日志保存在本地文件
file_handler = logging.FileHandler('nester.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


class Nester:
    def __init__(self, container=None, shapes=None):
        self.container = container  # 承载组件的容器
        self.shapes = shapes  # 组件信息
        self.contain_shapes = shapes  # 新增记录读取多边形数据的信息
        self.results = list()  # storage for the different results
        self.nfp_cache = {}  # 缓存中间计算结果
        self.rest_paths = []  # 新增成员变量用于保存剩余路径
        # 遗传算法的参数
        self.config = {
            'curveTolerance': 0.3,  # 允许的最大误差转换贝济耶和圆弧线段。在SVG的单位。更小的公差将需要更长的时间来计算
            'spacing': SPACING,  # 组件间的间隔
            'rotations': ROTATIONS,  # 旋转的颗粒度，360°的n份，如：4 = [0, 90 ,180, 270]
            'populationSize': POPULATION_SIZE,  # 基因群数量
            'mutationRate': MUTA_RATE,  # 变异概率
            'useHoles': False,  # 是否有洞，暂时都是没有洞
            'exploreConcave': False,  # 寻找凹面，暂时是否
        }
        # SADE算法的参数
        self.SADE_config = {
            'curveTolerance': 0.3,  # 允许的最大误差转换贝济耶和圆弧线段。在SVG的单位。更小的公差将需要更长的时间来计算
            'spacing': SPACING,  # 组件间的间隔
            'rotations': ROTATIONS,  # 旋转的颗粒度，360°的n份，如：4 = [0, 90 ,180, 270]
            'populationSize': POPULATION_SIZE,  # 基因群数量
            'mutationRate': SADE_MUTATION_RATE,  # 变异概率
            'mutagen': mutagen,  # 诱变率
            'radioactivity': radioactivity,  # 突变放射性
            'cross_rate': cross_rate,  # 交叉率
            'mutation_rate': mutation_rate,
            'SelectedSize': int(POPULATION_SIZE / 2),
            'useHoles': False,  # 是否有洞，暂时都是没有洞
            'exploreConcave': False,  # 寻找凹面，暂时是否
        }

        self.GA = None  # 遗传算法类
        self.SADE = None  # SADE算法
        self.best = None  # 记录最佳结果
        self.worker = None  # 根据NFP结果，计算每个图形的转移数据
        self.container_bounds = None  # 容器的最小包络矩形作为输出图的坐标

    def add_objects(self, objects):
        # if not isinstance(objects, list):
        #     objects = [objects]
        if not self.shapes:
            self.shapes = []
        if not self.contain_shapes:
            self.contain_shapes = []

        # p_id = 0
        total_area = 0
        for obj in objects:
            points = obj.get("points", [])
            better_points = self.clean_polygon(points)

            line_number = obj.get("line_number", None)
            # 如果 better_points 是 None 或为空，则跳过该对象
            if not better_points:
                logger.warning(f"Skipping invalid polygon, the linenumber is {line_number}")
                continue
            # points = self.clean_polygon(obj)

            if line_number is None:
                logger.warning(f"Skipping invalid object with line_number {line_number}")
                continue
            # if points is None:
            #     logger.warning("Skipping invalid polygon")
            #     continue
            shape = {
                'area': 0,
                'p_id': str(line_number),
                'points': [{'x': p[0], 'y': p[1]} for p in better_points],
            }
            area = nfp_utls.polygon_area(shape['points'])
            if area > 0:
                shape['points'].reverse()

            shape['area'] = abs(area)
            total_area += shape['area']
            self.shapes.append(shape)
            self.contain_shapes.append(shape)
            # p_id += 1


    def add_container(self):
        if not self.container:
            self.container = {}

        max_polygon_width = 0
        max_polygon_result=next((item for item in self.shapes if item['p_id'] == '1'), None)
        if max_polygon_result:
            points=max_polygon_result.get("points", [])
            x_coords = [p['x'] for p in points]
            y_coords = [p['y'] for p in points]
            min_x, max_x = min(x_coords), max(x_coords)
            max_polygon_width = max_x - min_x

        total_area = sum(shape['area'] for shape in self.shapes)
        total_bounding_box_area = 0  # 初始化包围盒总面积

        # 首先检查 self.shapes 的数量
        if len(self.shapes) <= 7:
            # 计算每个多边形的包围盒面积，并累加
            for shape in self.shapes:
                # 获取多边形的点
                points = shape['points']
                # 计算包围盒
                bounds = nfp_utls.get_polygon_bounds(points)
                # 计算包围盒面积
                bounding_box_area = bounds['width'] * bounds['height']
                total_bounding_box_area += bounding_box_area

            # 计算面积比例
            area_ratio = total_area / total_bounding_box_area if total_bounding_box_area > 0 else 0

            if area_ratio <= 0.8:
                # 如果面积比例不超过 0.8，使用包围盒总面积代替 total_area
                total_area = total_bounding_box_area

        # 计算容器的边长
        side_length = math.ceil(math.sqrt(total_area))
        # 确保边长是 2 的幂次方
        side_length = 2 ** math.ceil(math.log2(side_length))

        width = side_length
        height = side_length

        # # 如果多边形填充不到容器面积的一半，调整高度
        # if height > 1 and total_area <= (area_1 / 2):
        #     height //= 2

        # # 如果总体的占用率不超过88%，调整宽度
        # if width > 1 and total_area <= (area_1 * 0.96):
        #     width //= 2
         if not self.rest_paths:
            while width > 1 and total_area / (width * height) < 0.9 and max_polygon_width < width:
                width //= 2
        else:
            while height > 1 and total_area <= (width * height / 2):
                height //= 2

        self.container['points'] = [
            {'x': 0, 'y': 0},
            {'x': width, 'y': 0},
            {'x': width, 'y': height},
            {'x': 0, 'y': height}
        ]
        self.container['p_id'] = '-1'
        self.container['width'] = width
        self.container['height'] = height
        self.container_bounds = nfp_utls.get_polygon_bounds(self.container['points'])
        logger.info(f'width:{width}, height:{height}')



    def clear(self):
        self.shapes = None

    def run(self):
        if not self.container:
            logger.error("Empty container. Aborting")
            return
        if not self.shapes:
            logger.error("Empty shapes. Aborting")
            return

        faces = list()
        sort_time_start = time.time()
        for i in range(len(self.shapes)):
            shape = copy.deepcopy(self.shapes[i])
            shape['points'] = self.polygon_offset(shape['points'], self.config['spacing'])
            faces.append([str(i), shape])
        sort_time_end = time.time()
        logger.info(f'降序时间：{sort_time_end - sort_time_start}')

        faces = sorted(faces, reverse=True, key=lambda face: face[1]['area'])

        return self.launch_workers(faces)

    def launch_workers(self, adam):
        nester_instance = Nester(self.config)
        if self.SADE is None:
            offset_bin = copy.deepcopy(self.container)
            offset_bin['points'] = self.polygon_offset(offset_bin['points'], self.config['spacing'])
            SADE_initial_start = time.time()
            self.SADE = SADE(adam, offset_bin, self.SADE_config, nester_instance=nester_instance)
            SADE_initial_end = time.time()
            logger.info(f'初始化SADE算法的时间: {SADE_initial_end - SADE_initial_start}')
        else:
            generation_start = time.time()
            self.SADE.generation()
            generation_end = time.time()
            logger.info(f'SADE算法迭代时间: {generation_end - generation_start}')

        SADE_start = time.time()
        for i in range(self.SADE.config['populationSize']):
            res = self.find_fitness(self.SADE.CH[i])
            self.SADE.CH[i]['fitness'] = res['fitness']
            self.results.append(res)
        SADE_end = time.time()
        logger.info(f'计算每组适应值时间:{SADE_end - SADE_start}')

        best_result_start = time.time()
        if len(self.results) > 0:
            best_result = self.results[0]

            for p in self.results:
                if p['fitness'] < best_result['fitness']:
                    best_result = p

            if self.best is None or best_result['fitness'] < self.best['fitness']:
                self.best = best_result
        best_result_end = time.time()
        logger.info(f'找到最佳结果的时间:{best_result_end - best_result_start}')
        

    def find_fitness(self, individual):
        place_list = copy.deepcopy(individual['placement'])
        rotations = copy.deepcopy(individual['rotation'])
        ids = [p[0] for p in place_list]
        for i in range(len(place_list)):
            place_list[i].append(rotations[i])

        nfp_pairs = list()
        new_cache = dict()
        calculate_polygon_start = time.time()
        for i in range(len(place_list)):
            part = place_list[i]
            key = {
                'A': '-1',
                'B': part[0],
                'inside': True,
                'A_rotation': 0,
                'B_rotation': rotations[i],
            }
            tmp_json_key = json.dumps(key)

            if tmp_json_key not in self.nfp_cache:
                nfp_pairs.append({'A': self.container, 'B': part[1], 'key': key})
            else:
                new_cache[tmp_json_key] = self.nfp_cache[tmp_json_key]

            for j in range(i):
                placed = place_list[j]
                key = {
                    'A': placed[0],
                    'B': part[0],
                    'inside': False,
                    'A_rotation': rotations[j],
                    'B_rotation': rotations[i],
                }
                tmp_json_key = json.dumps(key)
                if tmp_json_key not in self.nfp_cache:
                    nfp_pairs.append({'A': placed[1], 'B': part[1], 'key': key})
                else:
                    new_cache[tmp_json_key] = self.nfp_cache[tmp_json_key]
        calculate_polygon_end = time.time()
        logger.info(f'计算NFP多边形和确保容器与多边形内切的时间:{calculate_polygon_end - calculate_polygon_start}')

        self.nfp_cache = new_cache
        placement_worker_start = time.time()
        self.worker = placement_worker.PlacementWorker(
            self.container, place_list, ids, rotations, self.config, self.nfp_cache
        )
        placement_worker_end = time.time()
        logger.info(f'计算图形的转移量和适应值的类时间:{placement_worker_end - placement_worker_start}')

        process_nfp_start = time.time()
        pair_list = self.process_nfp_parallel(nfp_pairs)
        process_nfp_end = time.time()
        logger.info(f'计算所有图形两两组合的NFP总时间:{process_nfp_end - process_nfp_start}')

        return self.generate_nfp(pair_list)

    def process_nfp_parallel(self, pairs):
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:  # 增加工作线程
            futures = {executor.submit(self.process_nfp, pair): pair for pair in pairs}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results

    def process_nfp(self, pair):
        if pair is None or len(pair) == 0:
            return None

        search_edges = self.config['exploreConcave']
        use_holes = self.config['useHoles']

        # A = copy.deepcopy(pair['A'])
        # A['points'] = nfp_utls.rotate_polygon(A['points'], pair['key']['A_rotation'])['points']
        # B = copy.deepcopy(pair['B'])
        # B['points'] = nfp_utls.rotate_polygon(B['points'], pair['key']['B_rotation'])['points']

        # 只对 points 进行深拷贝，避免拷贝整个 A 和 B 对象
        A_points = nfp_utls.rotate_polygon(pair['A']['points'], pair['key']['A_rotation'])['points']
        B_points = nfp_utls.rotate_polygon(pair['B']['points'], pair['key']['B_rotation'])['points']

        if pair['key']['inside']:
            if nfp_utls.is_rectangle(A_points, 0.0001):
                nfp = nfp_utls.nfp_rectangle(A_points, B_points)
            else:
                nfp = nfp_utls.nfp_polygon({'points': A_points}, {'points': B_points}, True, search_edges)
            if nfp and len(nfp) > 0:
                for i in range(len(nfp)):
                    if nfp_utls.polygon_area(nfp[i]) > 0:
                        nfp[i].reverse()
        else:
            if search_edges:
                nfp = nfp_utls.nfp_polygon({'points': A_points}, {'points': B_points}, False, search_edges)
            else:
                nfp = minkowski_difference({'points': A_points}, {'points': B_points})
            if nfp is None or len(nfp) == 0:
                return None
            for i in range(len(nfp)):
                if not search_edges or i == 0:
                    if abs(nfp_utls.polygon_area(nfp[i])) < abs(nfp_utls.polygon_area(A_points)):
                        nfp.pop(i)
                        return None
            if len(nfp) == 0:
                return None
            for i in range(len(nfp)):
                if nfp_utls.polygon_area(nfp[i]) > 0:
                    nfp[i].reverse()
                if i > 0:
                    if nfp_utls.point_in_polygon(nfp[i][0], nfp[0]):
                        if nfp_utls.polygon_area(nfp[i]) < 0:
                            nfp[i].reverse()
        return {'key': pair['key'], 'value': nfp}

        def generate_nfp(self, nfp):
        if nfp:
            for i in range(len(nfp)):
                if nfp[i]:
                    key = json.dumps(nfp[i]['key'])
                    self.nfp_cache[key] = nfp[i]['value']
        self.worker.nfpCache = copy.deepcopy(self.nfp_cache)

        result = self.worker.place_paths()
        # 更新 Nester 的 rest_paths
        self.rest_paths = self.worker.rest_paths
        total_length = sum(len(sublist) for sublist in self.rest_paths)
        logger.info(f"剩余多边形的个数：{total_length}")
        return result
        # return self.worker.place_paths()

    # def show_result(self):
    #     draw_result(self.best['placements'], self.shapes, self.container, self.container_bounds)

    def polygon_offset(self, polygon, offset):
        # shape['points'] = self.polygon_offset(shape['points'], self.config['spacing'])
        is_list = True
        if isinstance(polygon[0], dict):
            polygon = [[p['x'], p['y']] for p in polygon]
            is_list = False
        miter_limit = 2
        co = pyclipper.PyclipperOffset(miter_limit, self.config['curveTolerance'])
        co.AddPath(polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        result = co.Execute(1 * offset)
        if not is_list:
            result = [{'x': p[0], 'y': p[1]} for p in result[0]]
        return result

    def clean_polygon(self, polygon):
        if len(polygon) == 3:
            return [(int(point[0]), int(point[1])) for point in polygon]
        simple = pyclipper.SimplifyPolygon(polygon, pyclipper.PFT_NONZERO)
        if simple is None or len(simple) == 0:
            logger.warning("返回None")
            return None
        biggest = simple[0]
        biggest_area = pyclipper.Area(biggest)
        for i in range(1, len(simple)):
            area = abs(pyclipper.Area(simple[i]))
            if area > biggest_area:
                biggest = simple[i]
                biggest_area = area
        clean = pyclipper.CleanPolygon(biggest, self.config['curveTolerance'])
        if clean is None or len(clean) == 0:
            return None
        return clean

def plot_shapes(shapes, output_folder):
    """
    读取 p_id 和 points，并将每个多边形绘制到独立的 PNG 图片中，文件名为 {p_id}.png
    shapes 的数据结构参考示例：
    [
      {
        'area': 960929.0,
        'p_id': '1',
        'points': [
          {'x': 1007, 'y': -690},
          {'x': 1000, 'y': -689},
          ...
        ]
      },
      ...
    ]
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for shape in shapes:
        p_id = shape['p_id']  # 多边形 ID
        points = shape['points']  # 多边形坐标列表

        # 将 points 中的 x 和 y 分别取出，方便绘图
        xs = [pt['x'] for pt in points]
        ys = [pt['y'] for pt in points]

        # 如果多边形需要闭合，首尾点不一致时可将首点再加入末尾
        if (xs[0], ys[0]) != (xs[-1], ys[-1]):
            xs.append(xs[0])
            ys.append(ys[0])

        # 创建绘图
        fig, ax = plt.subplots(figsize=(6, 6))  # 可根据需要调整图像尺寸
        ax.plot(xs, ys, marker='o', color='blue', linewidth=2)  # 绘制多边形边界
        ax.fill(xs, ys, color='cyan', alpha=0.3)                # 填充多边形内部颜色

        ax.set_title(f"Polygon p_id: {p_id}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        # 设置坐标系等比例显示
        ax.set_aspect('equal', adjustable='box')

        # 保存图片，文件名使用 p_id
        output_path = os.path.join(output_folder, f"{p_id}.png")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)  # 关闭图像，释放内存
        print(f"已保存 {output_path}")
def draw_result(shift_data, polygons, bin_polygon, bin_bounds, file_name):
    # output_floder=r'/home/yujingle/no_fit_polygon/polygons'
    # plot_shapes(polygons,output_floder)
    # 将 shift_data 写入一个 txt 文件
    with open(f"shift_data_{file_name}.txt", "w", encoding="utf-8") as f:
        json.dump(shift_data, f, ensure_ascii=False, indent=2)

    # 将 polygons 写入另一个 txt 文件
    with open(f"polygons_{file_name}.txt", "w", encoding="utf-8") as f:
        json.dump(polygons, f, ensure_ascii=False, indent=2)
        
    shapes = list()
    for polygon in polygons:
        contour = [[p['x'], p['y']] for p in polygon['points']]
        shapes.append(Polygon(contour))

    bin_shape = Polygon([[p['x'], p['y']] for p in bin_polygon['points']])
    shape_area = bin_shape.area(0)
    ids=[]
    p_ids=[]
    solution = list()
    rates = list()
    for s_data in shift_data:
        tmp_bin = list()
        total_area = 0.0
        for move_step in s_data:
            if move_step['rotation'] != 0:
                shapes[int(move_step['p_id'])].rotate(math.pi / 180 * move_step['rotation'], 0, 0)
            shapes[int(move_step['p_id'])].shift(move_step['x'], move_step['y'])
            ids.append(move_step['p_id'])
            tmp_bin.append(shapes[int(move_step['p_id'])])
            total_area += shapes[int(move_step['p_id'])].area(0)
        rates.append(total_area / shape_area)
        solution.append(tmp_bin)
    for id in ids:
        polygon=polygons[int(id)]
        p_id=polygon.get('p_id')
        p_ids.append(p_id)
    # output_folder = '/home/yujingle/no_fit_polygon/single_polygon'
    # draw_polygons(solution, output_folder, ids, p_ids)
    draw_polygon(solution, rates, bin_bounds, bin_shape, file_name)

def draw_polygons(solution, output_folder, ids, p_ids):
    """
    绘制 solution 中的多边形，并将图片保存到指定文件夹。

    Args:
        solution (list): 包含多边形的对象列表，每个对象支持 contour(0) 方法。
        output_folder (str): 图片保存的输出路径。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    new_soulution=solution[0]
    # 遍历 solution 中的多边形
    for idx, polygon in enumerate(new_soulution):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')  # 保持坐标系比例
        # middle_id=ids[idx]
        actual_id=p_ids[int(idx)]
        # 使用 polygon.contour(0) 获取多边形的点坐标并转换为二维数组
        polygon_points = polygon.contour(0)  # 假设 contour(0) 返回 [[x1, y1], [x2, y2], ...]
        clean_polygon = np.array(polygon_points, dtype=float)
        # 绘制多边形
        poly_patch = MplPolygon(clean_polygon, closed=True, edgecolor='blue', facecolor='lightblue', linewidth=2)
        ax.add_patch(poly_patch)


        # 设置坐标轴范围
        x_coords, y_coords = clean_polygon[:, 0], clean_polygon[:, 1]
        ax.set_xlim(min(x_coords) - 10, max(x_coords) + 10)
        ax.set_ylim(min(y_coords) - 10, max(y_coords) + 10)

        # 移除坐标轴
        plt.axis('off')

        # 保存图片
        output_path = os.path.join(output_folder, f"{actual_id}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)  # 关闭图像，释放内存

        print(f"已保存多边形图片：{output_path}")

# SADE算法
class SADE():
    def __init__(self, adam, bin_polygon, SADE_config, nester_instance=None):
        # 初始化
        self.Nester = nester_instance
        self.bin_bounds = bin_polygon['points']
        self.bin_bounds = {
            'width': bin_polygon['width'],
            'height': bin_polygon['height'],
        }
        self.config = SADE_config
        self.bin_polygon = bin_polygon

        # SADE算法的参数
        self.PoolSize = self.config['populationSize']
        self.SelectdSize = int(self.PoolSize / 2)
        self.cross_rate = self.config['cross_rate']
        self.Dim = 2
        self.xmin = [-10.0, -10.0]
        self.xmax = [10.0, 10.0]

        self.ActualSize = 0
        self.Force = [0.0] * self.PoolSize
        self.reduced_radioactivity = 0.0
        self.top = 0.0

        angles = list()
        shapes = copy.deepcopy(adam)
        for shape in shapes:
            angles.append(self.random_angle(shape))
        self.CH = [{'placement': shapes, 'rotation': angles}]
        for i in range(1, self.config['populationSize']):
            mutant = self.MUTATE(self.CH[0])
            self.CH.append(mutant)

    def random_angle(self, shape):
        angle_list = list()
        for i in range(self.config['rotations']):
            angle_list.append(i * (360 / self.config['rotations']))

        def shuffle_array(data):
            for i in range(len(data) - 1, 0, -1):
                j = random.randint(0, i)
                data[i], data[j] = data[j], data[i]
            return data

        angle_list = shuffle_array(angle_list)
        for angle in angle_list:
            rotate_part = nfp_utls.rotate_polygon(shape[1]['points'], angle)
            if (rotate_part['width'] < self.bin_bounds['width'] and rotate_part['height'] < self.bin_bounds['height']):
                return angle
        return 0

    def new_point(self):
        return [random.uniform(self.xmin[i], self.xmax[i]) for i in range(self.Dim)]

    def compute_radioactivity(self):
        p = self.config['radioactivity'] * self.config['SelectedSize']
        rn = math.ceil(p)
        self.reduced_radioactivity = p / rn
        self.mutants = int(rn)

    def EVALUATE_GENERATION(self):
        btg = -float('inf')
        for i in range(self.ActualSize):
            res = self.Nester.find_fitness(self.CH[i])
            self.CH[i]['fitness'] = res['fitness']
            if self.CH[i]['fitness'] > btg:
                btg = self.CH[i]['fitness']  # 更新最优解
        return btg

    def SELECT(self):
        while self.ActualSize > self.SelectdSize:
            i1 = random.randint(0, self.ActualSize - 1)
            i2 = random.randint(1, self.ActualSize - 1)
            if i1 == i2:
                i2 -= 1
            dead = i2 if self.Force[i1] >= self.Force[i2] else i1
            last = self.ActualSize - 1
            self.CH[last], self.CH[dead] = self.CH[dead], self.CH[last]
            self.Force[dead] = self.Force[last]
            self.ActualSize -= 1

    def MUTATE(self, individual):
        self.compute_radioactivity()
        clone = {
            'placement': individual['placement'][:],
            'rotation': individual['rotation'][:],
        }
        for _ in range(self.mutants):
            p = random.uniform(0, 1)
            if p <= self.reduced_radioactivity:
                index = random.randint(0, self.config['SelectedSize'] - 1)
                x = self.new_point()
                for j in range(self.Dim):
                    # 获取 placement 中的 points 列表并修改其 x 和 y 坐标
                    for point in clone['placement'][index][1]['points']:
                        point['x'] += self.config['mutation_rate'] * (x[0] - point['x'])  # 变异x坐标
                        point['y'] += self.config['mutation_rate'] * (x[1] - point['y'])  # 变异y坐标
                self.ActualSize += 1
        return clone

    def CROSS(self):
        while self.ActualSize < self.PoolSize:
            i1 = random.randint(0, self.SelectdSize)
            i2 = random.randint(1, self.SelectdSize)
            if i1 == i2:
                i2 -= 1
            i3 = random.randint(0, self.SelectdSize)

            self.CH[self.ActualSize]['placement'].append([self.CH[i3]['placement'][j] + self.cross_rate * (
                        self.CH[i2]['placement'][j] - self.CH[i1]['placement'][j]) for j in range(self.Dim)])
            self.ActualSize += 1

    def to_continue(self, bsf):
        return abs(self.top - bsf) > 0.001

    def generation(self):
        self.CH = sorted(self.CH, key=lambda a: a['fitness'])
        new_CH = [self.CH[0]]
        while len(new_CH) < self.config['populationSize']:
            index = random.randint(0, len(self.CH) - 1)
            individual = self.CH[index]
            mutated_individual = self.MUTATE(individual)
            new_CH.append(mutated_individual)  # 添加变异后的个体到新种群
        self.CH = new_CH
        self.SELECT()



# GA算法
class genetic_algorithm:
    def __init__(self, adam, bin_polygon, config):
        self.bin_bounds = bin_polygon['points']
        self.bin_bounds = {
            'width': bin_polygon['width'],
            'height': bin_polygon['height'],
        }
        self.config = config
        self.bin_polygon = bin_polygon
        angles = list()
        shapes = copy.deepcopy(adam)
        for shape in shapes:
            angles.append(self.random_angle(shape))
        self.population = [{'placement': shapes, 'rotation': angles}]
        for i in range(1, self.config['populationSize']):
            mutant = self.mutate(self.population[0])
            self.population.append(mutant)

    def random_angle(self, shape):
        angle_list = list()
        for i in range(self.config['rotations']):
            angle_list.append(i * (360 / self.config['rotations']))

        def shuffle_array(data):
            for i in range(len(data) - 1, 0, -1):
                j = random.randint(0, i)
                data[i], data[j] = data[j], data[i]
            return data

        angle_list = shuffle_array(angle_list)
        for angle in angle_list:
            rotate_part = nfp_utls.rotate_polygon(shape[1]['points'], angle)
            if (rotate_part['width'] < self.bin_bounds['width'] and rotate_part['height'] < self.bin_bounds['height']):
                return angle
        return 0

    def mutate(self, individual):
        clone = {
            'placement': individual['placement'][:],
            'rotation': individual['rotation'][:],
        }
        for i in range(len(clone['placement'])):
            if random.random() < 0.01 * self.config['mutationRate']:
                if i + 1 < len(clone['placement']):
                    clone['placement'][i], clone['placement'][i + 1] = clone['placement'][i + 1], clone['placement'][i]
        if random.random() < 0.01 * self.config['mutationRate']:
            clone['rotation'][i] = self.random_angle(clone['placement'][i])
        return clone

    def generation(self):
        self.population = sorted(self.population, key=lambda a: a['fitness'])
        new_population = [self.population[0]]
        while len(new_population) < self.config['populationSize']:
            male = self.random_weighted_individual()
            female = self.random_weighted_individual(male)
            children = self.mate(male, female)
            new_population.append(self.mutate(children[0]))
            if len(new_population) < self.config['populationSize']:
                new_population.append(self.mutate(children[1]))
        self.population = new_population

    def random_weighted_individual(self, exclude=None):
        # 选择
        pop = self.population
        if exclude and pop.index(exclude) >= 0:
            pop.remove(exclude)
        rand = random.random()
        lower = 0
        weight = 1.0 / len(pop)
        upper = weight
        pop_len = len(pop)
        for i in range(pop_len):
            if (rand > lower) and (rand < upper):
                return pop[i]
            lower = upper
            upper += 2 * weight * float(pop_len - i) / pop_len
        return pop[0]

    def mate(self, male, female):
        # 交叉
        cutpoint = random.randint(0, len(male['placement']) - 1)
        gene1 = male['placement'][:cutpoint]
        rot1 = male['rotation'][:cutpoint]

        gene2 = female['placement'][:cutpoint]
        rot2 = female['rotation'][:cutpoint]

        def contains(gene, shape_id):
            for i in range(len(gene)):
                if gene[i][0] == shape_id:
                    return True
            return False

        for i in range(len(female['placement']) - 1, -1, -1):
            if not contains(gene1, female['placement'][i][0]):
                gene1.append(female['placement'][i])
                rot1.append(female['rotation'][i])

        for i in range(len(male['placement']) - 1, -1, -1):
            if not contains(gene2, male['placement'][i][0]):
                gene2.append(male['placement'][i])
                rot2.append(male['rotation'][i])

        return [{'placement': gene1, 'rotation': rot1}, {'placement': gene2, 'rotation': rot2}]


def minkowski_difference(A, B):
    Ac = [[p['x'], p['y']] for p in A['points']]
    Bc = [[p['x'] * -1, p['y'] * -1] for p in B['points']]
    solution = pyclipper.MinkowskiSum(Ac, Bc, True)
    largest_area = None
    clipper_nfp = None
    for p in solution:
        p = [{'x': i[0], 'y': i[1]} for i in p]
        sarea = nfp_utls.polygon_area(p)
        if largest_area is None or largest_area > sarea:
            clipper_nfp = p
            largest_area = sarea
    clipper_nfp = [{'x': clipper_nfp[i]['x'] + Bc[0][0] * -1, 'y': clipper_nfp[i]['y'] + Bc[0][1] * -1} for i in
                   range(len(clipper_nfp))]
    return [clipper_nfp]

def draw_polygon(solution, rates, bin_bounds, bin_shape, file_name):

    # 替换为实际字体的文件路径
    font_path = '/System/Library/AssetsV2/com_apple_MobileAsset_Font7/62032b9b64a0e3a9121c50aeb2ed794e3e2c201f.asset/AssetData/Hei.ttf'
    font_prop = fm.FontProperties(fname = font_path)
    mpl.rcParams['font.family'] = font_prop.get_name()
    mpl.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题

    num_bin = len(solution)

    bin_bounds_list = []
    bin_shape_list = []

    # 打开一个文本文件，用于写入多边形坐标数据
    with open(f'polygon_coordinates_{file_name}.txt', 'w', encoding='utf-8') as coord_file:

        for i in range(num_bin):
            if i == 0:
                # 第一个容器使用传入的 bin_bounds 和 bin_shape
                bin_bounds_list.append(bin_bounds)
                bin_shape_list.append(bin_shape)
            else:
                # 计算后续容器的尺寸
                shapes_in_bin = solution[i]
                current_bin_bounds = calculate_container_size(shapes_in_bin)
                current_bin_shape = Polygon([
                    [0, 0],
                    [current_bin_bounds['width'], 0],
                    [current_bin_bounds['width'], current_bin_bounds['height']],
                    [0, current_bin_bounds['height']]
                ])
                bin_bounds_list.append(current_bin_bounds)
                bin_shape_list.append(current_bin_shape)

        base_width = 20  # 增加每个子图的宽度

        # 计算每个子图的高度，保持比例
        base_heights = []
        for bounds in bin_bounds_list:
            base_height = base_width * bounds['height'] / bounds['width']
            base_heights.append(base_height)

        if not base_heights:
            print("base_heights 为空，无法计算最大高度。")
            return

        max_base_height = max(base_heights)
        fig_width = num_bin * base_width  # 横向排列，因此需要增加宽度

        fig1, axes = plt.subplots(1, num_bin, figsize=(fig_width, max_base_height), dpi=300)  # 调整为1行多列

        fig1.suptitle('多边形排样结果', fontweight='bold', fontsize=18, y=1.05)  # 将 y 参数调整为更大的值，确保标题完全显示

        if num_bin == 1:
            axes = [axes]  # 确保 axes 是一个可迭代的列表

        # 初始化调用次数
        if not hasattr(draw_polygon,"call_count"):
            draw_polygon.call_count = 0
        # 更新调用次数
        draw_polygon.call_count += 1

        for i_pic, (ax, shapes, bounds, bin_shape_current) in enumerate(
                zip(axes, solution, bin_bounds_list, bin_shape_list), start=1):
            ax.set_xlim(bounds['x'] - 10, bounds['width'] + 50)
            ax.set_ylim(bounds['y'] - 10, bounds['height'] + 50)
            ax.set_aspect('equal')

            # 改进颜色和边框对比度
            output_obj = []
            # 绘制容器
            output_obj.append(patches.Polygon(bin_shape_current.contour(0), fc='lightgreen', ec='green', alpha=0.5))
            # 绘制多边形
            for idx, s in enumerate(shapes):
                output_obj.append(patches.Polygon(s.contour(0), fc='yellow', lw=0.5, edgecolor='black'))

                # 获取多边形的坐标点
                polygon_points = s.contour(0)
                # 将坐标数据写入文件
                coord_file.write(f'容器 {i_pic} 中的多边形 {idx + 1}:\n')
                for point in polygon_points:
                    coord_file.write(f'{point[0]}, {point[1]}\n')
                coord_file.write('\n')  # 每个多边形之间空一行

            for p in output_obj:
                ax.add_patch(p)

            # 添加子图标题
            ax.set_title(f'The {draw_polygon.call_count} container，Utilization: {rates[i_pic - 1] * 100:.2f}%', fontsize=14, y=1.02)

        # 调整子图之间的间距，防止重叠
        plt.subplots_adjust(wspace=0.4)  # 增加子图之间的宽度间隔
        plt.tight_layout(pad=4.0, rect=[0, 0, 1, 0.96])  # 调整布局，增加全局边距
        plt.show()
        fig1.savefig(f'figure{file_name}.png')

def draw_and_save_shapes(self, output_folder):
    """
    从 self.contain_shapes 中读取每行的数据，绘制多边形，并保存图片。

    Args:
        self: 包含 contain_shapes 属性的对象，contain_shapes 是列表，每个元素是一个包含 'points' 的字典。
        output_folder (str): 图片保存的输出路径。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历 contain_shapes 中的每一行数据
    for idx, shape in enumerate(self.contain_shapes):
        points = shape['points']
        p_id = shape.get('p_id', f"polygon_{idx}")  # 获取多边形ID，默认为索引

        # 提取坐标点
        polygon_points = [(point['x'], point['y']) for point in points]

        # 创建绘图
        fig, ax = plt.subplots()
        ax.set_aspect('equal')  # 保持坐标系比例

        # 绘制多边形
        poly_patch = MplPolygon(polygon_points, closed=True, edgecolor='blue', facecolor='lightblue', linewidth=2)
        ax.add_patch(poly_patch)

        # 设置坐标轴范围
        x_coords, y_coords = zip(*polygon_points)
        ax.set_xlim(min(x_coords) - 10, max(x_coords) + 10)
        ax.set_ylim(min(y_coords) - 10, max(y_coords) + 10)

        # 移除坐标轴
        ax.axis('off')

        # 保存图片
        output_path = os.path.join(output_folder, f"{p_id}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)  # 关闭图像，释放内存

        print(f"已保存多边形图片：{output_path}")

def calculate_container_size(polygons):
    # 初始化全局最大的 x 和 y 坐标
    global_x_max = float('-inf')
    global_y_max = float('-inf')

    for polygon in polygons:
        # 获取多边形的所有点
        points = polygon.contour(0)  # 假设 contour(0) 返回点的列表 [[x1, y1], [x2, y2], ...]

        for p in points:
            x, y = p[0], p[1]
            # 更新全局最大的 x 和 y 坐标
            if x > global_x_max:
                global_x_max = x
            if y > global_y_max:
                global_y_max = y

    # 取 x_max 和 y_max 中的较大值
    max_coordinate = max(global_x_max, global_y_max)

    # 将边长调整为不小于 max_coordinate 的 2 的整数次幂
    side_length = 2 ** math.ceil(math.log2(max_coordinate))

    width = side_length
    height = side_length

    container_bounds = {'x': 0, 'y': 0, 'width': width, 'height': height}
    return container_bounds


def content_loop_rate(best, n, file_name, loop_time=0):
    res = best
    run_time = loop_time
    count = 1
    while run_time:
        count += 1
        startime = time.time()
        n.run()
        newtime = time.time()
        logger.info(f'第{count}次循环时间{newtime - startime}')
        best = n.best
        if best['fitness'] <= res['fitness']:
            res = best
        run_time -= 1
    draw_result(res['placements'], n.shapes, n.container, n.container_bounds, file_name)

def set_target_loop(best, nest):
    res = best
    total_area = 0
    rate = None
    num_placed = 0
    while 1:
        nest.run()
        best = nest.best
        if best['fitness'] <= res['fitness']:
            res = best
            for s_data in res['placements']:
                tmp_total_area = 0.0
                tmp_num_placed = 0

                for move_step in s_data:
                    tmp_total_area += nest.shapes[int(move_step['p_id'])]['area']
                    tmp_num_placed += 1

                tmp_rates = tmp_total_area / abs(nfp_utls.polygon_area(nest.container['points']))

                if (num_placed < tmp_num_placed or total_area < tmp_total_area or rate < tmp_rates):
                    num_placed = tmp_num_placed
                    total_area = tmp_total_area
                    rate = tmp_rates
        if num_placed == len(nest.shapes):
            break
    draw_result(res['placements'], nest.shapes, nest.container, nest.container_bounds)
