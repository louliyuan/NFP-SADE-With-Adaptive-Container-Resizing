# coding=utf8
import dxfgrabber
import csv
import concurrent.futures
import logging
from tools import nfp_utls
from tools.nfp_utls import polygon_area

def find_shape_from_dxf(file_name):
    """
    读取DXF文档，从LINE里面找出多边形
    :param file_name: 文档路径
    :return:
    """
    dxf = dxfgrabber.readfile(file_name)
    all_shapes = list()
    new_polygon = dict()
    for e in dxf.entities:
        if e.dxftype == 'LINE':
            # print (e.start, e.end)
            # 找封闭的多边形
            # 线条不按顺序画
            end_key = '{}x{}'.format(e.end[0], e.end[1])
            star_key = '{}x{}'.format(e.start[0], e.start[1])
            if end_key in new_polygon:
                # 找到闭合的多边形
                all_shapes.append(new_polygon[end_key])
                new_polygon.pop(end_key)
                continue

            # 开始和结束点转换
            if star_key in new_polygon:
                # 找到闭合的多边形
                all_shapes.append(new_polygon[star_key])
                new_polygon.pop(star_key)
                continue

            # 找连接的点
            has_find = False
            for key, points in new_polygon.items():
                if points[-1][0] == e.start[0] and points[-1][1] == e.start[1]:
                    new_polygon[key].append([e.end[0], e.end[1]])
                    has_find = True
                    break
                if points[-1][0] == e.end[0] and points[-1][1] == e.end[1]:
                    new_polygon[key].append([e.start[0], e.start[1]])
                    has_find = True
                    break

            if not has_find:
                new_polygon['{}x{}'.format(e.start[0], e.start[1])] = [
                    [e.start[0], e.start[1]],
                    [e.end[0], e.end[1]],
                ]
    return all_shapes


def input_polygon(dxf_file):
    """
    :param dxf_file: 文件地址
    :param is_class: 返回Polygon 类，或者通用的 list
    :return:
    """
    # 从dxf文件中提取数据
    datas = find_shape_from_dxf(dxf_file)
    shapes = list()

    for i in range(0, len(datas)):
        shapes.append(datas[i])

    print(f'shapes:{shapes}')
    return shapes

def parse_point(point_str, line_number):
    try:
        x_str, y_str = point_str.strip("()").split(", ")
        return float(x_str), float(y_str)
    except ValueError as e:
        print(f"解析点时出错: 行 {line_number}, 点字符串: '{point_str}'。错误: {e}")
        return None
def transform_polygon(points_list):
    """根据给定的点列表，转换为期望的格式（示例中并未说明具体转换逻辑，此处假设为直接使用）"""
    # 此处需要根据具体的转换逻辑来实现，示例中没有提供明确的转换规则
    # 假设直接返回相同的坐标点列表
    return points_list

def parse_csv_to_list(file_path):
    shapes = []
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for line_number, row in enumerate(csv_reader, start=1):  # 从第二行开始
            points_str = row["Polygon Points"].split("; ")
            points = [parse_point(point, line_number) for point in points_str]
            points = [point for point in points if point is not None]  # 过滤掉 None 值
            shape={
                'line_number':line_number,
                'points':points
            }
            shapes.append(shape)
    return shapes

def process_batch(nester, batch):
    nester.add_objects(batch)

def batch_process(nester, polygons, batch_size=100):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(polygons), batch_size):
            batch = polygons[i:i+batch_size]
            futures.append(executor.submit(process_batch, nester, batch))

        for future in concurrent.futures.as_completed(futures):
            future.result()

def process_all(nester, polygons):
    """
    将所有多边形直接添加到 Nester 对象中，无需分批处理。

    :param nester: Nester 对象
    :param polygons: 包含多边形的列表
    """
    if not polygons or not isinstance(polygons, list):
        logging.error("输入的多边形数据无效，必须是一个包含多边形的列表。")
        return

    try:
        nester.add_objects(polygons)
        logging.info(f"成功添加 {len(polygons)} 个多边形到 Nester 对象中。")
    except Exception as e:
        logging.error(f"添加多边形到 Nester 对象时发生错误: {e}")


def convert_rest_paths_to_add_objects_format(rest_paths):
    """
    将复杂的 rest_paths 数据格式转换为 add_objects 所需的格式。
    每个包含 points 的列表数据将转换为 [(x1, y1), (x2, y2), ...] 的格式。

    :param rest_paths: [{'points': [{'x': 2844.0, 'y': 1034.0}, ...]}, ...]
    :return: [[(x1, y1), (x2, y2), ...], ...]
    """
    if not isinstance(rest_paths, list):
        raise ValueError("rest_paths 应为列表格式，但收到的格式为: {}".format(type(rest_paths)))

    converted_data = []

    for entry in rest_paths:
        if 'points' in entry and isinstance(entry['points'], list):
            points = entry['points']
            converted_points = [(point['x'], point['y']) for point in points if 'x' in point and 'y' in point]
            if converted_points:
                converted_data.append(converted_points)
        else:
            print(f"跳过无效的 entry: {entry}")

    # 检查是否成功转换
    if not converted_data:
        raise ValueError("未能成功转换任何数据，请检查 rest_paths 的格式是否正确")

    return converted_data

def process_data(self, data_list):
    if not self.shapes:
        self.shapes = []
    total_area=0
    for data in data_list:
        better_points = [[point['x'], point['y']] for point in data['points']]
        line_number = data.get('p_id', '0')
        shape = {
            'area': 0,
            'p_id': str(line_number),
            'points': [{'x': p[0], 'y': p[1]} for p in better_points],
        }
        area = polygon_area(shape['points'])
        if area > 0:
            shape['points'].reverse()
        shape['area'] = abs(area)
        total_area += shape['area']
        area = nfp_utls.polygon_area(shape['points'])
        if area > 0:
            shape['points'].reverse()
        shape['area'] = abs(area)
        self.shapes.append(shape)
        total_area += shape['area']

if __name__ == '__main__':
    s = find_shape_from_dxf('T2.dxf')
    print(s)
    print(len(s))
    shapes_list = parse_csv_to_list('polygons.csv')
    print(shapes_list)