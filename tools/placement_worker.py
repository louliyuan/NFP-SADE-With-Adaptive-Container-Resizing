# -*- coding: utf-8 -*-
import json
from tools.nfp_utls import (
    almost_equal,
    rotate_polygon,
    get_polygon_bounds,
    polygon_area,
)
import copy
import pyclipper
import time
import logging

class PlacementWorker:
    def __init__(self, bin_polygon, paths, ids, rotations, config, nfp_cache):
        self.bin_polygon = bin_polygon
        self.paths = paths
        self.ids = ids
        self.rotations = rotations
        self.config = config
        self.nfpCache = nfp_cache or {}
        self.rotated_paths = self.preprocess_paths()
        self.rest_paths = []  # 新增变量用于存储剩余路径

    def preprocess_paths(self):
        rotated_paths = []
        for path in self.paths:
            rotated_path = rotate_polygon(path[1]['points'], path[2])
            rotated_path['rotation'] = path[2]
            rotated_path['source'] = path[1]['p_id']
            rotated_path['p_id'] = path[0]
            rotated_paths.append(rotated_path)
        return rotated_paths

    def place_paths(self):
        nfp_start = time.time()
        if self.bin_polygon is None:
            return None

        paths = self.rotated_paths.copy()
        all_placements = []
        fitness = 0
        bin_area = abs(polygon_area(self.bin_polygon['points']))
        min_width = None

        while paths:
            placed = []
            placements = []
            fitness += 1

            # 顺序处理每个路径
            for path in paths:
                result = self.process_path(path, placed, placements)
                if result:
                    placed.append(path)
                    placements.append(result)

            if min_width:
                fitness += min_width / bin_area

            # paths = [path for path in paths if path not in placed] #把paths中不在placed列表中的路径保留下来，形成一个新的列表，代替原来的paths
            # 保存未放置路径到 rest_paths，并清空 paths
            self.rest_paths = [path for path in paths if path not in placed]
            paths = []  # 清空 paths

            if placements:
                all_placements.append(placements)
            else:
                break

        fitness += 2 * len(paths)
        nfp_end = time.time()
        logging.info(f'求解图形分布的时间: {nfp_end - nfp_start}')
        return {
            'placements': all_placements,
            'fitness': fitness,
            'paths': paths,
            'area': bin_area,
        }

    def process_path(self, path, placed, placements):
        key = json.dumps({
            'A': '-1',
            'B': path['p_id'],
            'inside': True,
            'A_rotation': 0,
            'B_rotation': path['rotation'],
        })
        bin_nfp = self.nfpCache.get(key)
        if bin_nfp is None or not bin_nfp:
            return None

        # 检查路径之间的重叠情况
        error = False
        for p in placed:
            key = json.dumps({
                'A': p['p_id'],
                'B': path['p_id'],
                'inside': False,
                'A_rotation': p['rotation'],
                'B_rotation': path['rotation'],
            })
            nfp = self.nfpCache.get(key)
            if nfp is None:
                error = True
                break
        if error:
            return None

        # 计算最佳放置位置
        position = None
        if not placed:
            for nfp in bin_nfp:
                for point in nfp:
                    if position is None or point['x'] - path['points'][0]['x'] < position['x']:
                        position = {
                            'x': point['x'] - path['points'][0]['x'],
                            'y': point['y'] - path['points'][0]['y'],
                            'p_id': path['p_id'],
                            'rotation': path['rotation'],
                        }
            return position

        # 后续路径处理
        clipper_bin_nfp = [[(p['x'], p['y']) for p in nfp] for nfp in bin_nfp]
        clipper = pyclipper.Pyclipper()
        for j, p in enumerate(placed):
            key = json.dumps({
                'A': p['p_id'],
                'B': path['p_id'],
                'inside': False,
                'A_rotation': p['rotation'],
                'B_rotation': path['rotation'],
            })
            nfp = self.nfpCache.get(key)
            if nfp is None:
                continue
            for n in nfp:
                clone = [(np['x'] + placements[j]['x'], np['y'] + placements[j]['y']) for np in n]
                clone = pyclipper.CleanPolygon(clone)
                if len(clone) > 2:
                    clipper.AddPath(clone, pyclipper.PT_SUBJECT, True)

        combine_nfp = clipper.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
        if not combine_nfp:
            return None

        clipper = pyclipper.Pyclipper()
        clipper.AddPaths(combine_nfp, pyclipper.PT_CLIP, True)
        try:
            clipper.AddPaths(clipper_bin_nfp, pyclipper.PT_SUBJECT, True)
        except:
            print('图形坐标出错', clipper_bin_nfp)

        final_nfp = clipper.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
        if not final_nfp:
            return None

        final_nfp = pyclipper.CleanPolygons(final_nfp)
        final_nfp = [polygon for polygon in final_nfp if len(polygon) >= 3]
        if not final_nfp:
            return None

        final_nfp = [[{'x': p[0], 'y': p[1]} for p in polygon] for polygon in final_nfp]
        # 更新最小面积和放置位置
        min_area = None
        min_x = None
        position = None

        for nf in final_nfp:
            if abs(polygon_area(nf)) < 2:
                continue

            all_points = [{'x': p['x'] + placements[m]['x'], 'y': p['y'] + placements[m]['y']}
                          for m in range(len(placed)) for p in placed[m]['points']]

            for p_nf in nf:
                shift_vector = {
                    'x': p_nf['x'] - path['points'][0]['x'],
                    'y': p_nf['y'] - path['points'][0]['y'],
                    'p_id': path['p_id'],
                    'rotation': path['rotation'],
                }
                combined_points = all_points + [
                    {'x': point['x'] + shift_vector['x'], 'y': point['y'] + shift_vector['y']}
                    for point in path['points']
                ]
                rect_bounds = get_polygon_bounds(combined_points)
                area = rect_bounds['width'] * 2 + rect_bounds['height']

                if (min_area is None or area < min_area or almost_equal(min_area, area)) and (
                        min_x is None or shift_vector['x'] <= min_x):
                    min_area = area
                    min_width = rect_bounds['width']
                    position = shift_vector
                    min_x = shift_vector['x']

        return position
