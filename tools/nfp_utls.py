# -*- coding: utf-8 -*-
import copy
import math

# ==== 整合 Shapely 与 R 树加速依赖 ====
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union
from rtree import index

# =====================================================================
# 全局浮点误差允许值
# =====================================================================
TOL = 1e-7


# =====================================================================
# 1) 改进版 almost_equal：使用相对 / 绝对误差判断
# =====================================================================
def almost_equal(a, b, rel_tol=1e-7, abs_tol=1e-9):
    """
    更稳健的浮点比较，避免单纯使用固定阈值.
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# =====================================================================
# 2) 基础函数：is_rectangle, normalize_vector, on_segment, 等
# =====================================================================
def is_rectangle(poly, tolerance=None):
    bb = get_polygon_bounds(poly)
    tolerance = tolerance or TOL
    for point in poly:
        if not almost_equal(point['x'], bb['x'], tolerance) and not almost_equal(
                point['x'], bb['x'] + bb['width'], tolerance
        ):
            return False
        if not almost_equal(point['y'], bb['y'], tolerance) and not almost_equal(
                point['y'], bb['y'] + bb['height'], tolerance
        ):
            return False
    return True


def normalize_vector(v):
    length_squared = v['x'] ** 2 + v['y'] ** 2
    if almost_equal(length_squared, 1.0):
        return v
    inverse = 1.0 / math.sqrt(length_squared)
    return {'x': v['x'] * inverse, 'y': v['y'] * inverse}

def polygon_area(polygon):
    area = 0
    j = len(polygon) - 1
    for i in range(0, len(polygon)):
        area += (polygon[j]['x'] + polygon[i]['x']) * (
            polygon[j]['y'] - polygon[i]['y']
        )
        j = i
    return 0.5 * area

def on_segment(A, B, p):
    # 垂直线判断
    if almost_equal(A['x'], B['x']) and almost_equal(p['x'], A['x']):
        if (not almost_equal(p['y'], B['y'])
                and not almost_equal(p['y'], A['y'])
                and max(B['y'], A['y']) > p['y'] > min(B['y'], A['y'])):
            return True
        else:
            return False
    # 水平线判断
    if almost_equal(A['y'], B['y']) and almost_equal(p['y'], A['y']):
        if (not almost_equal(p['x'], B['x'])
                and not almost_equal(p['x'], A['x'])
                and max(B['x'], A['x']) > p['x'] > min(B['x'], A['x'])):
            return True
        else:
            return False
    # 范围检查
    if ((p['x'] < A['x'] and p['x'] < B['x']) or (p['x'] > A['x'] and p['x'] > B['x'])
            or (p['y'] < A['y'] and p['y'] < B['y']) or (p['y'] > A['y'] and p['y'] > B['y'])):
        return False
    # 排除端点
    if ((almost_equal(p['x'], A['x']) and almost_equal(p['y'], A['y']))
            or (almost_equal(p['x'], B['x']) and almost_equal(p['y'], B['y']))):
        return False
    cross = (p['y'] - A['y']) * (B['x'] - A['x']) - (p['x'] - A['x']) * (B['y'] - A['y'])
    if abs(cross) > TOL:
        return False
    dot = ((p['x'] - A['x']) * (B['x'] - A['x'])
           + (p['y'] - A['y']) * (B['y'] - A['y']))
    if dot < 0 or almost_equal(dot, 0):
        return False
    len2 = ((B['x'] - A['x']) ** 2 + (B['y'] - A['y']) ** 2)
    if dot > len2 or almost_equal(dot, len2):
        return False
    return True


# =====================================================================
# 3) 矩形 NFP 计算 (原函数保留)
# =====================================================================
def nfp_rectangle(A, B):
    min_ax = min(A, key=lambda p: p['x'])['x']
    max_ax = max(A, key=lambda p: p['x'])['x']
    min_ay = min(A, key=lambda p: p['y'])['y']
    max_ay = max(A, key=lambda p: p['y'])['y']
    min_bx = min(B, key=lambda p: p['x'])['x']
    max_bx = max(B, key=lambda p: p['x'])['x']
    min_by = min(B, key=lambda p: p['y'])['y']
    max_by = max(B, key=lambda p: p['y'])['y']
    if (max_bx - min_bx) > (max_ax - min_ax) or (max_by - min_by) > (max_ay - min_ay):
        return None
    return [[
        {'x': min_ax - min_bx + B[0]['x'], 'y': min_ay - min_by + B[0]['y']},
        {'x': max_ax - max_bx + B[0]['x'], 'y': min_ay - min_by + B[0]['y']},
        {'x': max_ax - max_bx + B[0]['x'], 'y': max_ay - max_by + B[0]['y']},
        {'x': min_ax - min_bx + B[0]['x'], 'y': max_ay - max_by + B[0]['y']},
    ]]


# =====================================================================
# 新增：分段滑动辅助函数
# =====================================================================
def find_slide_parameter_for_edge_and_vertex(P, Q, direction, v, tol=1e-7):
    """
    求解参数 t 和 s，使得：
      P + t*direction + s*(Q-P) == v
    当 s ∈ [0,1] 且 t > 0 时返回 t，否则返回 None。
    """
    dx = Q['x'] - P['x']
    dy = Q['y'] - P['y']
    det = direction['x'] * dy - direction['y'] * dx
    if almost_equal(det, 0, abs_tol=tol):
        return None
    t = ((v['x'] - P['x']) * dy - (v['y'] - P['y']) * dx) / det
    s = ((v['y'] - P['y']) * direction['x'] - (v['x'] - P['x']) * direction['y']) / det
    if 0 <= s <= 1 and t > tol:
        return t
    return None


def compute_sub_nfp(A, B, offset):
    """
    对 B 进行平移 offset 后，与 A 计算子 NFP。
    这里示例简单：若平移后 B 与 A 不相交，返回平移后的 B 点列表；否则返回 None。
    """
    moved_B = copy.deepcopy(B)
    moved_B['offsetx'] = offset['x']
    moved_B['offsety'] = offset['y']
    if not intersect(A, moved_B):
        return [{'x': p['x'] + offset['x'], 'y': p['y'] + offset['y']} for p in B['points']]
    else:
        return None


def segmented_polygon_slide(A, B, direction):
    """
    对轨道多边形 B 在方向 direction 上进行分段滑动求解，
    求解从 t=0 到 t_total（方向向量 translate 的长度）过程中，
    每当 B 上的边与 A 的顶点接触时将滑动过程分段。
    返回一个列表，每个元素包含：
      - t_range: (t_start, t_end)
      - offset: 对应分段中点的平移偏移量
      - nfp: 该分段下计算得到的子 NFP（此处示例返回平移后的 B 点）
    """
    t_total = math.sqrt(direction['x'] ** 2 + direction['y'] ** 2)
    unit_dir = normalize_vector(direction)
    t_candidates = [0.0]
    for i in range(len(B['points']) - 1):
        P = B['points'][i]
        Q = B['points'][i + 1]
        for v in A['points']:
            t_val = find_slide_parameter_for_edge_and_vertex(P, Q, unit_dir, v)
            if t_val is not None and t_val < t_total:
                t_candidates.append(t_val)
    t_candidates.append(t_total)
    t_candidates = sorted(set(t_candidates))

    segments = []
    for i in range(len(t_candidates) - 1):
        t_start = t_candidates[i]
        t_end = t_candidates[i + 1]
        t_mid = (t_start + t_end) / 2.0
        offset = {'x': t_mid * unit_dir['x'], 'y': t_mid * unit_dir['y']}
        sub_nfp = compute_sub_nfp(A, B, offset)
        segments.append({'t_range': (t_start, t_end), 'offset': offset, 'nfp': sub_nfp})
    return segments


# =====================================================================
# 4) NFP 多边形计算 (基于 orbiting 方法，并加入分段求解)
# 修改后的函数命名为 nfp_polygon_segmented
# =====================================================================
def nfp_polygon_segmented(A, B, inside=True, search_edges=False):
    """
    基于 orbiting 方法计算 NFP，同时对轨道多边形在滑动过程中分段求解，
    考虑当轨道多边形的边经过固定多边形的顶点时产生子 NFP。
    返回一个列表，每个元素包含分段信息及相应的子 NFP 数据。
    """
    if A is None or len(A['points']) < 3 or B is None or len(B['points']) < 3:
        return None

    A['offsetx'] = 0
    A['offsety'] = 0

    # 初始化 A 和 B 的标记
    for pt in A['points']:
        pt['marked'] = False
    for pt in B['points']:
        pt['marked'] = False

    # 确定起始点
    if not inside:
        min_a = min(A['points'], key=lambda p: p['y'])
        max_b = max(B['points'], key=lambda p: p['y'])
        start_point = {
            'x': min_a['x'] - max_b['x'],
            'y': min_a['y'] - max_b['y'],
        }
    else:
        start_point = search_start_point(A, B, inside)

    segmented_results = []  # 用于保存所有分段求解的结果

    while start_point:
        # 将 B 平移到起始位置
        B['offsetx'] = start_point['x']
        B['offsety'] = start_point['y']

        current_NFP = [{
            'x': B['points'][0]['x'] + B['offsetx'],
            'y': B['points'][0]['y'] + B['offsety'],
        }]
        referencex = current_NFP[0]['x']
        referencey = current_NFP[0]['y']
        startx, starty = referencex, referencey
        counter = 0
        len_a = len(A['points'])
        len_b = len(B['points'])
        prevvector = None

        # 本次 orbiting 过程中，记录分段信息
        orbit_segments = []

        while counter < 10 * (len_a + len_b):
            touching = []
            for i in range(len_a):
                nexti = 0 if i == len_a - 1 else i + 1
                for j in range(len_b):
                    nextj = 0 if j == len_b - 1 else j + 1

                    # 触点判定
                    if (almost_equal(A['points'][i]['x'], B['points'][j]['x'] + B['offsetx'])
                            and almost_equal(A['points'][i]['y'], B['points'][j]['y'] + B['offsety'])):
                        touching.append({'type': 0, 'A': i, 'B': j})
                    elif on_segment(
                            A['points'][i],
                            A['points'][nexti],
                            {'x': B['points'][j]['x'] + B['offsetx'],
                             'y': B['points'][j]['y'] + B['offsety']},
                    ):
                        touching.append({'type': 1, 'A': nexti, 'B': j})
                    elif on_segment(
                            {'x': B['points'][j]['x'] + B['offsetx'],
                             'y': B['points'][j]['y'] + B['offsety']},
                            {'x': B['points'][nextj]['x'] + B['offsetx'],
                             'y': B['points'][nextj]['y'] + B['offsety']},
                            A['points'][i],
                    ):
                        touching.append({'type': 2, 'A': i, 'B': nextj})

            vectors = []
            for t in touching:
                vertex_a = {'A': A['points'][t['A']], 'marked': True}
                prev_a_index = t['A'] - 1 if t['A'] - 1 >= 0 else len_a - 1
                next_a_index = t['A'] + 1 if t['A'] + 1 < len_a else 0
                prev_a = A['points'][prev_a_index]
                next_a = A['points'][next_a_index]

                vertex_b = {'A': B['points'][t['B']]}
                prev_b_index = t['B'] - 1 if t['B'] - 1 >= 0 else len_b - 1
                next_b_index = t['B'] + 1 if t['B'] + 1 < len_b else 0
                prev_b = B['points'][prev_b_index]
                next_b = B['points'][next_b_index]

                if t['type'] == 0:
                    v_a1 = {
                        'x': prev_a['x'] - vertex_a['A']['x'],
                        'y': prev_a['y'] - vertex_a['A']['y'],
                        'start': vertex_a['A'],
                        'end': prev_a,
                    }
                    v_a2 = {
                        'x': next_a['x'] - vertex_a['A']['x'],
                        'y': next_a['y'] - vertex_a['A']['y'],
                        'start': vertex_a['A'],
                        'end': next_a,
                    }
                    v_b1 = {
                        'x': vertex_b['A']['x'] - prev_b['x'],
                        'y': vertex_b['A']['y'] - prev_b['y'],
                        'start': prev_b,
                        'end': vertex_b['A'],
                    }
                    v_b2 = {
                        'x': vertex_b['A']['x'] - next_b['x'],
                        'y': vertex_b['A']['y'] - next_b['y'],
                        'start': next_b,
                        'end': vertex_b['A'],
                    }
                    vectors.extend([v_a1, v_a2, v_b1, v_b2])
                elif t['type'] == 1:
                    vectors.append({
                        'x': vertex_a['A']['x'] - (vertex_b['A']['x'] + B['offsetx']),
                        'y': vertex_a['A']['y'] - (vertex_b['A']['y'] + B['offsety']),
                        'start': prev_a,
                        'end': vertex_a['A'],
                    })
                    vectors.append({
                        'x': prev_a['x'] - (vertex_b['A']['x'] + B['offsetx']),
                        'y': prev_a['y'] - (vertex_b['A']['y'] + B['offsety']),
                        'start': vertex_a['A'],
                        'end': prev_a,
                    })
                elif t['type'] == 2:
                    vectors.append({
                        'x': vertex_a['A']['x'] - (vertex_b['A']['x'] + B['offsetx']),
                        'y': vertex_a['A']['y'] - (vertex_b['A']['y'] + B['offsety']),
                        'start': prev_b,
                        'end': vertex_b['A'],
                    })
                    vectors.append({
                        'x': vertex_a['A']['x'] - (prev_b['x'] + B['offsetx']),
                        'y': vertex_a['A']['y'] - (prev_b['y'] + B['offsety']),
                        'start': vertex_b['A'],
                        'end': prev_b,
                    })

            translate = None
            max_d = 0
            for v in vectors:
                if almost_equal(v['x'], 0.0) and almost_equal(v['y'], 0.0):
                    continue
                if prevvector:
                    dotv = v['x'] * prevvector['x'] + v['y'] * prevvector['y']
                    if dotv < 0:
                        vectorlength = math.sqrt(v['x'] ** 2 + v['y'] ** 2)
                        unitv = {'x': v['x'] / vectorlength, 'y': v['y'] / vectorlength}
                        prevlength = math.sqrt(prevvector['x'] ** 2 + prevvector['y'] ** 2)
                        prevunit = {'x': prevvector['x'] / prevlength, 'y': prevvector['y'] / prevlength}
                        cross_ = abs(unitv['y'] * prevunit['x'] - unitv['x'] * prevunit['y'])
                        if cross_ < 0.0001:
                            continue
                d = polygon_slide_distance(A, B, v, True)
                vecd2 = v['x'] ** 2 + v['y'] ** 2
                if d is None or d ** 2 > vecd2:
                    vecd = math.sqrt(vecd2)
                    d = vecd
                if d and d > max_d:
                    max_d = d
                    translate = v

            if not translate or almost_equal(max_d, 0.0):
                current_NFP = None
                break

            translate['start']['marked'] = True
            translate['end']['marked'] = True
            prevvector = translate

            # 对本次计算得到的滑动向量 translate 进行分段求解
            segs = segmented_polygon_slide(A, B, translate)
            orbit_segments.append({
                'translation': translate,
                'segments': segs
            })

            # 按照原算法，应用全量滑动向量 translate
            referencex += translate['x']
            referencey += translate['y']
            if almost_equal(referencex, startx) and almost_equal(referencey, starty):
                break
            current_NFP.append({'x': referencex, 'y': referencey})
            B['offsetx'] += translate['x']
            B['offsety'] += translate['y']
            counter += 1

        # 保存本次 orbiting 得到的分段信息及当前 NFP 点序列
        segmented_results.append({
            'start_point': start_point,
            'NFP': current_NFP,
            'orbit_segments': orbit_segments
        })

        if not search_edges:
            break
        start_point = search_start_point(A, B, inside, segmented_results)

    return segmented_results

def search_start_point(A, B, inside=True, NFP=None):
    A = copy.deepcopy(A)
    B = copy.deepcopy(B)

    for i in range(0, len(A['points']) - 1):
        if not A['points'][i].get('marked'):
            A['points'][i]['marked'] = True
            for j in range(len(B['points'])):
                B['offsetx'] = A['points'][i]['x'] - B['points'][j]['x']
                B['offsety'] = A['points'][i]['y'] - B['points'][j]['y']

                bin_side = None
                for k in range(len(B['points'])):
                    inpoly = point_in_polygon(
                        {'x': B['points'][k]['x'] + B['offsetx'],
                         'y': B['points'][k]['y'] + B['offsety']},
                        A,
                    )
                    if inpoly is not None:
                        bin_side = inpoly
                        break

                if bin_side is None:
                    return None

                start_point = {'x': B['offsetx'], 'y': B['offsety']}
                if ((bin_side and inside) or (not bin_side and not inside)) \
                    and (not intersect(A, B)) and (not inNfp(start_point, NFP)):
                    return start_point

                vx = A['points'][i+1]['x'] - A['points'][i]['x']
                vy = A['points'][i+1]['y'] - A['points'][i]['y']

                d1 = polygon_projection_distance(A, B, {'x': vx, 'y': vy})
                d2 = polygon_projection_distance(B, A, {'x': -vx, 'y': -vy})
                d = None
                if d1 is not None and d2 is not None:
                    d = min(d1, d2)
                elif d1 is None and d2 is not None:
                    d = d2
                elif d1 is not None and d2 is None:
                    d = d1

                if not (d is not None and not almost_equal(d, 0) and d > 0):
                    continue

                vd2 = vx*vx + vy*vy
                if d*d < vd2 and not almost_equal(d*d, vd2):
                    vd = math.sqrt(vd2)
                    vx *= d/vd
                    vy *= d/vd

                B['offsetx'] += vx
                B['offsety'] += vy

                for k in range(len(B['points'])):
                    inpoly = point_in_polygon(
                        {'x': B['points'][k]['x'] + B['offsetx'],
                         'y': B['points'][k]['y'] + B['offsety']},
                        A,
                    )
                    if inpoly is not None:
                        bin_side = inpoly
                        break

                start_point = {'x': B['offsetx'], 'y': B['offsety']}
                if ((bin_side and inside) or (not bin_side and not inside)) \
                    and (not intersect(A, B)) and (not inNfp(start_point, NFP)):
                    return start_point

    return None

def inNfp(p, nfp):
    if not nfp or len(nfp) == 0:
        return False

    for poly in nfp:
        for pt in poly:
            if almost_equal(p['x'], pt['x']) and almost_equal(p['y'], pt['y']):
                return True
    return False

# =====================================================================
# 5) 点在多边形内 (射线法) 以及 intersect 相关函数
# =====================================================================
def point_in_polygon(point, polygon):
    if isinstance(polygon, list):
        polygon = {'points': polygon}
    if len(polygon.get('points', [])) < 3:
        return None

    inside = False
    offsetx = polygon.get('offsetx', 0)
    offsety = polygon.get('offsety', 0)
    pts = polygon['points']
    j = len(pts) - 1
    for i in range(len(pts)):
        xi = pts[i]['x'] + offsetx
        yi = pts[i]['y'] + offsety
        xj = pts[j]['x'] + offsetx
        yj = pts[j]['y'] + offsety

        if almost_equal(xi, point['x']) and almost_equal(yi, point['y']):
            return None
        if on_segment({'x': xi, 'y': yi}, {'x': xj, 'y': yj}, point):
            return None
        if almost_equal(xi, xj) and almost_equal(yi, yj):
            j = i
            continue

        intersect_flag = ((yi > point['y']) != (yj > point['y'])) and \
                         (point['x'] < (xj - xi) * (point['y'] - yi) / (yj - yi) + xi)
        if intersect_flag:
            inside = not inside
        j = i
    return inside


def intersect(A, B):
    a_offsetx = A.get('offsetx', 0)
    a_offsety = A.get('offsety', 0)
    b_offsetx = B.get('offsetx', 0)
    b_offsety = B.get('offsety', 0)

    A = copy.deepcopy(A)
    B = copy.deepcopy(B)
    len_a = len(A['points'])
    len_b = len(B['points'])

    for i in range(len_a - 1):
        for j in range(len_b - 1):
            a1 = {'x': A['points'][i]['x'] + a_offsetx,
                  'y': A['points'][i]['y'] + a_offsety}
            a2 = {'x': A['points'][i + 1]['x'] + a_offsetx,
                  'y': A['points'][i + 1]['y'] + a_offsety}
            b1 = {'x': B['points'][j]['x'] + b_offsetx,
                  'y': B['points'][j]['y'] + b_offsety}
            b2 = {'x': B['points'][j + 1]['x'] + b_offsetx,
                  'y': B['points'][j + 1]['y'] + b_offsety}

            pre_vb_index = len_b - 1 if j == 0 else j - 1
            pre_va_index = len_a - 1 if i == 0 else i - 1
            next_b_index = 0 if (j + 1) == (len_b - 1) else j + 2
            next_a_index = 0 if (i + 1) == (len_a - 1) else i + 2

            if (on_segment(a1, a2, b1)
                    or (almost_equal(a1['x'], b1['x']) and almost_equal(a1['y'], b1['y']))):
                b0in = point_in_polygon(
                    {'x': B['points'][pre_vb_index]['x'] + b_offsetx,
                     'y': B['points'][pre_vb_index]['y'] + b_offsety},
                    A)
                b2in = point_in_polygon(b2, A)
                if (b0in and not b2in) or (not b0in and b2in):
                    return True
                else:
                    continue

            if (on_segment(a1, a2, b2)
                    or (almost_equal(a2['x'], b2['x']) and almost_equal(a2['y'], b2['y']))):
                b1in = point_in_polygon(b1, A)
                b3 = {'x': B['points'][next_b_index]['x'] + b_offsetx,
                      'y': B['points'][next_b_index]['y'] + b_offsety}
                b3in = point_in_polygon(b3, A)
                if (b1in and not b3in) or (not b1in and b3in):
                    return True
                else:
                    continue

            if (on_segment(b1, b2, a1)
                    or (almost_equal(a1['x'], b2['x']) and almost_equal(a1['y'], b2['y']))):
                a0 = {'x': A['points'][pre_va_index]['x'] + a_offsetx,
                      'y': A['points'][pre_va_index]['y'] + a_offsety}
                a2in = point_in_polygon(a2, B)
                a0in = point_in_polygon(a0, B)
                if (a0in and not a2in) or (not a0in and a2in):
                    return True
                else:
                    continue

            if (on_segment(b1, b2, a2)
                    or (almost_equal(a2['x'], b1['x']) and almost_equal(a2['y'], b1['y']))):
                a1in = point_in_polygon(a1, B)
                a3 = {'x': A['points'][next_a_index]['x'] + a_offsetx,
                      'y': A['points'][next_a_index]['y'] + a_offsety}
                a3in = point_in_polygon(a3, B)
                if (a1in and not a3in) or (not a1in and a3in):
                    return True
                else:
                    continue

            if line_intersect(b1, b2, a1, a2):
                return True

    return False


def line_intersect(A, B, E, F, infinite=None):
    a1 = B['y'] - A['y']
    b1 = A['x'] - B['x']
    c1 = B['x'] * A['y'] - A['x'] * B['y']
    a2 = F['y'] - E['y']
    b2 = E['x'] - F['x']
    c2 = F['x'] * E['y'] - E['x'] * F['y']
    denom = a1 * b2 - a2 * b1
    if almost_equal(denom, 0.0):
        return None
    x = (b1 * c2 - b2 * c1) / denom
    y = (a2 * c1 - a1 * c2) / denom

    if infinite is None:
        if abs(A['x'] - B['x']) > TOL:
            tmp = (x < A['x'] or x > B['x']) if A['x'] < B['x'] else (x > A['x'] or x < B['x'])
            if tmp:
                return None
            tmp = (y < A['y'] or y > B['y']) if A['y'] < B['y'] else (y > A['y'] or y < B['y'])
            if tmp:
                return None
        if abs(E['x'] - F['x']) > TOL:
            tmp = (x < E['x'] or x > F['x']) if E['x'] < F['x'] else (x > E['x'] or x < F['x'])
            if tmp:
                return None
            tmp = (y < E['y'] or y > F['y']) if E['y'] < F['y'] else (y > E['y'] or y < F['y'])
            if tmp:
                return None

    return {'x': x, 'y': y}


# =====================================================================
# 6) polygon_projection_distance, point_distance, polygon_slide_distance 等
# =====================================================================
def polygon_projection_distance(A, B, direction):
    b_offsetx = B.get('offsetx', 0)
    b_offsety = B.get('offsety', 0)
    a_offsetx = A.get('offsetx', 0)
    a_offsety = A.get('offsety', 0)

    A = copy.deepcopy(A)
    B = copy.deepcopy(B)
    edge_a = A['points']
    edge_b = B['points']
    distance = None
    p = {}
    s1 = {}
    s2 = {}

    for i in range(len(edge_b)):
        min_projection = None
        for j in range(len(edge_a) - 1):
            p['x'] = edge_b[i]['x'] + b_offsetx
            p['y'] = edge_b[i]['y'] + b_offsety
            s1['x'] = edge_a[j]['x'] + a_offsetx
            s1['y'] = edge_a[j]['y'] + a_offsety
            s2['x'] = edge_a[j + 1]['x'] + a_offsetx
            s2['y'] = edge_a[j + 1]['y'] + a_offsety

            cross_ = abs((s2['y'] - s1['y']) * direction['x'] - (s2['x'] - s1['x']) * direction['y'])
            if cross_ < TOL:
                continue

            d = point_distance(p, s1, s2, direction)
            if d and (min_projection is None or d < min_projection):
                min_projection = d

        if min_projection and (distance is None or min_projection > distance):
            distance = min_projection

    return distance


def point_distance(p, s1, s2, normal, infinite=None):
    normal = normalize_vector(normal)
    dir_point = {'x': normal['y'], 'y': -normal['x']}

    pdot = p['x'] * dir_point['x'] + p['y'] * dir_point['y']
    s1dot = s1['x'] * dir_point['x'] + s1['y'] * dir_point['y']
    s2dot = s2['x'] * dir_point['x'] + s2['y'] * dir_point['y']

    pdotnorm = p['x'] * normal['x'] + p['y'] * normal['y']
    s1dotnorm = s1['x'] * normal['x'] + s1['y'] * normal['y']
    s2dotnorm = s2['x'] * normal['x'] + s2['y'] * normal['y']

    if infinite is None:
        if (((pdot < s1dot or almost_equal(pdot, s1dot))
             and (pdot < s2dot or almost_equal(pdot, s2dot)))
                or ((pdot > s1dot or almost_equal(pdot, s1dot))
                    and (pdot > s2dot or almost_equal(pdot, s2dot)))):
            return None

        if (almost_equal(pdot, s1dot) and almost_equal(pdot, s2dot)):
            if pdotnorm > s1dotnorm and pdotnorm > s2dotnorm:
                return min(pdotnorm - s1dotnorm, pdotnorm - s2dotnorm)
            if pdotnorm < s1dotnorm and pdotnorm < s2dotnorm:
                return -min(s1dotnorm - pdotnorm, s2dotnorm - pdotnorm)

    return -(
            pdotnorm - s1dotnorm
            + (s1dotnorm - s2dotnorm) * (s1dot - pdot) / (s1dot - s2dot)
    )


def polygon_slide_distance(A, B, direction, ignorenegative):
    b_offsetx = B.get('offsetx', 0)
    b_offsety = B.get('offsety', 0)
    a_offsetx = A.get('offsetx', 0)
    a_offsety = A.get('offsety', 0)

    A = copy.deepcopy(A)
    B = copy.deepcopy(B)

    if A['points'][-1] != A['points'][0]:
        A['points'].append(A['points'][0])
    if B['points'][-1] != B['points'][0]:
        B['points'].append(B['points'][0])

    edge_a = A['points']
    edge_b = B['points']
    distance = None

    dir_point = normalize_vector(direction)

    for i in range(len(edge_b) - 1):
        for j in range(len(edge_a) - 1):
            A1 = {'x': edge_a[j]['x'] + a_offsetx, 'y': edge_a[j]['y'] + a_offsety}
            A2 = {'x': edge_a[j + 1]['x'] + a_offsetx, 'y': edge_a[j + 1]['y'] + a_offsety}
            B1 = {'x': edge_b[i]['x'] + b_offsetx, 'y': edge_b[i]['y'] + b_offsety}
            B2 = {'x': edge_b[i + 1]['x'] + b_offsetx, 'y': edge_b[i + 1]['y'] + b_offsety}

            if (almost_equal(A1['x'], A2['x']) and almost_equal(A1['y'], A2['y'])):
                continue
            if (almost_equal(B1['x'], B2['x']) and almost_equal(B1['y'], B2['y'])):
                continue

            d = segment_distance(A1, A2, B1, B2, dir_point)
            if d and (distance is None or d < distance):
                if not ignorenegative or d > 0 or almost_equal(d, 0):
                    distance = d

    return distance


def segment_distance(A, B, E, F, direction):
    normal = {'x': direction['y'], 'y': -direction['x']}
    reverse = {'x': -direction['x'], 'y': -direction['y']}

    dot_a = A['x'] * normal['x'] + A['y'] * normal['y']
    dot_b = B['x'] * normal['x'] + B['y'] * normal['y']
    dot_e = E['x'] * normal['x'] + E['y'] * normal['y']
    dot_f = F['x'] * normal['x'] + F['y'] * normal['y']

    cross_a = A['x'] * direction['x'] + A['y'] * direction['y']
    cross_b = B['x'] * direction['x'] + B['y'] * direction['y']
    cross_e = E['x'] * direction['x'] + E['y'] * direction['y']
    cross_f = F['x'] * direction['x'] + F['y'] * direction['y']

    ab_min = min(dot_a, dot_b)
    ab_max = max(dot_a, dot_b)
    ef_min = min(dot_e, dot_f)
    ef_max = max(dot_e, dot_f)

    if almost_equal(ab_max, ef_min, TOL) or almost_equal(ab_min, ef_max, TOL):
        return None
    if ab_max < ef_min or ab_min > ef_max:
        return None

    if ((ab_max > ef_max and ab_min < ef_min) or (ef_max > ab_max and ef_min < ab_min)):
        overlap = 1
    else:
        min_max = min(ab_max, ef_max)
        max_min = max(ab_min, ef_min)
        max_max = max(ab_max, ef_max)
        min_min = min(ab_min, ef_min)
        overlap = (min_max - max_min) / (max_max - min_min)

    distances = []

    if almost_equal(dot_a, dot_e):
        distances.append(cross_a - cross_e)
    elif almost_equal(dot_a, dot_f):
        distances.append(cross_a - cross_f)
    elif ef_min < dot_a < ef_max:
        d = point_distance(A, E, F, reverse)
        if d and almost_equal(d, 0):
            db = point_distance(B, E, F, reverse, True)
            if db < 0 or almost_equal(db * overlap, 0):
                d = None
        if d:
            distances.append(d)

    if almost_equal(dot_b, dot_e):
        distances.append(cross_b - cross_e)
    elif almost_equal(dot_b, dot_f):
        distances.append(cross_b - cross_f)
    elif ef_min < dot_b < ef_max:
        d = point_distance(B, E, F, reverse)
        if d and almost_equal(d, 0):
            da = point_distance(A, E, F, reverse, True)
            if da < 0 or almost_equal(da * overlap, 0):
                d = None
        if d:
            distances.append(d)

    if dot_e > ab_min and dot_e < ab_max:
        d = point_distance(E, A, B, direction)
        if d and almost_equal(d, 0):
            df = point_distance(F, A, B, direction, True)
            if df < 0 or almost_equal(df * overlap, 0):
                d = None
        if d:
            distances.append(d)

    if dot_f > ab_min and dot_f < ef_max:
        d = point_distance(F, A, B, direction)
        if d and almost_equal(d, 0):
            de = point_distance(E, A, B, direction, True)
            if de < 0 or almost_equal(de * overlap, 0):
                d = None
        if d:
            distances.append(d)

    if not distances:
        return None

    return min(distances)


# =====================================================================
# 7) 多边形旋转 & 外包矩形计算 (原函数保留)
# =====================================================================
def rotate_polygon(polygon, angle):
    rotated = {'points': []}
    rad = angle * math.pi / 180.0
    for p in polygon:
        x, y = p['x'], p['y']
        rx = x * math.cos(rad) - y * math.sin(rad)
        ry = x * math.sin(rad) + y * math.cos(rad)
        rotated['points'].append({'x': rx, 'y': ry})

    bounds = get_polygon_bounds(rotated['points'])
    if bounds:
        rotated['x'] = bounds['x']
        rotated['y'] = bounds['y']
        rotated['width'] = bounds['width']
        rotated['height'] = bounds['height']
    return rotated


def get_polygon_bounds(polygon):
    if polygon is None or len(polygon) < 3:
        return None
    xs = [p['x'] for p in polygon]
    ys = [p['y'] for p in polygon]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return {
        'x': xmin, 'y': ymin,
        'width': xmax - xmin, 'height': ymax - ymin
    }


# =====================================================================
# 8) Shapely 与 R 树辅助函数
# =====================================================================
def dicts_to_shapely_polygon(poly_points):
    coords = [(p['x'], p['y']) for p in poly_points]
    return Polygon(coords)


def polygons_intersect_shapely(A_points, B_points):
    polyA = dicts_to_shapely_polygon(A_points)
    polyB = dicts_to_shapely_polygon(B_points)
    return polyA.intersects(polyB)


def build_rtree_for_polygon_edges(polygon_points):
    """
    将多边形的每条边存入 R 树索引，使用批量加载进行优化
    返回: (rtree_idx, edges_list)
    """
    # 保证多边形闭合
    if polygon_points[-1] != polygon_points[0]:
        polygon_points = polygon_points + [polygon_points[0]]

    items = []
    edges_list = []
    for i in range(len(polygon_points) - 1):
        x1, y1 = polygon_points[i]['x'], polygon_points[i]['y']
        x2, y2 = polygon_points[i + 1]['x'], polygon_points[i + 1]['y']
        minx, maxx = min(x1, x2), max(x1, x2)
        miny, maxy = min(y1, y2), max(y1, y2)
        bbox = (minx, miny, maxx, maxy)
        items.append((i, bbox, None))
        edges_list.append(((x1, y1), (x2, y2)))
    p = index.Property()
    p.dimension = 2
    rtree_idx = index.Index(items, properties=p)
    return rtree_idx, edges_list


def segment_intersect_with_polygon(segment, rtree_idx, edges_list):
    (x1, y1), (x2, y2) = segment
    minx, maxx = min(x1, x2), max(x1, x2)
    miny, maxy = min(y1, y2), max(y1, y2)
    seg_line = LineString([segment[0], segment[1]])
    candidate_ids = list(rtree_idx.intersection((minx, miny, maxx, maxy)))
    for i in candidate_ids:
        edge = edges_list[i]
        edge_line = LineString([edge[0], edge[1]])
        if seg_line.intersects(edge_line):
            return True
    return False


# =====================================================================
# 9) 拆分 Orbiting（示例）
# =====================================================================
def orbit_polygon_edge(A_points, B_points, edge_index, inside=True):
    # 示例：对多边形A的某条边进行 orbiting
    return Polygon([])


def nfp_polygon_refined(A_points, B_points, inside=True):
    # 保证闭合
    if A_points[-1] != A_points[0]:
        A_points = A_points + [A_points[0]]
    union_list = []
    for i in range(len(A_points) - 1):
        edge_nfp = orbit_polygon_edge(A_points, B_points, i, inside=inside)
        if not edge_nfp.is_empty:
            union_list.append(edge_nfp)
    if union_list:
        final_nfp = unary_union(union_list)
        return final_nfp  # 可能返回 Polygon 或 MultiPolygon
    else:
        return Polygon([])


# =====================================================================
# 示例调用：使用 nfp_polygon_segmented 进行分段求解
# =====================================================================
if __name__ == "__main__":
    # 构造示例多边形 A（固定）和 B（轨道）
    A = {
        'points': [
            {'x': 0, 'y': 0},
            {'x': 100, 'y': 0},
            {'x': 100, 'y': 100},
            {'x': 0, 'y': 100},
        ]
    }
    B = {
        'points': [
            {'x': 0, 'y': 0},
            {'x': 30, 'y': 0},
            {'x': 30, 'y': 20},
            {'x': 0, 'y': 20},
        ]
    }
    # 计算分段 NFP，inside 参数视具体情况而定
    segmented_nfp = nfp_polygon_segmented(A, B, inside=True, search_edges=False)
    # 输出分段求解结果
    for orbit in segmented_nfp:
        print("起始偏移:", orbit['start_point'])
        print("NFP 路径点:", orbit['NFP'])
        for seg in orbit['orbit_segments']:
            print("  本次滑动向量:", seg['translation'])
            for s in seg['segments']:
                print("    t 范围:", s['t_range'], "偏移:", s['offset'], "子 NFP:", s['nfp'])
