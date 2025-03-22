from collections import defaultdict
from PIL import Image, ImageDraw
import random
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np
import os
import re
import csv
import ast
import math

def rotate_polygon(points, angle, center_x=0.0, center_y=0.0, angle_in_degrees=True):
    """
    将多边形坐标绕 (center_x, center_y) 进行逆时针旋转。
    """
    if angle_in_degrees:
        angle = math.radians(angle)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rotated = []
    for (x, y) in points:
        dx = x - center_x
        dy = y - center_y
        x_new = center_x + dx * cos_a - dy * sin_a
        y_new = center_y + dx * sin_a + dy * cos_a
        rotated.append((x_new, y_new))
    return rotated

def load_shapes_from_txt(txt_file_path):
    """
    从绝对路径txt_file_path中读取文本内容，
    利用 ast.literal_eval 将其解析为 Python 对象（列表）。
    返回 shapes 列表。
    """
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"文件不存在：{txt_file_path}")

    with open(txt_file_path, 'r', encoding='utf-8') as f:
        data_str = f.read()
    shapes = ast.literal_eval(data_str)
    return shapes

def create_new_data_structure(shapes, shift_data):
    """
    根据 shift_data 的嵌套结构，逐步处理每个 move_step。
    使用 p_ids 和 shapes 构建新的数据结构，不进行平移和旋转。
    """
    new_data_structure = []

    for s_data in shift_data:  # 外层循环，遍历 shift_data 中的每个条目
        for move_step in s_data:  # 内层循环，逐步处理 move_step
            p_id_str = move_step.get("p_id")
            tmp_shape = shapes[int(p_id_str)]

            if tmp_shape is None:
                print(f"警告：在 shapes 中未找到匹配的 p_id {p_id_str}，跳过。")
                continue

            # 构建新条目
            new_entry = {
                "p_id": tmp_shape.get("p_id"),  # 从 shapes 中获取的 p_id
                "rotation": move_step.get("rotation", 0),  # 从 shift_data 中获取的旋转
                "x_translation": move_step.get("x", 0),  # 从 shift_data 中获取的 x 平移
                "y_translation": move_step.get("y", 0),  # 从 shift_data 中获取的 y 平移
                "original_shape_data": tmp_shape  # 保留 shapes 中的原始数据
            }
            new_data_structure.append(new_entry)

    print(f"新数据结构生成完成，共包含 {len(new_data_structure)} 条记录。")
    return new_data_structure

def parse_obj_file(file_path):
    print(f"Parsing OBJ file: {file_path}")
    vertices = []
    textures = []
    faces = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt':
                textures.append([float(parts[1]), float(parts[2])])
            elif parts[0] == 'f':
                texture_indices = []
                for part in parts[1:]:
                    tokens = part.split('/')
                    if len(tokens) >= 2 and tokens[1]:
                        texture_indices.append(int(tokens[1]) - 1)
                if len(texture_indices) == 3:
                    faces.append(texture_indices)
    print(f"Parsed {len(vertices)} vertices, {len(textures)} texture coords, {len(faces)} faces.")
    return vertices, textures, faces

def calculate_pixel_coordinates(textures, faces, img_width, img_height):
    print("Calculating pixel coordinates...")
    pixel_coords = []
    for face in faces:
        face_coords = []
        for texture_index in face:
            texture_coord = textures[texture_index]
            pixel_x = texture_coord[0] * img_width
            pixel_y = (1 - texture_coord[1]) * img_height
            face_coords.append((pixel_x, pixel_y))
        pixel_coords.append(face_coords)
    print(f"Calculated pixel coords for {len(faces)} faces.")
    return pixel_coords

def group_connected_triangles(pixel_coords):
    print("Grouping connected triangles...")
    edge_to_triangles = defaultdict(set)

    for tri_index, tri in enumerate(pixel_coords):
        for i in range(3):
            edge = tuple(sorted((tuple(tri[i]), tuple(tri[(i + 1) % 3]))))
            edge_to_triangles[edge].add(tri_index)

    groups = []
    ungrouped_triangles = set(range(len(pixel_coords)))

    while ungrouped_triangles:
        group = set()
        queue = {ungrouped_triangles.pop()}
        while queue:
            tri_index = queue.pop()
            group.add(tri_index)
            for i in range(3):
                edge = tuple(sorted((tuple(pixel_coords[tri_index][i]), tuple(pixel_coords[tri_index][(i + 1) % 3]))))
                for neighbor in edge_to_triangles[edge]:
                    if neighbor in ungrouped_triangles:
                        queue.add(neighbor)
                        ungrouped_triangles.remove(neighbor)
        groups.append(group)

    print(f"Formed {len(groups)} groups of connected triangles.")
    return groups

def extract_outer_boundary_and_holes(triangles, pixel_coords):
    polygons = []
    for tri_index in triangles:
        coords = pixel_coords[tri_index]
        polygon = Polygon(coords)
        polygons.append(polygon)

    merged_polygon = unary_union(polygons)

    exterior_coords = []
    interiors_coords = []

    if isinstance(merged_polygon, Polygon):
        exterior_coords = list(merged_polygon.exterior.coords)
        interiors_coords = [list(interior.coords) for interior in merged_polygon.interiors]
    elif isinstance(merged_polygon, MultiPolygon):
        ex_coords = []
        in_coords = []
        for poly in merged_polygon:
            ex_coords.extend(list(poly.exterior.coords))
            for interior in poly.interiors:
                in_coords.append(list(interior.coords))
        exterior_coords = ex_coords
        interiors_coords = in_coords
    return exterior_coords, interiors_coords

def generate_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def parse_points_string(points_str):
    points_str = points_str.strip()
    if not points_str:
        return []
    points = points_str.split(';')
    coords = []
    for p in points:
        p = p.strip().strip('"\'')
        match = re.match(r'^\(?\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*\)?$', p)
        if match:
            x, y = match.groups()
            x = float(x)
            y = float(y)
            y = abs(y)  # 将y坐标变为正数
            coords.append((x, y))
    return coords

def load_corners(corners_csv_path):
    print(f"Loading corners from {corners_csv_path}")
    corners_dict = {}
    with open(corners_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            no_str = row['NO']
            corner_points_str = row['Corner Points'].strip() if 'Corner Points' in row and row['Corner Points'] else ''
            if '|' in corner_points_str:
                sub_sets = corner_points_str.split('|')
                all_corners = []
                for s in sub_sets:
                    all_corners.extend(parse_points_string(s.strip()))
                corners_dict[no_str] = all_corners
            else:
                corners_dict[no_str] = parse_points_string(corner_points_str)
            count += 1
        print(f"Loaded corners for {count} polygons.")
    return corners_dict

def load_simplified_polygons(simplified_csv_path):
    print(f"Loading simplified polygons from {simplified_csv_path}")
    polygons_dict = {}
    with open(simplified_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            no_str = row['NO']
            polygon_str = row['Polygon Points'].strip() if 'Polygon Points' in row and row['Polygon Points'] else ''
            polygon_coords = parse_points_string(polygon_str)
            polygons_dict[no_str] = polygon_coords
            count += 1
        print(f"Loaded {count} simplified polygons.")
    return polygons_dict

def draw_simplified_polygons(polygons_dict, output_path):
    print("Drawing simplified polygons to create base image...")
    all_x = []
    all_y = []
    for coords in polygons_dict.values():
        if coords:
            for (x,y) in coords:
                all_x.append(x)
                all_y.append(y)

    if not all_x or not all_y:
        print("No polygon data found, cannot draw base image.")
        return None, 0, 0

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width = int(max_x - min_x) + 100
    height = int(max_y - min_y) + 100

    base_img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(base_img)

    for no_str, coords in polygons_dict.items():
        if coords:
            poly_int = [(int(x - min_x + 50), int(y - min_y + 50)) for (x,y) in coords]
            draw.polygon(poly_int, outline='black', fill='lightgray')

    base_img.save(output_path)
    base_img.show()
    print(f"Base image saved to {output_path}")
    return (base_img, min_x, min_y)


def draw_full_textures(image_path, pixel_coords, groups, output_folder, test_folder, polygon_csv_path, new_solution, bin_polygon):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 打开原始图像，并转换为 RGBA 格式（确保有透明度通道）
    original_image = Image.open(image_path).convert('RGBA')

    # 从 polygon_csv_path 中读取所有多边形数据（NO和Polygon Points）
    polygons_dict = load_simplified_polygons(polygon_csv_path)

    # 读取 bin_polygon 里提供的画布尺寸（如 2048 x 2048）
    bin_width = bin_polygon.get('width', 2048)
    bin_height = bin_polygon.get('height', 2048)

    # 创建一张新的图像，用于绘制所有的多边形和纹理图像
    unified_img = Image.new('RGBA', (bin_width, bin_height), 'white')
    unified_draw = ImageDraw.Draw(unified_img)

    for i, group in enumerate(groups):
        # 提取该组面片合并后的外边界和内部孔洞坐标
        exterior_coords, interiors_coords = extract_outer_boundary_and_holes(group, pixel_coords)
        if not exterior_coords:
            continue

        # 创建裁剪掩膜
        mask = Image.new('L', original_image.size, 0)
        draw_mask = ImageDraw.Draw(mask)

        exterior_int = [(int(x), int(y)) for x, y in exterior_coords]
        draw_mask.polygon(exterior_int, fill=255)
        for hole_coords in interiors_coords:
            hole_int = [(int(x), int(y)) for x, y in hole_coords]
            draw_mask.polygon(hole_int, fill=0)

        bbox = mask.getbbox()
        if not bbox:
            continue

        # 裁剪纹理和掩膜
        cropped_image = original_image.crop(bbox)  # 原图裁出所需区域
        cropped_mask = mask.crop(bbox)  # 也同步裁出对应的 mask
        cropped_image.putalpha(cropped_mask)  # 将 mask 作为 alpha 通道

        no_str = str(i + 1)  # groups 的编号，通常对应 CSV 里的 NO

        # 读取与此 no_str 对应的 new_solution 条目
        matching_entry = next(
            (entry for entry in new_solution if str(entry.get("p_id", "")) == no_str),
            None
        )
        # 如果没找到，就跳过绘制
        if matching_entry is None:
            print(f"在 new_solution 中未找到 p_id={no_str} 的记录，跳过绘制。")
            continue

        # 如果找到了，就获取旋转 & 平移参数
        rotation_angle = matching_entry.get("rotation", 0.0)
        x_translation = matching_entry.get("x_translation", 0.0)
        y_translation = matching_entry.get("y_translation", 0.0)

        # 从 polygons_dict 中获取与 NO 对应的多边形坐标
        polygon_coords = polygons_dict.get(no_str, [])
        if not polygon_coords:
            print(f"在 {polygon_csv_path} 中未找到 NO={no_str} 的多边形数据。")
            continue

        # 将多边形坐标平移到统一图像坐标中
        poly_transformed = [(x + x_translation, y + y_translation) for (x, y) in polygon_coords]
        if rotation_angle != 0.0:
            rotated_poly = rotate_polygon(
                points=poly_transformed,
                angle=rotation_angle,
                center_x=0,
                center_y=0,
                angle_in_degrees=True
            )
        else:
            rotated_poly=poly_transformed

        # 在统一图上绘制多边形
        # unified_draw.polygon(rotated_poly, outline='black', fill='lightgray')

        if rotation_angle != 0.0:
            # 我们先把 Pillow 的 rotate 视作顺时针 => 传入 -rotation_angle 变成逆时针
            rotated_cropped_image = cropped_image.rotate(
                -rotation_angle,  # 注意取负号
                expand=True,
                center=(0, 0)  # 以左上角为中心旋转
            )
        else:
            rotated_cropped_image = cropped_image

        texture_offset_x = bbox[0] + x_translation
        texture_offset_y = bbox[1] + y_translation
        unified_img.alpha_composite(rotated_cropped_image, (int(texture_offset_x), int(texture_offset_y)))

    flipped_img = unified_img.transpose(Image.FLIP_TOP_BOTTOM)
    # 将合成结果保存到 test_folder 中
    test_output_file = os.path.join(test_folder, f'final.png')
    flipped_img.save(test_output_file)
    print(f'多边形与裁剪纹理在统一坐标系下合成图已保存：{test_output_file}')

def draw_groups_with_boundaries(pixel_coords, groups, img_width, img_height, output_path, corners_csv_path):
    corners_dict = load_corners(corners_csv_path)

    output_img = Image.new('RGB', (int(img_width), int(img_height)), 'white')
    draw = ImageDraw.Draw(output_img)
    group_colors = {i: generate_random_color() for i in range(len(groups))}

    # 绘制每组的三角形（填充色）
    for i, group in enumerate(groups):
        color = group_colors[i]
        for tri_index in group:
            draw.polygon(pixel_coords[tri_index], outline=color, fill=color)

    # 绘制外边界和孔洞，并在此处绘制角点
    for i, group in enumerate(groups):
        exterior_coords, interiors_coords = extract_outer_boundary_and_holes(group, pixel_coords)
        if exterior_coords:
            exterior_int = [(int(x), int(y)) for x, y in exterior_coords]
            draw.line(exterior_int + [exterior_int[0]], fill='black', width=5, joint='curve')
            for hole_coords in interiors_coords:
                hole_int = [(int(x), int(y)) for x, y in hole_coords]
                draw.line(hole_int + [hole_int[0]], fill='black', width=5, joint='curve')

            no_str = str(i + 1)
            corners = corners_dict.get(no_str, [])
            radius = 10
            for (cx, cy) in corners:
                cx_int, cy_int = int(cx), int(cy)
                draw.ellipse((cx_int - radius, cy_int - radius, cx_int + radius, cy_int + radius),
                             fill='red', outline='red')
    output_img.save(output_path)
    output_img.show()
    print(f"Groups with boundaries and corners drawn to {output_path}")

def main():
    obj_file_path = r'D:\CC\whu\Productions\WHU\Data\Tile_+001_+002\Tile_+001_+002.obj'
    image_path = r'D:\CC\whu\Productions\WHU\Data\Tile_+001_+002\Tile_+001_+002_0.jpg'
    output_image_path = r'C:/Users/Zhan/PycharmProjects/NFP/output_with_colored_groups_and_boundaries_fixed.png'
    output_folder = r'C:/Users/Zhan\PycharmProjects\NFP\cropped_textures'
    corners_csv_path = r'C:\Users\Zhan\PycharmProjects\NFP\corners_coordinates.csv'
    test_folder = r'C:\Users\Zhan\PycharmProjects\NFP\test_polygon'
    polygon_folder = r'C:\Users\Zhan\PycharmProjects\NFP\convex_hulls.csv'
    # CC的
    shapes_txt_file_path = r"C:\Users\Zhan\PycharmProjects\NFP\shapes.txt"
    shift_data_txt_file_path = r"C:\Users\Zhan\PycharmProjects\NFP\shift_data.txt"
    p_ids_txt_file_path = r"C:\Users\Zhan\PycharmProjects\NFP\p_ids.txt"

    # 加载 shapes / shift_data / p_ids  (若需要，可在后续加用)
    shapes = load_shapes_from_txt(shapes_txt_file_path)
    shift_data = load_shapes_from_txt(shift_data_txt_file_path)
    p_ids = load_shapes_from_txt(p_ids_txt_file_path)
    bin_polygon = {
        'points': [(0, 0), (0, 2048), (4096, 2048), (4096, 0)],  # 一个简单的2048x2048方形
        'width': 4096,
        'height': 2048
    }

    new_solution = create_new_data_structure(shapes, shift_data)

    print("Starting main process...")
    image = Image.open(image_path)
    img_width, img_height = image.size
    print(f"Image size: {img_width}x{img_height}")

    vertices, textures, faces = parse_obj_file(obj_file_path)
    pixel_coords = calculate_pixel_coordinates(textures, faces, img_width, img_height)
    groups = group_connected_triangles(pixel_coords)

    print("Drawing groups with boundaries and corners...")

    draw_groups_with_boundaries(pixel_coords, groups, img_width, img_height, output_image_path, corners_csv_path)

    draw_full_textures(image_path, pixel_coords, groups, output_folder, test_folder, polygon_folder, new_solution, bin_polygon)

if __name__ == "__main__":
    main()
