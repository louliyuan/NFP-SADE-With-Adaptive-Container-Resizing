# 项目交接文档 / Project Handover Document

**项目名称 / Project Name:** NFP-SADE With Adaptive Container Resizing
**项目类型 / Project Type:** 不规则图形嵌套优化算法 / Irregular Shape Nesting Optimization
**文档版本 / Doc Version:** v1.0
**创建日期 / Created:** 2026-03-16
**GitHub:** https://github.com/louliyuan/NFP-SADE-With-Adaptive-Container-Resizing

---

## 目录 / Table of Contents

1. [项目概述](#1-项目概述)
2. [技术架构](#2-技术架构)
3. [环境依赖](#3-环境依赖)
4. [安装与配置](#4-安装与配置)
5. [项目结构详解](#5-项目结构详解)
6. [核心算法说明](#6-核心算法说明)
7. [数据格式规范](#7-数据格式规范)
8. [运行指南](#8-运行指南)
9. [参数配置说明](#9-参数配置说明)
10. [输出结果说明](#10-输出结果说明)
11. [已知问题与限制](#11-已知问题与限制)
12. [后续开发建议](#12-后续开发建议)

---

## 1. 项目概述

### 1.1 业务背景

本项目解决将大量不规则形状的布料/零件多边形，以最高面积利用率排列到有限尺寸的矩形容器中，从而减少材料浪费，降低生产成本。

### 1.2 核心功能

| 功能 | 说明 |
|------|------|
| 不规则多边形嵌套 | 基于 NFP 算法，支持任意形状多边形 |
| 自适应容器调整 | 根据输入多边形总面积自动计算容器尺寸 |
| 旋转优化 | 支持任意角度旋转（默认 6° 步进，即 60 档） |
| 多容器支持 | 放不下的零件自动进入下一个容器 |
| 进化算法优化 | 使用 SADE（自适应差分进化）寻找全局最优排列 |
| 可视化输出 | 输出 PNG 图形展示排版结果 |
| 多格式输入 | 支持 CSV 多边形数据 和 DXF CAD 文件 |

### 1.3 算法核心

**NFP (No-Fit Polygon) + SADE (Self-Adaptive Differential Evolution)**

- **NFP**：精确计算两个多边形之间的无碰撞区域，是嵌套问题的经典几何方法
- **SADE**：进化算法，优化所有零件的排列顺序和旋转角度，寻找最小化材料浪费的解

---

## 2. 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        输入层 (Input)                        │
│           CSV 多边形数据  /  DXF CAD 文件                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    预处理层 (Preprocessing)                   │
│     tools/input_utls.py                                      │
│     • 解析CSV/DXF → 标准多边形格式                           │
│     • 多边形清洗（去除重复点、自交）                          │
│     • 间距偏移（Polygon Offset）                              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   核心算法层 (Core Engine)                    │
│     nfp_function.py                                          │
│                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │   Nester     │    │    SADE      │    │  NFP Cache   │  │
│   │  (调度器)    │◄──►│  (进化优化)  │◄──►│  (结果缓存)  │  │
│   └──────────────┘    └──────────────┘    └──────────────┘  │
│          │                                                   │
│          ▼                                                   │
│   ┌──────────────┐    ┌──────────────┐                       │
│   │  NFP Compute │    │  Placement   │                       │
│   │ tools/nfp_   │◄──►│  Worker      │                       │
│   │ utls.py      │    │  (贪心放置)  │                       │
│   └──────────────┘    └──────────────┘                       │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     输出层 (Output)                           │
│   shift_data_X.txt  │  polygons_X.txt  │  figure_X.png       │
│   (位置+旋转JSON)   │  (多边形定义)    │  (可视化图)         │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 关键依赖关系

```
test_nfp.py (入口)
    ├── nfp_function.py (核心)
    │       ├── tools/nfp_utls.py (几何计算)
    │       ├── tools/placement_worker.py (放置策略)
    │       └── settings.py (配置)
    └── tools/input_utls.py (数据输入)
```

---

## 3. 环境依赖

### 3.1 Python 版本

**推荐：Python 3.7 ~ 3.10**
> 注意：`Polygon3` 库在 Python 3.11+ 可能有兼容性问题，建议使用 Python 3.8/3.9

### 3.2 核心依赖库

| 库名 | 版本建议 | 用途 | 安装难度 |
|------|---------|------|---------|
| `Polygon3` | ≥ 3.0.9 | 多边形面积/布尔运算 | ⚠️ 需要C编译器 |
| `pyclipper` | ≥ 1.3.0 | 多边形裁剪/偏移/布尔操作 | 一般 |
| `shapely` | ≥ 1.8.0 | 高精度几何操作 | 简单 |
| `rtree` | ≥ 1.0.0 | 空间索引加速 | 需要libspatialindex |
| `matplotlib` | ≥ 3.4.0 | 可视化输出 | 简单 |
| `numpy` | ≥ 1.20.0 | 数值计算 | 简单 |
| `dxgrabber` | ≥ 0.7.0 | DXF文件解析 | 简单 |
| `Pillow` | ≥ 8.0.0 | 图像处理 | 简单 |

### 3.3 系统级依赖（rtree 需要）

```bash
# macOS
brew install spatialindex

# Ubuntu/Debian
sudo apt-get install libspatialindex-dev

# CentOS/RHEL
sudo yum install spatialindex-devel
```

---

## 4. 安装与配置

### 4.1 推荐方式：使用 conda 虚拟环境

```bash
# 创建虚拟环境
conda create -n nfp_env python=3.9
conda activate nfp_env

# 安装系统依赖（macOS）
brew install spatialindex

# 安装 Python 依赖
pip install Polygon3 pyclipper shapely rtree matplotlib numpy dxgrabber Pillow
```

### 4.2 使用 pip + venv

```bash
# 创建虚拟环境
python3.9 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
pip install shapely rtree numpy Pillow  # requirements.txt 未完整列出的隐式依赖
```

### 4.3 常见安装问题

**问题1：`Polygon3` 安装失败（缺少编译器）**
```bash
# macOS
xcode-select --install
pip install Polygon3

# Linux
sudo apt-get install python3-dev build-essential
pip install Polygon3
```

**问题2：`rtree` 找不到 libspatialindex**
```bash
# 确认先安装了系统库再安装 rtree
brew install spatialindex  # macOS
pip install rtree
```

**问题3：`pyclipper` 版本兼容**
```bash
pip install pyclipper==1.3.0
```

---

## 5. 项目结构详解

```
no_fit_polygon_py3-master/
│
├── nfp_function.py          # 核心：Nester + SADE 类，主算法逻辑
├── test_nfp.py              # 入口：加载数据、执行算法、保存结果
├── settings.py              # 配置：所有可调参数集中在此
├── requirements.txt         # 依赖列表（不完整，见第3章）
│
├── tools/                   # 工具模块
│   ├── __init__.py
│   ├── input_utls.py        # 输入处理：CSV/DXF 解析，批量加载
│   ├── nfp_utls.py          # NFP 几何计算：轨道法、投影、相交判断
│   └── placement_worker.py  # 放置策略：贪心最小化边界宽度
│
├── dxf_file/                # DXF 格式输入文件（E6.dxf）
│   └── E6.dxf
│
├── 测试数据1-1/              # CSV 格式测试数据
│   ├── convex_hulls0011.csv  ← test_nfp.py 使用此文件
│   ├── convex_hulls001.csv
│   └── ... (共 10+ 个数据集)
│
├── best_data/               # 历史最优解存档（CSV）
│   └── *.csv
│
├── new_data/                # 较新解数据（CSV）
│   └── *.csv
│
├── polygons/                # 算法输出：可视化 PNG 图片
│   └── *.png
│
└── docs/                    # 文档目录
    ├── PROJECT_HANDOVER.md  ← 本文件
    └── ENVIRONMENT_SETUP.md
```

### 5.1 核心文件说明

#### `nfp_function.py` — 核心引擎（1029行）

| 类/函数 | 职责 |
|---------|------|
| `class Nester` | 主调度器：管理输入多边形、调用SADE优化、缓存NFP |
| `class SADE` | 自适应差分进化算法：维护种群、变异、交叉、选择 |
| `class genetic_algorithm` | 备用遗传算法实现（功能类似SADE） |
| `minkowski_difference()` | Minkowski差计算（NFP核心几何操作） |
| `draw_result()` | 渲染排版结果为PNG |
| `content_loop_rate()` | 迭代优化主循环 |
| `calculate_container_size()` | 根据总面积自动计算容器尺寸 |

#### `tools/nfp_utls.py` — 几何工具库（969行）

| 函数 | 职责 |
|------|------|
| `nfp_polygon_segmented()` | **主NFP算法**：轨道法计算B绕A的无碰撞边界 |
| `polygon_slide_distance()` | 计算多边形沿方向的滑动距离 |
| `point_in_polygon()` | 射线法点在多边形内判断 |
| `intersect()` | 多边形相交检测 |
| `rotate_polygon()` | 多边形旋转变换 |
| `build_rtree_for_polygon_edges()` | R树空间索引加速边相交检测 |

#### `tools/placement_worker.py` — 放置引擎（202行）

**放置逻辑：**
1. 对每个待放置零件，从NFP缓存取得其与容器的约束区域
2. 用 pyclipper 对已放置零件的NFP求并集
3. 与容器约束取差集，得到有效放置区域
4. 在有效区域中选择使边界宽度最小的位置

**适应度公式：**
```
fitness = 容器数量 + (最小宽度 / 容器面积) + 2 × 未放置数量
```
越小越好。

---

## 6. 核心算法说明

### 6.1 NFP（无适多边形）算法

**目标：** 给定固定多边形A和移动多边形B，计算B的参考点能合法移动的所有位置构成的边界。

**原理（轨道法）：**
```
1. 翻转B（180°旋转）
2. 找A和翻转B的起始接触点
3. B沿A轮廓"爬行"，每次接触都记录参考点位置
4. 爬行一圈后形成闭合曲线——即NFP
```

**应用：**
- `inside=False`：外部NFP，用于防止B与A重叠
- `inside=True`：内部NFP，用于约束B在A（容器）内部

### 6.2 SADE（自适应差分进化）算法

**种群表示：** 每个个体是一组零件的排列顺序 + 旋转角度列表

**迭代过程：**
```
初始化种群
While 未收敛:
    1. MUTATE()：基于变异率 F 对个体进行差分变异
    2. CROSS()：以交叉率 CR 决定是否采用变异个体的基因
    3. EVALUATE()：用 PlacementWorker 评估每个个体的放置适应度
    4. SELECT()：保留适应度更好的个体（锦标赛选择）
    5. 自适应调整 F 和 CR
```

### 6.3 自适应容器调整

当不指定容器尺寸时，系统自动计算：
```python
total_area = sum(polygon_area(p) for p in all_polygons)
# 以 1.3 倍冗余系数估算容器尺寸
side = sqrt(total_area * 1.3)
container = [0, side, side, 0]  # 近似正方形
```

---

## 7. 数据格式规范

### 7.1 输入格式：CSV 多边形数据

```csv
NO,Polygon Points
1,"(3196.613, 347.222); (3194.088, 371.074); (3188.426, 374.960); ..."
2,"(100.0, 200.0); (150.0, 250.0); (120.0, 300.0)"
```

- 第一列：序号（从1开始）
- 第二列：多边形顶点列表，格式 `(x, y)` 以 `;` 分隔
- 坐标单位：毫米（mm）
- 最少 3 个顶点

### 7.2 输入格式：DXF 文件

- 仅支持 `LINE` 实体（不支持 ARC, SPLINE）
- 由连续的线段首尾相连构成多边形
- 函数：`tools/input_utls.py → find_shape_from_dxf()`

### 7.3 输出格式：排版结果

**shift_data_X.txt（放置坐标，JSON）：**
```json
[
  [
    {"p_id": "0", "x": 100.5, "y": 200.3, "rotation": 0},
    {"p_id": "1", "x": 350.0, "y": 150.0, "rotation": 60}
  ]
]
```
外层数组 = 容器列表，内层数组 = 该容器内的放置列表

**polygons_X.txt（多边形定义，JSON）：**
```json
[
  {
    "area": 12345.6,
    "p_id": "0",
    "points": [{"x": 10.0, "y": 20.0}, {"x": 30.0, "y": 40.0}]
  }
]
```

**polygon_coordinates_X.txt（可读坐标）：**
```
Polygon 0: (100.5, 200.3), (130.2, 200.3), ...
```

---

## 8. 运行指南

### 8.1 基本运行

```bash
# 激活环境
conda activate nfp_env

# 进入项目目录
cd /path/to/no_fit_polygon_py3-master

# 运行主程序
python test_nfp.py
```

### 8.2 修改输入数据

编辑 `test_nfp.py` 第5行：
```python
# 修改此路径指向你的CSV文件
s = input_utls.parse_csv_to_list('测试数据1-1/convex_hulls0011.csv')
```

### 8.3 典型运行流程

```
程序启动
  ↓
加载 CSV → 解析多边形（可能耗时30~120秒，取决于数据量）
  ↓
add_container() → 自动计算容器尺寸
  ↓
run() → SADE 进化优化（默认3代种群，可能需要数分钟）
  ↓
content_loop_rate() → 迭代优化（循环改进）
  ↓
处理 rest_paths（未放置零件 → 创建第二个容器）
  ↓
输出 figure_7.png, shift_data_7.txt, polygons_7.txt
```

### 8.4 监控运行状态

程序会输出日志到 `nester.log`：
```bash
tail -f nester.log
```

---

## 9. 参数配置说明

编辑 `settings.py` 调整所有参数：

```python
# ===== 进化算法参数 =====
POPULATION_SIZE = 3      # 种群大小。增大可提高解质量，但显著增加运行时间
                         # 建议范围: 3~20，生产环境建议 ≥5

MUTA_RATE = 20           # 变异概率（百分比）
SADE_MUTATION_RATE = 0.5 # SADE变异率 F，控制差分向量步长
mutagen = 1.0            # 变异因子，放大差分向量
radioactivity = 0.05     # 微扰动因子，防止收敛过早
cross_rate = 0.1         # 交叉率 CR（较低值保守）
mutation_rate = 0.5      # 通用变异率

# ===== 几何参数 =====
ROTATIONS = 60           # 旋转角度档数。60 = 每6°一档（360°/60档）
                         # 减小可加速，增大可提高紧密度
                         # 建议: 测试用4, 生产用16~60

SPACING = 2              # 零件间最小间距（mm）
                         # 增大可提高裁切可行性，减小提高面积利用率

# ===== 容器尺寸（固定容器时使用）=====
BIN_HEIGHT = 2048        # 容器高度（mm）
BIN_WIDTH = 2048         # 容器宽度（mm）

# ===== 文件序号 =====
file_name = 7            # 输出文件编号，影响输出文件名: figure_7.png 等
```

### 9.1 性能调优建议

| 场景 | POPULATION_SIZE | ROTATIONS | SPACING |
|------|----------------|-----------|---------|
| 快速测试 | 3 | 4 | 2 |
| 开发调试 | 5 | 16 | 2 |
| 生产高质量 | 10~20 | 60 | 1~3 |
| 极端精度 | 20+ | 360 | 1 |

---

## 10. 输出结果说明

### 10.1 文件清单

运行结束后生成（X 为 `settings.py` 中的 `file_name`）：

| 文件 | 说明 |
|------|------|
| `figure_X.png` | 排版结果可视化，每个容器一张图 |
| `shift_data_X.txt` | JSON：每个零件的放置位置和旋转角度 |
| `polygons_X.txt` | JSON：所有零件的多边形定义（带面积和ID） |
| `polygon_coordinates_X.txt` | 人类可读的放置后坐标 |
| `nester.log` | 运行日志（含时间戳、fitness变化、错误信息） |

### 10.2 结果质量指标

从日志 `nester.log` 中查看：
- **fitness 值**：越小越好，0表示完美（无未放置件，容器利用率100%）
- **rest_paths 数量**：未能放置的零件数，越少越好
- **容器数量**：使用了几个容器

---

## 11. 已知问题与限制

### 11.1 已知问题

| 问题 | 现象 | 临时解决方案 |
|------|------|-------------|
| NFP计算超时 | 极端复杂多边形（>50顶点）可能导致计算时间极长 | 预处理时进行多边形简化 |
| 内存占用大 | NFP缓存随多边形数量指数增长 | 限制批次大小（batch_size=100） |
| 浮点精度 | 极小多边形可能因精度问题被清理掉 | 增大 curveTolerance |
| 并行竞争 | ThreadPoolExecutor 在某些平台可能死锁 | 减少线程数或改为顺序执行 |

### 11.2 已知限制

- **不支持弧线/曲线**：DXF输入仅支持LINE实体
- **不支持孔洞多边形**：所有多边形视为简单无孔多边形
- **容器必须为矩形**：非矩形容器需手动修改代码
- **无GUI界面**：所有操作通过修改代码/配置文件完成
- **单机运行**：无分布式计算支持

---

## 12. 后续开发建议

### 12.1 近期可改进项

1. **完善 requirements.txt**
   ```
   # 现有文件缺少 shapely, rtree, numpy, Pillow 等隐式依赖
   # 建议生成完整列表:
   pip freeze > requirements_full.txt
   ```

2. **增加命令行参数支持**
   ```python
   # 当前需修改代码才能换数据文件，建议增加argparse:
   python test_nfp.py --input data/polygons.csv --rotations 16 --output results/
   ```

3. **增加进度显示**
   - SADE迭代进度条（使用tqdm）
   - 实时fitness变化曲线

### 12.2 中期架构改进

1. **NFP缓存持久化**：将计算过的NFP保存到磁盘，避免重复计算
2. **支持自定义容器形状**：不规则容器（如整卷布料）
3. **REST API封装**：将算法包装为HTTP服务，方便集成到生产系统
4. **批量作业系统**：支持多个嵌套任务的队列处理

### 12.3 长期规划

1. 引入深度强化学习替代SADE（参考 DRL-based nesting 论文）
2. GPU并行化NFP计算
3. 支持3D嵌套（堆叠问题）

---

## 附录：快速参考卡

```
# 运行项目
conda activate nfp_env
cd no_fit_polygon_py3-master
python test_nfp.py

# 查看日志
tail -f nester.log

# 修改输入
vim test_nfp.py  → 第5行改路径

# 修改参数
vim settings.py  → POPULATION_SIZE, ROTATIONS, SPACING

# 输出文件
figure_7.png         ← 可视化结果
shift_data_7.txt     ← 位置数据
nester.log           ← 运行日志
```

---

*本文档由项目原始开发者整理，如有疑问请联系项目 GitHub 仓库。*
