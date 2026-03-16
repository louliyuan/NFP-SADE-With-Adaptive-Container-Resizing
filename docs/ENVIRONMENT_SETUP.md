# 环境搭建指南 / Environment Setup Guide

**适用项目：** NFP-SADE With Adaptive Container Resizing
**文档版本：** v1.0
**更新日期：** 2026-03-16

---

## 目录

1. [系统要求](#1-系统要求)
2. [macOS 搭建步骤](#2-macos-搭建步骤)
3. [Windows 搭建步骤](#3-windows-搭建步骤)
4. [Linux 搭建步骤](#4-linux-搭建步骤)
5. [验证安装](#5-验证安装)
6. [IDE 配置](#6-ide-配置)
7. [常见错误排查](#7-常见错误排查)

---

## 1. 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| Python | 3.7 | 3.9 |
| 内存 (RAM) | 4 GB | 16 GB |
| 磁盘空间 | 2 GB | 10 GB |
| CPU | 双核 | 8核以上 |
| 操作系统 | macOS 10.14 / Ubuntu 18.04 / Win10 | macOS 12+ / Ubuntu 22.04 |

> **内存说明：** 处理大型数据集（1000+多边形）时，NFP缓存可能占用 4~8 GB 内存。

---

## 2. macOS 搭建步骤

### Step 1: 安装 Homebrew（如未安装）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: 安装系统依赖

```bash
# 安装 spatialindex（rtree 依赖）
brew install spatialindex

# 安装 Xcode 命令行工具（Polygon3 编译需要）
xcode-select --install
```

### Step 3: 安装 Conda（推荐）

```bash
# 下载 Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
# Apple Silicon (M1/M2) 用户：
# curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

bash Miniconda3-latest-*.sh
source ~/.zshrc  # 或 ~/.bashrc
```

### Step 4: 创建虚拟环境

```bash
conda create -n nfp_env python=3.9 -y
conda activate nfp_env
```

### Step 5: 安装 Python 依赖

```bash
pip install Polygon3==3.0.9
pip install pyclipper==1.3.0
pip install shapely==2.0.1
pip install rtree==1.0.1
pip install matplotlib==3.7.1
pip install numpy==1.24.3
pip install dxgrabber==0.7.0
pip install Pillow==9.5.0
```

### Step 6: 验证安装

```bash
python -c "
import Polygon
import pyclipper
import shapely
from rtree import index
import matplotlib
import numpy
import dxgrabber
from PIL import Image
print('All dependencies installed successfully!')
print(f'Python: {__import__(\"sys\").version}')
print(f'shapely: {shapely.__version__}')
print(f'numpy: {numpy.__version__}')
"
```

---

## 3. Windows 搭建步骤

### Step 1: 安装 Python 3.9

1. 下载：https://www.python.org/downloads/release/python-3913/
2. 安装时勾选 **"Add Python to PATH"**
3. 验证：`python --version`

### Step 2: 安装 Visual C++ Build Tools（Polygon3 编译需要）

1. 下载：https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. 安装时选择 **"C++ build tools"** 工作负载

### Step 3: 创建虚拟环境

```cmd
python -m venv nfp_env
nfp_env\Scripts\activate
```

### Step 4: 安装依赖

```cmd
pip install Polygon3 pyclipper shapely rtree matplotlib numpy dxgrabber Pillow
```

> **Windows 注意：** `rtree` 在 Windows 上通常会自动包含 spatialindex.dll，无需额外安装。

### Step 5: 如果 rtree 安装失败

```cmd
# 使用 conda 替代
conda install -c conda-forge rtree
```

---

## 4. Linux 搭建步骤

### Ubuntu / Debian

```bash
# 系统依赖
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-dev python3.9-venv
sudo apt-get install -y libspatialindex-dev build-essential

# 虚拟环境
python3.9 -m venv nfp_env
source nfp_env/bin/activate

# Python 依赖
pip install Polygon3 pyclipper shapely rtree matplotlib numpy dxgrabber Pillow
```

### CentOS / RHEL

```bash
# 系统依赖
sudo yum install -y python39 python39-devel
sudo yum install -y spatialindex-devel gcc gcc-c++

# 虚拟环境
python3.9 -m venv nfp_env
source nfp_env/bin/activate

# Python 依赖
pip install Polygon3 pyclipper shapely rtree matplotlib numpy dxgrabber Pillow
```

---

## 5. 验证安装

### 5.1 完整验证脚本

将以下内容保存为 `verify_env.py` 并运行：

```python
#!/usr/bin/env python3
"""环境验证脚本 - 运行后确认所有依赖正常"""

import sys
import importlib

print(f"Python version: {sys.version}")
print("-" * 50)

dependencies = [
    ("Polygon", "Polygon3"),
    ("pyclipper", "pyclipper"),
    ("shapely", "shapely"),
    ("rtree", "rtree"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("dxgrabber", "dxgrabber"),
    ("PIL", "Pillow"),
    ("json", "built-in"),
    ("csv", "built-in"),
    ("concurrent.futures", "built-in"),
]

all_ok = True
for module_name, package_name in dependencies:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'N/A')
        print(f"[OK] {module_name:20s} ({package_name}) - version: {version}")
    except ImportError as e:
        print(f"[FAIL] {module_name:20s} ({package_name}) - ERROR: {e}")
        all_ok = False

print("-" * 50)
if all_ok:
    print("All dependencies OK! Environment is ready.")
else:
    print("Some dependencies are MISSING. Please install them.")
    sys.exit(1)
```

```bash
python verify_env.py
```

### 5.2 算法功能验证

```bash
cd no_fit_polygon_py3-master
python -c "
from tools import input_utls
from nfp_function import Nester
print('Core modules import: OK')
n = Nester()
print('Nester initialization: OK')
"
```

---

## 6. IDE 配置

### PyCharm

1. File → Settings → Project → Python Interpreter
2. 点击齿轮 → Add → Conda Environment → Existing
3. 选择 `nfp_env` 的 Python 解释器路径

macOS 路径示例：
```
~/miniconda3/envs/nfp_env/bin/python
```

### VS Code

1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. 选择 `nfp_env` 对应的 Python 3.9

在 `.vscode/settings.json` 中添加：
```json
{
    "python.defaultInterpreterPath": "~/miniconda3/envs/nfp_env/bin/python",
    "python.terminal.activateEnvironment": true
}
```

---

## 7. 常见错误排查

### 错误 1：`ModuleNotFoundError: No module named 'Polygon'`

```bash
pip install Polygon3
# 如果失败，检查是否有 C 编译器：
xcode-select --install  # macOS
sudo apt-get install build-essential python3-dev  # Ubuntu
```

### 错误 2：`OSError: Could not find libspatialindex`

```bash
# macOS
brew install spatialindex
pip install --force-reinstall rtree

# Ubuntu
sudo apt-get install libspatialindex-dev
pip install --force-reinstall rtree
```

### 错误 3：`pyclipper.ClipperException: Error in OffsetPaths`

通常由极端细长或自交多边形引起：
- 检查输入CSV数据是否有无效多边形
- 调大 `settings.py` 中的 `curveTolerance`

### 错误 4：内存不足（MemoryError）

```python
# 在 test_nfp.py 中减小批次大小：
input_utls.batch_process(n, s, batch_size=50)  # 从100改为50
```

### 错误 5：Python 3.11+ 兼容性问题

`Polygon3` 库与 Python 3.11 存在兼容性问题，建议降级到 Python 3.9：
```bash
conda create -n nfp_env python=3.9
```

---

## 附录：一键安装脚本（macOS/Linux）

将以下内容保存为 `setup.sh`：

```bash
#!/bin/bash
set -e

echo "=== NFP-SADE Environment Setup ==="

# 检测系统
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    brew install spatialindex || true
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux"
    sudo apt-get install -y libspatialindex-dev build-essential python3-dev || \
    sudo yum install -y spatialindex-devel gcc gcc-c++ python39-devel
fi

# 创建 conda 环境
conda create -n nfp_env python=3.9 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nfp_env

# 安装依赖
pip install Polygon3==3.0.9 pyclipper==1.3.0 shapely==2.0.1 \
            rtree==1.0.1 matplotlib==3.7.1 numpy==1.24.3 \
            dxgrabber==0.7.0 Pillow==9.5.0

echo "=== Setup Complete ==="
echo "Activate with: conda activate nfp_env"
echo "Run with: python test_nfp.py"
```

```bash
chmod +x setup.sh
./setup.sh
```
