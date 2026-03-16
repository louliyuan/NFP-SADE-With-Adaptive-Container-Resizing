# An Efficient Irregular Texture Nesting Method via Hybrid NFP-SADE with Adaptive Container Resizing
[![Paper](https://img.shields.io/badge/Paper-PERS%202025-blue?style=flat-square&logo=bookstack)](https://doi.org/10.14358/PERS.25-00038R3)
[![Open Access](https://img.shields.io/badge/Open%20Access-CC%20BY--NC--ND-green?style=flat-square)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

> An efficient Python implementation of irregular polygon nesting using the **No-Fit Polygon (NFP)** algorithm combined with **Self-Adaptive Differential Evolution (SADE)**, featuring automatic container size optimization.

---

## Overview

This project solves the **2D irregular bin-packing (nesting) problem**: arranging large numbers of arbitrarily shaped polygons into rectangular containers with maximum material utilization. The primary application is in **textile and manufacturing industries** — minimizing fabric waste when cutting irregular pattern pieces.

### Key Features

- **NFP-Based Collision Detection** — Uses the No-Fit Polygon orbiting algorithm for exact, overlap-free polygon placement
- **SADE Evolutionary Optimization** — Self-Adaptive Differential Evolution finds globally optimal arrangement order and rotation angles
- **Adaptive Container Sizing** — Automatically calculates container dimensions from total polygon area
- **Arbitrary Rotation Support** — Configurable rotation increments (default: 6° steps)
- **Multi-Container Handling** — Overflow polygons automatically cascade to additional containers
- **Parallel NFP Computation** — ThreadPoolExecutor for concurrent NFP calculations
- **Multiple Input Formats** — CSV polygon data and DXF CAD files
- **Visualization Output** — PNG rendering of final nesting layouts

---

## Algorithm

### No-Fit Polygon (NFP)

Given a stationary polygon **A** and a moving polygon **B**, the NFP defines the complete set of positions where B's reference point can be placed without overlapping A:

```
1. Reflect B through its reference point (180° rotation)
2. Find an initial contact configuration between A and reflected-B
3. "Orbit" B around A, maintaining edge-to-edge contact
4. Record reference point positions to trace the NFP boundary
```

Two NFP modes are used:
- **External NFP** (`inside=False`): prevents B from overlapping A
- **Internal NFP** (`inside=True`): constrains B to remain inside container A

### SADE Optimization

Each individual in the population encodes a **permutation of polygon indices + rotation angles**. The fitness function evaluates actual placement quality:

```
fitness = num_containers + (min_width / bin_area) + 2 x num_unplaced
```

Lower fitness = better material utilization. The algorithm self-adapts its mutation factor **F** and crossover rate **CR** during evolution.

### Placement Strategy (Greedy)

For each polygon in order:
1. Retrieve bin-path NFP from cache
2. Compute union of already-placed polygon NFPs (adjusted to placement positions)
3. Subtract from bin NFP to get valid placement region
4. Select position that **minimizes the current bounding box width**

---

## Project Structure

```
no_fit_polygon_py3-master/
├── nfp_function.py          # Core: Nester + SADE classes, main algorithm
├── test_nfp.py              # Entry point: load data, run algorithm, save results
├── settings.py              # Configuration: all tunable parameters
├── requirements.txt         # Python dependencies
│
├── tools/
│   ├── input_utls.py        # Input: CSV/DXF parsing, batch loading
│   ├── nfp_utls.py          # NFP geometry: orbiting, projection, intersection
│   └── placement_worker.py  # Placement: greedy width-minimization strategy
│
├── dxf_file/                # DXF input files
├── best_data/               # Archived best solutions
├── new_data/                # Recent solution data
├── polygons/                # Output PNG visualizations
└── docs/                    # Project documentation
    ├── PROJECT_HANDOVER.md  # Complete handover document (Chinese/English)
    └── ENVIRONMENT_SETUP.md # Detailed setup guide
```

---

## Quick Start

### Requirements

- Python 3.7–3.10 (Python 3.9 recommended)
- See [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) for full setup instructions

### Installation

```bash
# 1. Install system dependency (macOS)
brew install spatialindex

# 2. Create virtual environment
conda create -n nfp_env python=3.9 -y
conda activate nfp_env

# 3. Install Python dependencies
pip install Polygon3 pyclipper shapely rtree matplotlib numpy dxgrabber Pillow
```

### Run

```bash
python test_nfp.py
```

The program will:
1. Load polygon data from the CSV file specified in `test_nfp.py`
2. Automatically compute optimal container size
3. Run SADE evolutionary optimization
4. Output results to `figure_7.png`, `shift_data_7.txt`, `polygons_7.txt`

---

## Configuration

Edit `settings.py` to tune the algorithm:

```python
POPULATION_SIZE = 3       # SADE population size (increase for better quality)
ROTATIONS = 60            # Rotation increments (360/ROTATIONS degrees each)
SPACING = 2               # Minimum gap between polygons (mm)
BIN_HEIGHT = 2048         # Container height (mm)
BIN_WIDTH = 2048          # Container width (mm)
SADE_MUTATION_RATE = 0.5  # Differential evolution mutation factor F
cross_rate = 0.1          # Crossover rate CR
file_name = 7             # Output file numbering index
```

### Performance Guide

| Use Case | POPULATION_SIZE | ROTATIONS |
|----------|----------------|-----------|
| Quick test | 3 | 4 |
| Development | 5 | 16 |
| Production quality | 10-20 | 60 |

---

## Input Data Format

### CSV Format

```csv
NO,Polygon Points
1,"(3196.613, 347.222); (3194.088, 371.074); (3188.426, 374.960); ..."
2,"(100.0, 200.0); (150.0, 250.0); (120.0, 300.0)"
```

- Column 1: Sequential index (starting from 1)
- Column 2: Polygon vertices as `(x, y)` pairs separated by `;`
- Coordinates in millimeters; minimum 3 vertices required

### DXF Format

Supported entity: `LINE` only (ARC and SPLINE not supported). Connected line segments must form closed polygons.

---

## Output Files

| File | Description |
|------|-------------|
| `figure_X.png` | Visual layout of nested polygons per container |
| `shift_data_X.txt` | JSON: placement coordinates and rotation for each polygon |
| `polygons_X.txt` | JSON: polygon definitions with area and ID |
| `polygon_coordinates_X.txt` | Human-readable final coordinates |
| `nester.log` | Execution log with timestamps and fitness progression |

### Placement Result Format (`shift_data_X.txt`)

```json
[
  [
    {"p_id": "0", "x": 100.5, "y": 200.3, "rotation": 0},
    {"p_id": "1", "x": 350.0, "y": 150.0, "rotation": 60}
  ]
]
```

Outer array = containers; inner array = placements within each container.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `Polygon3` | Polygon area and boolean operations |
| `pyclipper` | Polygon clipping, offsetting, and boolean ops |
| `shapely` | High-precision geometry computations |
| `rtree` | Spatial indexing for edge intersection acceleration |
| `matplotlib` | Visualization and PNG output |
| `numpy` | Numerical computing |
| `dxgrabber` | DXF file parsing |
| `Pillow` | Image processing |

---

## Documentation

For complete project documentation, see the `docs/` directory:

- **[docs/PROJECT_HANDOVER.md](docs/PROJECT_HANDOVER.md)** — Full technical handover document including architecture, algorithm details, data formats, known issues, and development roadmap
- **[docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md)** — Step-by-step environment setup for macOS, Windows, and Linux

---

## References

- SVGNest: A browser-based vector nesting tool — [https://github.com/Jack000/SVGnest](https://github.com/Jack000/SVGnest)
- Storn, R., & Price, K. (1997). "Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces"

---

## 📄 Citation

If you find this work useful in your research, please consider citing:
```bibtex
@article{lou2025nfpsade,
  title     = {An Efficient Irregular Texture Nesting Method via Hybrid {NFP}-{SADE} with Adaptive Container Resizing},
  author    = {Lou, Liyuan and Li, Wanyun and Yu, Jingle and Wang, Xin and Zhan, Zongqian},
  journal   = {Photogrammetric Engineering \& Remote Sensing},
  volume    = {91},
  number    = {11},
  pages     = {681--691},
  year      = {2025},
  publisher = {American Society for Photogrammetry and Remote Sensing},
  doi       = {10.14358/PERS.25-00038R3},
  url       = {https://doi.org/10.14358/PERS.25-00038R3}
}```

## License

MIT License
