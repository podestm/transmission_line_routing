"""Navigation graph construction and path utility functions."""

import math
import sys
import time

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep


# prevede smer hrany do jednoho z osmi smerovych id
def _dir_id(dx, dy):
    return int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8


# sklada primou a skokovou sousednost nad costmap centroidy
def build_navigation_graph(
    coords,
    costs,
    cell_size,
    max_jump_edge_m=200.0,
    uncrossable_gdf=None,
    use_turns=True,
    label="",
):
    """Build adjacency lists for direct and jump edges from costmap centroids.

    Returns (adj_direct, adj_jump) where each is a list-of-lists.
    Each entry: (neighbor_idx, distance_ratio, direction_id).
    """
    n = len(costs)

    print(f"  [{label}] Building graph ({n:,} nodes)...", end=" ")
    sys.stdout.flush()
    t_graph = time.time()

    tree = cKDTree(coords)
    direct_radius = cell_size * 1.5
    jump_radius = max(direct_radius, float(max_jump_edge_m))
    direct_pairs = tree.query_pairs(direct_radius)
    all_close = tree.query_pairs(jump_radius)
    jump_pairs = all_close - direct_pairs

    adj_direct = [[] for _ in range(n)]
    adj_jump = [[] for _ in range(n)]
    for i, j in direct_pairs:
        dx = coords[j, 0] - coords[i, 0]
        dy = coords[j, 1] - coords[i, 1]
        dist = math.sqrt(dx * dx + dy * dy)
        ratio = dist / cell_size
        d_ij = _dir_id(dx, dy) if use_turns else 0
        d_ji = _dir_id(-dx, -dy) if use_turns else 0
        adj_direct[i].append((j, ratio, d_ij))
        adj_direct[j].append((i, ratio, d_ji))

    uncrossable_prepared = None
    if uncrossable_gdf is not None and len(uncrossable_gdf) > 0:
        uncrossable_prepared = prep(unary_union(uncrossable_gdf.geometry.values))

    blocked_count = 0
    for i, j in jump_pairs:
        dx = coords[j, 0] - coords[i, 0]
        dy = coords[j, 1] - coords[i, 1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > max_jump_edge_m:
            continue
        if uncrossable_prepared is not None:
            edge_line = LineString([(coords[i, 0], coords[i, 1]), (coords[j, 0], coords[j, 1])])
            if uncrossable_prepared.intersects(edge_line):
                blocked_count += 1
                continue
        ratio = dist / cell_size
        d_ij = _dir_id(dx, dy) if use_turns else 0
        d_ji = _dir_id(-dx, -dy) if use_turns else 0
        adj_jump[i].append((j, ratio, d_ij))
        adj_jump[j].append((i, ratio, d_ji))

    jump_edge_count = sum(len(neighbors) for neighbors in adj_jump) // 2
    if blocked_count > 0:
        print(f"{len(direct_pairs):,} direct + {jump_edge_count:,} jump edges ({blocked_count:,} blocked by can_cross) ({time.time() - t_graph:.1f} s)")
    else:
        print(f"{len(direct_pairs):,} direct + {jump_edge_count:,} jump edges ({time.time() - t_graph:.1f} s)")
    sys.stdout.flush()

    return adj_direct, adj_jump


# vybere costmap bunku, ktera je nejbliz startu nebo cili ve wgs84
def find_nearest_cell(costmap, lon, lat, crs):
    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    centroids = costmap.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    dists = np.sqrt((coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2)
    idx = int(np.argmin(dists))
    return idx, int(costmap.iloc[idx]["cell_id"])


# zkontroluje, zda zadany bod nelezi v exkluzi nebo na bunce s inf costem
def check_exclusion(lon, lat, crs, exclusion_gdf, costmap, cell_idx, label="Point"):
    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    warnings = []
    cell_cost = costmap.iloc[cell_idx]["cost"]
    if cell_cost == float("inf"):
        warnings.append(f"  {label} cell (idx={cell_idx}) has cost=inf (excluded layer)")
    if exclusion_gdf is not None and len(exclusion_gdf) > 0:
        x, y = transformer.transform(lon, lat)
        point = Point(x, y)
        hits = exclusion_gdf[exclusion_gdf.geometry.contains(point)]
        if len(hits) > 0:
            layers = hits["layer"].unique().tolist() if "layer" in hits.columns else ["?"]
            warnings.append(f"  {label} ({lon}, {lat}) is inside exclusion zone: {layers}")
    return warnings


# secte skutecnou delku trasy po souradnicich navstivenych bunek
def compute_path_distance(coords, path):
    if len(path) < 2:
        return 0.0
    pts = coords[path]
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)))


# spocita kolikrat se trasa natocila do jineho smeru
def count_turns(coords, path):
    if len(path) < 3:
        return 0
    turns = 0
    prev_dir = None
    for idx in range(1, len(path)):
        dx = coords[path[idx], 0] - coords[path[idx - 1], 0]
        dy = coords[path[idx], 1] - coords[path[idx - 1], 1]
        direction = int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8
        if prev_dir is not None and direction != prev_dir:
            turns += 1
        prev_dir = direction
    return turns


# shrne prunik trasy s gradientnimi ochrannymi pasmy
def summarize_gradient_overlap(coords, path, crs, inward_gradient_info):
    if not path or not inward_gradient_info:
        return {}

    points = gpd.GeoSeries(gpd.points_from_xy(coords[path, 0], coords[path, 1]), crs=crs)
    overlap = {}
    for layer_name, grad_data in inward_gradient_info.items():
        dist_max = float(grad_data.get("distance", 0.0) or 0.0)
        zones = [zone for zone in grad_data.get("zones", []) if zone is not None and not zone.is_empty]
        if dist_max <= 0 or not zones:
            continue

        hits = np.zeros(len(points), dtype=bool)
        for zone in zones:
            hits |= points.within(zone).to_numpy()
        overlap[layer_name] = int(hits.sum())
    return overlap
