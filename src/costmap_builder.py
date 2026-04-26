"""Config loading and costmap building for notebook scenarios."""

import importlib
import inspect
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import box, shape


# prepne aktivni config a znovu nacte costmap_grid s odpovidajicimi pravidly
def load_config_context(config_module):
    qualified_cfg = config_module if config_module.startswith("config.") else f"config.{config_module}"
    for module_name in [config_module, qualified_cfg, "config.config_costs", "src.costmap_grid"]:
        sys.modules.pop(module_name, None)

    cfg = importlib.import_module(qualified_cfg)
    sys.modules["config.config_costs"] = cfg
    cmg = importlib.import_module("src.costmap_grid")
    return cfg, cmg


# z geojsonu pripravi polygon, bbox a pomocne transformovane vrstvy oblasti
def load_area_context(geojson_path, crs):
    geojson_path = Path(geojson_path)
    with open(geojson_path, "r", encoding="utf-8") as handle:
        gj = json.load(handle)

    geom = shape(gj["features"][0]["geometry"])
    gdf_sel = gpd.GeoDataFrame(geometry=[geom], crs=4326)
    lon_min, lat_min, lon_max, lat_max = geom.bounds
    gdf_sel_5514 = gdf_sel.to_crs(crs)
    clip_polygon = gdf_sel_5514.geometry.iloc[0]
    bbox = clip_polygon.bounds
    return {
        "geom": geom,
        "gdf_sel": gdf_sel,
        "gdf_sel_5514": gdf_sel_5514,
        "clip_polygon": clip_polygon,
        "bbox": bbox,
        "lon_min": lon_min,
        "lat_min": lat_min,
        "lon_max": lon_max,
        "lat_max": lat_max,
    }


# drzi kompatibilitu mezi starsi a novejsi signaturou load_zabaged_layers
def _call_load_zabaged_layers(cmg, gpkg_path, bbox, clip_polygon):
    signature = inspect.signature(cmg.load_zabaged_layers)
    if "clip_polygon" in signature.parameters:
        return cmg.load_zabaged_layers(gpkg_path, bbox, clip_polygon=clip_polygon)
    return cmg.load_zabaged_layers(gpkg_path, bbox)


# postavi jednu costmapu pro zvoleny config a pripadne ji ulozi do processed
def build_costmap_for_config(config_module, geojson_path, gpkg_path, output_dir, cell_size, export_prefix=None, force_rebuild=False):
    cfg, cmg = load_config_context(config_module)
    crs = cfg.GRAPH_CONFIG.get("crs", 5514)
    area = load_area_context(geojson_path, crs)

    # Fast path: load from cache if GPKG already exists and has exclusion layer
    if not force_rebuild and export_prefix:
        out_gpkg = Path(output_dir) / f"{export_prefix}_{cell_size}m.gpkg"
        if out_gpkg.exists():
            try:
                costmap = gpd.read_file(out_gpkg, layer="costmap")
                exclusion_gdf = gpd.read_file(out_gpkg, layer="exclusion")
                try:
                    uncrossable_gdf = gpd.read_file(out_gpkg, layer="uncrossable")
                except Exception:
                    uncrossable_gdf = gpd.GeoDataFrame()
                print(f"  [cache] Loading {out_gpkg.name} from disk (use force_rebuild=True to rebuild)")
                return {
                    "config_module": config_module,
                    "config": cfg,
                    "cmg": cmg,
                    "crs": crs,
                    "cell_size": cell_size,
                    "geojson_path": Path(geojson_path),
                    "gpkg_path": Path(gpkg_path),
                    "output_dir": Path(output_dir),
                    "output_gpkg": out_gpkg,
                    "output_tif": None,
                    "costmap": costmap,
                    "zabaged": None,
                    "exclusion_gdf": exclusion_gdf,
                    "uncrossable_gdf": uncrossable_gdf,
                    "layer_stats": {},
                    "inward_gradient_info": [],
                    "area": area,
                    "label": export_prefix,
                }
            except Exception:
                pass  # layer missing → fall through to full rebuild

    print(f"\n{'=' * 70}")
    print(f"SCENARIO {config_module}")
    print(f"{'=' * 70}")

    for layer_name in ["VelkoplosneZvlasteChraneneUzemi", "EvropskyVyznamnaLokalita"]:
        rules = cfg.LAYER_RULES.get(layer_name, {})
        print(f"Config {config_module} | {layer_name}: inward_gradient={rules.get('inward_gradient')}")

    print(
        f"Area WGS84: lon [{area['lon_min']:.6f}, {area['lon_max']:.6f}], "
        f"lat [{area['lat_min']:.6f}, {area['lat_max']:.6f}]"
    )
    print(f"Area (EPSG:{crs}): {area['bbox']}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = Path(gpkg_path)

    t0_all = time.time()
    zabaged, exclusion_gdf, layer_stats, inward_gradient_info, uncrossable_gdf, outward_gradient_info = _call_load_zabaged_layers(
        cmg, gpkg_path, area["bbox"], area["clip_polygon"]
    )
    slope_grid, slope_xs, slope_ys = cmg.load_slope_raster(area["bbox"], cell_size)

    mid_lat = (area["lat_min"] + area["lat_max"]) / 2
    m_to_lat = 1.0 / 111_320.0
    m_to_lon = 1.0 / (111_320.0 * np.cos(np.radians(mid_lat)))
    transformer_fwd = Transformer.from_crs(4326, crs, always_xy=True)
    transformer_inv = Transformer.from_crs(crs, 4326, always_xy=True)

    print("\n" + "=" * 60)
    print("SQUARE COSTMAP")
    print("=" * 60)
    t0 = time.time()

    d_lat = cell_size * m_to_lat
    d_lon = cell_size * m_to_lon
    xs_wgs = np.arange(area["lon_min"] + d_lon / 2, area["lon_max"], d_lon)
    ys_wgs = np.arange(area["lat_min"] + d_lat / 2, area["lat_max"], d_lat)
    xx, yy = np.meshgrid(xs_wgs, ys_wgs)
    cx_wgs = xx.ravel()
    cy_wgs = yy.ravel()
    print(f"Generated {len(cx_wgs):,} centers in WGS84 ({len(xs_wgs)}x{len(ys_wgs)})")

    cx_5514, cy_5514 = transformer_fwd.transform(cx_wgs, cy_wgs)
    cx_5514 = np.array(cx_5514)
    cy_5514 = np.array(cy_5514)

    costmap = cmg.assign_costs(
        cx_5514,
        cy_5514,
        zabaged,
        exclusion_gdf,
        crs,
        clip_polygon=area["clip_polygon"],
        grid_type="square",
        cell_size=cell_size,
        slope_grid=slope_grid,
        slope_xs=slope_xs,
        slope_ys=slope_ys,
        inward_gradient_info=inward_gradient_info,
        outward_gradient_info=outward_gradient_info,
    )

    sq_cx = np.array(costmap.geometry.centroid.x)
    sq_cy = np.array(costmap.geometry.centroid.y)
    sq_cx_wgs, sq_cy_wgs = transformer_inv.transform(sq_cx, sq_cy)
    half_lon = d_lon / 2
    half_lat = d_lat / 2
    sq_polys_wgs = [
        box(x - half_lon, y - half_lat, x + half_lon, y + half_lat)
        for x, y in zip(sq_cx_wgs, sq_cy_wgs)
    ]
    costmap = gpd.GeoDataFrame(
        costmap.drop(columns="geometry").reset_index(drop=True),
        geometry=sq_polys_wgs,
        crs=4326,
    ).to_crs(crs)

    costmap.insert(0, "cell_id", range(1, len(costmap) + 1))
    costmap.index = range(len(costmap))
    print(f"Done in {time.time() - t0:.1f} s - {len(costmap):,} cells")

    if export_prefix:
        out_gpkg = output_dir / f"{export_prefix}_{cell_size}m.gpkg"
        out_tif = output_dir / f"{export_prefix}_{cell_size}m.tif"
        costmap.to_file(out_gpkg, driver="GPKG", layer="costmap")
        if exclusion_gdf is not None and len(exclusion_gdf) > 0:
            exclusion_gdf.to_file(out_gpkg, driver="GPKG", layer="exclusion")
        if uncrossable_gdf is not None and len(uncrossable_gdf) > 0:
            uncrossable_gdf.to_file(out_gpkg, driver="GPKG", layer="uncrossable")
        cmg.export_raster(costmap, area["bbox"], cell_size, out_tif, crs)
        print(f"Saved: {out_gpkg}")
    else:
        out_gpkg = None
        out_tif = None

    print(f"Total build time: {time.time() - t0_all:.1f} s")

    return {
        "config_module": config_module,
        "config": cfg,
        "cmg": cmg,
        "crs": crs,
        "cell_size": cell_size,
        "geojson_path": Path(geojson_path),
        "gpkg_path": gpkg_path,
        "output_dir": output_dir,
        "output_gpkg": out_gpkg,
        "output_tif": out_tif,
        "costmap": costmap,
        "zabaged": zabaged,
        "exclusion_gdf": exclusion_gdf,
        "uncrossable_gdf": uncrossable_gdf,
        "layer_stats": layer_stats,
        "inward_gradient_info": inward_gradient_info,
        "area": area,
    }


# opakuje stavbu costmapy pro vice scenaru se stejnou oblasti a vstupnim gpkg
def build_costmaps_for_scenarios(scenarios, geojson_path, gpkg_path, output_dir, cell_size):
    results = []
    for scenario in scenarios:
        module_name = scenario["module"]
        display_name = scenario["label"]
        export_prefix = scenario.get("export_prefix") or display_name.replace(" ", "_")
        result = build_costmap_for_config(
            module_name,
            geojson_path,
            gpkg_path,
            output_dir,
            cell_size,
            export_prefix=export_prefix,
        )
        result["label"] = display_name
        results.append(result)
    return results
