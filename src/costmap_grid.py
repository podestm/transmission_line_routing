# ============================================
# costmap_grid.py
# JednoduchÃ¡ cost mapa bez grafu
# Dva vÃ½stupy: ÄtvercovÃ¡ mÅ™Ã­Å¾ka  â†’ GeoTIFF + GPKG
#               hexagonÃ¡lnÃ­ mÅ™Ã­Å¾ka  â†’ GeoTIFF + GPKG
# ============================================
import geopandas as gpd
import pandas as pd
import sys
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union
from folium import Map
from folium.plugins import Draw
from pathlib import Path
from datetime import datetime
import numpy as np
import json
import time
import rasterio
from rasterio.transform import from_bounds
from config.config_costs import LAYER_COSTS, GRAPH_CONFIG
from scipy.interpolate import griddata

try:
    from config.config_costs import LAYER_RULES
except ImportError:
    LAYER_RULES = None

try:
    from config.config_costs import ELEVATION_CONFIG
except ImportError:
    ELEVATION_CONFIG = None

RULES_ENABLED = isinstance(LAYER_RULES, dict) and len(LAYER_RULES) > 0
ELEVATION_ENABLED = isinstance(ELEVATION_CONFIG, dict) and len(ELEVATION_CONFIG) > 0

# ------------------------------------------------------------
# KONFIGURACE
# ------------------------------------------------------------
GPKG_PATH = Path("data/raw/ZABAGED_RESULTS.gpkg")
OUTPUT_DIR = Path("data/processed")
OUTPUT_SQUARES_GPKG = OUTPUT_DIR / "costmap_squares.gpkg"
OUTPUT_SQUARES_TIF  = OUTPUT_DIR / "costmap_squares.tif"
OUTPUT_HEXAGONS_GPKG = OUTPUT_DIR / "costmap_hexagons.gpkg"
OUTPUT_HEXAGONS_TIF  = OUTPUT_DIR / "costmap_hexagons.tif"
LOG_PATH = Path("logs/costmap_grid.log")

CRS = GRAPH_CONFIG.get("crs", 5514)
CELL_SIZE = 30  # rozliÅ¡enÃ­ v metrech

NODATA = -9999.0  # nodata hodnota pro rastr


# ------------------------------------------------------------
# LogovÃ¡nÃ­
# ------------------------------------------------------------
# zapisuje zpravy soucasne do konzole i log souboru
def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


# ------------------------------------------------------------
# 1) InteraktivnÃ­ vÃ½bÄ›r oblasti
# ------------------------------------------------------------
# otevre mapu pro rucni vyber oblasti a pocka na export geojsonu
def select_bbox_interactive():
    print("OtevÅ™ena interaktivnÃ­ mapa â€” nakresli obdÃ©lnÃ­k vÃ½bÄ›ru (rectangle).")
    print("Soubor se uloÅ¾Ã­ jako 'select_area.html'. Po nakreslenÃ­ klikni na Export â†’ Download as GeoJSON (data.geojson).")

    m = Map(location=[49.8, 15.5], zoom_start=7, tiles="CartoDB positron")
    Draw(export=True, draw_options={"polyline": False, "circle": False}).add_to(m)
    m.save("select_area.html")

    geojson_path = Path("data.geojson")
    print("\nÄŒekÃ¡m na soubor data.geojson ...")
    while not geojson_path.exists():
        time.sleep(2)

    print("Nalezen soubor data.geojson â€” naÄÃ­tÃ¡m...")
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    geom = shape(gj["features"][0]["geometry"])

    # Transformace celÃ© geometrie do EPSG:5514
    # KrovÃ¡k je pootoÄenÃ½ â†’ obdÃ©lnÃ­k ve WGS84 se stane kosodÃ©lnÃ­kem
    # UchovÃ¡me celÃ½ polygon pro oÅ™ez bunÄ›k, bbox jen pro generovÃ¡nÃ­ mÅ™Ã­Å¾ky
    gdf_sel = gpd.GeoDataFrame(geometry=[geom], crs=4326)
    gdf_sel = gdf_sel.to_crs(CRS)
    clip_polygon = gdf_sel.geometry.iloc[0]
    xmin, ymin, xmax, ymax = clip_polygon.bounds
    bbox = (xmin, ymin, xmax, ymax)
    log(f"VybranÃ¡ oblast (EPSG:{CRS}): {bbox}")
    return bbox, clip_polygon


# ------------------------------------------------------------
# 2) NaÄtenÃ­ vrstev ZABAGED (polygonovÃ© vrstvy)
# ------------------------------------------------------------
# nacte vrstvy zabaged a rozdeli je na cost, exkluzi a neprekrocitelne zony
def load_zabaged_layers(path_gpkg, bbox):
    log("=== NAÄŒÃTÃNÃ VRSTEV ZE ZABAGED ===")
    polygon_frames = []
    exclusion_frames = []  # polygony kam NESMI padnout bunka
    uncrossable_frames = []  # polygony pres ktere NESMI vest skokova hrana
    stats = {"polygons": 0, "lines": 0}
    line_types = {"LineString", "MultiLineString"}
    inward_gradient_info = {}
    outward_gradient_info = {}

    for name, cost in LAYER_COSTS.items():
        try:
            gdf = gpd.read_file(path_gpkg, layer=name, bbox=bbox)
            if gdf.empty:
                continue

            gdf["layer"] = name
            gdf["cost"] = cost
            geom_types = set(gdf.geometry.geom_type)
            is_linear = geom_types.issubset(line_types)

            if is_linear:
                stats["lines"] += len(gdf)
            else:
                stats["polygons"] += len(gdf)

            # --- Zjistit, zda vrstva patÅ™Ã­ do exkluznÃ­ masky ---
            rules = LAYER_RULES.get(name, {}) if RULES_ENABLED else {}
            can_place = rules.get("can_place_nodes", True)
            buffer_dist = rules.get("buffer_exclusion", 0) or 0
            can_cross = rules.get("can_cross", True)
            is_inf = (cost == float("inf"))
            inward_gradient = rules.get("inward_gradient") if isinstance(rules, dict) else None
            outward_gradient = rules.get("outward_gradient") if isinstance(rules, dict) else None

            # Vrstva jde do exkluze pokud: cost=inf NEBO can_place_nodes=False
            if is_inf or not can_place:
                exc_geom = gdf.geometry

                # Inward gradient: jadro je pouze vnitrni cast polygonu,
                # prstenec od okraje dovnitr zustava pruchodny s graduelnim costem.
                if inward_gradient:
                    grad_dist = float(inward_gradient.get("distance", 500.0))
                    core_geoms = []
                    zone_geoms = []
                    for geom in gdf.geometry:
                        if geom.is_empty:
                            continue
                        inner = geom.buffer(-grad_dist)
                        if inner.is_empty:
                            # Polygon je uzsi nez 2*distance -> cele uzemi je gradient zona
                            zone_geoms.append(geom)
                        else:
                            core_geoms.append(inner)
                            zone_geoms.append(geom)

                    if core_geoms:
                        exclusion_frames.append(
                            gpd.GeoDataFrame({"geometry": core_geoms}, crs=CRS)
                        )
                    inward_gradient_info[name] = {
                        "zones": zone_geoms,
                        "distance": grad_dist,
                        "cost_outer": float(inward_gradient.get("cost_outer", 4.0)),
                        "cost_inner": float(inward_gradient.get("cost_inner", 10.0)),
                    }
                else:
                    if buffer_dist != 0:
                        exc_geom = exc_geom.buffer(buffer_dist)
                    exclusion_frames.append(
                        gpd.GeoDataFrame({"geometry": exc_geom}, crs=CRS)
                    )

                # Vrstva s can_cross=False jde do uncrossable masky
                if not can_cross:
                    uncrossable_frames.append(
                        gpd.GeoDataFrame({"geometry": gdf.geometry}, crs=CRS)
                    )

                # Outward gradient: gradientni buffer okolo prekazky (smerem ven)
                if outward_gradient:
                    og_dist = float(outward_gradient.get("distance", 100.0))
                    if og_dist > 0:
                        outward_gradient_info[name] = {
                            "obstacle_geoms": [g for g in gdf.geometry if not g.is_empty],
                            "distance": og_dist,
                            "cost_outer": float(outward_gradient.get("cost_outer", 4.0)),
                            "cost_inner": float(outward_gradient.get("cost_inner", 10.0)),
                        }

                reason = []
                if is_inf:
                    reason.append("cost=inf")
                if not can_place:
                    reason.append("can_place_nodes=False")
                if buffer_dist != 0:
                    reason.append(f"buffer={buffer_dist}m")
                if not can_cross:
                    reason.append("can_cross=False")
                if inward_gradient:
                    reason.append(f"inward_gradient={inward_gradient.get('distance', 500)}m")
                if outward_gradient:
                    reason.append(f"outward_gradient={outward_gradient.get('distance', 100)}m")
                log(f"  {name} -> EXKLUZE ({', '.join(reason)}, {len(gdf)} prvku)")

                # Inward gradient vrstvy nesmi jit do polygon_frames s inf costem
                if inward_gradient:
                    continue

            # Vrstva s platnÃ½m cost jde do cost polygonÅ¯ (jen ne-liniovÃ©)
            elif not is_linear:
                polygon_frames.append(gdf[["layer", "cost", "geometry"]])
                log(f"  {name} naÄteno ({len(gdf)} prvkÅ¯, cost={cost})")
            else:
                log(f"  {name} naÄteno ({len(gdf)} liniÃ­, cost={cost})")

        except Exception as e:
            log(f"  {name} pÅ™eskoÄeno ({e})")

    if not polygon_frames:
        raise ValueError("Å½Ã¡dnÃ¡ data v danÃ©m vÃ½Å™ezu!")

    combined = gpd.GeoDataFrame(pd.concat(polygon_frames, ignore_index=True), crs=CRS)
    log(f"Celkem {len(combined)} polygonÅ¯ s platnÃ½m cost.")

    # ExkluznÃ­ maska jako GeoDataFrame (sjoin pÅ™es prostorovÃ½ index)
    exclusion_gdf = None
    if exclusion_frames:
        exclusion_gdf = gpd.GeoDataFrame(
            pd.concat(exclusion_frames, ignore_index=True), crs=CRS
        ).reset_index(drop=True)
        log(f"ExkluznÃ­ maska: {len(exclusion_gdf):,} geometriÃ­ (budovy, silnice+buffer, zahrady â€¦)")
    else:
        log("ExkluznÃ­ maska: prÃ¡zdnÃ¡")

    # Uncrossable maska (vrstvy s can_cross=False)
    uncrossable_gdf = None
    if uncrossable_frames:
        uncrossable_gdf = gpd.GeoDataFrame(
            pd.concat(uncrossable_frames, ignore_index=True), crs=CRS
        ).reset_index(drop=True)
        log(f"Uncrossable maska: {len(uncrossable_gdf):,} geometriÃ­ (nelze pÅ™eskoÄit jump hranou)")
    else:
        log("Uncrossable maska: prÃ¡zdnÃ¡ (vÅ¡echny exkluznÃ­ zÃ³ny lze pÅ™eskoÄit)")

    if inward_gradient_info:
        log(f"Inward gradient zony: {len(inward_gradient_info)} vrstev")
    if outward_gradient_info:
        log(f"Outward gradient zony: {len(outward_gradient_info)} vrstev")

    return combined, exclusion_gdf, stats, inward_gradient_info, uncrossable_gdf, outward_gradient_info


# ------------------------------------------------------------
# 2b) NaÄtenÃ­ vrstevnic a vÃ½poÄet sklonovÃ©ho rastru
# ------------------------------------------------------------
# nacte nebo pripravi raster sklonu pro penalizaci terenu
def load_slope_raster(bbox, cell_size):
    """
    NaÄte vrstevnice, interpoluje DMR (griddata â†’ linear)
    a spoÄÃ­tÃ¡ sklon ve stupnÃ­ch.  VrÃ¡tÃ­:
        slope_grid   â€“ 2D numpy pole sklonu (Â°), shape (ny, nx)
        slope_xs     â€“ 1D pole x souÅ™adnic (stÅ™edy pixelÅ¯)
        slope_ys     â€“ 1D pole y souÅ™adnic (stÅ™edy pixelÅ¯)
    nebo (None, None, None) pokud data nejsou dostupnÃ¡.
    """
    if not ELEVATION_ENABLED:
        log("ELEVATION_CONFIG nenÃ­ povolena â€” sklon pÅ™eskoÄen.")
        return None, None, None

    contour_path = Path(ELEVATION_CONFIG.get("contour_path", ""))
    if not contour_path.exists():
        log(f"Vrstevnice nenalezeny: {contour_path} â€” sklon pÅ™eskoÄen.")
        return None, None, None

    log("=== NAÄŒÃTÃNÃ VRSTEVNIC A VÃPOÄŒET SKLONU ===")
    t0 = time.time()
    gdf = gpd.read_file(contour_path, bbox=bbox)
    if gdf.empty:
        log("Vrstevnice: prÃ¡zdnÃ½ vÃ½Å™ez â€” sklon pÅ™eskoÄen.")
        return None, None, None
    if gdf.crs and gdf.crs.to_epsg() != CRS:
        gdf = gdf.to_crs(CRS)

    # Autodetekce sloupce s vÃ½Å¡kou
    import unicodedata
    height_col = ELEVATION_CONFIG.get("height_field")
    if not height_col:
        candidates = ELEVATION_CONFIG.get("height_field_candidates",
                                          ["VYSKA", "vyska__m_", "vyska_m", "height", "elevation", "z"])
        for col in gdf.columns:
            norm = unicodedata.normalize("NFKD", str(col)).encode("ascii", "ignore").decode("ascii").lower()
            if any(c.lower() in norm for c in candidates):
                height_col = col
                break
    if not height_col:
        log(f"Nenalezen sloupec s vÃ½Å¡kou. Sloupce: {list(gdf.columns)} â€” sklon pÅ™eskoÄen.")
        return None, None, None

    gdf["elev"] = pd.to_numeric(gdf[height_col], errors="coerce")
    gdf = gdf.dropna(subset=["elev"])
    log(f"Vrstevnice: {len(gdf):,} prvkÅ¯, vÃ½Å¡ka: {height_col}, "
        f"rozsah {gdf.elev.min():.0f}â€“{gdf.elev.max():.0f} m n.m.")

    # Vzorkovat body z liniÃ­ vrstevnic (hustÄ›jÅ¡Ã­ = lepÅ¡Ã­ interpolace)
    sample_dist = cell_size  # vzdÃ¡lenost vzorkÅ¯ podÃ©l Äar
    pts_x, pts_y, pts_z = [], [], []
    for _, row in gdf.iterrows():
        geom = row.geometry
        elev = row.elev
        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        elif geom.geom_type == "LineString":
            lines = [geom]
        else:
            continue
        for line in lines:
            length = line.length
            n_samples = max(2, int(length / sample_dist))
            for frac in np.linspace(0, 1, n_samples):
                pt = line.interpolate(frac, normalized=True)
                pts_x.append(pt.x)
                pts_y.append(pt.y)
                pts_z.append(elev)

    pts_x = np.array(pts_x)
    pts_y = np.array(pts_y)
    pts_z = np.array(pts_z)
    log(f"VzorkovÃ¡no {len(pts_x):,} bodÅ¯ z vrstevnic.")

    # Interpolace DMR na mÅ™Ã­Å¾ku shodnou s cell_size
    xmin, ymin, xmax, ymax = bbox
    slope_xs = np.arange(xmin + cell_size / 2, xmax, cell_size)
    slope_ys = np.arange(ymin + cell_size / 2, ymax, cell_size)
    grid_xx, grid_yy = np.meshgrid(slope_xs, slope_ys)

    log(f"Interpoluji DMR na mÅ™Ã­Å¾ku {len(slope_xs)}Ã—{len(slope_ys)} ...")
    dem = griddata(
        np.column_stack([pts_x, pts_y]),
        pts_z,
        (grid_xx, grid_yy),
        method="linear",
        fill_value=np.nan,
    )

    # VÃ½poÄet sklonu z gradientu
    dy, dx = np.gradient(dem, cell_size, cell_size)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # NaN z interpolace â†’ slope = 0 (bezpeÄnÃ© â€” jde o okraje bez vrstevnic)
    slope_deg = np.nan_to_num(slope_deg, nan=0.0)

    elapsed = time.time() - t0
    log(f"Sklon spoÄÃ­tÃ¡n: min={np.nanmin(slope_deg):.1f}Â°, max={np.nanmax(slope_deg):.1f}Â°, "
        f"prÅ¯mÄ›r={np.nanmean(slope_deg):.1f}Â° ({elapsed:.1f} s)")
    return slope_deg, slope_xs, slope_ys


# odebere hodnotu sklonu v mistech stredu generovanych bunek
def sample_slope_at_points(cx, cy, slope_grid, slope_xs, slope_ys):
    """
    Pro pole souÅ™adnic (cx, cy) vrÃ¡tÃ­ sklon ve stupnÃ­ch
    z pÅ™edpoÄÃ­tanÃ©ho slope gridu (nearest-neighbor lookup).
    """
    if slope_grid is None:
        return None

    # Pixel indexy (nearest neighbor)
    x_step = slope_xs[1] - slope_xs[0] if len(slope_xs) > 1 else 1.0
    y_step = slope_ys[1] - slope_ys[0] if len(slope_ys) > 1 else 1.0
    col_idx = np.clip(((cx - slope_xs[0]) / x_step).astype(int), 0, len(slope_xs) - 1)
    row_idx = np.clip(((cy - slope_ys[0]) / y_step).astype(int), 0, len(slope_ys) - 1)
    return slope_grid[row_idx, col_idx]


# ------------------------------------------------------------
# 3) PomocnÃ© funkce pro tvorbu geometriÃ­
# ------------------------------------------------------------
_HEX_ANGLES_RAD = np.radians([0, 60, 120, 180, 240, 300])
_HEX_COS = np.cos(_HEX_ANGLES_RAD)
_HEX_SIN = np.sin(_HEX_ANGLES_RAD)


# vytvori geometrii jedne ctvercove bunky
def _make_square(x, y, s):
    return box(x - s / 2, y - s / 2, x + s / 2, y + s / 2)


# vytvori geometrii jedne hexagonalni bunky
def _make_hexagon(cx, cy, size):
    coords = [(cx + size * c, cy + size * s) for c, s in zip(_HEX_COS, _HEX_SIN)]
    coords.append(coords[0])
    return Polygon(coords)


# ------------------------------------------------------------
# 4) GenerovÃ¡nÃ­ mÅ™Ã­Å¾ky stÅ™edÅ¯ (numpy) â€” bez geometriÃ­
# ------------------------------------------------------------
# vygeneruje stredu ctvercove mrizky pro danou oblast
def generate_square_centers(bbox, cell_size):
    """Vygeneruje numpy pole stÅ™edÅ¯ ÄtvercovÃ© mÅ™Ã­Å¾ky."""
    log(f"=== GENEROVÃNÃ STÅ˜EDÅ® ÄŒTVERCOVÃ‰ MÅ˜ÃÅ½KY ({cell_size}Ã—{cell_size} m) ===")
    xmin, ymin, xmax, ymax = bbox
    xs = np.arange(xmin + cell_size / 2, xmax, cell_size)
    ys = np.arange(ymin + cell_size / 2, ymax, cell_size)
    xx, yy = np.meshgrid(xs, ys)
    cx = xx.ravel()
    cy = yy.ravel()
    log(f"VygenerovÃ¡no {len(cx):,} stÅ™edÅ¯")
    return cx, cy


# vygeneruje stredu hexagonalni mrizky pro danou oblast
def generate_hex_centers(bbox, cell_size):
    """Vygeneruje numpy pole stÅ™edÅ¯ hexagonÃ¡lnÃ­ mÅ™Ã­Å¾ky."""
    log(f"=== GENEROVÃNÃ STÅ˜EDÅ® HEXAGONÃLNÃ MÅ˜ÃÅ½KY (~{cell_size} m) ===")
    xmin, ymin, xmax, ymax = bbox
    size = cell_size / np.sqrt(3)
    col_step = 1.5 * size
    row_step = np.sqrt(3) * size

    cols = np.arange(xmin, xmax + col_step, col_step)
    rows = np.arange(ymin, ymax + row_step, row_step)

    cx_list = []
    cy_list = []
    for ci, x in enumerate(cols):
        y_off = row_step / 2 if ci % 2 else 0.0
        for y in rows:
            cx_list.append(x)
            cy_list.append(y + y_off)

    cx = np.array(cx_list)
    cy = np.array(cy_list)
    log(f"VygenerovÃ¡no {len(cx):,} stÅ™edÅ¯")
    return cx, cy


# ------------------------------------------------------------
# 5) PÅ™iÅ™azenÃ­ nÃ¡kladÅ¯ a tvorba bunÄ›k
# ------------------------------------------------------------
# priradi kazdemu stredu cost, vrstvu a dalsi atributy pro export
def assign_costs(cx, cy, zabaged_gdf, exclusion_gdf, crs,
                 clip_polygon=None, grid_type="square", cell_size=20,
                 slope_grid=None, slope_xs=None, slope_ys=None,
                 inward_gradient_info=None, outward_gradient_info=None):
    """
    1. OÅ™ez stÅ™edÅ¯ na clip_polygon (transformovanÃ½ vÃ½bÄ›r)
    2. Exkluze pÅ™es sjoin (budovy, silnice s bufferem, zahrady â€¦)
    3. PÅ™iÅ™azenÃ­ cost ze ZABAGED
    4. Aplikace sklonovÃ½ch omezenÃ­ (penalizace / exkluze)
    5. Aplikace inward gradient costu (u vybranÃ½ch chrÃ¡nÄ›nÃ½ch vrstev)
    5b. Aplikace outward gradient costu (buffer okolo pÅ™ekÃ¡Å¾ek)
    6. Tvorba polygon geometriÃ­ jen pro vÃ½slednÃ© buÅˆky
    """
    total = len(cx)
    log(f"ZpracovÃ¡vÃ¡m {total:,} stÅ™edÅ¯ ({grid_type})...")

    # --- 1) OÅ™ez na polygon vÃ½bÄ›ru (kosodÃ©lnÃ­k v 5514) ---
    if clip_polygon is not None:
        from shapely.prepared import prep
        prepared = prep(clip_polygon)
        mask = np.array([prepared.contains_properly(p)
                         for p in gpd.points_from_xy(cx, cy)],
                        dtype=bool)
        cx, cy = cx[mask], cy[mask]
        log(f"OÅ™ez na polygon vÃ½bÄ›ru: {total:,} â†’ {len(cx):,} (odstranÄ›no {total - len(cx):,})")

    # GeoDataFrame se stÅ™edy
    centroids_gdf = gpd.GeoDataFrame(
        {"_idx": np.arange(len(cx))},
        geometry=gpd.points_from_xy(cx, cy),
        crs=crs,
    )

    # --- 2) Exkluze ---
    if exclusion_gdf is not None and len(exclusion_gdf) > 0:
        n_before = len(centroids_gdf)
        log(f"Kontroluji exkluzi ({n_before:,} bunÄ›k Ã— {len(exclusion_gdf):,} exkl. polygonÅ¯)...")
        hits = gpd.sjoin(centroids_gdf, exclusion_gdf, how="inner", predicate="within")
        excluded_ids = set(hits["_idx"].values)
        if excluded_ids:
            centroids_gdf = centroids_gdf[~centroids_gdf["_idx"].isin(excluded_ids)].reset_index(drop=True)
        log(f"VylouÄeno {len(excluded_ids):,} bunÄ›k v exkluznÃ­ zÃ³nÄ›")

    # --- 3) PÅ™iÅ™azenÃ­ cost ---
    log("Spatial join s cost vrstvami...")
    joined_all = gpd.sjoin(
        centroids_gdf,
        zabaged_gdf[["cost", "layer", "geometry"]],
        how="inner",
        predicate="within",
    )

    # Pri prekryvu vrstev se costy scitaji.
    overlap_counts = joined_all.groupby("_idx").size()
    max_overlap = int(overlap_counts.max()) if len(overlap_counts) > 0 else 0
    if max_overlap > 1:
        multi_cnt = int((overlap_counts > 1).sum())
        log(f"Prekryv vrstev: {multi_cnt:,} bunek ma vice vrstev (max {max_overlap})")

    aggregated = joined_all.groupby("_idx", as_index=False).agg(
        cost=("cost", "sum"),
        layer=("layer", lambda s: "|".join(sorted(set(map(str, s))))),
    )
    joined = centroids_gdf[["_idx", "geometry"]].merge(aggregated, on="_idx", how="inner")

    # --- 4) SklonovÃ¡ omezenÃ­ ---
    final_cx = np.array(joined.geometry.x)
    final_cy = np.array(joined.geometry.y)
    slopes = sample_slope_at_points(final_cx, final_cy, slope_grid, slope_xs, slope_ys)

    if slopes is not None and ELEVATION_ENABLED:
        max_slope = float(ELEVATION_CONFIG.get("max_slope_deg", 90.0))
        penalty_from = float(ELEVATION_CONFIG.get("penalty_from_deg", 90.0))
        penalty_max_mult = float(ELEVATION_CONFIG.get("penalty_max_multiplier", 2.0))

        # Exkluze bunÄ›k nad max_slope
        steep_mask = slopes > max_slope
        n_steep = int(steep_mask.sum())
        if n_steep > 0:
            keep = ~steep_mask
            joined = joined.iloc[keep].reset_index(drop=True)
            final_cx = final_cx[keep]
            final_cy = final_cy[keep]
            slopes = slopes[keep]
            log(f"VylouÄeno {n_steep:,} bunÄ›k s pÅ™Ã­liÅ¡ strmÃ½m sklonem (>{max_slope:.0f}Â°)")

        # Penalizace sklonu: lineÃ¡rnÃ­ interpolace multiplikÃ¡toru
        # penalty_from â†’ Ã—1.0, max_slope â†’ Ã—penalty_max_multiplier
        cost_values = joined["cost"].values.astype(float)
        penalty_mask = slopes > penalty_from
        n_penalized = int(penalty_mask.sum())
        if n_penalized > 0:
            # LineÃ¡rnÃ­ interpolace: pÅ™i penalty_from â†’ 1.0, pÅ™i max_slope â†’ penalty_max_mult
            slope_range = max_slope - penalty_from
            if slope_range > 0:
                t = np.clip((slopes[penalty_mask] - penalty_from) / slope_range, 0.0, 1.0)
                multiplier = 1.0 + t * (penalty_max_mult - 1.0)
                cost_values[penalty_mask] *= multiplier
            log(f"PenalizovÃ¡no {n_penalized:,} bunÄ›k se sklonem >{penalty_from:.0f}Â° "
                f"(multiplikÃ¡tor aÅ¾ Ã—{penalty_max_mult:.1f})")
        joined = joined.copy()
        joined["cost"] = cost_values
    else:
        slopes = None

    # --- 5) Aplikace inward gradient costu (od okraje dovnitÅ™) ---
    if inward_gradient_info and len(inward_gradient_info) > 0 and len(joined) > 0:
        log(f"Aplikuji inward gradient pro {len(inward_gradient_info)} vrstvy...")
        pts = gpd.GeoSeries(gpd.points_from_xy(final_cx, final_cy), crs=crs)
        cost_values = joined["cost"].values.astype(float)

        for layer_name, grad_data in inward_gradient_info.items():
            zones = grad_data.get("zones", [])
            dist_max = float(grad_data.get("distance", 500.0))
            cost_outer = float(grad_data.get("cost_outer", 4.0))
            cost_inner = float(grad_data.get("cost_inner", 10.0))
            if dist_max <= 0 or not zones:
                continue

            layer_hits = 0
            for zone in zones:
                if zone.is_empty:
                    continue

                inside_zone = pts.within(zone).to_numpy()
                if not np.any(inside_zone):
                    continue

                # Vzdalenost od hranice dovnitr: 0 na okraji, roste smerem k jadru.
                dists = np.array([p.distance(zone.boundary) for p in pts[inside_zone]])
                dists = np.clip(dists, 0.0, dist_max)
                t = dists / dist_max
                grad_cost = cost_outer + t * (cost_inner - cost_outer)
                cost_values[inside_zone] += grad_cost
                layer_hits += int(np.sum(inside_zone))

            log(f"  {layer_name}: gradient aplikovan na {layer_hits:,} bunek")

        joined = joined.copy()
        joined["cost"] = cost_values

    # --- 5b) Outward gradient (od pÅ™ekÃ¡Å¾ky ven) ---
    if outward_gradient_info and len(outward_gradient_info) > 0 and len(joined) > 0:
        log(f"Aplikuji outward gradient pro {len(outward_gradient_info)} vrstvy...")
        pts = gpd.GeoSeries(gpd.points_from_xy(final_cx, final_cy), crs=crs)
        cost_values = joined["cost"].values.astype(float)

        for layer_name, grad_data in outward_gradient_info.items():
            obstacle_geoms = grad_data["obstacle_geoms"]
            dist_max = float(grad_data["distance"])
            cost_outer = float(grad_data["cost_outer"])
            cost_inner = float(grad_data["cost_inner"])
            if not obstacle_geoms or dist_max <= 0:
                continue

            obstacles_union = unary_union(obstacle_geoms)
            buffer_zone = obstacles_union.buffer(dist_max)

            inside_buffer = pts.within(buffer_zone).to_numpy()
            if not np.any(inside_buffer):
                log(f"  {layer_name}: outward gradient â€” 0 bunÄ›k v zÃ³nÄ›")
                continue

            # VzdÃ¡lenost od pÅ™ekÃ¡Å¾ky: 0 na hranici, roste smÄ›rem ven
            dists = np.array([p.distance(obstacles_union) for p in pts[inside_buffer]])
            dists = np.clip(dists, 0.0, dist_max)
            # t=0 u pÅ™ekÃ¡Å¾ky â†’ cost_inner, t=1 na vnÄ›jÅ¡Ã­m okraji â†’ cost_outer
            t = dists / dist_max
            grad_cost = cost_inner + t * (cost_outer - cost_inner)
            cost_values[inside_buffer] += grad_cost

            log(f"  {layer_name}: outward gradient na {int(np.sum(inside_buffer)):,} bunÄ›k")

        joined = joined.copy()
        joined["cost"] = cost_values

    # --- 6) Tvorba geometriÃ­ jen pro platnÃ© buÅˆky ---
    log(f"VytvÃ¡Å™Ã­m geometrie pro {len(joined):,} bunÄ›k...")

    if grid_type == "square":
        polys = [_make_square(x, y, cell_size) for x, y in zip(final_cx, final_cy)]
    else:
        hex_size = cell_size / np.sqrt(3)
        polys = [_make_hexagon(x, y, hex_size) for x, y in zip(final_cx, final_cy)]

    data = {
            "cost": joined["cost"].values,
            "layer": joined["layer"].values,
        }
    if slopes is not None:
        data["slope_deg"] = np.round(slopes, 1)

    gdf_result = gpd.GeoDataFrame(
        data,
        geometry=polys,
        crs=crs,
    )
    log(f"PÅ™iÅ™azeno {len(gdf_result):,} bunÄ›k s platnÃ½m nÃ¡kladem (z {total:,})")
    return gdf_result


# ------------------------------------------------------------
# 6) Export do rastru (GeoTIFF)
# ------------------------------------------------------------
# rasterizuje vektorovou costmapu do geotiffu
def export_raster(gdf_costmap, bbox, cell_size, output_path, crs_epsg):
    """
    Rasterizuje cost mapu do GeoTIFF.
    PouÅ¾Ã­vÃ¡ pÅ™Ã­mÃ© vepsÃ¡nÃ­ stÅ™edÅ¯ do rastru (rychlÃ© i pro miliony bunÄ›k).
    BuÅˆky bez pÅ™iÅ™azenÃ©ho cost zÅ¯stanou jako NODATA.
    """
    log(f"Rasterizuji do {output_path} ...")
    xmin, ymin, xmax, ymax = bbox

    width = int(np.ceil((xmax - xmin) / cell_size))
    height = int(np.ceil((ymax - ymin) / cell_size))

    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    # PÅ™Ã­mÃ© vepsÃ¡nÃ­ â€” z centroidÅ¯ bunÄ›k spoÄÃ­tÃ¡m pixelovÃ© indexy
    centroids = gdf_costmap.geometry.centroid
    cx = np.array(centroids.x)
    cy = np.array(centroids.y)
    costs = np.array(gdf_costmap["cost"], dtype=np.float32)

    # Pixel souÅ™adnice (col, row) z afinnÃ­ transformace
    inv_transform = ~transform
    col_f, row_f = inv_transform * (cx, cy)
    cols_px = np.floor(col_f).astype(int)
    rows_px = np.floor(row_f).astype(int)

    # Filtrovat body mimo rozsah
    valid = (cols_px >= 0) & (cols_px < width) & (rows_px >= 0) & (rows_px < height)
    cols_px = cols_px[valid]
    rows_px = rows_px[valid]
    costs = costs[valid]

    raster = np.full((height, width), NODATA, dtype=np.float32)
    raster[rows_px, cols_px] = costs

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=f"EPSG:{crs_epsg}",
        transform=transform,
        nodata=NODATA,
        compress="deflate",
    ) as dst:
        dst.write(raster, 1)

    log(f"Rastr uloÅ¾en: {output_path} ({width}Ã—{height} px, {cell_size} m/px)")


# ------------------------------------------------------------
# 6) HlavnÃ­ pipeline
# ------------------------------------------------------------
# samostatny beh vytvori ctvercovou i hexagonalni costmapu
def main():
    t_start = time.time()
    log("=== SPUÅ TÄšNÃ GENERÃTORU COST MAPY ===")

    bbox, clip_polygon = select_bbox_interactive()

    zabaged, exclusion_gdf, layer_stats, inward_gradient_info, _uncrossable_gdf, outward_gradient_info = load_zabaged_layers(GPKG_PATH, bbox)

    # --- SKLON ---
    slope_grid, slope_xs, slope_ys = load_slope_raster(bbox, CELL_SIZE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- ÄŒTVERCOVÃ MÅ˜ÃÅ½KA ---
    sq_cx, sq_cy = generate_square_centers(bbox, CELL_SIZE)
    costmap_sq = assign_costs(sq_cx, sq_cy, zabaged, exclusion_gdf, CRS,
                              clip_polygon=clip_polygon,
                              grid_type="square", cell_size=CELL_SIZE,
                              slope_grid=slope_grid, slope_xs=slope_xs,
                              slope_ys=slope_ys,
                              inward_gradient_info=inward_gradient_info,
                              outward_gradient_info=outward_gradient_info)
    costmap_sq.to_file(OUTPUT_SQUARES_GPKG, driver="GPKG", layer="costmap")
    log(f"Vektor uloÅ¾en: {OUTPUT_SQUARES_GPKG} ({len(costmap_sq):,} bunÄ›k)")
    export_raster(costmap_sq, bbox, CELL_SIZE, OUTPUT_SQUARES_TIF, CRS)

    # --- HEXAGONÃLNÃ MÅ˜ÃÅ½KA ---
    hex_cx, hex_cy = generate_hex_centers(bbox, CELL_SIZE)
    costmap_hex = assign_costs(hex_cx, hex_cy, zabaged, exclusion_gdf, CRS,
                               clip_polygon=clip_polygon,
                               grid_type="hex", cell_size=CELL_SIZE,
                               slope_grid=slope_grid, slope_xs=slope_xs,
                               slope_ys=slope_ys,
                               inward_gradient_info=inward_gradient_info,
                               outward_gradient_info=outward_gradient_info)
    costmap_hex.to_file(OUTPUT_HEXAGONS_GPKG, driver="GPKG", layer="costmap")
    log(f"Vektor uloÅ¾en: {OUTPUT_HEXAGONS_GPKG} ({len(costmap_hex):,} bunÄ›k)")
    export_raster(costmap_hex, bbox, CELL_SIZE, OUTPUT_HEXAGONS_TIF, CRS)

    # --- SOUHRN ---
    elapsed = time.time() - t_start
    log(f"Vrstev naÄteno: {layer_stats['polygons']:,} polygonÅ¯, {layer_stats['lines']:,} liniÃ­")
    log(f"ÄŒtverce: {len(costmap_sq):,} bunÄ›k ({CELL_SIZE}Ã—{CELL_SIZE} m)")
    log(f"Hexagony: {len(costmap_hex):,} bunÄ›k (~{CELL_SIZE} m)")
    log(f"CelkovÃ½ Äas: {elapsed:.1f} s")
    log("=== HOTOVO ===")


if __name__ == "__main__":
    main()

