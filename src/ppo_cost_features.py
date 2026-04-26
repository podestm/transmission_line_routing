"""Cost feature extraction for PPO notebook inference."""

import numpy as np

from src.ppo_common import ensure_cost_components


# premapuje textove vrstvy na zakladni cenovy kanal pro ppo pozorovani
def compute_land_costs(layers: np.ndarray, layer_definitions: dict) -> np.ndarray:
    """Look up base land-use cost per node from LAYER_DEFINITIONS."""
    land_costs = np.ones(len(layers), dtype=np.float64)
    for i, layer in enumerate(layers):
        if layer in layer_definitions:
            c = layer_definitions[layer]["cost"]
            land_costs[i] = c if np.isfinite(c) else 100.0
        else:
            land_costs[i] = 1.0
    return land_costs


# prevedeni sklonu na nasobek ceny, ktery se pouzije v observation patchi
def compute_slope_factors(slope_degs: np.ndarray, elev_config: dict) -> np.ndarray:
    """Compute slope penalty multiplier per node from ELEVATION_CONFIG."""
    from_deg = elev_config["penalty_from_deg"]
    max_deg = elev_config["max_slope_deg"]
    max_mult = elev_config["penalty_max_multiplier"]

    factors = np.ones(len(slope_degs), dtype=np.float64)
    slope_degs = np.nan_to_num(slope_degs, nan=0.0)
    mask = slope_degs > from_deg
    if mask.any():
        t = np.clip((slope_degs[mask] - from_deg) / max(max_deg - from_deg, 1.0), 0, 1)
        factors[mask] = 1.0 + t * (max_mult - 1.0)
    return factors


# z vysledku scenare vytahne land a slope kanaly v konzistentnim formatu
def get_result_cost_components(result, costs):
    """Build PPO observation cost components for a notebook scenario result."""
    costmap = result["costmap"]
    config = result.get("config")

    if config is None or "layer" not in costmap.columns or "slope_deg" not in costmap.columns:
        return ensure_cost_components(costs)

    land_costs = compute_land_costs(costmap["layer"].values, config.LAYER_DEFINITIONS)
    slope_factors = compute_slope_factors(
        costmap["slope_deg"].values.astype(float),
        config.ELEVATION_CONFIG,
    )
    return land_costs, slope_factors
