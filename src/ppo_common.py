"""Graph building and environment wiring for PPO routing."""

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from src.ppo_environment import CostmapRoutingEnv


# sklada ridky smerovy graf nad coarse costmapou pro trenink a inference
def build_coarse_direct_graph(coords, step_m=300, tol_factor=0.4, costs=None,
                              uncrossable_gdf=None, cell_size_m=50.0):
    """Build 8-directional adjacency where every step is ~step_m metres."""
    coords_arr = np.asarray(coords, dtype=np.float64)
    n = len(coords_arr)
    tree = cKDTree(coords_arr)
    tol = step_m * tol_factor

    costs_arr = None
    passable = None
    if costs is not None:
        costs_arr = np.asarray(costs, dtype=np.float64)
        passable = np.isfinite(costs_arr)

    n_samples = max(2, round(step_m / cell_size_m))

    barrier_tree = None
    barrier_geoms = None
    if uncrossable_gdf is not None and len(uncrossable_gdf) > 0:
        from shapely.strtree import STRtree
        barrier_geoms = list(uncrossable_gdf.geometry)
        barrier_tree = STRtree(barrier_geoms)

    angles = np.arange(8) * (np.pi / 4)
    deltas = np.column_stack([np.cos(angles), np.sin(angles)]) * step_m

    adj_direct = [[] for _ in range(n)]
    for d_idx, delta in enumerate(deltas):
        targets = coords_arr + delta
        dists, indices = tree.query(targets, k=1)
        for i, (dist, j) in enumerate(zip(dists, indices.tolist())):
            if dist <= tol and j != i:
                if passable is not None and (not passable[i] or not passable[j]):
                    continue
                if barrier_tree is not None:
                    edge = LineString([coords_arr[i], coords_arr[j]])
                    hits = barrier_tree.query(edge)
                    if len(hits) > 0 and any(
                        barrier_geoms[k].intersects(edge) for k in hits
                    ):
                        continue

                if costs_arr is not None:
                    t_vals = np.linspace(0.0, 1.0, n_samples)
                    sample_pts = coords_arr[i] + t_vals[:, None] * (coords_arr[j] - coords_arr[i])
                    _, nearest = tree.query(sample_pts)
                    sample_costs = costs_arr[nearest]
                    valid = np.isfinite(sample_costs)
                    n_valid = int(valid.sum())
                    avg_cost = float(sample_costs[valid].mean()) if n_valid > 0 else float(costs_arr[j])
                    dest_cost = float(costs_arr[j])
                    ratio = avg_cost / dest_cost if dest_cost > 1e-9 else 1.0
                else:
                    ratio = 1.0

                adj_direct[i].append((int(j), ratio, d_idx))

    covered = sum(1 for nb in adj_direct if nb)
    avg_dirs = sum(len(nb) for nb in adj_direct) / max(n, 1)
    print(f"  Coarse graph ({step_m:.0f} m): {covered}/{n} nodes reachable, "
          f"avg {avg_dirs:.1f} directions/node")

    adj_jump = [[] for _ in range(n)]
    return adj_direct, adj_jump


DISTANCE_BOUNDS_PER_LEVEL = [
    (2000.0, 8000.0),
    (1000.0, 5000.0),
    (1500.0, 7000.0),
]


# vraci pevne intervaly vzdalenosti pro curriculum podle indexu levelu
def get_distance_bounds_for_level(level_idx: int) -> tuple[float, float]:
    """Return fixed start-goal distance bounds for a curriculum level."""
    bounds_idx = min(level_idx, len(DISTANCE_BOUNDS_PER_LEVEL) - 1)
    return DISTANCE_BOUNDS_PER_LEVEL[bounds_idx]


# wrapper sb3 vola tuto funkci pri sestaveni action masky
def action_mask_fn(env):
    """Action mask callback for ActionMasker wrapper."""
    return env.action_masks()


# sjednoti chybejici cost kanaly do tvaru, ktery ceka ppo prostredi
def ensure_cost_components(costs, land_costs=None, slope_factors=None):
    """Return observation-compatible land/slope arrays."""
    costs = np.asarray(costs, dtype=np.float32)

    if land_costs is None:
        land_costs = np.where(np.isfinite(costs), costs, 100.0)
    if slope_factors is None:
        slope_factors = np.ones_like(costs, dtype=np.float32)

    return (
        np.asarray(land_costs, dtype=np.float32),
        np.asarray(slope_factors, dtype=np.float32),
    )


# vytvori prostredi a obali ho maskovanim akci i monitorem pro sb3
def make_masked_env(
    graph_data,
    *,
    randomize=True,
    max_steps=600,
    patch_radius=12,
    reward_scale=0.01,
    proximity_coef=1.0,
    min_start_goal_dist=1000.0,
    max_start_goal_dist=5000.0,
    momentum_bonus=0.0,
    revisit_penalty=10.0,
    goal_bonus=100.0,
    goal_radius=0.0,
    step_penalty=0.0,
    wrap_monitor=True,
):
    """Create a masked PPO routing environment from graph-data dict."""
    land_costs, slope_factors = ensure_cost_components(
        graph_data["costs"],
        graph_data.get("land_costs"),
        graph_data.get("slope_factors"),
    )

    env = CostmapRoutingEnv(
        coords=graph_data["coords"],
        costs=graph_data["costs"],
        land_costs=land_costs,
        slope_factors=slope_factors,
        adj_direct=graph_data["adj_direct"],
        adj_jump=graph_data["adj_jump"],
        cell_size=graph_data["cell_size"],
        start_idx=graph_data["start_idx"],
        goal_idx=graph_data["goal_idx"],
        max_jumps=graph_data["max_jumps"],
        penalty=graph_data["penalty"],
        turn_penalty=graph_data["turn_penalty"],
        max_steps=max_steps,
        patch_radius=patch_radius,
        randomize_start_goal=randomize,
        min_start_goal_dist=min_start_goal_dist,
        max_start_goal_dist=max_start_goal_dist,
        reward_scale=reward_scale,
        proximity_coef=proximity_coef,
        momentum_bonus=momentum_bonus,
        revisit_penalty=revisit_penalty,
        goal_bonus=goal_bonus,
        goal_radius=goal_radius,
        step_penalty=step_penalty,
    )
    env = ActionMasker(env, action_mask_fn)
    if wrap_monitor:
        env = Monitor(env)
    return env
