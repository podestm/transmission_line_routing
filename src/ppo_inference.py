"""PPO inference routines for notebook scenarios."""

import time

import numpy as np

from src.graph import (
    build_navigation_graph,
    check_exclusion,
    compute_path_distance,
    count_turns,
    find_nearest_cell,
    summarize_gradient_overlap,
)

from src.ppo_common import build_coarse_direct_graph, ensure_cost_components, make_masked_env
from src.ppo_cost_features import get_result_cost_components
from src.ppo_model_loader import load_ppo_model


# pusti jednu epizodu inference nad pevnym startem a cilem
def run_ppo_inference(
    model,
    coords,
    costs,
    adj_direct,
    adj_jump,
    cell_size,
    start_idx,
    goal_idx,
    land_costs=None,
    slope_factors=None,
    max_jumps=10,
    penalty=10.0,
    turn_penalty=0.0,
    max_steps=600,
    reward_scale=0.01,
    deterministic=False,
    label="",
):
    """Run a single PPO episode from start to goal. Returns path, cost, jumps."""
    land_costs, slope_factors = ensure_cost_components(
        costs,
        land_costs=land_costs,
        slope_factors=slope_factors,
    )
    env = make_masked_env(
        {
            "coords": coords,
            "costs": costs,
            "land_costs": land_costs,
            "slope_factors": slope_factors,
            "adj_direct": adj_direct,
            "adj_jump": adj_jump,
            "cell_size": cell_size,
            "start_idx": start_idx,
            "goal_idx": goal_idx,
            "max_jumps": max_jumps,
            "penalty": penalty,
            "turn_penalty": turn_penalty,
        },
        randomize=False,
        max_steps=max_steps,
        reward_scale=reward_scale,
        wrap_monitor=False,
    )

    obs, _ = env.reset()
    path = [start_idx]
    total_reward = 0.0
    done = False
    jumps = 0
    step_count = 0

    inner_env = env.env

    t0 = time.time()
    while not done:
        action_masks = env.action_masks()

        action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

        path.append(inner_env.current_node)

        if terminated:
            break

    elapsed = time.time() - t0
    reached_goal = info.get("reached_goal", False) if done else False

    path_cost = 0.0
    for i in range(1, len(path)):
        u, v = path[i - 1], path[i]
        edge_found = False
        for nb, ratio, d in adj_direct[u]:
            if nb == v:
                path_cost += costs[v] * ratio
                edge_found = True
                break
        if not edge_found:
            for nb, ratio, d in adj_jump[u]:
                if nb == v:
                    path_cost += costs[v] * ratio + penalty
                    jumps += 1
                    edge_found = True
                    break
        if not edge_found:
            path_cost += costs[v]

    turns = count_turns(coords, path) if len(path) >= 3 else 0
    distance_m = compute_path_distance(coords, path)

    if reached_goal:
        print(
            f"  [{label}] PPO: cost={path_cost:.1f} | {len(path)} cells | "
            f"~{distance_m:,.0f} m | jumps={jumps} | turns={turns} | "
            f"steps={step_count} | {elapsed:.2f}s"
        )
    else:
        print(
            f"  [{label}] PPO: FAILED to reach goal in {step_count} steps "
            f"({elapsed:.2f}s)"
        )

    return {
        "path": path if reached_goal else [],
        "total_cost": path_cost if reached_goal else float("inf"),
        "jumps": jumps,
        "turns": turns,
        "distance_m": distance_m if reached_goal else 0.0,
        "reached_goal": reached_goal,
        "total_reward": total_reward,
        "steps": step_count,
        "elapsed": elapsed,
    }


# pripravi graf, nacte model a vrati ppo vysledek pro jeden scenar
def run_ppo_for_result(
    result,
    start_lonlat,
    goal_lonlat,
    exclusion_penalty,
    max_jumps,
    turn_penalty,
    model_path="ppo_models/ppo_routing_final",
    max_jump_edge_m=200.0,
    max_steps=600,
    deterministic=False,
):
    """Run PPO inference on one scenario result."""
    costmap = result["costmap"]
    crs = result["crs"]
    exclusion_gdf = result["exclusion_gdf"]
    label = result["label"]

    start_idx, start_cid = find_nearest_cell(costmap, *start_lonlat, crs)
    goal_idx, goal_cid = find_nearest_cell(costmap, *goal_lonlat, crs)

    print(f"\n{'=' * 60}")
    print(f"PPO {label}")
    print(f"{'=' * 60}")
    print(f"Start coord: {start_lonlat} -> cell_id={start_cid}")
    print(f"Goal  coord: {goal_lonlat} -> cell_id={goal_cid}")

    warnings = []
    warnings += check_exclusion(*start_lonlat, crs, exclusion_gdf, costmap, start_idx, "Start")
    warnings += check_exclusion(*goal_lonlat, crs, exclusion_gdf, costmap, goal_idx, "Goal")
    if warnings:
        print("Exclusion warnings:")
        for w in warnings:
            print(w)

    centroids = costmap.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    costs = costmap["cost"].values.astype(float)
    land_costs, slope_factors = get_result_cost_components(result, costs)

    use_turns = turn_penalty > 0
    adj_direct, adj_jump = build_navigation_graph(
        coords, costs, result["cell_size"],
        max_jump_edge_m=max_jump_edge_m,
        uncrossable_gdf=result.get("uncrossable_gdf"),
        use_turns=use_turns,
        label=label,
    )

    model = load_ppo_model(model_path)
    print(f"  Model loaded: {model_path}")

    ppo_result = run_ppo_inference(
        model, coords, costs, adj_direct, adj_jump,
        result["cell_size"], start_idx, goal_idx,
        land_costs=land_costs,
        slope_factors=slope_factors,
        max_jumps=max_jumps, penalty=exclusion_penalty,
        turn_penalty=turn_penalty, max_steps=max_steps,
        deterministic=deterministic, label=label,
    )

    gradient_overlap = {}
    if ppo_result["path"]:
        gradient_overlap = summarize_gradient_overlap(
            coords, ppo_result["path"], crs,
            result.get("inward_gradient_info"),
        )

    return {
        **result,
        "start_idx": start_idx,
        "start_cid": start_cid,
        "goal_idx": goal_idx,
        "goal_cid": goal_cid,
        "warnings": warnings,
        "coords": coords,
        "costs": costs,
        "path": ppo_result["path"],
        "total_cost": ppo_result["total_cost"],
        "jumps": ppo_result["jumps"],
        "distance_m": ppo_result["distance_m"],
        "turns": ppo_result["turns"],
        "gradient_overlap": gradient_overlap,
        "elapsed": ppo_result["elapsed"],
        "ppo_reached_goal": ppo_result["reached_goal"],
        "ppo_steps": ppo_result["steps"],
        "ppo_reward": ppo_result["total_reward"],
    }


# obali inference pro vice scenaru stejne oblasti
def run_ppo_for_scenarios(
    results,
    start_lonlat,
    goal_lonlat,
    exclusion_penalty,
    max_jumps,
    turn_penalty,
    model_path="ppo_models/ppo_routing_final",
    max_jump_edge_m=200.0,
    max_steps=600,
    deterministic=False,
):
    """Run PPO on all scenario results."""
    return [
        run_ppo_for_result(
            result, start_lonlat, goal_lonlat,
            exclusion_penalty, max_jumps, turn_penalty,
            model_path=model_path,
            max_jump_edge_m=max_jump_edge_m,
            max_steps=max_steps,
            deterministic=deterministic,
        )
        for result in results
    ]


# opakovanou inferenci pouziva pro ziskani vice ruznych sample tras
def run_ppo_samples_for_result(
    result,
    model_path="ppo_models/ppo_routing_final",
    n_samples=3,
    max_attempts=12,
    exclusion_penalty=10.0,
    max_jumps=10,
    turn_penalty=0.0,
    max_jump_edge_m=200.0,
    max_steps=2500,
    coarse_step_m=None,
):
    """Generate up to n successful PPO routes for one scenario result."""
    costmap = result["costmap"]
    centroids = costmap.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    costs = costmap["cost"].values.astype(float)
    land_costs, slope_factors = get_result_cost_components(result, costs)

    if coarse_step_m:
        print(f"  [{result['label']}] Building coarse graph ({coarse_step_m} m steps)...")
        adj_direct, adj_jump = build_coarse_direct_graph(
            coords, step_m=coarse_step_m, costs=costs,
            uncrossable_gdf=result.get("uncrossable_gdf"),
        )
        exclusion_penalty = 0.0
    else:
        use_turns = turn_penalty > 0
        adj_direct, adj_jump = build_navigation_graph(
            coords,
            costs,
            result["cell_size"],
            max_jump_edge_m=max_jump_edge_m,
            uncrossable_gdf=result.get("uncrossable_gdf"),
            use_turns=use_turns,
            label=result["label"],
        )

    model = load_ppo_model(model_path)

    all_samples = []
    seen_paths = set()
    for attempt in range(max_attempts):
        sample = run_ppo_inference(
            model,
            coords,
            costs,
            adj_direct,
            adj_jump,
            result["cell_size"],
            result["start_idx"],
            result["goal_idx"],
            land_costs=land_costs,
            slope_factors=slope_factors,
            max_jumps=max_jumps,
            penalty=exclusion_penalty,
            turn_penalty=turn_penalty,
            max_steps=max_steps,
            deterministic=False,
            label=f"{result['label']} attempt {attempt + 1}",
        )
        if not sample["reached_goal"]:
            continue
        path_key = tuple(sample["path"])
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        all_samples.append(sample)

    print(f"  [{result['label']}] {len(all_samples)} unique successful routes from {max_attempts} attempts")

    if len(all_samples) > 1:
        costs_arr = np.array([s["total_cost"] for s in all_samples], dtype=float)
        lens_arr = np.array([s["distance_m"] for s in all_samples], dtype=float)

        def _norm(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-9)

        score = 0.5 * _norm(costs_arr) + 0.5 * _norm(lens_arr)
        ranked = sorted(zip(score, all_samples), key=lambda x: x[0])
        best = [s for _, s in ranked[:n_samples]]
    else:
        best = all_samples[:n_samples]

    for idx, s in enumerate(best):
        s["sample_id"] = idx + 1

    return {
        **result,
        "coords": coords,
        "costs": costs,
        "ppo_samples": best,
    }


# sampling workflow aplikuje na celou sadu scenaru
def run_ppo_samples_for_scenarios(
    results,
    model_path="ppo_models/ppo_routing_final",
    n_samples=3,
    max_attempts=12,
    exclusion_penalty=10.0,
    max_jumps=10,
    turn_penalty=0.0,
    max_jump_edge_m=200.0,
    max_steps=2500,
    coarse_step_m=None,
):
    """Generate up to n successful PPO routes for each scenario result."""
    return [
        run_ppo_samples_for_result(
            result,
            model_path=model_path,
            n_samples=n_samples,
            max_attempts=max_attempts,
            exclusion_penalty=exclusion_penalty,
            max_jumps=max_jumps,
            turn_penalty=turn_penalty,
            max_jump_edge_m=max_jump_edge_m,
            max_steps=max_steps,
            coarse_step_m=coarse_step_m,
        )
        for result in results
    ]
