"""Dijkstra routing algorithm over costmap navigation graphs."""

import heapq
import sys
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


# hlavni shortest path routine nad navigacnim grafem a stavem skoku
def run_dijkstra(
    coords,
    costs,
    cell_size,
    start_idx,
    goal_idx,
    max_jumps,
    penalty,
    turn_penalty=0.0,
    label="",
    max_jump_edge_m=200.0,
    uncrossable_gdf=None,
):
    n = len(costs)
    n_jumps = max_jumps + 1
    use_turns = turn_penalty > 0
    n_dirs = 9 if use_turns else 1
    start_dir = 8 if use_turns else 0
    n_per_node = n_jumps * n_dirs

    adj_direct, adj_jump = build_navigation_graph(
        coords, costs, cell_size,
        max_jump_edge_m=max_jump_edge_m,
        uncrossable_gdf=uncrossable_gdf,
        use_turns=use_turns,
        label=label,
    )

    total_states = n * n_per_node
    if use_turns:
        mem_mb = total_states * 17 / 1024 / 1024
        print(
            f"  [{label}] Turn penalty active: {n_dirs} dirs x {n_jumps} jumps "
            f"= {total_states:,} states (~{mem_mb:.0f} MB)"
        )

    inf = float("inf")
    dist = np.full(total_states, inf)
    prev = np.full(total_states, -1, dtype=np.int64)
    visited = np.zeros(total_states, dtype=bool)

    s0 = start_idx * n_per_node + start_dir
    dist[s0] = 0.0
    heap = [(0.0, s0)]

    print(f"  [{label}] Running Dijkstra ({total_states:,} states)...", end=" ")
    sys.stdout.flush()
    t_dijk = time.time()
    expansions = 0

    while heap:
        d_cost, state = heapq.heappop(heap)
        if visited[state]:
            continue
        visited[state] = True
        expansions += 1

        u = state // n_per_node
        rem = state % n_per_node
        jumps_used = rem // n_dirs
        d_in = rem % n_dirs

        if u == goal_idx:
            break

        for v, ratio, d_out in adj_direct[u]:
            turn_cost = turn_penalty if (use_turns and d_in != start_dir and d_in != d_out) else 0.0
            new_dist = d_cost + costs[v] * ratio + turn_cost
            next_state = v * n_per_node + jumps_used * n_dirs + d_out
            if new_dist < dist[next_state]:
                dist[next_state] = new_dist
                prev[next_state] = state
                heapq.heappush(heap, (new_dist, next_state))

        if jumps_used < max_jumps:
            new_jumps = jumps_used + 1
            for v, ratio, d_out in adj_jump[u]:
                turn_cost = turn_penalty if (use_turns and d_in != start_dir and d_in != d_out) else 0.0
                new_dist = d_cost + costs[v] * ratio + penalty + turn_cost
                next_state = v * n_per_node + new_jumps * n_dirs + d_out
                if new_dist < dist[next_state]:
                    dist[next_state] = new_dist
                    prev[next_state] = state
                    heapq.heappush(heap, (new_dist, next_state))

    print(f"{expansions:,} expansions ({time.time() - t_dijk:.1f} s)")
    sys.stdout.flush()

    goal_base = goal_idx * n_per_node
    goal_dists = dist[goal_base : goal_base + n_per_node]
    best_sub = int(np.argmin(goal_dists))
    total_cost = goal_dists[best_sub]
    best_jumps = best_sub // n_dirs

    if total_cost == inf:
        return [], inf, 0

    path = []
    state = goal_base + best_sub
    while state >= 0:
        path.append(state // n_per_node)
        state = prev[state]
    path.reverse()
    return path, total_cost, best_jumps


# pripravi vstupy jednoho scenare a spusti nad nimi dijkstra
def run_dijkstra_for_result(
    result,
    start_lonlat,
    goal_lonlat,
    exclusion_penalty,
    max_jumps,
    turn_penalty,
    max_jump_edge_m=200.0,
):
    costmap = result["costmap"]
    crs = result["crs"]
    exclusion_gdf = result["exclusion_gdf"]
    label = result["label"]

    start_idx, start_cid = find_nearest_cell(costmap, *start_lonlat, crs)
    goal_idx, goal_cid = find_nearest_cell(costmap, *goal_lonlat, crs)

    print(f"\n{'=' * 60}")
    print(f"DIJKSTRA {label}")
    print(f"{'=' * 60}")
    print(f"Start coord: {start_lonlat} -> square cell_id={start_cid}")
    print(f"Goal  coord: {goal_lonlat} -> square cell_id={goal_cid}")

    warnings = []
    warnings += check_exclusion(*start_lonlat, crs, exclusion_gdf, costmap, start_idx, "Start")
    warnings += check_exclusion(*goal_lonlat, crs, exclusion_gdf, costmap, goal_idx, "Goal")
    if warnings:
        print("Exclusion warnings:")
        for warning in warnings:
            print(warning)
    else:
        print("Start and Goal are outside exclusion zones.")

    t0 = time.time()
    centroids = costmap.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    costs = costmap["cost"].values.astype(float)
    path, total_cost, jumps = run_dijkstra(
        coords,
        costs,
        result["cell_size"],
        start_idx,
        goal_idx,
        max_jumps,
        exclusion_penalty,
        turn_penalty=turn_penalty,
        label=label,
        max_jump_edge_m=max_jump_edge_m,
        uncrossable_gdf=result.get("uncrossable_gdf"),
    )

    distance_m = compute_path_distance(coords, path) if path else 0.0
    turns = count_turns(coords, path) if path else 0
    gradient_overlap = summarize_gradient_overlap(coords, path, crs, result.get("inward_gradient_info"))
    elapsed = time.time() - t0
    if total_cost == float("inf"):
        print(f"Path not found even with {max_jumps} jumps!")
    else:
        print(
            f"Total cost: {total_cost:.1f} | {len(path)} cells | ~{distance_m:,.0f} m | "
            f"jumps: {jumps}x | turns: {turns} | {elapsed:.2f} s total"
        )
        if gradient_overlap:
            overlap_text = ", ".join(
                f"{layer}={count}" for layer, count in gradient_overlap.items()
            )
            print(f"Gradient-zone cells on path: {overlap_text}")

    return {
        **result,
        "start_idx": start_idx,
        "start_cid": start_cid,
        "goal_idx": goal_idx,
        "goal_cid": goal_cid,
        "warnings": warnings,
        "coords": coords,
        "costs": costs,
        "path": path,
        "total_cost": total_cost,
        "jumps": jumps,
        "distance_m": distance_m,
        "turns": turns,
        "gradient_overlap": gradient_overlap,
        "elapsed": elapsed,
    }


# opakuje dijkstra pro celou sadu scenaru se stejnymi body
def run_dijkstra_for_scenarios(
    results,
    start_lonlat,
    goal_lonlat,
    exclusion_penalty,
    max_jumps,
    turn_penalty,
    max_jump_edge_m=200.0,
):
    return [
        run_dijkstra_for_result(
            result,
            start_lonlat,
            goal_lonlat,
            exclusion_penalty,
            max_jumps,
            turn_penalty,
            max_jump_edge_m=max_jump_edge_m,
        )
        for result in results
    ]
