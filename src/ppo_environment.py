"""
Gymnasium environment for power-line routing on a costmap graph.

The agent walks node-by-node through the same graph that Dijkstra uses,
choosing at each step one of the available neighbors (direct or jump).
Action masking ensures only valid moves are presented to the policy.

Observation: Dict space for CNN policy
  "patch"  — (N_CHANNELS, PATCH_SIZE, PATCH_SIZE) spatial window around current node
  "global" — (N_GLOBAL,) scalar features (goal direction, distance, jumps, steps)
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Maximum number of actions: 8 direct neighbors + 8 best jump neighbors
MAX_DIRECT = 8
MAX_JUMP = 8
MAX_ACTIONS = MAX_DIRECT + MAX_JUMP

# CNN patch parameters (defaults — can be overridden per-instance via patch_radius=)
PATCH_RADIUS = 12
PATCH_SIZE = 2 * PATCH_RADIUS + 1  # 25
# Channels: land_cost, slope_factor, visited, is_current, is_exclusion
N_CHANNELS = 5
# Global features: goal_dx, goal_dy, goal_dist, jumps_used, steps_ratio
N_GLOBAL = 5


# prostredi drzi stav epizody a hlida validni pohyb po grafu
class CostmapRoutingEnv(gym.Env):
    """Step-by-step routing on a costmap graph with action masking."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        coords: np.ndarray,
        costs: np.ndarray,
        land_costs: np.ndarray,
        slope_factors: np.ndarray,
        adj_direct: list,
        adj_jump: list,
        cell_size: float,
        start_idx: int,
        goal_idx: int,
        max_jumps: int = 10,
        penalty: float = 10.0,
        turn_penalty: float = 0.0,
        max_steps: int = 600,
        patch_radius: int = PATCH_RADIUS,
        randomize_start_goal: bool = False,
        min_start_goal_dist: float = 2000.0,
        max_start_goal_dist: float = 0.0,
        reward_scale: float = 0.01,
        proximity_coef: float = 5.0,
        momentum_bonus: float = 0.0,
        revisit_penalty: float = 2.0,
        goal_bonus: float = 100.0,
        goal_radius: float = 0.0,
        step_penalty: float = 0.0,
    ):
        super().__init__()

        # Per-instance patch dimensions (derived from patch_radius)
        self._patch_radius = int(patch_radius)
        self._patch_size = 2 * self._patch_radius + 1

        self.coords = coords.astype(np.float32)
        self.costs = costs.astype(np.float32)
        self.land_costs = land_costs.astype(np.float32)
        self.slope_factors = slope_factors.astype(np.float32)
        self.adj_direct = adj_direct
        self.adj_jump = adj_jump
        self.cell_size = float(cell_size)
        self.n_nodes = len(costs)

        self.default_start = start_idx
        self.default_goal = goal_idx
        self.max_jumps = max_jumps
        self.penalty = float(penalty)
        self.turn_penalty = float(turn_penalty)
        self.max_steps = max_steps
        self.randomize = randomize_start_goal
        self.min_sg_dist = min_start_goal_dist
        self.max_sg_dist = max_start_goal_dist  # 0 = no upper limit
        self.reward_scale = reward_scale
        self.proximity_coef = proximity_coef
        self.momentum_bonus = float(momentum_bonus)
        self.revisit_penalty = float(revisit_penalty)
        self.goal_bonus = float(goal_bonus)
        # goal_radius > 0: agent reaches goal if within this many metres of goal node.
        # Useful for coarse-step training where exact goal hit is nearly impossible.
        self.goal_radius = float(goal_radius)
        # step_penalty > 0: fixed cost per step regardless of direction.
        # Directly penalizes route length and discourages circuitous paths.
        self.step_penalty = float(step_penalty)

        # Bounding box for coordinate normalization
        self.x_min = self.coords[:, 0].min()
        self.x_max = self.coords[:, 0].max()
        self.y_min = self.coords[:, 1].min()
        self.y_max = self.coords[:, 1].max()
        self.x_range = max(self.x_max - self.x_min, 1.0)
        self.y_range = max(self.y_max - self.y_min, 1.0)
        self.diag = math.sqrt(self.x_range**2 + self.y_range**2)

        # Pre-compute max cost for normalization (cap infinities)
        finite_mask = np.isfinite(self.costs)
        self.max_cost = float(self.costs[finite_mask].max()) if finite_mask.any() else 10.0
        self.max_land_cost = float(self.land_costs[self.land_costs < 100].max()) if (self.land_costs < 100).any() else 10.0
        self.max_slope_factor = float(self.slope_factors.max()) if len(self.slope_factors) else 2.0

        # Pre-compute direction angle table for turn cosine
        # Directions 0-7 are spaced 45° apart
        self._dir_angles = np.array([i * math.pi / 4 for i in range(8)])
        # Precompute 8×8 cosine lookup table — avoids math.cos() per action
        self._cos_table = np.cos(
            self._dir_angles[:, None] - self._dir_angles[None, :]
        ).astype(np.float32)

        # Build 2D grid for CNN patch extraction
        self._build_grid()

        # Dict observation space for CNN policy
        self.observation_space = spaces.Dict({
            "patch": spaces.Box(
                low=0.0, high=1.0,
                shape=(N_CHANNELS, self._patch_size, self._patch_size),
                dtype=np.float32,
            ),
            "global": spaces.Box(
                low=-2.0, high=2.0,
                shape=(N_GLOBAL,),
                dtype=np.float32,
            ),
        })

        # Action: pick one of MAX_ACTIONS neighbors (masked)
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        # Episode state (set in reset)
        self.current_node = 0
        self.goal_idx = 0
        self.start_idx = 0
        self.jumps_used = 0
        self.step_count = 0
        self.last_dir = -1
        self._visited_arr = np.zeros(len(costs), dtype=bool)  # fast visited tracking
        self._current_neighbors = []  # list of (node, ratio, dir, is_jump)
        self._goal_dists: np.ndarray | None = None  # precomputed per-node dist to goal
        self._best_dist_to_goal: float = 0.0  # best (min) dist to goal seen this episode
        self._goal_dists_cache: dict = {}  # cache goal distance arrays by goal_idx
        self._jump_cache: dict = {}  # cache sorted jump neighbors by goal_idx
        self._adj_jump_empty: bool = not any(adj_jump)  # True if all jump lists empty (coarse mode)
        self._refresh_valid_episode_nodes()
        self._precompute_direct_neighbors()

    # ------ Curriculum API ------
    def set_goal_bonus(self, bonus: float):
        """Hot-update goal bonus reward (called by curriculum callback on level transitions)."""
        self.goal_bonus = float(bonus)

    def set_curriculum(self, min_dist: float, max_dist: float = 0.0):
        """Update start-goal distance bounds (called by CurriculumCallback)."""
        self.min_sg_dist = min_dist
        self.max_sg_dist = max_dist

    def set_max_steps(self, max_steps: int):
        """Hot-update episode step limit (called by curriculum for per-level budgets)."""
        self.max_steps = int(max_steps)

    def set_graph(self, coords, costs, land_costs, slope_factors,
                  adj_direct, adj_jump, cell_size,
                  start_idx, goal_idx):
        """Hot-swap graph data (used by resolution curriculum)."""
        self.coords = coords.astype(np.float32)
        self.costs = costs.astype(np.float32)
        self.land_costs = land_costs.astype(np.float32)
        self.slope_factors = slope_factors.astype(np.float32)
        self.adj_direct = adj_direct
        self.adj_jump = adj_jump
        self.cell_size = float(cell_size)
        self.n_nodes = len(costs)
        self.default_start = start_idx
        self.default_goal = goal_idx
        # Recompute bounding box and normalization
        self.x_min = self.coords[:, 0].min()
        self.x_max = self.coords[:, 0].max()
        self.y_min = self.coords[:, 1].min()
        self.y_max = self.coords[:, 1].max()
        self.x_range = max(self.x_max - self.x_min, 1.0)
        self.y_range = max(self.y_max - self.y_min, 1.0)
        self.diag = math.sqrt(self.x_range**2 + self.y_range**2)
        finite_mask = np.isfinite(self.costs)
        self.max_cost = float(self.costs[finite_mask].max()) if finite_mask.any() else 10.0
        self.max_land_cost = float(self.land_costs[self.land_costs < 100].max()) if (self.land_costs < 100).any() else 10.0
        self.max_slope_factor = float(self.slope_factors.max()) if len(self.slope_factors) else 2.0
        self._refresh_valid_episode_nodes()
        # Rebuild 2D grid for new graph geometry
        self._build_grid()
        # Recompute goal distances for the new graph so _get_obs / step don't
        # use a stale array sized for the old (coarser) graph.
        self.goal_idx = self.default_goal
        self.current_node = self.default_start  # ensure valid node for new graph
        self._precompute_goal_dists()
        self._visited_arr = np.zeros(self.n_nodes, dtype=bool)
        self._goal_dists_cache = {}  # invalidate: new graph has different node count
        self._jump_cache = {}        # invalidate: new graph has different node count
        self._adj_jump_empty = not any(adj_jump)
        self._precompute_direct_neighbors()
        self._precompute_sorted_jumps()  # rebuild jump cache (uses new _goal_dists)
        self._build_neighbor_list()      # rebuild action list for new current_node

    # ------ CNN grid helpers ------

    def _build_grid(self):
        """Build 2D grid arrays (land_cost, slope, exclusion) for patch extraction.

        Each node is mapped to (row, col) by rounding its coordinates to the
        nearest cell_size multiple.  Out-of-graph cells are treated as exclusion.
        """
        cs = self.cell_size
        self._grid_rows = int(np.round(self.y_range / cs)) + 1
        self._grid_cols = int(np.round(self.x_range / cs)) + 1

        # Per-node grid coordinates
        self._node_row = np.clip(
            np.round((self.coords[:, 1] - self.y_min) / cs).astype(np.int32),
            0, self._grid_rows - 1,
        )
        self._node_col = np.clip(
            np.round((self.coords[:, 0] - self.x_min) / cs).astype(np.int32),
            0, self._grid_cols - 1,
        )

        # Default: everything is exclusion / high-cost until filled by nodes
        self._grid_land = np.ones((self._grid_rows, self._grid_cols), dtype=np.float32)
        self._grid_slope = np.ones((self._grid_rows, self._grid_cols), dtype=np.float32)
        self._grid_excl = np.ones((self._grid_rows, self._grid_cols), dtype=np.float32)

        ml = self.max_land_cost if self.max_land_cost > 0 else 1.0
        ms = self.max_slope_factor if self.max_slope_factor > 0 else 1.0

        # Vectorised fill — only passable (finite-cost) nodes
        finite = np.isfinite(self.costs)
        rows = self._node_row[finite]
        cols = self._node_col[finite]
        self._grid_land[rows, cols] = np.minimum(self.land_costs[finite] / ml, 1.0)
        self._grid_slope[rows, cols] = np.minimum(self.slope_factors[finite] / ms, 1.0)
        self._grid_excl[rows, cols] = 0.0

        # Visited grid — reset each episode in reset()
        self._grid_visited = np.zeros((self._grid_rows, self._grid_cols), dtype=np.float32)
        # Preallocated patch buffer — reused every step to avoid repeated allocations
        self._patch_buf = np.zeros((N_CHANNELS, self._patch_size, self._patch_size), dtype=np.float32)

    def _get_patch(self) -> np.ndarray:
        """Return (N_CHANNELS, PATCH_SIZE, PATCH_SIZE) spatial patch around current node.

        Channels: land_cost, slope_factor, visited, is_current, is_exclusion.
        Out-of-bounds cells are padded with exclusion defaults.
        Uses a preallocated buffer to avoid per-step array allocations.
        """
        r = int(self._node_row[self.current_node])
        c = int(self._node_col[self.current_node])
        half = self._patch_radius
        P = self._patch_size

        # Patch slices in grid coordinates
        gr_s = r - half;  gr_e = r + half + 1
        gc_s = c - half;  gc_e = c + half + 1

        # Corresponding patch slices (clamped to valid patch region)
        pr_s = max(0, -gr_s);       pr_e = P - max(0, gr_e - self._grid_rows)
        pc_s = max(0, -gc_s);       pc_e = P - max(0, gc_e - self._grid_cols)
        vgr_s = max(0, gr_s);       vgr_e = min(self._grid_rows, gr_e)
        vgc_s = max(0, gc_s);       vgc_e = min(self._grid_cols, gc_e)

        # Reset buffer to exclusion defaults
        buf = self._patch_buf
        buf[0] = 1.0  # land_cost default
        buf[1] = 1.0  # slope_factor default
        buf[2] = 0.0  # visited default
        buf[3] = 0.0  # is_current default
        buf[4] = 1.0  # is_exclusion default

        if pr_e > pr_s and pc_e > pc_s:
            buf[0, pr_s:pr_e, pc_s:pc_e] = self._grid_land   [vgr_s:vgr_e, vgc_s:vgc_e]
            buf[1, pr_s:pr_e, pc_s:pc_e] = self._grid_slope  [vgr_s:vgr_e, vgc_s:vgc_e]
            buf[2, pr_s:pr_e, pc_s:pc_e] = self._grid_visited[vgr_s:vgr_e, vgc_s:vgc_e]
            buf[4, pr_s:pr_e, pc_s:pc_e] = self._grid_excl   [vgr_s:vgr_e, vgc_s:vgc_e]

        buf[3, half, half] = 1.0  # mark agent position at center

        return buf.copy()  # copy required — SB3 holds reference to obs array

    # vybere jen bunky, ktere nejsou v exkluzi a maji kam jit
    def _refresh_valid_episode_nodes(self):
        """Cache nodes that can be used as episode start/goal candidates."""
        self.valid_episode_nodes = np.array(
            [
                i for i in range(self.n_nodes)
                if np.isfinite(self.costs[i]) and len(self.adj_direct[i]) > 0
            ],
            dtype=np.int32,
        )
        if len(self.valid_episode_nodes) == 0:
            raise ValueError("No valid non-exclusion nodes available for episode sampling")

        if self.default_start not in self.valid_episode_nodes:
            self.default_start = int(self.valid_episode_nodes[0])
        if self.default_goal not in self.valid_episode_nodes:
            fallback_idx = 1 if len(self.valid_episode_nodes) > 1 else 0
            self.default_goal = int(self.valid_episode_nodes[fallback_idx])

    def _precompute_sorted_jumps(self):
        """Precompute jump neighbors sorted by goal distance for the current episode.

        Called once per episode after _precompute_goal_dists() — eliminates
        per-step sorted() call in _build_neighbor_list(). Results are cached
        by goal_idx so repeated resets to the same goal are free.
        """
        if self.goal_idx in self._jump_cache:
            self._cached_jump = self._jump_cache[self.goal_idx]
            return
        # If adj_jump is entirely empty (coarse mode), share one instance across
        # all goals to avoid O(n_nodes * n_goals) memory usage.
        if self._adj_jump_empty:
            if not hasattr(self, '_empty_jump_list'):
                self._empty_jump_list = [[] for _ in range(self.n_nodes)]
            self._cached_jump = self._empty_jump_list
            return
        cached = []
        for node in range(self.n_nodes):
            raw = self.adj_jump[node]
            if raw:
                sorted_j = sorted(raw, key=lambda x: self._goal_dists[x[0]])[:MAX_JUMP]
                cached.append([(v, ratio, d, True) for v, ratio, d in sorted_j])
            else:
                cached.append([])
        self._cached_jump = cached
        self._jump_cache[self.goal_idx] = cached

    def _precompute_direct_neighbors(self):
        """Cache padded direct-neighbor lists for all nodes (goal-independent).

        Called once at graph load so _build_neighbor_list() only copies a list
        instead of looping over adj_direct each step.
        """
        cached = []
        for node in range(self.n_nodes):
            nb = []
            for v, ratio, d in self.adj_direct[node]:
                if len(nb) >= MAX_DIRECT:
                    break
                nb.append((v, ratio, d, False))
            while len(nb) < MAX_DIRECT:
                nb.append(None)
            cached.append(nb)
        self._cached_direct = cached

    # losuje start a cil tak, aby odpovidaly aktualnimu curriculum intervalu
    def _sample_start_goal(self):
        """Sample a valid start/goal pair that respects the current distance curriculum."""
        valid_nodes = self.valid_episode_nodes
        for _ in range(200):
            s = int(valid_nodes[self.np_random.integers(0, len(valid_nodes))])
            g = int(valid_nodes[self.np_random.integers(0, len(valid_nodes))])
            if s == g:
                continue

            dx = self.coords[s, 0] - self.coords[g, 0]
            dy = self.coords[s, 1] - self.coords[g, 1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < self.min_sg_dist:
                continue
            if self.max_sg_dist > 0 and distance > self.max_sg_dist:
                continue
            return s, g

        # Fallback: keep the episode valid even if the curriculum interval is tight.
        return int(self.default_start), int(self.default_goal)

    def _norm_x(self, x):
        return (x - self.x_min) / self.x_range

    def _norm_y(self, y):
        return (y - self.y_min) / self.y_range

    def _dist_to_goal(self, node):
        dx = self.coords[node, 0] - self.coords[self.goal_idx, 0]
        dy = self.coords[node, 1] - self.coords[self.goal_idx, 1]
        return math.sqrt(dx * dx + dy * dy)

    def _precompute_goal_dists(self):
        """Vectorised precomputation of Euclidean distance from every node to goal.
        Called once per episode; lookups are O(1) floats instead of per-step sqrt.
        Results are cached by goal_idx so repeated resets to the same goal are free."""
        if self.goal_idx in self._goal_dists_cache:
            self._goal_dists = self._goal_dists_cache[self.goal_idx]
            return
        gx = self.coords[self.goal_idx, 0]
        gy = self.coords[self.goal_idx, 1]
        dx = self.coords[:, 0] - gx
        dy = self.coords[:, 1] - gy
        self._goal_dists = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        self._goal_dists_cache[self.goal_idx] = self._goal_dists

    def _build_neighbor_list(self):
        """Build the padded action-to-neighbor mapping for current node."""
        # Direct neighbors: precomputed padded list (goal-independent)
        neighbors = list(self._cached_direct[self.current_node])

        # Jump neighbors: precomputed + sorted once per episode
        if self.jumps_used < self.max_jumps:
            neighbors.extend(self._cached_jump[self.current_node])

        # Pad jump slots
        while len(neighbors) < MAX_ACTIONS:
            neighbors.append(None)

        # Final sprint: when within 2x goal_radius, add goal node as a direct
        # reachable action in the first free slot. Lets the agent take a short
        # final step instead of circling around the goal area.
        if self.goal_radius > 0.0 and self.goal_idx != self.current_node:
            if self._goal_dists[self.current_node] <= self.goal_radius * 2.0:
                goal_dir = self.last_dir if self.last_dir >= 0 else 0
                for i, nb in enumerate(neighbors):
                    if nb is None:
                        neighbors[i] = (self.goal_idx, 1.0, goal_dir, False)
                        break

        self._current_neighbors = neighbors

    def _turn_cosine(self, dir_a, dir_b):
        """Cosine of angle between two directions (0-7).
        1.0 = straight, 0.0 = 90°, -1.0 = U-turn."""
        return float(self._cos_table[dir_a, dir_b])

    def _get_obs(self) -> dict:
        """Return Dict observation: CNN patch + global scalars."""
        cx, cy = self.coords[self.current_node]
        gx, gy = self.coords[self.goal_idx]
        goal_dist = self._goal_dists[self.current_node]

        global_f = np.array([
            (gx - cx) / self.diag,                    # goal direction x
            (gy - cy) / self.diag,                    # goal direction y
            goal_dist / self.diag,                    # normalized goal distance
            self.jumps_used / max(self.max_jumps, 1), # jumps used fraction
            self.step_count / self.max_steps,          # steps used fraction
        ], dtype=np.float32)

        return {"patch": self._get_patch(), "global": global_f}

    # maska drzi jen povolene akce a zakazuje prilis ostry obrat
    def action_masks(self):
        """Return boolean mask: True = action is valid. Masks sharp turns (>90°)."""
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        for i, nb in enumerate(self._current_neighbors):
            if nb is not None:
                if self.last_dir >= 0:
                    _, _, d, _ = nb
                    if self._turn_cosine(self.last_dir, d) < 0:
                        continue  # sharp turn (>90°) — always forbidden
                mask[i] = True
        # Safety: allow everything if no valid actions remain
        if not mask.any():
            for i, nb in enumerate(self._current_neighbors):
                if nb is not None:
                    mask[i] = True
        return mask

    # reset pripravi novou epizodu od losovaneho nebo defaultniho startu
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize:
            self.start_idx, self.goal_idx = self._sample_start_goal()
        else:
            self.start_idx = self.default_start
            self.goal_idx = self.default_goal

        self.current_node = self.start_idx
        self.jumps_used = 0
        self.step_count = 0
        self.last_dir = -1
        self._visited_arr[:] = False
        self._visited_arr[self.start_idx] = True
        # Reset and mark start node on visited grid
        self._grid_visited[:] = 0.0
        self._grid_visited[
            int(self._node_row[self.start_idx]),
            int(self._node_col[self.start_idx]),
        ] = 1.0
        self._precompute_goal_dists()
        self._best_dist_to_goal = float(self._goal_dists[self.start_idx])
        self._precompute_sorted_jumps()
        self._build_neighbor_list()

        return self._get_obs(), {}

    # jeden krok posune agenta a prepocita reward i konec epizody
    def step(self, action):
        self.step_count += 1

        # Validate action
        if action < 0 or action >= MAX_ACTIONS or self._current_neighbors[action] is None:
            # Invalid action — penalize and stay
            obs = self._get_obs()
            return obs, -10.0, False, self.step_count >= self.max_steps, {}

        v, ratio, d_out, is_jump = self._current_neighbors[action]

        # --- Compute reward (mirrors Dijkstra cost, negated) ---
        step_cost = self.costs[v] * ratio
        reward = -step_cost * self.reward_scale
        # Fixed per-step penalty to discourage circuitous routes
        if self.step_penalty > 0.0:
            reward -= self.step_penalty * self.reward_scale

        if is_jump:
            reward -= self.penalty * self.reward_scale
            self.jumps_used += 1

        if self.turn_penalty > 0 and self.last_dir >= 0 and d_out != self.last_dir:
            # Only penalize turn if continuing straight was a valid option
            if any(nb is not None and nb[2] == self.last_dir
                   for nb in self._current_neighbors[:MAX_DIRECT]):
                reward -= self.turn_penalty * self.reward_scale

        # Momentum bonus: reward for continuing in the same direction (straight segment)
        if self.momentum_bonus > 0 and self.last_dir >= 0 and d_out == self.last_dir:
            reward += self.momentum_bonus * self.reward_scale

        # Proximity shaping: reward for getting closer, penalty for moving away
        old_dist = self._goal_dists[self.current_node]
        new_dist = self._goal_dists[v]
        delta = old_dist - new_dist  # positive = toward goal, negative = away
        reward += self.proximity_coef * delta / self.cell_size * self.reward_scale
        # Extra punishment for moving away from goal (asymmetric — discourage detours)
        if delta < 0:
            reward += self.proximity_coef * delta / self.cell_size * self.reward_scale

        # Progress bonus: extra reward for achieving a new best distance to goal.
        # Potential-based shaping — prevents oscillation farming.
        if new_dist < self._best_dist_to_goal:
            progress = self._best_dist_to_goal - new_dist
            reward += self.proximity_coef * progress / self.cell_size * self.reward_scale
            self._best_dist_to_goal = new_dist

        # Revisit penalty
        if self._visited_arr[v]:
            reward -= self.revisit_penalty * self.reward_scale

        # Move
        self.current_node = v
        self.last_dir = d_out
        self._visited_arr[v] = True
        self._grid_visited[int(self._node_row[v]), int(self._node_col[v])] = 1.0

        # Check goal — exact hit or within goal_radius
        if self.goal_radius > 0.0:
            terminated = self._goal_dists[v] <= self.goal_radius
        else:
            terminated = (v == self.goal_idx)
        if terminated:
            # Efficiency bonus: reward for reaching goal quickly (max 2x goal_bonus if first step)
            efficiency = max(0.0, 1.0 - self.step_count / self.max_steps)
            reward += self.goal_bonus * (1.0 + efficiency)

        # Check max steps
        truncated = (self.step_count >= self.max_steps) and not terminated

        if truncated:
            # Penalty proportional to remaining distance — consistent with proximity reward scale.
            # Agent close to goal when time runs out pays less than agent far away.
            remaining = float(self._goal_dists[self.current_node])
            reward -= self.proximity_coef * remaining / self.cell_size * self.reward_scale

        # Rebuild neighbors for new position
        self._build_neighbor_list()

        # Dead end (no valid neighbors and not at goal)
        if not terminated and not truncated:
            if not any(nb is not None for nb in self._current_neighbors):
                reward -= 20.0 * self.reward_scale
                truncated = True

        obs = self._get_obs()
        info = {}
        if terminated or truncated:
            info["total_cost"] = self._compute_path_cost()
            info["path_length"] = int(self._visited_arr.sum())
            info["jumps"] = self.jumps_used
            info["reached_goal"] = terminated

        return obs, reward, terminated, truncated, info

    def _compute_path_cost(self):
        """Approximate total cost (we don't store the full path, just visited count)."""
        return int(self._visited_arr.sum())  # placeholder — real cost tracked externally
