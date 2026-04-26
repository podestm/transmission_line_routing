"""Training infrastructure for PPO routing: CNN policy, env factory, curriculum callback."""

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.ppo_common import build_coarse_direct_graph, make_masked_env


# ---------------------------------------------------------------------------
# CNN feature extractor
# ---------------------------------------------------------------------------

# extractor prevadi lokalni patch a globalni priznaky na vstup pro policy
class RoutingFeaturesExtractor(BaseFeaturesExtractor):
    """Small CNN for the (N_CHANNELS, PATCH_SIZE, PATCH_SIZE) spatial patch
    concatenated with the global scalar features vector."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_channels: int = 32,
        cnn_out_dim: int = 64,
        patch_radius: int = 12,  # informační — skutečný tvar se bere z observation_space
    ):
        patch_space = observation_space["patch"]    # (N_CHANNELS, patch_size, patch_size)
        global_space = observation_space["global"]  # (N_GLOBAL,)

        features_dim = cnn_out_dim + global_space.shape[0]
        super().__init__(observation_space, features_dim=features_dim)

        n_in = patch_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.cnn(torch.zeros(1, *patch_space.shape)).shape[1]

        self.cnn_head = nn.Linear(flat_dim, cnn_out_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        cnn_feats = torch.relu(self.cnn_head(self.cnn(observations["patch"])))
        return torch.cat([cnn_feats, observations["global"]], dim=1)


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------

# z processed costmapy pripravi coarse graf a fallback validni dvojici
def load_map_for_training(
    gpkg_path,
    *,
    coarse_step_m: float = 300.0,
    max_jumps: int = 0,
    penalty: float = 10.0,
    turn_penalty: float = 0.0,
    label: str = "",
) -> dict:
    """Load a costmap GPKG and build a coarse direct graph for training.

    Returns a graph_data dict compatible with make_masked_env / make_env_factory.
    """
    gpkg_path = Path(gpkg_path)
    costmap = gpd.read_file(gpkg_path, layer="costmap")
    crs_epsg = costmap.crs.to_epsg()

    centroids = costmap.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    costs = costmap["cost"].values.astype(np.float32)

    adj_direct, adj_jump = build_coarse_direct_graph(
        coords, step_m=coarse_step_m, costs=costs, cell_size_m=50.0,
    )

    valid_nodes = np.array(
        [
            i for i in range(len(costs))
            if np.isfinite(costs[i]) and len(adj_direct[i]) > 0
        ],
        dtype=np.int32,
    )
    if len(valid_nodes) < 2:
        raise ValueError(
            f"Map {label or gpkg_path.stem} does not contain at least two valid training nodes."
        )

    start_idx = int(valid_nodes[0])
    dx = coords[valid_nodes, 0] - coords[start_idx, 0]
    dy = coords[valid_nodes, 1] - coords[start_idx, 1]
    goal_idx = int(valid_nodes[np.argmax(dx * dx + dy * dy)])
    if goal_idx == start_idx:
        goal_idx = int(valid_nodes[1])

    print(
        f"  [{label or gpkg_path.stem}] {len(costs):,} nodes | "
        f"valid={len(valid_nodes):,} | fallback_start={start_idx}  "
        f"fallback_goal={goal_idx} | crs=EPSG:{crs_epsg}"
    )

    return {
        "label": label or gpkg_path.stem,
        "coords": coords,
        "costs": costs,
        "land_costs": None,
        "slope_factors": None,
        "adj_direct": adj_direct,
        "adj_jump": adj_jump,
        "cell_size": float(coarse_step_m),
        "start_idx": start_idx,
        "goal_idx": goal_idx,
        "max_jumps": max_jumps,
        "penalty": penalty,
        "turn_penalty": turn_penalty,
    }


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

# vraci nulovou fabriku, aby si vec env skladal vlastni env instance
def make_env_factory(graph_data: dict, *, max_steps: int, **kwargs):
    """Return a zero-argument callable that creates a fresh masked env from graph_data.

    Extra kwargs are forwarded to make_masked_env (reward_scale, proximity_coef, …).
    """
    def _factory():
        return make_masked_env(graph_data, max_steps=max_steps, randomize=True, **kwargs)
    return _factory


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

# callback postupne posouva curriculum od kratsich tras k delsim
class CurriculumCallback(BaseCallback):
    """Advances start-goal distance bounds when the agent crosses a success-rate threshold.

    Levels are defined as (min_dist_m, max_dist_m) pairs. The callback monitors
    ``reached_goal`` from episode info dicts and calls ``set_curriculum`` on all
    environments once the recent success rate exceeds ``success_threshold``.
    """

    def __init__(
        self,
        curriculum_levels: list,
        success_threshold: float = 0.5,
        min_episodes: int = 200,
        check_interval: int = 20_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum_levels = curriculum_levels
        self.success_threshold = success_threshold
        self.min_episodes = min_episodes
        self.check_interval = check_interval
        self.current_level = 0
        self._last_check = 0
        self._successes: list = []

    def _on_training_start(self) -> None:
        min_d, max_d = self.curriculum_levels[0]
        self.training_env.env_method("set_curriculum", min_d, max_d)
        if self.verbose:
            print(f"[Curriculum] Level 0: {min_d:.0f} – {max_d:.0f} m")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done:
                self._successes.append(bool(info.get("reached_goal", False)))

        if (
            self.num_timesteps - self._last_check >= self.check_interval
            and len(self._successes) >= self.min_episodes
            and self.current_level < len(self.curriculum_levels) - 1
        ):
            self._last_check = self.num_timesteps
            recent = self._successes[-self.min_episodes:]
            rate = sum(recent) / len(recent)

            if self.verbose:
                print(
                    f"\n[Curriculum] step={self.num_timesteps:,} | "
                    f"level={self.current_level} | "
                    f"success_rate={rate:.2f} (last {len(recent)} eps)"
                )

            if rate >= self.success_threshold:
                self.current_level += 1
                min_d, max_d = self.curriculum_levels[self.current_level]
                self.training_env.env_method("set_curriculum", min_d, max_d)
                if self.verbose:
                    print(
                        f"[Curriculum] -> Level {self.current_level}: "
                        f"{min_d:.0f} – {max_d:.0f} m"
                    )
        return True
