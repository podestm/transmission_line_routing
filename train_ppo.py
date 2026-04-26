from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import src.ppo_training as ppo_training


# vychozi konfigurace drzi vsechny parametry treninku na jednom miste
def make_default_config() -> dict:
    return {
        "save_dir": Path("ppo_models/coarse300_12km_v5"),
        "maps": [
            "data/processed/A1/costmap_A1_1_50m.gpkg",
            "data/processed/A1/costmap_A1_2_50m.gpkg",
            "data/processed/A2/costmap_A2_1_50m.gpkg",
            "data/processed/A2/costmap_A2_2_50m.gpkg",
            # "data/processed/A3/costmap_A3_1_50m.gpkg",
            # "data/processed/A3/costmap_A3_2_50m.gpkg",
            # "data/processed/A4/costmap_A4_1_50m.gpkg",
            # "data/processed/A4/costmap_A4_2_50m.gpkg",
        ],
        "coarse_step_m": 300.0,
        "max_route_dist": 12_000.0,
        "max_steps_multiplier": 3.0,
        "goal_radius_multiplier": 1.5,
        "patch_radius": 12,
        "curriculum_max_route_fracs": (0.38, 0.69, 1.0),
        "proximity_coef": 10.0,
        "revisit_penalty": 50.0,
        "step_penalty": 5.0,
        "goal_bonus": 100.0,
        "reward_scale": 0.01,
        "n_envs": 8,
        "total_timesteps": 10_000_000,
        "learning_rate": 3e-4,
        "ent_coef": 0.02,
        "n_steps": 2048,
        "batch_size": 256,
        "success_threshold": 0.8,
        "check_interval": 20_000,
        "min_episodes": 200,
        "checkpoint_freq": 100_000,
        "tb_log_name": "run",
        "progress_bar": True,
        "reset_num_timesteps": True,
        "verbose": 1,
        "device": "auto",
        "cnn_channels": 32,
        "cnn_out_dim": 64,
        "policy_pi": (128, 128),
        "policy_vf": (128, 128),
    }


# z maximalni delky trasy odvodi limit kroku pro jednu epizodu
def get_max_steps(cfg: dict) -> int:
    return int(cfg["max_route_dist"] / cfg["coarse_step_m"] * cfg["max_steps_multiplier"])


# cil se povazuje za dosazeny uz v okoli cilove bunky
def get_goal_radius(cfg: dict) -> float:
    return cfg["coarse_step_m"] * cfg["goal_radius_multiplier"]


# curriculum postupne rozsiruje povolenou delku start-cil dvojice
def get_curriculum_levels(cfg: dict) -> list[tuple[float, float]]:
    min_dist = cfg["coarse_step_m"] * 3
    return [
        (min_dist, float(cfg["max_route_dist"] * frac))
        for frac in cfg["curriculum_max_route_fracs"]
    ]


# jeden rollout sbira kroky ze vsech paralelnich prostredi
def get_rollout_timesteps(cfg: dict) -> int:
    return cfg["n_envs"] * cfg["n_steps"]


# sem se skladaji parametry predane do prostredi pri jeho tvorbe
def get_env_kwargs(cfg: dict) -> dict:
    first_min, first_max = get_curriculum_levels(cfg)[0]
    return {
        "patch_radius": cfg["patch_radius"],
        "reward_scale": cfg["reward_scale"],
        "proximity_coef": cfg["proximity_coef"],
        "revisit_penalty": cfg["revisit_penalty"],
        "goal_bonus": cfg["goal_bonus"],
        "step_penalty": cfg["step_penalty"],
        "goal_radius": get_goal_radius(cfg),
        "min_start_goal_dist": first_min,
        "max_start_goal_dist": first_max,
    }


# cli prepina nejbeznejsi override bez upravy zdrojaku
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO routing model from terminal.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit.")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Load maps, build VecEnv and model, then exit before training.",
    )
    parser.add_argument("--save-dir", type=Path, help="Override output directory.")
    parser.add_argument("--total-timesteps", type=int, help="Override total PPO timesteps.")
    parser.add_argument("--n-envs", type=int, help="Override number of parallel environments.")
    parser.add_argument("--n-steps", type=int, help="Override PPO rollout steps per environment.")
    parser.add_argument("--batch-size", type=int, help="Override PPO batch size.")
    parser.add_argument("--max-route-dist", type=float, help="Override max route distance in metres.")
    parser.add_argument("--patch-radius", type=int, help="Override CNN patch radius in cells.")
    parser.add_argument("--device", help="Torch device, e.g. auto, cpu, cuda.")
    parser.add_argument("--run-name", help="TensorBoard run base name.")
    parser.add_argument("--checkpoint-freq", type=int, help="Checkpoint frequency in timesteps.")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress-bar", dest="progress_bar", action="store_true")
    progress_group.add_argument("--no-progress-bar", dest="progress_bar", action="store_false")
    parser.set_defaults(progress_bar=None)
    return parser.parse_args()


# spoji vychozi konfiguraci s parametry predanymi z terminalu
def resolve_config(args: argparse.Namespace) -> dict:
    cfg = make_default_config()
    cfg["maps"] = list(cfg["maps"])

    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        cfg["n_envs"] = args.n_envs
    if args.n_steps is not None:
        cfg["n_steps"] = args.n_steps
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.max_route_dist is not None:
        cfg["max_route_dist"] = args.max_route_dist
    if args.patch_radius is not None:
        cfg["patch_radius"] = args.patch_radius
    if args.device is not None:
        cfg["device"] = args.device
    if args.run_name is not None:
        cfg["tb_log_name"] = args.run_name
    if args.checkpoint_freq is not None:
        cfg["checkpoint_freq"] = args.checkpoint_freq
    if args.progress_bar is not None:
        cfg["progress_bar"] = args.progress_bar

    cfg["save_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


# rychly prehled pred spustenim, aby bylo jasne co se bude trenovat
def print_config(cfg: dict) -> None:
    patch_size = 2 * cfg["patch_radius"] + 1
    print("=== PPO TRAINING CONFIG ===")
    print(f"SAVE_DIR:           {cfg['save_dir']}")
    print(f"MAPS:               {len(cfg['maps'])}")
    print(f"COARSE_STEP_M:      {cfg['coarse_step_m']:.0f}")
    print(f"MAX_ROUTE_DIST:     {cfg['max_route_dist']:.0f}")
    print(f"MAX_STEPS:          {get_max_steps(cfg)}")
    print(f"GOAL_RADIUS:        {get_goal_radius(cfg):.0f}")
    print(f"PATCH:              radius={cfg['patch_radius']}, size={patch_size}x{patch_size}")
    print(f"CURRICULUM:         {[(round(mn), round(mx)) for mn, mx in get_curriculum_levels(cfg)]}")
    print(f"N_ENVS:             {cfg['n_envs']}")
    print(f"TOTAL_TIMESTEPS:    {cfg['total_timesteps']:,}")
    print(f"N_STEPS:            {cfg['n_steps']}")
    print(f"BATCH_SIZE:         {cfg['batch_size']}")
    print(f"ROLLOUT_TIMESTEPS:  {get_rollout_timesteps(cfg):,}")
    print(f"LEARNING_RATE:      {cfg['learning_rate']}")
    print(f"ENT_COEF:           {cfg['ent_coef']}")
    print(f"CHECKPOINT_FREQ:    {cfg['checkpoint_freq']:,}")
    print(f"RUN_NAME:           {cfg['tb_log_name']}")
    print(f"DEVICE:             {cfg['device']}")
    print(f"PROGRESS_BAR:       {cfg['progress_bar']}")
    print()
    print("Maps:")
    for gpkg_path in cfg["maps"]:
        print(f"  - {Path(gpkg_path).stem}")
    print("Sampling mode:      pure random valid start-goal pairs per episode")


# nacte vsechny costmapy a pripravi jejich grafovou reprezentaci
def load_graphs(cfg: dict) -> list[dict]:
    if not cfg["maps"]:
        raise ValueError("Training config must contain at least one map.")

    print("Loading maps and building coarse graphs...")
    graph_list = []
    for gpkg_path in cfg["maps"]:
        graph = ppo_training.load_map_for_training(
            gpkg_path,
            coarse_step_m=cfg["coarse_step_m"],
            label=Path(gpkg_path).stem,
        )
        graph_list.append(graph)
    print(f"\nLoaded {len(graph_list)} maps.")
    return graph_list


# vytvori paralelni sadu gym prostredi pro sbirani rolloutu
def build_vec_env(cfg: dict, graph_list: list[dict]) -> DummyVecEnv:
    print(f"Creating {cfg['n_envs']} environments (round-robin over {len(graph_list)} maps)...")
    factories = [
        ppo_training.make_env_factory(
            graph_list[i % len(graph_list)],
            max_steps=get_max_steps(cfg),
            **get_env_kwargs(cfg),
        )
        for i in range(cfg["n_envs"])
    ]
    vec_env = DummyVecEnv(factories)
    print(f"VecEnv: {vec_env.num_envs} envs, obs_space={vec_env.observation_space}")
    return vec_env


# slozi policy, extractor a ppo hyperparametry do jednoho modelu
def build_model(cfg: dict, vec_env: DummyVecEnv) -> MaskablePPO:
    policy_kwargs = {
        "features_extractor_class": ppo_training.RoutingFeaturesExtractor,
        "features_extractor_kwargs": {
            "cnn_channels": cfg["cnn_channels"],
            "cnn_out_dim": cfg["cnn_out_dim"],
            "patch_radius": cfg["patch_radius"],
        },
        "net_arch": {"pi": list(cfg["policy_pi"]), "vf": list(cfg["policy_vf"])},
        "activation_fn": nn.ReLU,
    }

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        ent_coef=cfg["ent_coef"],
        policy_kwargs=policy_kwargs,
        verbose=cfg["verbose"],
        tensorboard_log=str(cfg["save_dir"] / "tensorboard"),
        device=cfg["device"],
    )
    total_params = sum(param.numel() for param in model.policy.parameters())
    print(f"Policy parameters: {total_params:,}")
    return model


# trenink kombinuje checkpoint callback a curriculum callback
def train_model(cfg: dict, model: MaskablePPO) -> None:
    checkpoint_cb = CheckpointCallback(
        save_freq=max(cfg["checkpoint_freq"] // cfg["n_envs"], 1),
        save_path=str(cfg["save_dir"] / "checkpoints"),
        name_prefix="ppo_routing",
        verbose=1,
    )
    curriculum_cb = ppo_training.CurriculumCallback(
        curriculum_levels=get_curriculum_levels(cfg),
        success_threshold=cfg["success_threshold"],
        min_episodes=cfg["min_episodes"],
        check_interval=cfg["check_interval"],
        verbose=1,
    )

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=[checkpoint_cb, curriculum_cb],
        reset_num_timesteps=cfg["reset_num_timesteps"],
        tb_log_name=cfg["tb_log_name"],
        progress_bar=cfg["progress_bar"],
    )
    print("Training finished.")


# ulozi textovy souhrn, aby byl beh dohledatelny i bez notebooku
def write_training_info(cfg: dict) -> Path:
    lines = [
        f"Model: {cfg['save_dir'].name}",
        f"Training date: {datetime.today().strftime('%Y-%m-%d')}",
        "",
        "=== MAPS ===",
    ]
    for gpkg_path in cfg["maps"]:
        lines.append(f"  {Path(gpkg_path).stem}")
    lines += [
        "",
        "=== CONFIGURATION ===",
        "sampling_mode:      pure random valid start-goal pairs",
        f"coarse_step_m:      {cfg['coarse_step_m']:.0f} m",
        f"max_route_dist:     {cfg['max_route_dist']:.0f} m",
        f"max_steps:          {get_max_steps(cfg)}",
        f"goal_radius:        {get_goal_radius(cfg):.0f} m",
        f"patch_radius:       {cfg['patch_radius']}",
        f"proximity_coef:     {cfg['proximity_coef']}",
        f"goal_bonus:         {cfg['goal_bonus']}",
        f"revisit_penalty:    {cfg['revisit_penalty']}",
        f"step_penalty:       {cfg['step_penalty']}",
        f"reward_scale:       {cfg['reward_scale']}",
        f"n_envs:             {cfg['n_envs']}",
        f"total_timesteps:    {cfg['total_timesteps']:,}",
        f"learning_rate:      {cfg['learning_rate']}",
        f"ent_coef:           {cfg['ent_coef']}",
        f"n_steps:            {cfg['n_steps']}",
        f"batch_size:         {cfg['batch_size']}",
        f"checkpoint_freq:    {cfg['checkpoint_freq']:,}",
        f"device:             {cfg['device']}",
        f"tb_log_name:        {cfg['tb_log_name']}",
        "",
        "=== CURRICULUM ===",
    ]
    for idx, (mn, mx) in enumerate(get_curriculum_levels(cfg), start=1):
        lines.append(f"Level {idx}: {mn:.0f} - {mx:.0f} m")

    info_path = cfg["save_dir"] / "training_info.txt"
    info_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Training info saved: {info_path}")
    return info_path


# finalni model se uklada oddelene od prubeznych checkpointu
def save_model(cfg: dict, model: MaskablePPO) -> Path:
    final_path = cfg["save_dir"] / "ppo_routing_final"
    model.save(str(final_path))
    saved_path = final_path.with_suffix(".zip")
    print(f"Model saved: {saved_path}")
    return saved_path


# hlavni vstupni bod drzi cely terminalovy workflow v jednom miste
def main() -> int:
    args = parse_args()
    cfg = resolve_config(args)
    print_config(cfg)

    if args.dry_run:
        print("\nDry run requested, exiting before build.")
        return 0

    graph_list = load_graphs(cfg)
    vec_env = build_vec_env(cfg, graph_list)
    try:
        model = build_model(cfg, vec_env)
        if args.build_only:
            print("\nBuild-only requested, exiting before training.")
            return 0

        train_model(cfg, model)
        save_model(cfg, model)
        write_training_info(cfg)
        return 0
    finally:
        vec_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
