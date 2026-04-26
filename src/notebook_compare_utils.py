"""Backward-compatibility shim.

All functions have been split into focused modules:
  - src.costmap_builder  – config loading and costmap construction
  - src.dijkstra         – Dijkstra routing algorithm
  - src.graph            – navigation graph building and path utilities
  - src.map_plot         – map visualization helpers and side-by-side plots
"""

# tento modul jen preposila puvodni importy do nove struktury souboru
from src.costmap_builder import (
    build_costmap_for_config,
    build_costmaps_for_scenarios,
    load_area_context,
    load_config_context,
)
from src.dijkstra import (
    run_dijkstra,
    run_dijkstra_for_result,
    run_dijkstra_for_scenarios,
)
from src.graph import (
    build_navigation_graph,
    check_exclusion,
    compute_path_distance,
    count_turns,
    find_nearest_cell,
    summarize_gradient_overlap,
)
from src.map_plot import (
    COLORS_LIST,
    EXCLUSION_COLOR,
    OUTLINE_COLOR,
    PANEL_FACE_COLOR,
    _make_comparison_figure,
    _plot_exclusion_layer,
    _shared_cost_scale,
    _style_map_axis,
    plot_costmaps_side_by_side,
    plot_dijkstra_side_by_side,
)

# exportuje stejne symboly jako drive, aby stare notebooky zustaly funkcni
__all__ = [
    "build_costmap_for_config",
    "build_costmaps_for_scenarios",
    "build_navigation_graph",
    "check_exclusion",
    "COLORS_LIST",
    "compute_path_distance",
    "count_turns",
    "EXCLUSION_COLOR",
    "find_nearest_cell",
    "load_area_context",
    "load_config_context",
    "OUTLINE_COLOR",
    "PANEL_FACE_COLOR",
    "plot_costmaps_side_by_side",
    "plot_dijkstra_side_by_side",
    "run_dijkstra",
    "run_dijkstra_for_result",
    "run_dijkstra_for_scenarios",
    "summarize_gradient_overlap",
    "_make_comparison_figure",
    "_plot_exclusion_layer",
    "_shared_cost_scale",
    "_style_map_axis",
]
