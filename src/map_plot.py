"""Map visualization helpers: color constants, figure layout, and side-by-side plots."""

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString


COLORS_LIST = [
    "#2d6a4f",
    "#52b788",
    "#b7e4c7",
    "#fefae0",
    "#faedcd",
    "#f4a261",
    "#e76f51",
    "#d62828",
    "#6a040f",
]

PANEL_FACE_COLOR = "#f7f5f0"
EXCLUSION_COLOR = "#101418"
OUTLINE_COLOR = "#4f5d56"


# pripravi figuru a pomocnou osu pro sdilenou colorbar
def _make_comparison_figure(panel_count, panel_width=7.4, panel_height=8.8):
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(panel_width * panel_count + 1.8, panel_height),
        sharex=True,
        sharey=True,
    )
    if panel_count == 1:
        axes = [axes]
    else:
        axes = list(axes)

    fig.subplots_adjust(left=0.06, bottom=0.10, top=0.88, right=0.87, wspace=0.08)
    for axis in axes:
        axis.set_facecolor(PANEL_FACE_COLOR)

    cax = fig.add_axes([0.89, 0.18, 0.018, 0.62])
    return fig, axes, cax


# prekresli exkluzni zony tak, aby byly citelne i nad costmapou
def _plot_exclusion_layer(ax, exclusion_gdf, clip_plot):
    if exclusion_gdf is None or len(exclusion_gdf) == 0:
        return

    excl_plot = exclusion_gdf.to_crs(4326).copy()
    excl_plot["geometry"] = excl_plot.geometry.intersection(clip_plot)
    excl_plot = excl_plot[~excl_plot.is_empty]
    if len(excl_plot) == 0:
        return

    excl_plot.plot(
        ax=ax,
        color=EXCLUSION_COLOR,
        edgecolor="none",
        alpha=1.0,
        zorder=4,
    )


# sjednoti rozsah os a vzhled mapoveho panelu
def _style_map_axis(ax, clip_plot, bounds, mid_lat):
    outline = gpd.GeoSeries([clip_plot], crs=4326)
    outline.boundary.plot(ax=ax, color=OUTLINE_COLOR, linewidth=0.9, zorder=7)
    ax.set_xlabel("Longitude (deg)", fontsize=10)
    ax.set_ylabel("Latitude (deg)", fontsize=10)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect(1.0 / np.cos(np.radians(mid_lat)))


# spocita jednotnou cost skalu pro vsechny zobrazene scenare
def _shared_cost_scale(results):
    plot_frames = [result["costmap"].to_crs(4326).copy() for result in results]
    all_costs = pd.concat([frame["cost"] for frame in plot_frames], ignore_index=True)
    vmin = float(all_costs.min())
    vmax = float(all_costs.max())
    q99 = float(all_costs.quantile(0.99))
    vmax_plot = min(vmax, q99 * 1.1) if q99 < vmax else vmax
    cmap = mcolors.LinearSegmentedColormap.from_list("cost_scale", COLORS_LIST, N=256)
    return plot_frames, cmap, vmin, vmax_plot, vmax


# vykresli vedle sebe vic costmap stejne oblasti
def plot_costmaps_side_by_side(results, area_label):
    plot_frames, cmap, vmin, vmax_plot, vmax = _shared_cost_scale(results)
    clip_plot = results[0]["area"]["gdf_sel"].geometry.iloc[0]
    bounds = clip_plot.bounds
    mid_lat = (bounds[1] + bounds[3]) / 2

    fig, axes, cax = _make_comparison_figure(len(results))

    for ax, result, cm_plot in zip(axes, results, plot_frames):
        cm_plot.plot(
            column="cost",
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_plot,
            edgecolor="none",
            linewidth=0,
            legend=False,
            zorder=2,
        )
        _plot_exclusion_layer(ax, result["exclusion_gdf"], clip_plot)
        stats_text = (
            f"min: {cm_plot['cost'].min():.1f}\n"
            f"max: {cm_plot['cost'].max():.1f}\n"
            f"mean: {cm_plot['cost'].mean():.2f}\n"
            f"median: {cm_plot['cost'].median():.1f}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
        )
        ax.set_title(
            f"{result['label']} | {len(result['costmap']):,} cells",
            fontsize=13,
            fontweight="bold",
        )
        _style_map_axis(ax, clip_plot, bounds, mid_lat)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax_plot))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Cost (traversal cost)", fontsize=11)
    fig.suptitle(
        f"Costmap comparison {area_label} | shared scale {vmin:.1f} to {vmax_plot:.1f} (raw max {vmax:.1f})",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    plt.show()


# pripravi vedle sebe srovnani dijkstra tras pro jednotlive scenare
def plot_dijkstra_side_by_side(
    route_results,
    area_label,
    start_lonlat,
    goal_lonlat,
    exclusion_penalty,
    max_jumps,
    turn_penalty,
    max_jump_edge_m=200.0,
):
    plot_frames, cmap, vmin, vmax_plot, vmax = _shared_cost_scale(route_results)
    clip_plot = route_results[0]["area"]["gdf_sel"].geometry.iloc[0]
    bounds = clip_plot.bounds
    mid_lat = (bounds[1] + bounds[3]) / 2

    fig, axes, cax = _make_comparison_figure(len(route_results))

    for ax, result, cm_plot in zip(axes, route_results, plot_frames):
        cm_plot.plot(
            column="cost",
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_plot,
            edgecolor="none",
            linewidth=0,
            legend=False,
            zorder=2,
        )
        _plot_exclusion_layer(ax, result["exclusion_gdf"], clip_plot)

        if result["path"] and result["total_cost"] < float("inf"):
            path_x = result["coords"][result["path"], 0]
            path_y = result["coords"][result["path"], 1]
            path_line = LineString(zip(path_x, path_y))
            path_gdf = gpd.GeoDataFrame(geometry=[path_line], crs=result["crs"]).to_crs(4326)
            path_gdf.plot(ax=ax, color="#00ffff", linewidth=3, zorder=5)

            start_geom = cm_plot.iloc[result["start_idx"]].geometry.centroid
            goal_geom = cm_plot.iloc[result["goal_idx"]].geometry.centroid
            ax.plot(
                start_geom.x,
                start_geom.y,
                "o",
                color="#00ff00",
                markersize=12,
                markeredgecolor="white",
                markeredgewidth=2,
                zorder=6,
                label="Start",
            )
            ax.plot(
                goal_geom.x,
                goal_geom.y,
                "s",
                color="#ff0000",
                markersize=12,
                markeredgecolor="white",
                markeredgewidth=2,
                zorder=6,
                label="Goal",
            )
            ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

        title = result["label"]
        if result["total_cost"] == float("inf"):
            title += "\nPath not found"
        else:
            title += (
                f"\ncost={result['total_cost']:.1f} | {len(result['path'])} cells | "
                f"~{result['distance_m']:,.0f} m | jumps {result['jumps']}x | turns {result['turns']}"
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        _style_map_axis(ax, clip_plot, bounds, mid_lat)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax_plot))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Cost (traversal cost)", fontsize=11)
    fig.suptitle(
        f"Dijkstra comparison {area_label} | Start {start_lonlat} | Goal {goal_lonlat} | "
        f"max jumps {max_jumps} | max jump edge {max_jump_edge_m:.0f} m | "
        f"penalty {exclusion_penalty} | turn penalty {turn_penalty} | "
        f"shared scale {vmin:.1f} to {vmax_plot:.1f} (raw max {vmax:.1f})",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    plt.show()
