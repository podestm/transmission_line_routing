"""PPO result plotting and reporting utilities."""

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

from src.map_plot import (
    _make_comparison_figure,
    _plot_exclusion_layer,
    _shared_cost_scale,
    _style_map_axis,
)

PPO_SAMPLE_COLORS = ["#ff006e", "#fb5607", "#8338ec"]


# kresli dijkstra trasu a nekolik sampled ppo tras pres stejnou costmapu
def plot_dijkstra_with_ppo_samples(sample_results, area_label, start_lonlat, goal_lonlat):
    """Overlay Dijkstra and up to three successful PPO routes for each scenario."""
    if not sample_results:
        return

    plot_frames, cmap, vmin, vmax_plot, _ = _shared_cost_scale(sample_results)
    clip_plot = sample_results[0]["area"]["gdf_sel"].geometry.iloc[0]
    bounds = clip_plot.bounds
    mid_lat = (bounds[1] + bounds[3]) / 2

    fig, axes, cax = _make_comparison_figure(len(sample_results))

    for ax, result, cm_plot in zip(axes, sample_results, plot_frames):
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

        if result.get("path"):
            d_path = result["path"]
            d_line = LineString(zip(result["coords"][d_path, 0], result["coords"][d_path, 1]))
            gpd.GeoDataFrame(geometry=[d_line], crs=result["crs"]).to_crs(4326).plot(
                ax=ax, color="#00b4d8", linewidth=3.2, zorder=5, label="Dijkstra"
            )

        for sample, color in zip(result.get("ppo_samples", []), PPO_SAMPLE_COLORS):
            p_path = sample["path"]
            p_line = LineString(zip(result["coords"][p_path, 0], result["coords"][p_path, 1]))
            gpd.GeoDataFrame(geometry=[p_line], crs=result["crs"]).to_crs(4326).plot(
                ax=ax,
                color=color,
                linewidth=2.4,
                alpha=0.85,
                zorder=6,
                label=f"PPO {sample['sample_id']}"
            )

        start_geom = cm_plot.iloc[result["start_idx"]].geometry.centroid
        goal_geom = cm_plot.iloc[result["goal_idx"]].geometry.centroid
        ax.plot(start_geom.x, start_geom.y, "o", color="#00ff00",
                markersize=12, markeredgecolor="white",
                markeredgewidth=2, zorder=7, label="Start")
        ax.plot(goal_geom.x, goal_geom.y, "s", color="#ff0000",
                markersize=12, markeredgecolor="white",
                markeredgewidth=2, zorder=7, label="Goal")

        subtitle = f"Dijkstra + {len(result.get('ppo_samples', []))} PPO successes"
        ax.set_title(f"{result['label']}\n{subtitle}", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        _style_map_axis(ax, clip_plot, bounds, mid_lat)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax_plot))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Cost", fontsize=11)
    fig.suptitle(
        f"Dijkstra and PPO route samples - {area_label}",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    plt.show()


# porovnava jednu dijkstra trasu a jednu ppo trasu v oddelenych panelech
def plot_ppo_vs_dijkstra(
    dijkstra_results,
    ppo_results,
    area_label,
    start_lonlat,
    goal_lonlat,
):
    """Plot Dijkstra routes vs PPO routes side by side for each scenario."""
    n_scenarios = len(dijkstra_results)

    for i in range(n_scenarios):
        d_res = dijkstra_results[i]
        p_res = ppo_results[i]
        label = d_res["label"]

        plot_frames, cmap, vmin, vmax_plot, vmax = _shared_cost_scale([d_res])
        clip_plot = d_res["area"]["gdf_sel"].geometry.iloc[0]
        bounds = clip_plot.bounds
        mid_lat = (bounds[1] + bounds[3]) / 2

        fig, axes, cax = _make_comparison_figure(2)

        for ax, res, title_prefix, color in zip(
            axes, [d_res, p_res], ["Dijkstra", "PPO"], ["#00ffff", "#ff00ff"]
        ):
            cm_plot = plot_frames[0]
            cm_plot.plot(
                column="cost", ax=ax, cmap=cmap,
                vmin=vmin, vmax=vmax_plot,
                edgecolor="none", linewidth=0, legend=False, zorder=2,
            )
            _plot_exclusion_layer(ax, d_res["exclusion_gdf"], clip_plot)

            if res["path"] and res["total_cost"] < float("inf"):
                path_x = res["coords"][res["path"], 0]
                path_y = res["coords"][res["path"], 1]
                path_line = LineString(zip(path_x, path_y))
                path_gdf = gpd.GeoDataFrame(
                    geometry=[path_line], crs=d_res["crs"]
                ).to_crs(4326)
                path_gdf.plot(ax=ax, color=color, linewidth=3, zorder=5)

                start_geom = cm_plot.iloc[res["start_idx"]].geometry.centroid
                goal_geom = cm_plot.iloc[res["goal_idx"]].geometry.centroid
                ax.plot(start_geom.x, start_geom.y, "o", color="#00ff00",
                        markersize=12, markeredgecolor="white",
                        markeredgewidth=2, zorder=6, label="Start")
                ax.plot(goal_geom.x, goal_geom.y, "s", color="#ff0000",
                        markersize=12, markeredgecolor="white",
                        markeredgewidth=2, zorder=6, label="Goal")
                ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

            title = f"{title_prefix} - {label}"
            if res["total_cost"] == float("inf"):
                title += "\nPath not found"
            else:
                title += (
                    f"\ncost={res['total_cost']:.1f} | {len(res['path'])} cells | "
                    f"~{res['distance_m']:,.0f} m | jumps {res['jumps']}x | "
                    f"turns {res['turns']}"
                )
            ax.set_title(title, fontsize=12, fontweight="bold")
            _style_map_axis(ax, clip_plot, bounds, mid_lat)

        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax_plot)
        )
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Cost", fontsize=11)
        fig.suptitle(
            f"Dijkstra vs PPO - {area_label} - {label}",
            fontsize=13, fontweight="bold", y=0.985,
        )
        plt.show()


# tiskne jednoduche textove srovnani, ktere se hodi do notebooku i logu
def print_comparison_table(dijkstra_results, ppo_results):
    """Print a markdown-style comparison table."""
    print(f"\n{'Label':<20} | {'Method':<8} | {'Cost':>10} | {'Cells':>6} | "
          f"{'Dist (m)':>10} | {'Jumps':>5} | {'Turns':>5}")
    print("-" * 80)
    for d_res, p_res in zip(dijkstra_results, ppo_results):
        label = d_res["label"]
        if d_res["total_cost"] < float("inf"):
            print(f"{label:<20} | {'Dijkstra':<8} | {d_res['total_cost']:>10.1f} | "
                  f"{len(d_res['path']):>6} | {d_res['distance_m']:>10,.0f} | "
                  f"{d_res['jumps']:>5} | {d_res['turns']:>5}")
        else:
            print(f"{label:<20} | {'Dijkstra':<8} | {'inf':>10} | {'--':>6} | "
                  f"{'--':>10} | {'--':>5} | {'--':>5}")

        if p_res["total_cost"] < float("inf"):
            print(f"{'':<20} | {'PPO':<8} | {p_res['total_cost']:>10.1f} | "
                  f"{len(p_res['path']):>6} | {p_res['distance_m']:>10,.0f} | "
                  f"{p_res['jumps']:>5} | {p_res['turns']:>5}")
        else:
            print(f"{'':<20} | {'PPO':<8} | {'FAILED':>10} | {'--':>6} | "
                  f"{'--':>10} | {'--':>5} | {'--':>5}")
