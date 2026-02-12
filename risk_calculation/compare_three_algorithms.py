from __future__ import annotations

import argparse
import io
import os
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Adult soccer comparison settings (aligned with sma_kids.py example)
SPORT = "soccer"
TG_DELTA = 5.0
HEIGHT = 1.8
WEIGHT = 75.0
T_CORE_EXTREME = 40.0
SWEAT_LOSS_G = 850.0
V_AIR = 1.0
T_RANGE = range(26, 45)  # 26..44
RH_RANGE = range(0, 101, 5)  # 0..100, step 5
DEFAULT_OUTPUT_DIR = "paper-olympics-heat-stress-risk/figures"


def _import_modules(repo_root: Path):
    """Import target modules from sibling projects with explicit paths."""
    import importlib.util
    import sys

    pythermalcomfort_root = repo_root / "pythermalcomfort"
    paper_project_root = repo_root / "paper-olympics-heat-stress-risk"
    if str(pythermalcomfort_root) not in sys.path:
        sys.path.insert(0, str(pythermalcomfort_root))
    if str(paper_project_root) not in sys.path:
        sys.path.insert(0, str(paper_project_root))

    from pythermalcomfort.models.sports_heat_stress_risk import (  # noqa: PLC0415
        Sports,
        _calc_risk_single_value,
    )
    from pythermalcomfort.utilities import mean_radiant_tmp  # noqa: PLC0415

    def load_module(name: str, file_path: Path, cwd: Path):
        prev_cwd = Path.cwd()
        os.chdir(cwd)
        try:
            spec = importlib.util.spec_from_file_location(name, str(file_path))
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)
            return module
        finally:
            os.chdir(prev_cwd)

    sma_kids_mod = load_module(
        "sma_kids_mod",
        repo_root / "tool-sma-extreme-heat-policy" / "sma_kids" / "sma_kids.py",
        repo_root / "tool-sma-extreme-heat-policy",
    )
    paper_mod = load_module(
        "paper_new_risk_eq_mod",
        repo_root
        / "paper-olympics-heat-stress-risk"
        / "risk_calculation"
        / "new_risk_eq_v2.py",
        repo_root / "paper-olympics-heat-stress-risk",
    )

    return (
        Sports,
        _calc_risk_single_value,
        mean_radiant_tmp,
        sma_kids_mod,
        paper_mod,
    )


def _pythermalcomfort_floor(x: float) -> int:
    # pythermalcomfort_floor means floor(pythermalcomfort_float):
    # convert continuous pythermalcomfort risk to a lower integer level.
    return int(np.floor(float(x)))


def run_comparison(repo_root: Path):
    (
        Sports,
        _calc_risk_single_value,
        mean_radiant_tmp,
        sma_kids_mod,
        paper_mod,
    ) = _import_modules(repo_root)

    sport_obj = Sports.SOCCER

    rows = []
    for tdb in T_RANGE:
        for rh in RH_RANGE:
            tg = tdb + TG_DELTA
            tr = float(mean_radiant_tmp(tdb=tdb, tg=tg, v=V_AIR))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                py_risk, _, _, _, _ = _calc_risk_single_value(
                    tdb=tdb,
                    tr=tr,
                    rh=rh,
                    vr=V_AIR,
                    sport=sport_obj,
                )

            with redirect_stdout(io.StringIO()):
                kids_risk = sma_kids_mod.get_sports_heat_stress_curves(
                    tdb=tdb,
                    tg=tg,
                    rh=rh,
                    v=V_AIR,
                    sport_id=SPORT,
                    sweat_loss_g=SWEAT_LOSS_G,
                    height=HEIGHT,
                    weight=WEIGHT,
                    t_cr_extreme=T_CORE_EXTREME,
                )
                paper_risk = paper_mod.get_sports_heat_stress_curves(
                    tdb=tdb,
                    tg=tg,
                    rh=rh,
                    v=V_AIR,
                    sport_id=SPORT,
                )

            rows.append(
                {
                    "tdb": tdb,
                    "rh": rh,
                    "risk_tool_sma_extreme_heat_policy": float(kids_risk),
                    "risk_pythermalcomfort_float": float(py_risk),
                    "risk_pythermalcomfort_floor": float(_pythermalcomfort_floor(py_risk)),
                    "risk_paper_olympics_heat_stress_risk": float(paper_risk),
                }
            )

    df = pd.DataFrame(rows)
    df["diff_tool_sma_minus_pythermalcomfort_floor"] = (
        df["risk_tool_sma_extreme_heat_policy"] - df["risk_pythermalcomfort_floor"]
    )
    df["diff_paper_minus_pythermalcomfort_floor"] = (
        df["risk_paper_olympics_heat_stress_risk"] - df["risk_pythermalcomfort_floor"]
    )
    df["diff_tool_sma_minus_paper"] = (
        df["risk_tool_sma_extreme_heat_policy"]
        - df["risk_paper_olympics_heat_stress_risk"]
    )
    df["diff_pythermalcomfort_float_minus_tool_sma"] = (
        df["risk_pythermalcomfort_float"] - df["risk_tool_sma_extreme_heat_policy"]
    )
    df["diff_pythermalcomfort_float_minus_paper"] = (
        df["risk_pythermalcomfort_float"]
        - df["risk_paper_olympics_heat_stress_risk"]
    )

    return df


def _pivot(df: pd.DataFrame, value_col: str):
    hm = df.pivot(index="rh", columns="tdb", values=value_col)
    hm = hm.sort_index(ascending=False)
    return hm


def save_outputs(df: pd.DataFrame, output_dir: Path, prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}_grid.csv"
    df.to_csv(csv_path, index=False)

    fig1, axs1 = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    sns.heatmap(
        _pivot(df, "risk_tool_sma_extreme_heat_policy"),
        ax=axs1[0, 0],
        cmap="viridis",
        vmin=0,
        vmax=3,
        cbar_kws={"label": "Risk"},
    )
    axs1[0, 0].set_title(
        "1) tool-sma-extreme-heat-policy (sma_kids.get_sports_heat_stress_curves)"
    )
    sns.heatmap(
        _pivot(df, "risk_pythermalcomfort_float"),
        ax=axs1[0, 1],
        cmap="viridis",
        vmin=0,
        vmax=3,
        cbar_kws={"label": "Risk"},
    )
    axs1[0, 1].set_title("2) pythermalcomfort (_calc_risk_single_value, float)")
    sns.heatmap(
        _pivot(df, "risk_pythermalcomfort_floor"),
        ax=axs1[1, 0],
        cmap="viridis",
        vmin=0,
        vmax=3,
        cbar_kws={"label": "Risk"},
    )
    axs1[1, 0].set_title("2b) pythermalcomfort_floor = floor(pythermalcomfort_float)")
    sns.heatmap(
        _pivot(df, "risk_paper_olympics_heat_stress_risk"),
        ax=axs1[1, 1],
        cmap="viridis",
        vmin=0,
        vmax=3,
        cbar_kws={"label": "Risk"},
    )
    axs1[1, 1].set_title(
        "3) paper-olympics-heat-stress-risk (new_risk_eq_v2.get_sports_heat_stress_curves)"
    )
    for ax in axs1.flat:
        ax.set_xlabel("tdb (degC)")
        ax.set_ylabel("rh (%)")
    fig1.tight_layout()
    fig1_path = output_dir / f"{prefix}_risk_heatmaps.png"
    fig1.savefig(fig1_path, dpi=220, bbox_inches="tight")
    plt.close(fig1)

    fig2, axs2 = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    diff_specs = [
        (
            "diff_tool_sma_minus_pythermalcomfort_floor",
            "tool-sma-extreme-heat-policy - pythermalcomfort_floor",
        ),
        (
            "diff_paper_minus_pythermalcomfort_floor",
            "paper-olympics-heat-stress-risk - pythermalcomfort_floor",
        ),
        ("diff_tool_sma_minus_paper", "tool-sma-extreme-heat-policy - paper-olympics"),
        (
            "diff_pythermalcomfort_float_minus_tool_sma",
            "pythermalcomfort_float - tool-sma-extreme-heat-policy",
        ),
        (
            "diff_pythermalcomfort_float_minus_paper",
            "pythermalcomfort_float - paper-olympics-heat-stress-risk",
        ),
    ]
    for idx, (col, title) in enumerate(diff_specs):
        r, c = divmod(idx, 3)
        vmax = max(1.0, float(np.nanmax(np.abs(df[col].values))))
        sns.heatmap(
            _pivot(df, col),
            ax=axs2[r, c],
            cmap="coolwarm",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            cbar_kws={"label": "Delta"},
        )
        axs2[r, c].set_title(title)
        axs2[r, c].set_xlabel("tdb (degC)")
        axs2[r, c].set_ylabel("rh (%)")
    axs2[1, 2].axis("off")
    fig2.tight_layout()
    fig2_path = output_dir / f"{prefix}_pairwise_diffs.png"
    fig2.savefig(fig2_path, dpi=220, bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "grid_points": int(len(df)),
        "tool_sma_vs_paper_equal_count": int(
            (
                df["risk_tool_sma_extreme_heat_policy"].values
                == df["risk_paper_olympics_heat_stress_risk"].values
            ).sum()
        ),
        "tool_sma_vs_paper_diff_count": int(
            (
                df["risk_tool_sma_extreme_heat_policy"].values
                != df["risk_paper_olympics_heat_stress_risk"].values
            ).sum()
        ),
        "tool_sma_vs_pythermalcomfort_floor_diff_count": int(
            (
                df["risk_tool_sma_extreme_heat_policy"].values
                != df["risk_pythermalcomfort_floor"].values
            ).sum()
        ),
        "paper_vs_pythermalcomfort_floor_diff_count": int(
            (
                df["risk_paper_olympics_heat_stress_risk"].values
                != df["risk_pythermalcomfort_floor"].values
            ).sum()
        ),
        "mean_abs_diff_pythermalcomfort_float_vs_tool_sma": float(
            np.mean(np.abs(df["diff_pythermalcomfort_float_minus_tool_sma"].values))
        ),
        "mean_abs_diff_pythermalcomfort_float_vs_paper": float(
            np.mean(np.abs(df["diff_pythermalcomfort_float_minus_paper"].values))
        ),
        "risk_heatmaps": str(fig1_path),
        "pairwise_diffs": str(fig2_path),
        "grid_csv": str(csv_path),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Three-way adult soccer comparison with fixed parameters aligned with "
            "sma_kids.py main example."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--prefix",
        default="strict_three_way_soccer_adult_fixed",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    df = run_comparison(repo_root=repo_root)
    summary = save_outputs(
        df=df,
        output_dir=repo_root / args.output_dir,
        prefix=args.prefix,
    )
    print("Comparison finished.")
    print(
        "fixed_params: "
        f"sport={SPORT}, tg_delta={TG_DELTA}, v={V_AIR}, "
        f"height={HEIGHT}, weight={WEIGHT}, "
        f"t_core_extreme={T_CORE_EXTREME}, sweat_loss_g={SWEAT_LOSS_G}, "
        f"t_range=[{T_RANGE.start},{T_RANGE.stop}), "
        f"rh_range=[{RH_RANGE.start},{RH_RANGE.stop}) step {RH_RANGE.step}"
    )
    print(
        "note: pythermalcomfort_floor = floor(pythermalcomfort_float), "
        "i.e., downward discretization to integer risk level."
    )
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
