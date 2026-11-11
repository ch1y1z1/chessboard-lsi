"""Independent saved-map checks and compact report assets; no inversion."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch
import numpy as np


def read_csv(path):
    return list(csv.DictReader(Path(path).open()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert head == a.expected_commit
    assert not subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    a.output.mkdir(parents=True, exist_ok=False)
    maps = np.load(a.analysis / "representative_maps.npz")
    curves = np.load(a.analysis / "unbinned_curves.npz")
    regions = read_csv(a.analysis / "regions.csv")
    stats = read_csv(a.analysis / "primary_statistics.csv")
    geometry = read_csv(a.analysis / "mask_geometry.csv")
    summary = json.loads((a.results / "summary.json").read_text())["results"]
    checks = 0

    def close(actual, expected, atol=1e-15):
        nonlocal checks
        assert np.allclose(actual, expected, rtol=1e-10, atol=atol), (actual, expected)
        checks += 1

    def metric(error, mask):
        square = error**2
        return {
            "region_rms": np.sqrt(square[mask].mean()),
            "interior_rms": np.sqrt(square[~mask].mean()),
            "squared_error_fraction": square[mask].sum() / square.sum(),
        }

    keys = ("single_ft", "single_lm", "multi64_ft", "multi64_lm")
    relative = []
    for na in (.02, .066, .11, .15):
        prefix = f"na{na:.3f}"
        rho = maps[prefix + "_rho"]
        n = len(rho)
        for key in keys:
            for condition in ("clean", "40db_seed42"):
                error = maps[f"{prefix}_{key}_{'seed42' if condition.startswith('40db') else 'clean'}"]
                original = next(r for r in summary if r["na_out"] == na
                                and f'{r["source_kind"]}_{r["method"]}' == key
                                and r["condition"] == condition)
                close(np.sqrt(np.mean(error**2)), original["raw_rms_waves"])
                for row in regions:
                    if float(row["na_out"]) != na or row["pipeline"] != key or row["condition"] != condition:
                        continue
                    fraction = float(row["requested_fraction"])
                    index = max(0, int(np.ceil(n * fraction)) - 1)
                    outer = rho >= np.sort(rho)[::-1][index]
                    count = int(outer.sum())
                    name = row["mask"]
                    if name == "geometric_outer":
                        selected = outer
                    elif name == "geometric_inner":
                        selected = rho <= np.sort(rho)[count - 1]
                    else:
                        source = error
                        if name.startswith("hot_clean"):
                            ref = "single_ft" if name.endswith("single_ft") else "multi64_ft"
                            source = maps[f"{prefix}_{ref}_clean"]
                        rank = np.lexsort((np.arange(n), -source**2))
                        selected = np.zeros(n, dtype=bool)
                        selected[rank[:count]] = True
                    close(selected.sum(), int(row["pixels"]))
                    for name, value in metric(error, selected).items():
                        close(value, float(row[name]), atol=1e-30)
            cdf = curves[f"{prefix}_{key}_cdf"]
            area = curves[prefix + "_outer_area"]
            cum = curves[f"{prefix}_{key}_rms"]
            close(cdf[:, -1], np.ones(5))
            for seed_index, seed in enumerate(range(41, 46)):
                for fraction in (.01, .02, .05, .1, .15, .2, .3):
                    row = next(r for r in regions if float(r["na_out"]) == na
                               and r["pipeline"] == key and r["condition"] == f"40db_seed{seed}"
                               and float(r["requested_fraction"]) == fraction
                               and r["mask"] == "geometric_outer")
                    shell = np.flatnonzero(np.isclose(area, float(row["pixel_fraction"]), atol=1e-14))
                    assert len(shell) == 1
                    close(cdf[seed_index, shell[0]], float(row["squared_error_fraction"]))
                    close(cum[seed_index, -1], float(row["full_rms"]))
            row = next(r for r in stats if float(r["na_out"]) == na
                       and r["pipeline"] == key and r["mask"] == "geometric_outer")
            sample = [r for r in regions if float(r["na_out"]) == na and r["pipeline"] == key
                      and r["mask"] == "geometric_outer" and float(r["requested_fraction"]) == .1
                      and r["condition"].startswith("40db")]
            for name in ("region_rms", "interior_rms", "full_rms", "squared_error_fraction"):
                values = [float(r[name]) for r in sample]
                close(np.mean(values), float(row[name + "_mean"]))
                close(np.std(values, ddof=1), float(row[name + "_sd"]))
            for item in sample:
                f = float(item["pixel_fraction"])
                true_edge = float(item["truth_region_rms"])
                true_interior = np.sqrt((.05**2 - f * true_edge**2) / (1 - f))
                relative.append(dict(
                    na_out=na, pipeline=key, condition=item["condition"],
                    edge_relative_rms=float(item["region_relative_rms"]),
                    interior_relative_rms=float(item["interior_rms"]) / true_interior,
                    relative_edge_over_interior=float(item["region_relative_rms"])
                    / (float(item["interior_rms"]) / true_interior),
                ))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.facecolor": "#fcfaf5", "figure.facecolor": "#fcfaf5"})

    def finish(fig, name):
        fig.savefig(a.output / (name + ".png"), dpi=170, bbox_inches="tight")
        fig.savefig(a.output / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(10, 12), constrained_layout=True)
    abs_max = max(np.max(abs(maps[f"na{na:.3f}_{key}_seed42"]))
                  for na in (.02, .066, .11, .15) for key in ("single_ft", "single_lm"))
    palette = ListedColormap(["#e8e4da", "#2b6e99", "#db8645", "#343b33"])
    for row_index, na in enumerate((.02, .066, .11, .15)):
        prefix = f"na{na:.3f}"
        pupil = maps[prefix + "_mask"]
        for col, key in enumerate(("single_ft", "single_lm")):
            image = np.full(pupil.shape, np.nan)
            image[pupil] = abs(maps[f"{prefix}_{key}_seed42"])
            cmap = plt.colormaps["magma"].copy()
            cmap.set_bad("#cfccc4")
            im = axes[row_index, col].imshow(
                image, origin="lower", cmap=cmap,
                norm=LogNorm(vmin=1e-6, vmax=float(abs_max)),
            )
            axes[row_index, col].set_title(f"NA_out={na:.3f} / {'Single FT' if col == 0 else 'Single LM'}")
        region = np.full(pupil.shape, np.nan)
        region[pupil] = maps[prefix + "_outer10"].astype(int) + 2 * maps[prefix + "_hot10"]
        axes[row_index, 2].imshow(region, origin="lower", cmap=palette, vmin=-.5, vmax=3.5)
        axes[row_index, 2].set_title("Frozen clean FT top 10% vs edge 10%")
        for ax in axes[row_index]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes[:, :2].ravel().tolist(), shrink=.5, fraction=.025,
                 label="Absolute error (waves), shared log scale; seed42")
    fig.legend(handles=[Patch(color=palette(i), label=label) for i, label in enumerate(
        ("Neither", "Edge only", "Hot only", "Overlap"))],
        loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Same observation, same pupil, raw wavefront errors; masks selected without noisy results")
    finish(fig, "maps_and_frozen_masks")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {"single_ft": "#aa4c26", "multi64_ft": "#bb8e24",
              "single_lm": "#21628c", "multi64_lm": "#27836e"}
    for key in keys:
        group = [r for r in stats if r["pipeline"] == key and r["mask"] == "geometric_outer"]
        axes[0].plot([float(r["na_out"]) for r in group],
                     [100 * float(r["squared_error_fraction_mean"]) for r in group],
                     color=colors[key], marker="o", label=key)
        values = [np.mean([r["relative_edge_over_interior"] for r in relative
                           if r["na_out"] == na and r["pipeline"] == key])
                  for na in (.02, .066, .11, .15)]
        axes[1].plot([.02, .066, .11, .15], values, color=colors[key], marker="o", label=key)
    baseline = read_csv(a.analysis / "modal_baseline.csv")
    group = [r for r in baseline if float(r["requested_fraction"]) == .1]
    axes[0].plot([float(r["na_out"]) for r in group],
                 [100 * float(r["equal_modal_variance_share"]) for r in group],
                 "k--", label="Equal independent modal variance")
    axes[0].axhline(10, color="gray", ls=":", label="Uniform pixel MSE")
    axes[1].axhline(1, color="gray", ls=":")
    axes[0].set(xlabel="System NA_out", ylabel="Outer 10% squared-error share (%)")
    axes[1].set(xlabel="System NA_out", ylabel="Relative RMS: edge / interior")
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("Basis-only baseline is not a fitted noise model; relative RMS uses each region's truth RMS")
    finish(fig, "basis_and_relative_control")

    with (a.output / "relative_error_control.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(relative[0]))
        writer.writeheader()
        writer.writerows(relative)
    receipt = dict(
        passed=True, checks=checks, commit=head,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command=sys.argv, map_max_abs_error=float(abs_max),
        scope="Independent saved-map reductions, mask reconstruction, CDF cross-checks and sample statistics; not independent physics",
    )
    (a.output / "independent_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
