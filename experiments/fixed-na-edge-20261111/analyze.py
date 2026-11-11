"""Analyze saved pupil errors with predeclared pixel-area and frozen masks."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/na-talbot-balanced-20261111"))
from report import COLORS, LABELS, pupil_basis


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def top_mask(values, count):
    order = np.argsort(-np.asarray(values), kind="stable")
    mask = np.zeros(len(values), dtype=bool)
    mask[order[:count]] = True
    return mask


def outer_mask(radii, fraction):
    ordered = np.sort(radii)[::-1]
    cutoff = ordered[max(0, int(np.ceil(len(radii) * fraction)) - 1)]
    return radii >= cutoff


def describe(error, truth, mask):
    squared = error**2
    n = int(mask.sum())
    fraction = n / len(error)
    share = float(squared[mask].sum() / squared.sum())
    interior = rms(error[~mask])
    return dict(
        pixels=n,
        pixel_fraction=fraction,
        region_rms=rms(error[mask]),
        interior_rms=interior,
        region_over_interior=rms(error[mask]) / interior,
        squared_error_fraction=share,
        concentration_enrichment=share / fraction,
        truth_region_rms=rms(truth[mask]),
        region_relative_rms=rms(error[mask]) / rms(truth[mask]),
        trimmed_over_full=interior / rms(error),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == args.expected_commit
    assert not subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = read(protocol_path)
    source_path = args.results / "summary.json"
    source = read(source_path)["results"]
    assert len(source) == 112
    inputs = {str(source_path): digest(source_path)}
    checks = []
    records, concentrations, cumulative, binned = [], [], [], []
    decomposition, baseline, holdouts, geometry, endpoints = [], [], [], [], []
    maps = {}
    curves = {}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "#fcfaf5", "axes.facecolor": "#fcfaf5",
    })

    def figure_done(fig, name):
        fig.savefig(args.output / f"{name}.png", dpi=170, bbox_inches="tight")
        fig.savefig(args.output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig_curve, ax_curve = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig_cdf, ax_cdf = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig_trim, ax_trim = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig_bin, ax_bin = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for panel, na in enumerate(protocol["na_out"]):
        cfg_path = args.results / f"na{na:.3f}_single/config.json"
        cfg = read(cfg_path)
        inputs[str(cfg_path)] = digest(cfg_path)
        Z, pupil, rho = pupil_basis(cfg)
        zp, rp = Z[:, pupil], rho[pupil]
        rows, cols = np.nonzero(pupil)
        angle = np.arctan2(rows - (pupil.shape[0] - 1) / 2,
                           cols - (pupil.shape[1] - 1) / 2)
        n = len(rp)
        order = np.argsort(rp, kind="stable")
        sorted_r = rp[order]
        shell_ends = np.r_[np.flatnonzero(np.diff(sorted_r)) + 1, n]
        shell_starts = np.r_[0, shell_ends[:-1]]
        shell_r = sorted_r[shell_ends - 1]
        outer_order = order[::-1]
        descending_r = rp[outer_order]
        outer_ends = np.r_[np.flatnonzero(np.diff(descending_r)) + 1, n]
        selected = [row for row in source if row["na_out"] == na]
        error, truths = {}, {}
        for row in selected:
            key = f'{row["source_kind"]}_{row["method"]}'
            condition = row["condition"]
            error[key, condition] = (
                np.asarray(row["coefficients"]) - row["truth"]
            ) @ zp
            truths[key, condition] = np.asarray(row["truth"]) @ zp
            assert abs(rms(error[key, condition]) - row["raw_rms_waves"]) < 1e-11
            assert n == row["evaluation_pixels"]
            assert abs(rms(truths[key, condition]) - 0.05) < 1e-12
            checks.append("saved full RMS, truth RMS, and pupil count")
        truth = truths["single_ft", "clean"]
        geom = {
            p: outer_mask(rp, p) for p in protocol["pixel_fractions"]
        }
        frozen = {
            (reference, p): top_mask(error[reference, "clean"] ** 2, int(mask.sum()))
            for p, mask in geom.items()
            for reference in ("single_ft", "multi64_ft")
        }
        for p, mask in geom.items():
            count = int(mask.sum())
            inner_r = np.sort(rp)[count - 1]
            inner = rp <= inner_r
            for reference in ("single_ft", "multi64_ft"):
                hot = frozen[reference, p]
                assert hot.sum() == count
                geometry.append(dict(
                    na_out=na, requested_fraction=p, actual_fraction=count / n,
                    evaluation_pixels=n, rho_low=float(rp[mask].min()),
                    na_low=float(na * rp[mask].min()), reference=reference,
                    hot_edge_overlap=float((hot & mask).sum() / count),
                    hot_edge_jaccard=float((hot & mask).sum() / (hot | mask).sum()),
                    hot_outer20_fraction=float((hot & geom[0.2]).sum() / count),
                    hot_median_rho=float(np.median(rp[hot])),
                    inner_actual_fraction=float(np.mean(inner)),
                ))
            for (key, condition), err in error.items():
                masks = {
                    "geometric_outer": mask, "geometric_inner": inner,
                    "hot_clean_single_ft": frozen["single_ft", p],
                    "hot_clean_multi64_ft": frozen["multi64_ft", p],
                    "oracle_self": top_mask(err**2, count),
                }
                for name, region in masks.items():
                    record = dict(
                        na_out=na, pipeline=key, condition=condition,
                        requested_fraction=p, mask=name, full_rms=rms(err),
                        **describe(err, truth, region),
                    )
                    identity = (record["pixel_fraction"] * record["region_rms"]**2
                                + (1 - record["pixel_fraction"])
                                * record["interior_rms"]**2)
                    assert np.isclose(identity, rms(err)**2, rtol=1e-12, atol=1e-35)
                    records.append(record)
                    checks.append("exact region/interior MSE partition")
                ideal = records[-1]["squared_error_fraction"]
                assert ideal >= records[-5]["squared_error_fraction"] - 1e-14
        modal_power = np.sum(zp**2, axis=0)
        for p, mask in geom.items():
            baseline.append(dict(
                na_out=na, requested_fraction=p,
                actual_fraction=float(np.mean(mask)),
                equal_pixel_error_share=float(np.mean(mask)),
                equal_modal_variance_share=float(
                    modal_power[mask].sum() / modal_power.sum()
                ),
            ))
        for key in LABELS:
            sample = np.stack([
                error[key, f"40db_seed{seed}"] for seed in protocol["seeds"]
            ])
            square = sample**2
            cs = np.cumsum(square[:, order], axis=1)[:, shell_ends - 1]
            curve_rms = np.sqrt(cs / shell_ends)
            cdf = np.cumsum(square[:, outer_order], axis=1)[:, outer_ends - 1]
            cdf /= square.sum(axis=1)[:, None]
            assert np.all(np.diff(cdf, axis=1) >= -1e-14)
            assert np.allclose(cdf[:, -1], 1)
            assert np.allclose(curve_rms[:, -1], np.sqrt(square.mean(axis=1)))
            checks.extend(["unbinned CDF monotonic/endpoint", "cumulative full endpoint"])
            curves[f"na{na:.3f}_{key}_rms"] = curve_rms
            curves[f"na{na:.3f}_{key}_cdf"] = cdf
            curves[f"na{na:.3f}_rho"] = shell_r
            curves[f"na{na:.3f}_outer_area"] = outer_ends / n
            curves[f"na{na:.3f}_inner_area"] = shell_ends / n
            ax = ax_curve.flat[panel]
            mean, sd = curve_rms.mean(axis=0), curve_rms.std(axis=0, ddof=1)
            ax.plot(shell_r * na, mean * 1e4, color=COLORS[key], label=LABELS[key])
            ax.fill_between(shell_r * na, (mean - sd) * 1e4, (mean + sd) * 1e4,
                            color=COLORS[key], alpha=0.13, linewidth=0)
            ax = ax_cdf.flat[panel]
            ax.plot(outer_ends / n * 100, cdf.mean(axis=0) * 100,
                    color=COLORS[key], label=LABELS[key])
            ax.fill_between(outer_ends / n * 100,
                            (cdf.mean(axis=0) - cdf.std(axis=0, ddof=1)) * 100,
                            (cdf.mean(axis=0) + cdf.std(axis=0, ddof=1)) * 100,
                            color=COLORS[key], alpha=0.12, linewidth=0)
            keep = outer_ends < n
            remainder_sse = square.sum(axis=1)[:, None] * (1 - cdf[:, keep])
            trim = np.sqrt(np.maximum(0, remainder_sse) / (n - outer_ends[keep]))
            trim_relative = trim / np.sqrt(square.mean(axis=1))[:, None]
            ax_trim.flat[panel].plot(
                outer_ends[keep] / n * 100, trim_relative.mean(axis=0),
                color=COLORS[key], label=LABELS[key],
            )
            for i, seed in enumerate(protocol["seeds"]):
                p80 = float(outer_ends[np.searchsorted(cdf[i], .8)] / n)
                endpoints.append(dict(
                    na_out=na, pipeline=key, seed=seed,
                    full_rms=rms(sample[i]), outer_area_for_80pct_sse=p80,
                ))
                for j in np.unique(np.linspace(0, len(shell_r) - 1, 201).astype(int)):
                    cumulative.append(dict(
                        na_out=na, pipeline=key, seed=seed,
                        rho=float(shell_r[j]), na_eval=float(shell_r[j] * na),
                        included_pixels=int(shell_ends[j]),
                        rms=float(curve_rms[i, j]),
                    ))
                for j in np.unique(np.linspace(0, len(outer_ends) - 1, 201).astype(int)):
                    concentrations.append(dict(
                        na_out=na, pipeline=key, seed=seed,
                        outer_pixel_fraction=float(outer_ends[j] / n),
                        squared_error_fraction=float(cdf[i, j]),
                    ))
            mean_error = sample.mean(axis=0)
            fluct = sample - mean_error
            clean = error[key, "clean"]
            perturbation = sample - clean
            for name, region in (
                ("full", np.ones(n, bool)), ("outer10", geom[.1]),
                ("interior90", ~geom[.1]),
            ):
                mse = float(square[:, region].mean())
                bias = float(np.mean(mean_error[region]**2))
                variance = float(np.mean(fluct[:, region]**2))
                clean_mse = float(np.mean(clean[region]**2))
                perturb_mse = float(np.mean(perturbation[:, region]**2))
                cross = float(2 * np.mean(clean[region] * perturbation[:, region]))
                assert np.isclose(mse, bias + variance, rtol=1e-11, atol=1e-30)
                assert np.isclose(mse, clean_mse + perturb_mse + cross,
                                  rtol=1e-11, atol=1e-30)
                decomposition.append(dict(
                    na_out=na, pipeline=key, region=name, mean_mse=mse,
                    mean_map_squared=bias, finite_seed_variance=variance,
                    variance_fraction=variance / mse, clean_mse=clean_mse,
                    perturbation_mse=perturb_mse, cross_term=cross,
                    clean_over_noisy_mse=clean_mse / mse,
                ))
                checks.append("exact mean/variance and clean/perturbation identities")
            for coordinate in ("equal_radius", "equal_area"):
                values = rp if coordinate == "equal_radius" else (
                    (rp**2 - (cfg["NA_in"] / na)**2)
                    / (1 - (cfg["NA_in"] / na)**2)
                )
                for bins in protocol["bin_counts"]:
                    edges = np.linspace(0, 1, bins + 1)
                    group = np.clip(np.searchsorted(edges, values, side="right") - 1,
                                    0, bins - 1)
                    counts = np.bincount(group, minlength=bins)
                    valid = counts > 0
                    sse = np.stack([
                        np.bincount(group, weights=e**2, minlength=bins)
                        for e in sample
                    ])
                    assert counts.sum() == n
                    assert np.allclose(sse.sum(axis=1), square.sum(axis=1), rtol=1e-12)
                    checks.append("disjoint bin coverage and SSE conservation")
                    for b in np.flatnonzero(valid):
                        rv = np.sqrt(sse[:, b] / counts[b])
                        binned.append(dict(
                            na_out=na, pipeline=key, coordinate=coordinate,
                            bins=bins, bin_index=int(b), low=float(edges[b]),
                            high=float(edges[b + 1]), pixels=int(counts[b]),
                            rms_mean=float(rv.mean()), rms_sd=float(rv.std(ddof=1)),
                        ))
                    if key == "single_ft" and coordinate == "equal_area":
                        ax_bin.flat[panel].plot(
                            (edges[:-1] + edges[1:])[valid] / 2,
                            np.sqrt(sse[:, valid] / counts[valid]).mean(axis=0) * 1e4,
                            label=f"{bins} equal-area bins", alpha=.8,
                        )
            maps[f"na{na:.3f}_{key}_seed42"] = error[key, "40db_seed42"]
            maps[f"na{na:.3f}_{key}_clean"] = clean
        maps[f"na{na:.3f}_mask"] = pupil
        maps[f"na{na:.3f}_rho"] = rp
        maps[f"na{na:.3f}_angle"] = angle
        maps[f"na{na:.3f}_truth"] = truth
        maps[f"na{na:.3f}_outer10"] = geom[.1]
        maps[f"na{na:.3f}_hot10"] = frozen["single_ft", .1]
        for kind in ("single", "multi64"):
            for seed in (1807, 1808, 1809):
                path = args.results / f"na{na:.3f}_{kind}/holdout_{seed}.json"
                item = read(path)
                inputs[str(path)] = digest(path)
                err = (np.asarray(item["coefficients"]) - item["truth"]) @ zp
                ht = np.asarray(item["truth"]) @ zp
                assert abs(rms(err) - item["raw_rms_waves"]) < 1e-11
                assert abs(rms(ht) - .05) < 1e-12
                holdouts.append(dict(
                    na_out=na, pipeline=f"{kind}_ft", truth_seed=seed,
                    full_rms=rms(err), **describe(err, ht, geom[.1]),
                ))
                checks.append("held-out truth and full RMS")
        for axes in (ax_curve, ax_cdf, ax_trim, ax_bin):
            axes.flat[panel].set_title(f"Fixed system NA_out = {na:.3f}")
            axes.flat[panel].grid(alpha=.16)
        ax_curve.flat[panel].set(xlabel="Evaluation cutoff NA",
                                 ylabel="Raw RMS (1e-4 waves)")
        ax_curve.flat[panel].set_ylim(bottom=0)
        ax_cdf.flat[panel].plot([0, 30], [0, 30], "k:", lw=1, label="Uniform error")
        ax_cdf.flat[panel].axvline(10, color="gray", ls=":", lw=1)
        ax_cdf.flat[panel].set(xlim=(0, 30), ylim=(0, 101),
                               xlabel="Outermost valid pixels (%)",
                               ylabel="Total squared error captured (%)")
        ax_trim.flat[panel].axhline(1, color="gray", ls=":", lw=1)
        ax_trim.flat[panel].set(xlim=(0, 30), ylim=(0, 1.2),
                                xlabel="Outermost pixels excluded (%)",
                                ylabel="Remaining RMS / full-pupil RMS")
        ax_bin.flat[panel].set(xlabel="Enclosed pupil area fraction",
                               ylabel="Single FT RMS (1e-4 waves)")
        save(args.output / "status.json", dict(
            state="running", completed_systems=panel + 1, total_systems=4,
            generated_utc=datetime.now(timezone.utc).isoformat(),
        ))

    for fig, axes, name, title in (
        (fig_curve, ax_curve, "unbinned_cumulative", "Every distinct native radius; mean +/- sample SD, 5 seeds"),
        (fig_cdf, ax_cdf, "edge_concentration", "No radial bins; geometric outer-pixel concentration"),
        (fig_trim, ax_trim, "edge_erosion", "No refit: remove evaluation pixels only"),
        (fig_bin, ax_bin, "bin_sensitivity", "Changing bins changes local curves; primary metrics use fixed pixel area"),
    ):
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(title, fontsize=12)
        figure_done(fig, name)

    primary = [
        row for row in records
        if row["requested_fraction"] == .1
        and row["condition"].startswith("40db_seed")
    ]
    stats = []
    metrics = (
        "full_rms", "region_rms", "interior_rms", "region_over_interior",
        "squared_error_fraction", "concentration_enrichment",
        "region_relative_rms", "trimmed_over_full",
    )
    for na in protocol["na_out"]:
        for key in LABELS:
            for mask in sorted({row["mask"] for row in primary}):
                group = [row for row in primary
                         if row["na_out"] == na and row["pipeline"] == key
                         and row["mask"] == mask]
                assert len(group) == 5
                record = dict(na_out=na, pipeline=key, mask=mask,
                              pixel_fraction=group[0]["pixel_fraction"], seeds=5)
                for metric in metrics:
                    values = np.asarray([row[metric] for row in group])
                    record[metric + "_mean"] = float(values.mean())
                    record[metric + "_sd"] = float(values.std(ddof=1))
                stats.append(record)

    # Paired across shared simulated seed labels, conditional on this system family.
    comparisons = []
    for low, high in ((.02, .15), (.11, .15)):
        for key in LABELS:
            for metric in ("region_rms", "squared_error_fraction",
                           "region_over_interior", "trimmed_over_full", "full_rms"):
                groups = []
                for na in (low, high):
                    group = sorted(
                        [row for row in primary if row["na_out"] == na
                         and row["pipeline"] == key and row["mask"] == "geometric_outer"],
                        key=lambda row: row["condition"],
                    )
                    groups.append(np.asarray([row[metric] for row in group]))
                difference = groups[1] - groups[0]
                comparisons.append(dict(
                    na_low=low, na_high=high, pipeline=key, metric=metric,
                    low_mean=float(groups[0].mean()), high_mean=float(groups[1].mean()),
                    paired_difference_mean=float(difference.mean()),
                    paired_difference_sd=float(difference.std(ddof=1)),
                    relative_change_of_mean=float(groups[1].mean() / groups[0].mean() - 1),
                ))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for key in LABELS:
        group = [r for r in stats if r["pipeline"] == key
                 and r["mask"] == "geometric_outer"]
        for ax, metric, factor, label in (
            (axes[0], "squared_error_fraction", 100, "Outer 10% squared-error share (%)"),
            (axes[1], "region_rms", 1e4, "Outer 10% RMS (1e-4 waves)"),
            (axes[2], "region_over_interior", 1, "Edge / interior RMS"),
        ):
            ax.errorbar(
                [r["na_out"] for r in group],
                [r[metric + "_mean"] * factor for r in group],
                yerr=[r[metric + "_sd"] * factor for r in group],
                color=COLORS[key], marker="o", capsize=3, label=LABELS[key],
            )
            ax.set(xlabel="System NA_out", ylabel=label)
            ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle("Same outer pixel fraction at every system; mean +/- sample SD (5 noise seeds)")
    figure_done(fig, "system_na_effect")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for col, na in enumerate(protocol["na_out"]):
        mask = maps[f"na{na:.3f}_mask"]
        outer = maps[f"na{na:.3f}_outer10"]
        hot = maps[f"na{na:.3f}_hot10"]
        err = abs(maps[f"na{na:.3f}_single_ft_seed42"])
        image = np.full(mask.shape, np.nan)
        image[mask] = err
        cmap = plt.colormaps["magma"].copy()
        cmap.set_bad("#d8d5ce")
        im = axes[0, col].imshow(image, origin="lower", cmap=cmap,
                                 norm=LogNorm(vmin=1e-6, vmax=3e-3))
        axes[0, col].set_title(f"NA_out={na:.3f}, Single FT seed42")
        regions = np.full(mask.shape, np.nan)
        regions[mask] = outer.astype(float) + 2 * hot
        axes[1, col].imshow(regions, origin="lower", cmap="viridis", vmin=0, vmax=3)
        axes[1, col].set_title("Geometry 10% + frozen clean-hot 10%")
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), fraction=.025, label="Absolute raw error (waves), shared log scale")
    fig.suptitle("Lower masks: purple=neither; blue=edge only; green=hot only; yellow=overlap")
    figure_done(fig, "error_and_masks")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    styles = {"geometric_outer": "-", "hot_clean_single_ft": "--", "oracle_self": ":"}
    for panel, na in enumerate(protocol["na_out"]):
        for key in ("single_ft", "single_lm"):
            for mask_name, style in styles.items():
                group = [r for r in records if r["na_out"] == na
                         and r["pipeline"] == key and r["mask"] == mask_name
                         and r["condition"].startswith("40db_seed")]
                xs, ys = [], []
                for p in protocol["pixel_fractions"]:
                    sample = [r for r in group if r["requested_fraction"] == p]
                    xs.append(100 * sample[0]["pixel_fraction"])
                    ys.append(100 * np.mean([r["squared_error_fraction"] for r in sample]))
                axes.flat[panel].plot(xs, ys, color=COLORS[key], ls=style, marker=".",
                                     label=f"{LABELS[key]} / {mask_name}")
        axes.flat[panel].set(
            title=f"Fixed NA_out={na:.3f}", xlabel="Selected valid pixels (%)",
            ylabel="Total squared error captured (%)", ylim=(0, 101),
        )
        axes.flat[panel].grid(alpha=.15)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Frozen clean FT mask vs geometric edge; self-ranked noisy masks are descriptive")
    figure_done(fig, "frozen_hot_concentration")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key in LABELS:
        group = [r for r in decomposition if r["pipeline"] == key and r["region"] == "outer10"]
        axes[0].plot([r["na_out"] for r in group],
                     [100 * r["variance_fraction"] for r in group],
                     color=COLORS[key], marker="o", label=LABELS[key])
    axes[0].set(xlabel="System NA_out", ylabel="Finite-seed variance / total edge MSE (%)")
    axes[0].legend(fontsize=8)
    for na in protocol["na_out"]:
        items = [r for r in holdouts if r["na_out"] == na]
        for r in items:
            axes[1].scatter(na, 100 * r["squared_error_fraction"],
                            color=COLORS[r["pipeline"]], alpha=.7, s=25)
    for key in ("single_ft", "multi64_ft"):
        items = [r for r in stats if r["pipeline"] == key and r["mask"] == "geometric_outer"]
        axes[1].plot([r["na_out"] for r in items],
                     [100 * r["squared_error_fraction_mean"] for r in items],
                     color=COLORS[key], label=f"{LABELS[key]} main truth")
    axes[1].set(xlabel="System NA_out", ylabel="Outer 10% squared-error share (%)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("Left: exact mean-map/variance decomposition; right: 24 saved clean FT holdouts (dots)")
    figure_done(fig, "bias_and_holdouts")

    outputs = {
        "regions.csv": records, "primary_statistics.csv": stats,
        "mask_geometry.csv": geometry, "modal_baseline.csv": baseline,
        "cumulative_samples.csv": cumulative, "concentration_samples.csv": concentrations,
        "bin_sensitivity.csv": binned, "bias_noise.csv": decomposition,
        "holdouts.csv": holdouts, "paired_na_comparisons.csv": comparisons,
        "unbinned_endpoints.csv": endpoints,
    }
    for name, data in outputs.items():
        write_csv(args.output / name, data)
    np.savez_compressed(args.output / "unbinned_curves.npz", **curves)
    np.savez_compressed(args.output / "representative_maps.npz", **maps)
    save(args.output / "protocol.json", protocol)
    save(args.output / "input_hashes.json", inputs)
    save(args.output / "verification.json", dict(
        passed=True, checks=len(checks), check_types=sorted(set(checks)),
        main_results=len(source), holdouts=len(holdouts), region_rows=len(records),
        primary_rows=len(stats), propagation=False, optimization=False,
        native_shells_preserved=True, hot_masks_frozen_from_clean=True,
    ))
    ended = datetime.now(timezone.utc).isoformat()
    save(args.output / "versions.json", dict(
        experiment_id=protocol["experiment_id"], run_id=protocol["run_id"],
        commit=head, script_sha256=digest(__file__),
        protocol_sha256=digest(protocol_path), command=sys.argv,
        python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__,
        finished_utc=ended, elapsed_seconds=time.monotonic() - started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        parent_summary_sha256=digest(source_path),
    ))
    save(args.output / "status.json", dict(
        state="complete", completed_systems=4, total_systems=4, generated_utc=ended,
    ))
    print(json.dumps(read(args.output / "verification.json")), flush=True)


if __name__ == "__main__":
    main()
