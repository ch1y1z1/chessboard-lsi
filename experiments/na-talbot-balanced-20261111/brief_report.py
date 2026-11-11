"""Make fixed-system aperture diagnostics from saved run-08 coefficients."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from report import COLORS, LABELS, pupil_basis


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(values):
    return float(np.sqrt(np.mean(values**2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == args.expected_commit
    assert not subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.results / "summary.json"
    rows = read(source)["results"]
    assert len(rows) == 112
    flight = read(args.preflight / "preflight.json")
    assert flight["passed"] and flight["completed_na"] == 4
    prior = list(csv.DictReader((args.reference_report / "statistics.csv").open()))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def figure_done(fig, name):
        fig.savefig(args.output / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(args.output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig_c, axes_c = plt.subplots(
        2, 2, figsize=(11, 7.2), sharey=True, constrained_layout=True
    )
    fig_r, axes_r = plt.subplots(
        2, 2, figsize=(11, 7.2), sharey=True, constrained_layout=True
    )
    cumulative, radial, spatial, stats, parameters, boundaries = [], [], [], [], [], []
    checks = 0
    for panel, na in enumerate((0.020, 0.066, 0.110, 0.150)):
        cfg = read(args.results / f"na{na:.3f}_single/config.json")
        geometry = read(args.results / f"na{na:.3f}_single/geometry.json")
        Z, pupil, rho = pupil_basis(cfg)
        zp = Z[:, pupil]
        rp = rho[pupil]
        order = np.argsort(rp)
        sorted_rho = rp[order]
        cuts = np.linspace(cfg["NA_in"] / na + 0.01, 1.0, 80)
        counts = np.searchsorted(sorted_rho, cuts, side="right")
        assert counts.min() > 0 and counts[-1] == int(pupil.sum())
        edges = np.linspace(0, 1, 51)
        # Each pupil pixel belongs to exactly one bin, including rho == 1.
        bins = np.clip(np.searchsorted(edges, rp, side="right") - 1, 0, 49)
        ring_counts = np.bincount(bins, minlength=50)
        assert ring_counts.sum() == pupil.sum()
        valid = ring_counts > 0
        ring_centers = (edges[:-1] + edges[1:]) / 2
        outer = rp >= 0.9
        inside = ~outer
        parameters.append(
            dict(
                na_out=na,
                grating_period_um=cfg["grating_period"] * 1e6,
                z_grating_um=cfg["z_grating"] * 1e6,
                z_cam_mm=cfg["z_cam"] * 1e3,
                source_pitch_um=cfg["super_lattice_period"] * 1e6,
                shift_z_um=cfg["shift_z"] * 1e6,
                effective_distance_um=geometry["effective_distance_m"] * 1e6,
                talbot_ratio=geometry["talbot_ratio"],
                epsilon=cfg["NA_in"] / na,
            )
        )
        boundary = next(row for row in flight["rows"] if row["na_out"] == na)
        boundaries.append(dict(na_out=na, **boundary["maxima"]))
        selected = [row for row in rows if row["na_out"] == na]
        errors = {}
        for row in selected:
            error = (np.asarray(row["coefficients"]) - row["truth"]) @ zp
            true = np.asarray(row["truth"]) @ zp
            evaluated = {
                "raw_rms_waves": rms(error),
                "raw_pv_waves": float(np.ptp(error)),
                "outer10_rms_waves": rms(error[outer]),
                "common_na002_rms_waves": rms(error[rp * na <= 0.02 + 1e-14]),
            }
            for key, value in evaluated.items():
                assert abs(value - row[key]) < 1e-10, (na, key)
                checks += 1
            assert abs(rms(true) - 0.05) < 1e-12
            checks += 1
            full_sse = float(np.sum(error**2))
            outer_sse = float(np.sum(error[outer] ** 2))
            key = f'{row["source_kind"]}_{row["method"]}'
            errors[(key, row["condition"])] = error
            spatial.append(
                dict(
                    na_out=na,
                    pipeline=key,
                    condition=row["condition"],
                    full_rms_waves=rms(error),
                    interior_rms_waves=rms(error[inside]),
                    outer_rms_waves=rms(error[outer]),
                    outer_over_interior=rms(error[outer]) / rms(error[inside]),
                    outer_squared_error_fraction=outer_sse / full_sse,
                    outer_pixel_fraction=float(np.mean(outer)),
                    evaluation_pixels=int(pupil.sum()),
                )
            )
        for key in LABELS:
            sample = np.stack([errors[(key, f"40db_seed{s}")] for s in range(41, 46)])
            cumulative_sse = np.cumsum(sample[:, order] ** 2, axis=1)
            cum_rms = np.sqrt(cumulative_sse[:, counts - 1] / counts[None, :])
            ring_rms = np.stack(
                [
                    np.sqrt(
                        np.divide(
                            np.bincount(bins, weights=error**2, minlength=50),
                            ring_counts,
                            out=np.zeros(50),
                            where=valid,
                        )
                    )
                    for error in sample
                ]
            )
            group = [
                row
                for row in spatial
                if row["na_out"] == na
                and row["pipeline"] == key
                and row["condition"].startswith("40db_")
            ]
            record = dict(na_out=na, pipeline=key, seeds=5)
            for metric in (
                "full_rms_waves",
                "interior_rms_waves",
                "outer_rms_waves",
                "outer_over_interior",
                "outer_squared_error_fraction",
            ):
                values = np.asarray([row[metric] for row in group])
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_sample_sd"] = float(values.std(ddof=1))
            reference = next(
                row for row in prior if float(row["na_out"]) == na and row["pipeline"] == key
            )
            assert abs(record["full_rms_waves_mean"] - float(reference["mean40"])) < 1e-12
            assert abs(record["full_rms_waves_sample_sd"] - float(reference["sd40"])) < 1e-12
            assert abs(cum_rms[:, -1].mean() - float(reference["mean40"])) < 1e-12
            checks += 3
            stats.append(record)
            for i, seed in enumerate(range(41, 46)):
                for cut, count, value in zip(cuts, counts, cum_rms[i]):
                    cumulative.append(
                        dict(
                            na_out=na, pipeline=key, seed=seed,
                            na_eval=float(cut * na), rho_cut=float(cut),
                            pixels=int(count), rms_waves=float(value),
                        )
                    )
                for b in np.flatnonzero(valid):
                    radial.append(
                        dict(
                            na_out=na, pipeline=key, seed=seed,
                            na_low=float(edges[b] * na),
                            na_high=float(edges[b + 1] * na),
                            rho_low=float(edges[b]), rho_high=float(edges[b + 1]),
                            pixels=int(ring_counts[b]), rms_waves=float(ring_rms[i, b]),
                        )
                    )
            for ax, x, values in (
                (axes_c.flat[panel], cuts * na, cum_rms),
                (axes_r.flat[panel], ring_centers[valid] * na, ring_rms[:, valid]),
            ):
                mean = values.mean(axis=0)
                sd = values.std(axis=0, ddof=1)
                ax.semilogy(x, mean, color=COLORS[key], label=LABELS[key], linewidth=1.8)
                ax.fill_between(
                    x, np.maximum(mean - sd, 1e-10), mean + sd,
                    color=COLORS[key], alpha=0.11, linewidth=0,
                )
        for ax in (axes_c.flat[panel], axes_r.flat[panel]):
            ax.axvspan(0.9 * na, na, facecolor="#d9a26a", alpha=0.16)
            ax.set(
                title=f"Fixed NA_out={na:.3f}",
                xlim=(cfg["NA_in"], na),
                ylabel="Wavefront error RMS (waves)",
            )
            ax.grid(alpha=0.18)
        axes_c.flat[panel].set_xlabel("Evaluation cutoff NA (cumulative pupil)")
        axes_r.flat[panel].set_xlabel("Local NA (disjoint annular bins)")
    axes_c.flat[0].legend(fontsize=8)
    axes_r.flat[0].legend(fontsize=8)
    fig_c.suptitle("Fixed-system cumulative RMS; 40 dB, mean +/- sample SD (5 seeds)")
    fig_r.suptitle("Local annular RMS; 40 dB, mean +/- sample SD (5 seeds)")
    figure_done(fig_c, "cumulative_rms_fixed_na")
    figure_done(fig_r, "annular_rms_fixed_na")

    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    for key in LABELS:
        group = [row for row in stats if row["pipeline"] == key]
        ax.errorbar(
            [row["na_out"] for row in group],
            [row["full_rms_waves_mean"] for row in group],
            yerr=[row["full_rms_waves_sample_sd"] for row in group],
            color=COLORS[key], marker="o", capsize=3, label=LABELS[key],
        )
    ax.set(
        xlabel="System NA_out", ylabel="Full-pupil error RMS (waves)",
        title="Four pipelines; 40 dB, mean +/- sample SD (5 seeds)", yscale="log",
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.18)
    figure_done(fig, "full_pupil_40db")
    for name, data in (
        ("cumulative_na.csv", cumulative), ("annular_na.csv", radial),
        ("spatial_metrics.csv", spatial), ("statistics_40db.csv", stats),
        ("parameters.csv", parameters), ("boundary_energy.csv", boundaries),
    ):
        write_csv(args.output / name, data)
    save(
        args.output / "facts.json",
        dict(parameters=parameters, boundaries=boundaries, statistics=stats),
    )
    save(
        args.output / "verification.json",
        dict(
            passed=True, checks=checks, input_results=112,
            evaluated_40db_results=80, cumulative_rows=len(cumulative),
            annular_rows=len(radial), disjoint_ring_count_checks=4,
            input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            commit=head, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            command=sys.argv, generated_utc=datetime.now(timezone.utc).isoformat(),
            new_propagation=False, new_optimization=False,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            cumulative_definition="original pupil intersect NA <= NA_eval",
            annular_definition="half-open rho bins; final bin includes rho=1",
            standard_deviation="sample standard deviation across seeds 41..45",
            configuration="four fixed saved systems; only evaluation region changes",
        ),
    )
    print(f"Verified {checks} checks; output {args.output}", flush=True)


if __name__ == "__main__":
    main()
