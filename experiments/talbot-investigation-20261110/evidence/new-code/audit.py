"""Independently recalculate reported metrics from saved numerical artifacts."""
from hashlib import sha256
import json

import numpy as np

from talbot_common import ROOT, RESULTS, digest, save, utc


def rms(array):
    return float(np.sqrt(np.mean(array*array)))


def read(path):
    return json.loads(path.read_text())


def main():
    checks = []

    def require(condition, name, **values):
        checks.append(dict(check=name, passed=bool(condition), **values))
        if not condition:
            save(RESULTS / "audit.json", dict(passed=False, checks=checks))
            raise AssertionError(name)

    manifest = read(ROOT / "provenance" / "input-manifest.json")
    for name, expected in manifest["files"].items():
        actual = sha256((ROOT/name).read_bytes()).hexdigest()
        require(actual == expected, f"input_sha256:{name}")
    total = 0
    truths = []
    summary = []
    for kind in ("single", "multi64"):
        folder = RESULTS/kind
        metadata = read(folder/"observation.json")
        clean = np.load(folder/"clean.npy")
        noisy = np.load(folder/"observed_40db.npy")
        require(digest(clean) == metadata["clean_sha256"], f"{kind}:clean_hash")
        require(digest(noisy) == metadata["observed_40db_sha256"], f"{kind}:noisy_hash")
        sigma = .01*rms(clean)
        expected = clean+sigma*np.random.default_rng(42).standard_normal(clean.shape)
        require(np.array_equal(expected, noisy), f"{kind}:noise_formula_seed42")
        require(abs(clean.max()-1) < 1e-15, f"{kind}:fixed_peak_normalization")
        require(len(metadata["sources"]) == (1 if kind == "single" else 64),
                f"{kind}:source_count")
        if kind == "multi64":
            require(metadata["sources"] == [[i,j] for i in range(3,11) for j in range(3,11)],
                    "multi64:fixed_physical_source_indices")
        else:
            require(metadata["sources"] == [[7,7]], "single:physical_source_index")
        truths.append(np.array(metadata["truth_coefficients_waves_36"]))
        for path in sorted(folder.glob("*.json")):
            row = read(path)
            if not isinstance(row, dict) or "raw_rms_waves" not in row:
                continue
            total += 1
            arrays = np.load(path.with_name(path.stem+"_wavefront.npz"))
            mask = arrays["pupil"]
            error = arrays["reconstruction"]-arrays["truth"]
            actual_rms = rms(error[mask])
            require(abs(actual_rms-row["raw_rms_waves"]) < 1e-12,
                    f"{kind}:{path.stem}:raw_rms", actual=actual_rms)
            require(abs(np.ptp(error[mask])-row["raw_pv_waves"]) < 1e-12,
                    f"{kind}:{path.stem}:raw_pv")
            require(abs(rms(arrays["truth"][mask])-.05) < 1e-13,
                    f"{kind}:{path.stem}:truth_rms_005")
            coefficients = np.array(row["coefficients_waves_36"])
            truth = np.array(row["truth_coefficients_waves_36"])
            require(coefficients[0] == 0 and truth[0] == 0,
                    f"{kind}:{path.stem}:piston_gauge")
            require(abs(rms((coefficients-truth)[1:])-row["coefficient_rms_waves"]) < 1e-13,
                    f"{kind}:{path.stem}:coefficient_rms")
            observed = clean if row["condition"] == "clean" else noisy
            require(digest(observed) == row["observed_sha256"],
                    f"{kind}:{path.stem}:shared_observation_hash")
            prediction = np.load(path.with_name(path.stem+"_predicted.npy"))
            require(abs(rms(prediction-observed)-row["full_intensity_rmse"]) < 1e-12,
                    f"{kind}:{path.stem}:intensity_rms")
            for radial in row["radial"]:
                radial_mask = mask & (arrays["rho"] >= radial["low"])
                radial_mask &= (arrays["rho"] <= radial["high"] if radial is row["radial"][-1]
                                else arrays["rho"] < radial["high"])
                require(int(radial_mask.sum()) == radial["pixels"],
                        f"{kind}:{path.stem}:radial_count:{radial['low']}")
                require(abs(rms(error[radial_mask])-radial["rms_waves"]) < 1e-12,
                        f"{kind}:{path.stem}:radial_rms:{radial['low']}")
            if path.stem.startswith("ft_"):
                tag = row["pipeline"]
                cal = np.load(folder/f"{tag}_calibration.npz")
                diagnostic = read(folder/f"{tag}_calibration.json")
                raw = np.array(row["raw_feature_coefficients"])
                solved = np.linalg.lstsq(cal["response"], raw-cal["baseline"],
                                         rcond=diagnostic["rcond"])[0]
                require(np.max(abs(solved-coefficients[1:])) < 1e-9,
                        f"{kind}:{path.stem}:independent_calibration_solution")
            summary.append(row)
    require(total == 10, "all_expected_results", actual=total, expected=10)
    require(np.array_equal(*truths), "shared_truth_coefficients")
    save(RESULTS/"summary.json", dict(timestamp_utc=utc(), result_count=total, results=summary))
    save(RESULTS/"audit.json", dict(timestamp_utc=utc(), passed=True,
                                   checks=checks, result_count=total))
    print(f"Audit passed: {len(checks)} checks; {total} results", flush=True)


if __name__ == "__main__":
    main()
