"""Validate the new full grids against the unchanged reference equations."""
from dataclasses import replace
import gc
import json
import os
import resource
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from lsi.forward import DiffractionModel
from lsi.execution import exclusive_study
from reference_check import check_reference, load_reference
from talbot_common import ROOT, RESULTS, Monitor, config, event, save, truth_for


def main():
    resource.setrlimit(resource.RLIMIT_AS, (20 * 1024**3, 20 * 1024**3))
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = dict(python=sys.version, executable=sys.executable,
                  threadpools=threadpool_info(), computation="CPU",
                  reference_adapter="Only unused torch import/base removed via AST; numeric code unchanged.",
                  errors={}, timings={})
    with exclusive_study(), Monitor("preflight") as monitor:
        cfg = config("single")
        started = time.monotonic()
        model = DiffractionModel(cfg, tuple(range(1, 36)), batch_sources=1, workers=1)
        truth = truth_for(model)
        output["truth_coefficients_waves_36"] = [0, *truth.tolist()]
        output["system"] = cfg.to_dict()
        output["source_locations_m"] = [
            ((7 + .5) - cfg.N_super_lattice / 2) * cfg.super_lattice_period
        ] * 2
        output["pupil_pixels"] = int(model.pupil.sum())
        output["pupil_box_shape"] = list(model.pupil_box.shape)
        output["timings"]["setup"] = time.monotonic() - started
        monitor.update(stage="single_reference_parity")
        for name, coefficients in (("zero", np.zeros(35)), ("truth", truth)):
            report = check_reference(model, ROOT / "reference", coefficients, all_sources=True)
            output["errors"][name] = report
            save(RESULTS / "preflight.json", output)
            assert report["passed"], report
            event("reference_parity", case=name, errors=report["errors"])
        rows = np.sort(np.random.default_rng(42).choice(cfg.n**2, 2048, replace=False))
        direction = np.random.default_rng(17).normal(size=35)
        direction /= np.linalg.norm(direction)
        monitor.update(stage="jacobian_validation")
        value, derivative = model.evaluate(truth, jacobian=True, rows=rows)
        h = 1e-6
        finite = (model.evaluate(truth + h * direction, rows=rows) -
                  model.evaluate(truth - h * direction, rows=rows)) / (2 * h)
        error = np.linalg.norm(direction @ derivative - finite) / np.linalg.norm(finite)
        sampled_error = np.linalg.norm(value - model.evaluate(truth).ravel()[rows]) / np.linalg.norm(value)
        output["errors"]["jacobian_directional_relative_l2"] = float(error)
        output["errors"]["sampled_vs_full_relative_l2"] = float(sampled_error)
        assert error < 1e-6 and sampled_error < 1e-10
        zero = model.evaluate(np.zeros(35))
        full = model.evaluate(truth)
        np.savez_compressed(RESULTS / "preflight_single.npz", zero=zero, truth=full)
        del model, value, derivative
        gc.collect()
        monitor.update(stage="multi64_reference_parity")
        model = DiffractionModel(config("multi64"), tuple(range(1, 36)), batch_sources=1)
        started = time.monotonic()
        report = check_reference(model, ROOT / "reference", truth, all_sources=True)
        output["errors"]["multi64_truth"] = report
        output["timings"]["multi64_original_parity"] = time.monotonic() - started
        assert report["passed"], report
        event("reference_parity", case="multi64", errors=report["errors"])
        full = model.evaluate(truth)
        zero = model.evaluate(np.zeros(35))
        np.savez_compressed(RESULTS / "preflight_multi64.npz", zero=zero, truth=full)
        for threads in (1, 4):
            with threadpool_limits(limits=threads):
                started = time.monotonic()
                image, derivative = model.evaluate(truth, jacobian=True, rows=rows)
                output["timings"][f"multi64_sampled_jacobian_threads_{threads}"] = time.monotonic() - started
                event("benchmark", threads=threads,
                      seconds=output["timings"][f"multi64_sampled_jacobian_threads_{threads}"])
                del image, derivative
        output["passed"] = True
        output["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        save(RESULTS / "preflight.json", output)
        event("preflight_complete", timings=output["timings"], peak_rss_gib=output["peak_rss_gib"])


if __name__ == "__main__":
    main()
