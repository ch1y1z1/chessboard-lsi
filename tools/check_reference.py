"""Run an independent parity check against user-supplied reference scripts.

Only the unused torch.nn.Module base is replaced with object, so checking the
NumPy reference does not require installing PyTorch. Numeric code is unmodified.
"""
import argparse
import ast
from dataclasses import asdict
from types import SimpleNamespace
from hashlib import sha256
from pathlib import Path
import numpy as np

from lsi.basis import mode_labels
from lsi.config import Scene, SystemConfig
from lsi.execution import exclusive_study, interrupt_on_termination, requires_large_run
from lsi.forward import DiffractionModel
from lsi.io import write_json


def load_reference(path):
    path = Path(path)
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if not isinstance(node, ast.Import) or not any(a.name == "torch" for a in node.names)]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            node.bases = [ast.Name(id="object", ctx=ast.Load())]
    namespace = {"__name__": "_reference_oracle"}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace


def check_reference(model, directory, coefficients, all_sources=False):
    directory = Path(directory)
    paths = [directory / name for name in ("point2plate.py", "obj_sys_ft.py", "grating2cam.py")]
    labels = [mode_labels(model.config.Nz)[i] for i in model.indices]
    # Only this adapter knows the original reference's unused parameters and
    # truth-in-config interface. They never enter the production forward model.
    original = SimpleNamespace(**asdict(model.config), distance_stage1=2., hole_diameter=50e-9,
        lattice_distance=150e-9, N_hole_rows=1, shift_x=0., shift_y=0., basis_factor=.01,
        c_Zernike=list(coefficients), n_radial=[(n + abs(l)) // 2 for n, l in labels],
        l_azimuthal=[l for n, l in labels])
    stage1 = load_reference(paths[0])["Stage1"](original)
    stage2 = load_reference(paths[1])["Stage2_ab"](original, mode="aberration")
    stage3 = load_reference(paths[2])["Stage3"](original, mode="camera")
    c = np.asarray(coefficients)
    records = {}

    def compare(name, actual, expected):
        error = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-300))
        records[name] = error
        return error

    src = model.config.sources[0]
    field = stage1.forward(*src)
    compare("source_field_relative_l2", model.source_field(src), field)
    H = model.pupil * np.exp(2j * np.pi * model.wavefront(c))
    predicted_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field))) * H)))
    reference_img = stage2.forward(field)
    compare("objective_field_relative_l2", predicted_img, reference_img)
    reference_g = stage3._img2grating(reference_img)
    actual_g = model.propagate_image(reference_img, plane="grating")
    compare("grating_field_relative_l2", actual_g, reference_g)
    compare("grating_mask_relative_l2", model.grating, stage3.grating)
    reference_cam = stage3._grating2cam(reference_g * stage3.grating)
    compare("camera_field_relative_l2", model.propagate_image(reference_img), reference_cam)
    if all_sources:
        image = np.zeros((model.n, model.n))
        for source in model.config.sources:
            image += abs(stage3.forward(stage2.forward(stage1.forward(*source))))**2
        compare("all_sources_intensity_relative_l2", model.evaluate(c), image)
    return {"errors": records, "passed": all(v < 1e-8 for v in records.values()),
            "tolerance": 1e-8, "all_sources_checked": all_sources,
            "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in paths},
            "config": model.config.to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('reference', 'retrieval', 'test'), required=True)
    parser.add_argument('--reference-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-large-run', action='store_true')
    args = parser.parse_args()
    system = SystemConfig.profile(args.profile)
    if requires_large_run(system) and not args.allow_large_run:
        parser.error('reference-sized verification requires --allow-large-run')
    if args.output.exists():
        parser.error('output file already exists')
    with exclusive_study(), interrupt_on_termination():
        model = DiffractionModel(system, batch_sources=1, workers=1)
        truth = Scene(rms=None).coefficients_for(model.indices, system.Nz)
        report = check_reference(model, args.reference_directory, truth, all_sources=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
        print(report['errors'])
        if not report['passed']:
            raise SystemExit('reference parity failed')


if __name__ == '__main__':
    main()
