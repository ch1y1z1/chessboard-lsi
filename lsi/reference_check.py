"""Run an independent parity check against user-supplied reference scripts.

Only the unused torch.nn.Module base is replaced with object, so checking the
NumPy reference does not require installing PyTorch. Numeric code is unmodified.
"""
import ast
from hashlib import sha256
from pathlib import Path
import numpy as np


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


def check_reference(model, directory, all_sources=False):
    directory = Path(directory)
    paths = [directory / name for name in ("point2plate.py", "obj_sys_ft.py", "grating2cam.py")]
    stage1 = load_reference(paths[0])["Stage1"](model.config)
    stage2 = load_reference(paths[1])["Stage2_ab"](model.config, mode="aberration")
    stage3 = load_reference(paths[2])["Stage3"](model.config, mode="camera")
    from .annular import configured_coeffs
    c = configured_coeffs(model.config)[list(model.indices)]
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
