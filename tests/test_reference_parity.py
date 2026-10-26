from pathlib import Path
import numpy as np

from lsi.config import Scene
from lsi.forward import DiffractionModel
from lsi.fourier import FourierExtractor, FourierSettings


def test_independent_reference_fields_and_sidebands(reference):
    system, expected = reference
    model = DiffractionModel(system, batch_sources=1, workers=1)
    truth = Scene(rms=None).coefficients_for(model.indices)
    np.testing.assert_allclose(model.source_field(system.sources[0]), expected['source'], atol=1e-13)
    H = model.pupil * np.exp(2j * np.pi * model.wavefront(truth))
    objective = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(expected['source']))) * H)))
    np.testing.assert_allclose(objective, expected['objective'], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(model.propagate_image(objective), expected['camera'], rtol=1e-8, atol=1e-16)
    np.testing.assert_allclose(model.evaluate(truth), expected['intensity'], rtol=1e-8, atol=1e-25)
    settings = FourierSettings(half_width=8, carrier_x=29, carrier_y=29, phase_sigma=1)
    extracted = FourierExtractor(system, settings).extract(expected['intensity'], details=True)
    with np.load(Path(__file__).parent / 'data/reference_sidebands.npz') as sidebands:
        np.testing.assert_allclose(np.exp(1j * extracted['wrapped']),
                                   np.exp(1j * sidebands['wrapped']), atol=1e-10)
