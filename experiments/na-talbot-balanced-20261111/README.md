# Balanced NA sweep, 2026-11-11

The four original NA_out values are 0.020, 0.066, 0.110 and 0.150.
Four pipelines are compared: single-source FT, single-source LM, 64-source FT
and 64-source LM. All 35 non-piston coefficients are free, including tilt and
defocus. The mixed j7/j25/j11/j15 truth has 0.05 waves raw RMS on each complete
discrete objective pupil. Piston is fixed at zero. Evaluation removes no modes.

## Physical design

This is a compensated system family, not an NA-only intervention in a fixed
instrument. Let q=NA_out/0.020. Scale source pitch, image/grid transverse
coordinates, grating period and camera pitch by 1/q; scale camera distance and
object axial offset by 1/q^2. Keep wavelength, f1/f2 and NA_in fixed. The changing
obscuration ratio NA_in/NA_out is recorded explicitly. The physical 8x8 source
pitch changes; the source indices and their normalized layout do not.

For each actual rasterized period p, set zT=2p^2/lambda and
R=zT*D/(D-zT), where D is the grating-camera distance. With magnification
M=f2/f1, set z_grating=R+M^2*shift_z. Thus (1/R+1/D)^(-1)=zT.
The reference's exact object-side angular-spectrum defocus remains enabled.
The R correction and Talbot identity apply to its quadratic phase approximation.
Increasing NA alone does not change this paraxial identity when p, D and the
axial offset remain fixed; the old baseline was already near a quarter Talbot,
and changing only p in the second old scan changed that identity again.

Numerics: 1680^2 camera/image and 8400^2 grating, 100 samples per grating period
and 50 per square aperture side. The camera width is 28.56 mm/q and the grating
width 56.28 micrometres/q. These deliberately differ from the 11/9 baseline
so that high diffraction orders, camera range and Fresnel sampling can be
checked together. The full camera interval is inside the grating quadrature
alias period, and a quadratic integrand phase-gradient check is required.

Preflight propagates all 64 sources independently for zero and truth, recording
outer 5% energy on the image, grating and camera planes. It requires positive
geometric margins even when NA_image is interpreted as a sine and converted to
tan(arcsin(NA_image)). This stricter envelope is a geometric diagnostic, not a
replacement of the reference Fresnel operator. Limits are predeclared in
protocol.json. Finite diffracted fields have nonzero tails, so no finite grid
can establish identically zero overflow. Edge energy is not a reconstruction
error bound. One source per NA is also propagated through the original
Stage1/Stage2/Stage3 reference and compared against the sampled operator.

## Reconstruction

FT selects the adjacent-order fundamental using geometry alone, constructs
Z(x-s)-Z(x) on the corresponding pupil intersection and subtracts an independent
zero-aberration reference in complex form. The real phase is a small-phase
principal-value feature, with branch diagnostics retained for every calibration
and observation. It is not advertised as a general large-aberration unwrapping
algorithm. Every NA and source count gets 70 independent +/-0.02-wave single-mode
calibrations; no test truth is used for selecting the band or fitting the response.
Three random 0.05-wave holdouts (seeds 1807..1809) are not used in calibration.

LM uses scipy least_squares(method="lm") with the analytic sampled-operator
Jacobian, a zero wavefront initial value, and independent gain/background.
4096 fixed random camera pixels (seed 62011) enter the objective. A separate
full-image prediction is used for the reported intensity residual. The report
must not describe this as all pixels participating in LM.

Each of the 16 pipelines has clean, 40 dB seeds 41..45, and 30 dB seed 42:
112 rows. Noise sigma=10^(-SNR/20)*RMS(clean/clean.max()). A source-count/NA
pair uses exactly the same observations for FT and LM. Report five-seed means
and sample standard deviations without averaging RMS over unrelated conditions.
The three random holdouts test FT calibration generalization separately.

## Execution and provenance

Only SJTU may run numerical preflight and production. Use the private Conda
environment /data/yk/conda/envs/shear-reference-lm-py312, CPU only, one numerical
process, four BLAS threads, eight FFT threads, a user-wide study lock, 30-second status/ETA logs,
SIGTERM preservation, 36 GiB RSS stop and a 20-hour invocation cap.

Before each invocation, commit and push its actual code/config and an immutable
exp/na-talbot-balanced-20261111/<run-id> tag in each participating repository.
Verify GitHub tag SHA and the server checkout SHA before starting. Do not alter
old result directories or retrospective tags. Retain the version manifest,
launch commands, source hashes and execution timestamps with the results.
Postprocessing receives a separate commit/tag if it changes after calculation.

The scalar 4f/Fresnel reference is preserved. NA_out/M reaches 0.75, where its
paraxial approximation is a substantive physical limitation. These experiments
quantify recovery inside that reference model; they do not validate vector
high-NA optics or establish actual instrument accuracy.
