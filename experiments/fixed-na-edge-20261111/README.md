# Fixed-system NA and edge concentration

This analysis reuses all 112 main results and 24 FT holdouts from
`na-talbot-balanced-20261111/run-08`. It reconstructs pupil error maps from saved
coefficients, with the original annular basis and raw wave units. It performs no
new propagation, calibration, or optimization.

The primary region is the outermost 10% of valid pupil pixels. This choice is
fixed in `protocol.json` before the formal run. Complete equal-radius shells
are retained, so actual fractions are recorded. Sensitivity fractions are
1%, 2%, 5%, 10%, 15%, 20%, and 30%. This removes the change in area caused by
using the same radial width at different obscuration ratios.

The main curves use every distinct native pupil radius, without radial bins.
Geometric outer-to-inner error concentration, RMS after erosion, and cumulative
inner-to-outer RMS answer complementary questions. Equal-radius and equal-area
bins with 10/20/50/100 bins are included only as sensitivity diagnostics.

A second diagnostic ranks pixels by absolute **clean FT** error, freezes this
mask, and evaluates all five noisy repetitions and all four pipelines on it.
Single-source and 64-source references are both retained. Each hot mask uses
the same pixel count as the corresponding geometric edge. Self-ranked noisy
top-error masks are also recorded as descriptive upper envelopes; they are
not independent validation or proof of an edge effect.

Map captions distinguish single seed42 images from five-seed mean statistics.
Repeated pixels are spatially correlated. Uncertainty uses noise seeds, never
the number of pixels. All comparisons are conditional on the saved main truth;
the 24 existing random-truth FT holdouts test its limits without retuning.

The clean/perturbation cross term is retained in the exact MSE decomposition.
The finite-seed mean/variance identity is also checked. An equal-variance,
independent coefficient-error model supplies a basis-only spatial baseline.
It is an algebraic control, not a fitted physical noise model or causal test.

Different `NA_out` settings belong to a compensated Fresnel system family.
`NA_in` stays fixed while obscuration, basis, physical truth and system
dimensions change. No result here isolates a pure physical high-NA effect,
proves Gibbs ringing, or establishes the cause of boundary-localized errors.

Formal execution requires a clean checkout at a pushed immutable experiment
tag. The script checks the expected commit, hashes all inputs, saves its
command and environment, and writes into a new output directory. Large
calculations remain on SJTU; this analysis needs no GPU and uses two CPU
threads.
