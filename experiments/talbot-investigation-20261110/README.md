# Talbot comparison and Fourier diagnosis, 2026-11-10

This is a retrospective archive of code already executed on SJTU. It is not a claim that this Git commit or its tag existed before the numerical run. The five executed scripts match results/code_sha256.json byte for byte. lsi comes from the repository base commit 6da7b93caf426f50f509edfe2dec474d1a5a483d; reference inputs come from shear-reference-lm 71ab81b18c9f017c6ddd3dfc302d18b0a2ba2542. Verification is recorded in provenance/archive-verification.json.

The original scripts deliberately retain their actual absolute BASE dependency path /data/yk/shear-reference-lm/talbot-rerun-20261109 and output directory convention. dependencies/ preserves the imported experiment-specific modules; lsi/ is at the repository root. Keep those paths available, or prepare a separate new run and record any path adaptations. Do not change the archived run code in place.

Runtime: private Conda /data/yk/conda/envs/shear-reference-lm-py312; one numerical process, four BLAS threads, CPU only. Sequential entries: forward_compare.py; PROBE_ONLY=1 phase_probe.py; calibrate_ablation.py; validate_ablation.py; order_mixing.py. The executed probe only evaluates the saved subset, rather than the optional full phase-probe calibration.

Large raw arrays remain in /data/yk/shear-reference-lm/talbot-investigation-20261110/results and the original local task directory. Postprocessing and reports are archived in a separate later commit/tag.
