# 04 — Mask generation

Generate **synthetic semantic masks** (same 3 classes as segmentation) to condition
the diffusion model. This lets us sample unlimited, seamlessly tileable (toroidal)
layouts instead of only the few real masks we hand-labeled.

These are **interchangeable methods**, not sequential steps — pick one. Each folder
has its own detailed README.

| Folder | Method |
|--------|--------|
| [generate-masks-WFC](generate-masks-WFC) | Wave Function Collapse — supports torus tiling. |
| [generate-masks-MRF](generate-masks-MRF) | Markov Random Field / Gibbs sampling. |
| [generate-test-masks](generate-test-masks) | Utility: hand-designed masks for sanity-testing diffusion conditioning. |
