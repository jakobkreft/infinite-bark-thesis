# infinite-bark-thesis

Master's thesis pipeline for generating **seamless (toroidal) tree-bark textures**
with a semantically-guided diffusion model (DiffInfinite).

The pipeline goes from raw field photos of logs to synthetic, tileable bark
textures rendered in 3D. Folders are numbered in pipeline order.

| Stage | What it does |
|-------|--------------|
| [01_capture](01_capture) | Photograph bark on logs in the field (protocol + dataset link). |
| [02_preprocess](02_preprocess) | Straighten/unwrap logs, fix lighting, crop, cut into patches. |
| [03_segmentation](03_segmentation) | Label and segment bark damage (knots, mechanical damage). |
| [04_masks](04_masks) | Generate synthetic semantic masks to condition the diffusion model. |
| [05_diffusion](05_diffusion) | DiffInfinite texture synthesis (will be the `diffinfinite-bark` submodule). |
| [06_evaluation](06_evaluation) | Evaluate generated textures (TBD). |
| [07_render](07_render) | Render generated textures in 3D (Blender normal maps). |
| [website](website) | Project page. |

**Classes used throughout:** `0` background / smooth bark · `1` slepice (pruning wounds / branch knots) · `2` mehanske poškodbe (mechanical damage).

> Each numbered subfolder is one step; most have their own short README.
> Datasets, model weights and environments are **not** committed.
