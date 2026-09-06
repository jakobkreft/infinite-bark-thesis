# infinite-bark-thesis

Master's thesis pipeline for generating **seamless (toroidal) tree-bark textures**
with a semantically-guided diffusion model (DiffInfinite).

📄 **Thesis:** *Semantično vodeni difuzijski modeli za neskončno velike toroidne teksture*,
Jakob Kreft, University of Ljubljana, Faculty of Electrical Engineering, 2026 &mdash;
published in the [Repository of the University of Ljubljana](https://hdl.handle.net/20.500.12556/RUL-185875).
🌐 **Project page:** <https://jakobkreft.github.io/infinite-bark-thesis/>

The pipeline goes from raw field photos of logs to synthetic, tileable bark
textures rendered in 3D. Folders are numbered in pipeline order.

| Stage | What it does |
|-------|--------------|
| [01_capture](01_capture) | Photograph bark on logs in the field (protocol + dataset link). |
| [02_preprocess](02_preprocess) | Segment the log out, straighten/unwrap it, normalize lighting, crop edges. |
| [03_segmentation](03_segmentation) | Label and segment bark damage (knots, mechanical damage). |
| [04_masks](04_masks) | Generate synthetic semantic masks to condition the diffusion model. |
| [05_diffusion](05_diffusion) | DiffInfinite texture synthesis (`diffinfinite-bark` submodule) + training-patch prep. |
| [06_evaluation](06_evaluation) | Evaluate generated textures (TBD). |
| [07_render](07_render) | Render generated textures in 3D (Blender normal maps). |
| [docs](docs) | Project web page (GitHub Pages). |

**Classes used throughout:** `0` background / smooth bark · `1` slepice (pruning wounds / branch knots) · `2` mehanske poškodbe (mechanical damage).

> Each numbered subfolder is one step; most have their own short README.
> Datasets, model weights and environments are **not** committed.

## Citation

If you find this work useful, please cite:

```bibtex
@mastersthesis{kreft2026toroidal,
  author  = {Kreft, Jakob},
  title   = {Semantično vodeni difuzijski modeli za neskončno velike toroidne teksture},
  school  = {University of Ljubljana, Faculty of Electrical Engineering},
  type    = {Master's thesis},
  address = {Ljubljana, Slovenia},
  year    = {2026},
  url     = {https://hdl.handle.net/20.500.12556/RUL-185875}
}
```

## AI assistance

AI tools (large language models) were used to assist with the development of this project.
Primarily for code scaffolding, debugging, refactoring and documentation.
All AI-assisted output was reviewed, tested and adapted by the author.
