from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class PipelineConfig:
    dataset_dir: Path
    output_dir: Path
    pattern_size: int = 3
    target_size: int = 100
    tile_mode: str = "none"
    num_generations: int = 3
    seed: int = 42
    max_restarts: int = 20
    snapshot_interval: int = 150
    augment_symmetry: bool = True
    min_pattern_weight: int = 2
    max_patterns: int = 512
    max_patterns_for_gallery: int = 64

    def to_json(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["dataset_dir"] = str(self.dataset_dir)
        payload["output_dir"] = str(self.output_dir)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def from_json(path: Path) -> "PipelineConfig":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["dataset_dir"] = Path(payload["dataset_dir"])
        payload["output_dir"] = Path(payload["output_dir"])
        return PipelineConfig(**payload)
