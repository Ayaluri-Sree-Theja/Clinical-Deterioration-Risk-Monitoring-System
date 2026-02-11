from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized repo paths."""
    project_root: Path

    # Data layers
    data_dir: Path
    raw_dir: Path
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path

    data_generation_dir: Path
    etl_dir: Path
    features_dir: Path
    labeling_dir: Path
    modeling_dir: Path
    inference_dir: Path
    dashboards_dir: Path
    notebooks_dir: Path

    artifacts_dir: Path

    @staticmethod
    def from_repo_root(repo_root: Optional[str] = None) -> "ProjectPaths":
        root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[1]

        data_dir = root / "data"
        return ProjectPaths(
            project_root=root,

            data_dir=data_dir,
            raw_dir=data_dir / "raw",
            bronze_dir=data_dir / "bronze",
            silver_dir=data_dir / "silver",
            gold_dir=data_dir / "gold",

            data_generation_dir=root / "data_generation",
            etl_dir=root / "etl",
            features_dir=root / "features",
            labeling_dir=root / "labeling",
            modeling_dir=root / "modeling",
            inference_dir=root / "inference",
            dashboards_dir=root / "dashboards",
            notebooks_dir=root / "notebooks",

            artifacts_dir=root / "artifacts",
        )


@dataclass(frozen=True)
class PipelineSpec:
    """
    Locked project parameters to avoid drift across scripts.
    """
    timezone: str = "UTC"

    # Anchor-time design
    anchor_cadence_minutes: int = 60        # hourly anchors
    warmup_hours: int = 6                   # ignore anchors in first 6h after admission

    # Feature windows (hours)
    feature_windows_hours: List[int] = None  # set in __post_init__

    # Label horizon
    label_horizon_start_hours: int = 24      # (anchor + 24h
    label_horizon_end_hours: int = 48        #  anchor + 48h]

    # Split strategy
    split_strategy: str = "patient_id"       # lock patient-level split

    def __post_init__(self) -> None:
        if self.feature_windows_hours is None:
            object.__setattr__(self, "feature_windows_hours", [6, 12, 24])


@dataclass(frozen=True)
class RunConfig:
    """
    Runtime configuration for a single pipeline run.
    """
    paths: ProjectPaths
    spec: PipelineSpec
    run_id: str

    raw_format: str = "parquet"

    partition_col: Optional[str] = "admission_id"

    @staticmethod
    def default(repo_root: Optional[str] = None) -> "RunConfig":
        paths = ProjectPaths.from_repo_root(repo_root)
        spec = PipelineSpec()

        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        return RunConfig(paths=paths, spec=spec, run_id=run_id)


def ensure_dirs(cfg: RunConfig) -> None:
    dirs = [
        cfg.paths.raw_dir,
        cfg.paths.bronze_dir,
        cfg.paths.silver_dir,
        cfg.paths.gold_dir,
        cfg.paths.artifacts_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def raw_table_path(cfg: RunConfig, table_name: str) -> Path:
    return cfg.paths.raw_dir / table_name


def layer_table_path(cfg: RunConfig, layer: str, table_name: str) -> Path:
    if layer not in {"bronze", "silver", "gold"}:
        raise ValueError(f"Invalid layer '{layer}'. Must be one of: bronze, silver, gold.")
    return cfg.paths.data_dir / layer / table_name
