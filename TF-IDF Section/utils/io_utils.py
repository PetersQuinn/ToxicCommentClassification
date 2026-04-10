from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import DATA_FILES, PipelineConfig, to_serializable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataset_path(config: PipelineConfig, split_name: str) -> Path:
    return config.data_dir / DATA_FILES[split_name]


def load_parquet_frame(
    config: PipelineConfig,
    split_name: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    return pd.read_parquet(dataset_path(config, split_name), columns=columns)


def choose_text_column(frame: pd.DataFrame, priority: list[str]) -> str:
    for column in priority:
        if column not in frame.columns:
            continue
        series = frame[column].fillna("").astype(str).str.strip()
        if series.ne("").mean() >= 0.95:
            return column
    raise ValueError("No usable TF-IDF text column was found.")


def coerce_binary_labels(frame: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in label_cols:
        output[column] = output[column].fillna(0).astype(int)
    return output


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(data), handle, indent=2, sort_keys=True)


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_frame(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=index)
        return
    frame.to_csv(path, index=index)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def format_storage_size(num_bytes: int) -> str:
    value = float(max(num_bytes, 0))
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.2f} {units[unit_index]}"


def remove_file(path: Path) -> int:
    if not path.exists():
        return 0
    size = int(path.stat().st_size)
    path.unlink()
    return size


def package_versions(package_names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            module = importlib.import_module(package_name)
            versions[package_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[package_name] = f"unavailable: {exc}"
    return versions


def runtime_metadata(config: PipelineConfig, step_name: str) -> dict[str, Any]:
    return {
        "step_name": step_name,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python_executable": sys.executable,
        "project_root": str(config.project_root),
        "data_dir": str(config.data_dir),
        "outputs_dir": str(config.outputs_dir),
        "data_files": DATA_FILES,
    }
