from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Dict[str, Path]) -> None:
    for path in paths.values():
        ensure_dir(path)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(text: str, path: Path) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension for {path}")
