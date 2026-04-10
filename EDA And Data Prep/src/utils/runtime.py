from __future__ import annotations

import importlib
from typing import Iterable


def check_dependencies(modules: Iterable[str]) -> None:
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required packages: {joined}. Install requirements.txt before running this script."
        )
