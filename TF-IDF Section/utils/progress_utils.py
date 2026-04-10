from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO

from utils.io_utils import ensure_dir


def format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def format_key_values(**kwargs: object) -> str:
    items = [f"{key}={_format_value(value)}" for key, value in kwargs.items()]
    return " | ".join(items)


@dataclass
class StepLogger:
    step_name: str
    project_root: Path
    logs_dir: Path
    log_to_file: bool = True
    start_time: float = field(default_factory=time.perf_counter)
    log_path: Path | None = None
    handle: TextIO | None = None

    def __post_init__(self) -> None:
        if not self.log_to_file:
            return
        ensure_dir(self.logs_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.logs_dir / f"{self.step_name}_{timestamp}.log"
        self.handle = self.log_path.open("a", encoding="utf-8")

    def _prefix(self) -> str:
        return f"{datetime.now().strftime('%H:%M:%S')} [{self.step_name}]"

    def _emit(self, message: str, indent: int = 0) -> None:
        line = f"{self._prefix()} {'  ' * indent}{message}"
        print(line, flush=True)
        if self.handle is not None:
            self.handle.write(line + "\n")
            self.handle.flush()

    def relative_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root.resolve()))
        except Exception:
            return str(path)

    def info(self, message: str, indent: int = 0) -> None:
        self._emit(message, indent=indent)

    def event(self, message: str, indent: int = 0, **kwargs: object) -> None:
        if kwargs:
            message = f"{message} | {format_key_values(**kwargs)}"
        self._emit(message, indent=indent)

    def saved(self, path: Path, indent: int = 0) -> None:
        self._emit(f"Saved {self.relative_path(path)}", indent=indent)

    def elapsed(self) -> str:
        return format_elapsed(time.perf_counter() - self.start_time)

    @contextmanager
    def timed(self, message: str, indent: int = 0, **kwargs: object) -> Iterator[None]:
        self.event(f"{message} started", indent=indent, **kwargs)
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.event(
                f"{message} failed",
                indent=indent,
                elapsed=format_elapsed(time.perf_counter() - start),
                error=type(exc).__name__,
            )
            raise
        self.event(
            f"{message} finished",
            indent=indent,
            elapsed=format_elapsed(time.perf_counter() - start),
        )

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def build_step_logger(config, step_name: str, log_to_file: bool = True) -> StepLogger:
    logger = StepLogger(
        step_name=step_name,
        project_root=config.project_root,
        logs_dir=config.logs_dir,
        log_to_file=log_to_file,
    )
    if logger.log_path is not None:
        logger.event("Log file ready", log_file=logger.relative_path(logger.log_path))
    return logger
