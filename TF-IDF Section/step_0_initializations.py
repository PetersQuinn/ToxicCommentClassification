from __future__ import annotations

from utils.config import build_config, config_snapshot, ensure_project_dirs
from utils.io_utils import dataset_path, package_versions, runtime_metadata, save_json
from utils.progress_utils import build_step_logger


def main() -> None:
    config = build_config()
    ensure_project_dirs(config)
    logger = build_step_logger(config, "step_0")
    logger.info("Starting initialization")

    try:
        missing = [
            str(dataset_path(config, split_name))
            for split_name in ["train", "test", "test_labeled"]
            if not dataset_path(config, split_name).exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing required parquet files: {missing}")
        logger.event(
            "Validated parquet inputs",
            train=dataset_path(config, "train").name,
            test=dataset_path(config, "test").name,
            test_labeled=dataset_path(config, "test_labeled").name,
        )

        save_json(config_snapshot(config), config.logs_dir / "config_snapshot.json")
        save_json(
            runtime_metadata(config, "step_0_initializations"),
            config.logs_dir / "run_metadata.json",
        )
        save_json(
            package_versions(
                ["pandas", "numpy", "pyarrow", "sklearn", "scipy", "matplotlib", "joblib", "lightgbm"]
            ),
            config.logs_dir / "environment_snapshot.json",
        )
        logger.saved(config.logs_dir / "config_snapshot.json")
        logger.saved(config.logs_dir / "run_metadata.json")
        logger.saved(config.logs_dir / "environment_snapshot.json")
        logger.event("Initialization complete", elapsed=logger.elapsed())
    finally:
        logger.close()


if __name__ == "__main__":
    main()
