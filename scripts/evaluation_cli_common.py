"""Shared helpers for recorded-data evaluation commands."""

import json
from pathlib import Path
from typing import Any

from carvision.evaluation import EvaluationDataset, sgbm_config_from_dict
from carvision.recorded_stereo import SGBMConfig


def load_dataset_and_config(manifest: str, config_path: str | None
                            ) -> tuple[EvaluationDataset, SGBMConfig]:
    dataset = EvaluationDataset.load(manifest)
    values: dict[str, Any] = dict(dataset.default_sgbm)
    if config_path:
        loaded = json.loads(Path(config_path).read_text())
        values.update(loaded.get("sgbm", loaded))
    return dataset, sgbm_config_from_dict(values)


def load_named_configs(path: str, dataset: EvaluationDataset) -> dict[str, SGBMConfig]:
    data = json.loads(Path(path).read_text())
    raw_configs = data.get("configurations", data)
    if not isinstance(raw_configs, dict):
        raise ValueError("comparison config must map names to SGBM parameter objects")
    base = sgbm_config_from_dict(dataset.default_sgbm)
    return {str(name): sgbm_config_from_dict(dict(values), base)
            for name, values in raw_configs.items()}

