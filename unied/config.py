# -*- coding: utf-8 -*-
"""YAML configuration loader with attribute-style access."""
from pathlib import Path

import yaml


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def _wrap(cls, obj):
        if isinstance(obj, dict):
            return cls({k: cls._wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls._wrap(v) for v in obj]
        return obj


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config._wrap(raw or {})
    cfg.config_path = str(Path(path).resolve())
    return cfg
