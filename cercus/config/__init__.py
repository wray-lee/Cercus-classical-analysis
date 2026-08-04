"""
Cercus Framework — Configuration Loader
=========================================

A unified configuration system that:
- Auto-loads project root config.yaml (if exists)
- Merges user config with defaults (defaults/*.yaml)
- Provides attribute-style access: config.thresholds.escape.start_threshold

Usage:
    from cercus.config import config

    # Access via attributes
    vmax = config.thresholds.escape.start_threshold
    radius = config.geometry.RADIUS_MM
    escape_color = config.colors.escape

    # Force reload
    reload_config()
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_DEFAULT_SECTIONS = (
    "thresholds", "geometry", "colors", "escape", "trajectory", "visualization", "analysis",
)


class ConfigProxy:
    """Attribute-style access to a config dict.

    - exact key lookup (case-insensitive)
    - nested dicts are wrapped in a nested ``ConfigProxy``

    Missing keys raise ``AttributeError``.
    """

    def __init__(self, data: dict[str, Any], prefix: str = ""):
        self._data = data
        self._prefix = prefix
        self._lower_map: dict[str, str] = {k.lower(): k for k in data}

    def _find_key(self, name: str) -> str | None:
        if name in self._data:
            return name
        return self._lower_map.get(name.lower())

    def __getattr__(self, name: str) -> Any:
        key = self._find_key(name)
        if key is not None:
            value = self._data[key]
            return ConfigProxy(value, f"{self._prefix}.{key}") if isinstance(value, dict) else value
        raise AttributeError(f"Config key '{name}' not found in '{self._prefix}' section")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data or key.lower() in self._lower_map

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return self._data.copy()

    def __repr__(self) -> str:
        return f"ConfigProxy({self._prefix or 'root'})"


class ConfigManager:
    """Loads ``defaults/*.yaml`` and merges the root ``config.yaml`` on top.

    Attribute access on the manager returns a :class:`ConfigProxy` for each
    config section (``config.thresholds``, ``config.geometry``, ...).
    """

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or self._discover_project_root()
        self.defaults_dir = Path(__file__).resolve().parent / "defaults"
        self.user_config_path = self.project_root / "config.yaml"
        self._data: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._load_all()

    @staticmethod
    def _discover_project_root() -> Path:
        """Nearest ancestor of cwd that holds a config.yaml (else .git, else cwd)."""
        current = Path.cwd().resolve()
        for parent in [current, *current.parents]:
            if (parent / "config.yaml").exists() or (parent / ".git").exists():
                return parent
        return current

    def _load_all(self) -> None:
        with self._lock:
            merged: dict[str, Any] = {}
            if self.defaults_dir.is_dir():
                for yaml_file in sorted(self.defaults_dir.glob("*.yaml")):
                    data = self._load_yaml_file(yaml_file)
                    section = yaml_file.stem
                    if section in merged and isinstance(merged[section], dict):
                        merged[section].update(data)
                    else:
                        merged[section] = data
            if self.user_config_path.exists():
                merged = self._merge(merged, self._load_yaml_file(self.user_config_path))
            self._data = merged
            log.info("Configuration loaded: %d sections", len(self._data))

    @staticmethod
    def _load_yaml_file(path: Path) -> dict[str, Any]:
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:  # noqa: BLE001 - config must never crash the pipeline
            log.warning("Failed to load %s: %s — using defaults.", path, exc)
            return {}

    @staticmethod
    def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        """Deep-merge *overlay* into a copy of *base* (dict values merged per key)."""
        merged = base.copy()
        for key, value in overlay.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
        return merged

    def __getattr__(self, name: str) -> ConfigProxy:
        if name in _DEFAULT_SECTIONS:
            return ConfigProxy(self._data.get(name, {}), name)
        raise AttributeError(f"ConfigManager has no section '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def reload(self) -> None:
        """Force reload all configuration files."""
        self._load_all()
        log.info("Configuration reloaded successfully")

    def to_dict(self) -> dict[str, Any]:
        return self._data.copy()

    def __repr__(self) -> str:
        return f"ConfigManager(sections={list(self._data.keys())})"


# ═══════════════════════════════════════════════════════════════════════
# Module-Level Singleton
# ═══════════════════════════════════════════════════════════════════════

_config_manager: ConfigManager | None = None
_config_lock = threading.Lock()


def get_config(project_root: str | Path | None = None) -> ConfigManager:
    """Get or create the global configuration manager singleton."""
    global _config_manager
    with _config_lock:
        if _config_manager is None:
            kwargs = {"project_root": Path(project_root)} if project_root is not None else {}
            _config_manager = ConfigManager(**kwargs)
    return _config_manager


def reload_config() -> None:
    """Force reload the global configuration."""
    get_config().reload()


config: ConfigManager = get_config()


def get_thresholds() -> ConfigProxy:
    return config.thresholds


def get_geometry() -> ConfigProxy:
    return config.geometry


def get_colors() -> ConfigProxy:
    return config.colors


def get_analysis() -> ConfigProxy:
    return config.analysis


def get_visualization() -> ConfigProxy:
    return config.visualization


# ═══════════════════════════════════════════════════════════════════════
# Legacy Exports (for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════

from cercus.config.settings import BarLabelStyle, TrajectoryConfig  # noqa: E402, F401

__all__ = [
    "config", "get_config", "reload_config",
    "get_thresholds", "get_geometry", "get_colors", "get_analysis", "get_visualization",
    "BarLabelStyle", "TrajectoryConfig",
]
