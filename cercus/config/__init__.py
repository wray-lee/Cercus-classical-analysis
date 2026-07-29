"""
Cercus Framework — Configuration Loader
=========================================

A unified configuration system that:
- Auto-loads project root config.yaml (if exists)
- Merges user config with defaults (defaults/*.yaml)
- Provides attribute-style access: config.thresholds.ESCAPE_VMAX_THRESHOLD
- Supports hot-reload with file watching
- Validates config with Pydantic models

Usage:
    from cercus.config import config

    # Access via attributes
    vmax = config.thresholds.ESCAPE_VMAX_THRESHOLD
    radius = config.geometry.RADIUS_MM
    escape_color = config.colors.COLOR_ESCAPE

    # Force reload
    config.reload()

    # Check if config changed
    if config.has_changed():
        config.reload()

The config object is a singleton - importing it from anywhere returns the same instance.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Pydantic Models for Configuration Validation
# ═══════════════════════════════════════════════════════════════════════

class ThresholdsConfig(BaseModel):
    """Threshold configuration for escape detection."""
    ESCAPE_VMAX_THRESHOLD: float = Field(default=98.0, gt=0, description="Burst floor for valid escape (mm/s)")
    ESCAPE_START_THRESHOLD: float = Field(default=10.0, gt=0, description="Latency onset anchor (mm/s)")
    PREWALK_THRESHOLD: float = Field(default=10.0, gt=0, description="Pre-stimulus spontaneous activity threshold (mm/s)")
    PREWALK_WINDOW_MS: float = Field(default=1000.0, gt=0, description="Pre-stimulus validation window (ms)")
    POST_STIM_BUFFER_MS: float = Field(default=50.0, ge=0, description="Post-stimulus tail buffer (ms)")
    ESCAPE_WINDOW_MS: float = Field(default=250.0, gt=0, description="Post-stimulus burst detection window (ms)")


class GeometryConfig(BaseModel):
    """Geometry and timing configuration."""
    RADIUS_MM: float = Field(default=30.0, gt=0, description="Arena radius in mm")
    TRAJECTORY_MAX_RADIUS_MM: float = Field(default=120.0, gt=0, description="Max trajectory plot radius (mm)")
    TRAJECTORY_STEP_MM: float = Field(default=10.0, gt=0, description="Trajectory ring step size (mm)")
    SPEED_WINDOW_MS: float = Field(default=100.0, gt=0, description="Speed calculation window (ms)")
    LEGACY_TRIAL_DURATION_MS: float = Field(default=5829.6, gt=0, description="Legacy trial duration (ms)")


class ColorsConfig(BaseModel):
    """Color palette configuration."""
    COLOR_ESCAPE: str = Field(default="#ED0000", pattern=r"^#[0-9A-Fa-f]{6}$")
    COLOR_PREWALK: str = Field(default="#00468B", pattern=r"^#[0-9A-Fa-f]{6}$")
    COLOR_NO_RESPONSE: str = Field(default="#7C878E", pattern=r"^#[0-9A-Fa-f]{6}$")
    COLOR_LEFT: str = Field(default="#00468B", pattern=r"^#[0-9A-Fa-f]{6}$")
    COLOR_RIGHT: str = Field(default="#ED0000", pattern=r"^#[0-9A-Fa-f]{6}$")
    COLOR_CONTROL: str = Field(default="#7C878E", pattern=r"^#[0-9A-Fa-f]{6}$")
    COLOR_OSCI_VIS: str = Field(default="#ADB6B6", pattern=r"^#[0-9A-Fa-f]{6}$")
    COLOR_OSCI_HW: str = Field(default="#E69F00", pattern=r"^#[0-9A-Fa-f]{6}$")
    COLOR_WITH_STILLNESS: str = Field(default="#00468B", pattern=r"^#[0-9A-Fa-f]{6}$")
    COLOR_NO_STILLNESS: str = Field(default="#5B6770", pattern=r"^#[0-9A-Fa-f]{6}$")
    NPG_PALETTE: List[str] = Field(
        default=[
            "#E69F00", "#56B4E9", "#009E73", "#F0E442",
            "#0072B2", "#D55E00", "#CC79A7", "#999999"
        ]
    )


class EscapeConfig(BaseModel):
    """Escape detection configuration."""
    start_threshold: float = Field(default=10.0, gt=0)
    vmax_threshold: float = Field(default=98.0, gt=0)
    window_ms: float = Field(default=250.0, gt=0)


class TrajectoryConfigModel(BaseModel):
    """Trajectory rendering configuration."""
    use_z_degree_to_draw: bool = True
    use_escape_onset_heading: bool = True
    use_escape_onset_only_xy: bool = True
    use_angular_velocity_offset: bool = False
    use_rigid_rotation: bool = False
    dz_integration_range: str = "escape_interval"


class VisualizationConfig(BaseModel):
    """Visualization style configuration."""
    bar_label_style: str = "inline"


# ═══════════════════════════════════════════════════════════════════════
# Configuration Proxy Classes (for attribute access)
# ═══════════════════════════════════════════════════════════════════════

class ConfigProxy:
    """
    A proxy object that provides attribute-style access to configuration values.
    Supports both flat and nested access patterns.

    Usage:
        thresholds = ConfigProxy(config_dict, 'thresholds')
        # Flat access: thresholds.ESCAPE_VMAX_THRESHOLD
        # Nested access: thresholds.escape.vmax_threshold
    """

    def __init__(self, data: dict[str, Any], prefix: str = ""):
        self._data = data
        self._prefix = prefix
        self._lower_map: dict[str, str] = {k.lower(): k for k in data}

    def _find_key(self, name: str) -> str | None:
        """Case-insensitive key lookup, returns actual key or None."""
        if name in self._data:
            return name
        lower = name.lower()
        return self._lower_map.get(lower)

    def __getattr__(self, name: str) -> Any:
        """Get configuration value by attribute name."""
        data = self._data
        name_lower = name.lower()

        # Special handling for colors: COLOR_XXX → xxx
        if self._prefix == "colors" and name_lower.startswith("color_"):
            bare = name_lower[6:]  # strip "color_"
            key = self._find_key(bare)
            if key is not None:
                return data[key]

        # Direct or case-insensitive match
        key = self._find_key(name)
        if key is not None:
            value = data[key]
            return ConfigProxy(value, f"{self._prefix}.{key}") if isinstance(value, dict) else value

        # Flatten nested dict and try underscore-joined lookup
        for k, v in data.items():
            if isinstance(v, dict):
                for inner_k, inner_v in v.items():
                    flat_key = f"{k}_{inner_k}".lower()
                    if flat_key == name_lower:
                        return inner_v

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

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __repr__(self) -> str:
        return f"ConfigProxy({self._prefix or 'root'})"


class DynamicConfigProxy:
    """Dynamic proxy that reloads configuration on access if changed."""

    def __init__(self, manager: "ConfigManager", section: str):
        self._manager = manager
        self._section = section

    def _section_data(self) -> dict[str, Any]:
        if self._manager.auto_reload:
            self._manager.check_and_reload()
        return self._manager._data.get(self._section, {})

    def __getattr__(self, name: str) -> Any:
        proxy = ConfigProxy(self._section_data(), self._section)
        return getattr(proxy, name)

    def __getitem__(self, key: str) -> Any:
        return self._section_data()[key]

    def __contains__(self, key: str) -> bool:
        return key in self._section_data()

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return self._section_data().copy()


# ═══════════════════════════════════════════════════════════════════════
# Configuration Manager
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ConfigManager:
    """
    Centralized configuration manager with auto-loading, merging, and hot-reload.

    Features:
    - Auto-discovers project root config.yaml
    - Merges user config with defaults
    - File watching for hot-reload
    - Pydantic validation
    - Thread-safe singleton

    Usage:
        from cercus.config import config

        # Access values
        vmax = config.thresholds.ESCAPE_VMAX_THRESHOLD

        # Check for changes
        if config.has_changed():
            config.reload()
    """

    # Configuration sources
    project_root: Path = field(default_factory=lambda: Path.cwd())
    defaults_dir: Path = field(default_factory=lambda: Path(__file__).parent / "defaults")
    user_config_path: Path | None = None

    # Behavior settings
    auto_reload: bool = True
    cache_validated: bool = True
    _watch_thread: threading.Thread | None = field(default=None, repr=False)
    _stop_watching: threading.Event = field(default_factory=threading.Event, repr=False)

    # Internal state
    _data: dict[str, Any] = field(default_factory=dict, repr=False)
    _file_mtimes: dict[Path, float] = field(default_factory=dict, repr=False)
    _validated_cache: dict[str, Any] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _initialized: bool = field(default=False, repr=False)

    # Proxies for attribute access
    _proxies: dict[str, DynamicConfigProxy] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Initialize the configuration manager."""
        self._discover_project_root()
        self._setup_paths()
        self._load_all()
        self._setup_proxies()
        self._initialized = True

        if self.auto_reload:
            self._start_watching()

    def _discover_project_root(self) -> None:
        """Discover project root by looking for config.yaml or git repository."""
        # Start from current working directory and traverse up
        current = Path.cwd().resolve()

        for parent in [current] + list(current.parents):
            # Check for config.yaml
            if (parent / "config.yaml").exists():
                self.project_root = parent
                log.debug(f"Found project root (config.yaml): {parent}")
                return

            # Check for .git
            if (parent / ".git").exists():
                self.project_root = parent
                log.debug(f"Found project root (.git): {parent}")
                return

        # Fallback to cwd
        self.project_root = current
        log.debug(f"Using current directory as project root: {current}")

    def _setup_paths(self) -> None:
        """Set up configuration file paths."""
        if self.user_config_path is None:
            self.user_config_path = self.project_root / "config.yaml"

        # Resolve defaults directory relative to project root
        if not self.defaults_dir.is_absolute():
            self.defaults_dir = self.project_root / self.defaults_dir

    def _load_all(self) -> None:
        """Load all configuration sources and merge them."""
        with self._lock:
            merged: dict[str, Any] = {}
            merged = self._load_defaults(merged)
            if self.user_config_path.exists():
                merged = self._load_yaml(self.user_config_path, merged)
                self._file_mtimes[self.user_config_path] = self.user_config_path.stat().st_mtime
            self._data = merged
            self._validated_cache.clear()
            log.info(f"Configuration loaded: {len(self._data)} sections")

    def _load_defaults(self, base: dict[str, Any]) -> dict[str, Any]:
        """Load all default configuration files."""
        if not self.defaults_dir.exists():
            log.warning(f"Defaults directory not found: {self.defaults_dir}")
            return base

        merged = base.copy()
        for yaml_file in sorted(self.defaults_dir.glob("*.yaml")):
            data = self._load_yaml_file(yaml_file)
            section = yaml_file.stem
            if section in merged and isinstance(merged[section], dict):
                merged[section].update(data)
            else:
                merged[section] = data
            self._file_mtimes[yaml_file] = yaml_file.stat().st_mtime
            log.debug(f"Loaded defaults: {yaml_file.name}")
        return merged

    def _load_yaml_file(self, path: Path) -> dict[str, Any]:
        """Load a single YAML file."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            log.error(f"Failed to load {path}: {e}")
            return {}

    def _load_yaml(self, path: Path, base: dict[str, Any] | None = None) -> dict[str, Any]:
        """Load YAML file and merge with base configuration."""
        data = self._load_yaml_file(path)
        if base is None:
            return data
        merged = base.copy()
        for key, value in data.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
        return merged

    def _setup_proxies(self) -> None:
        """Set up proxy objects for attribute access."""
        for name in ("thresholds", "geometry", "colors", "escape", "trajectory", "visualization"):
            self._proxies[name] = DynamicConfigProxy(self, name)

    def __getattr__(self, name: str) -> Any:
        if name in self._proxies:
            return self._proxies[name]
        raise AttributeError(f"ConfigManager has no section '{name}'")

    def _start_watching(self) -> None:
        """Start background thread for file watching."""
        if self._watch_thread is not None and self._watch_thread.is_alive():
            return

        self._stop_watching.clear()
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        log.debug("Started configuration file watcher")

    def _watch_loop(self) -> None:
        """Background loop to watch for file changes."""
        while not self._stop_watching.is_set():
            self._stop_watching.wait(timeout=1.0)
            if self.has_changed():
                log.info("Configuration files changed, reloading...")
                try:
                    self.reload()
                except Exception as e:
                    log.error(f"Failed to reload configuration: {e}")

    def _stop_watching_thread(self) -> None:
        """Stop the file watching thread."""
        if self._watch_thread is not None:
            self._stop_watching.set()
            self._watch_thread.join(timeout=2.0)

    # ═══════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════

    def reload(self) -> None:
        """Force reload all configuration files."""
        with self._lock:
            self._load_all()
            log.info("Configuration reloaded successfully")

    def check_and_reload(self) -> bool:
        """Check if files changed and reload if necessary."""
        if self.has_changed():
            self.reload()
            return True
        return False

    def has_changed(self) -> bool:
        """Check if any configuration file has been modified."""
        with self._lock:
            for path, last_mtime in self._file_mtimes.items():
                if not path.exists():
                    return True

                current_mtime = path.stat().st_mtime
                if current_mtime != last_mtime:
                    return True

            # Check if user config was added
            if self.user_config_path.exists() and self.user_config_path not in self._file_mtimes:
                return True

            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by dotted path (e.g. 'thresholds.ESCAPE_VMAX_THRESHOLD')."""
        if self.auto_reload:
            self.check_and_reload()
        parts = key.split(".")
        section_data = None
        for k in self._data:
            if k.lower() == parts[0].lower():
                section_data = self._data[k]
                break
        if section_data is None:
            return default
        proxy = ConfigProxy(section_data, parts[0])
        try:
            current = proxy
            for attr in parts[1:]:
                current = getattr(current, attr)
            return current
        except AttributeError:
            return default

    def get_validated(self, section: str, model_class: type | None = None) -> BaseModel | None:
        """Get validated configuration section as Pydantic model."""
        if self.auto_reload:
            self.check_and_reload()

        # Use cache if available
        if self.cache_validated and section in self._validated_cache:
            return self._validated_cache[section]

        # Map section names to models
        model_map = {
            "thresholds": ThresholdsConfig,
            "geometry": GeometryConfig,
            "colors": ColorsConfig,
            "escape": EscapeConfig,
            "trajectory": TrajectoryConfigModel,
            "visualization": VisualizationConfig,
        }

        model_class = model_class or model_map.get(section)
        if model_class is None:
            raise ValueError(f"No model class available for section: {section}")

        section_data = self._data.get(section, {})

        try:
            validated = model_class(**section_data)
            if self.cache_validated:
                self._validated_cache[section] = validated
            return validated
        except ValidationError as e:
            log.error(f"Configuration validation error in '{section}': {e}")
            # Return default model
            return model_class()

    def to_dict(self) -> dict[str, Any]:
        return self._data.copy()

    def __getitem__(self, key: str) -> Any:
        if self.auto_reload:
            self.check_and_reload()
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"ConfigManager(sections={list(self._data.keys())}, auto_reload={self.auto_reload})"


# ═══════════════════════════════════════════════════════════════════════
# Module-Level Singleton
# ═══════════════════════════════════════════════════════════════════════

# Global singleton instance
_config_manager: ConfigManager | None = None
_config_lock = threading.Lock()


def get_config(
    project_root: str | Path | None = None,
    auto_reload: bool = True
) -> ConfigManager:
    """Get or create the global configuration manager singleton."""
    global _config_manager
    with _config_lock:
        if _config_manager is None:
            kwargs = {"auto_reload": auto_reload}
            if project_root is not None:
                kwargs["project_root"] = Path(project_root)
            _config_manager = ConfigManager(**kwargs)
    return _config_manager


def reload_config() -> None:
    """Force reload the global configuration."""
    get_config().reload()


config: ConfigManager = get_config()


# ═══════════════════════════════════════════════════════════════════════
# Convenience Functions for Constants Module
# ═══════════════════════════════════════════════════════════════════════

def get_thresholds() -> DynamicConfigProxy:
    return config.thresholds

def get_geometry() -> DynamicConfigProxy:
    return config.geometry

def get_colors() -> DynamicConfigProxy:
    return config.colors


# ═══════════════════════════════════════════════════════════════════════
# Legacy Exports (for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════

from cercus.config.settings import BarLabelStyle, TrajectoryConfig, ComparisonConfig, UnifiedPreset  # noqa: E402, F401

__all__ = [
    "config", "get_config", "reload_config",
    "get_thresholds", "get_geometry", "get_colors",
    "BarLabelStyle", "TrajectoryConfig", "ComparisonConfig", "UnifiedPreset",
]
