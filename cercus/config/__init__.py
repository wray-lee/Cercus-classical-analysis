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
import os
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

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

    def __init__(self, data: Dict[str, Any], prefix: str = ""):
        self._data = data
        self._prefix = prefix

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
        """Flatten a nested dictionary with underscore separator."""
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def __getattr__(self, name: str) -> Any:
        """Get configuration value by attribute name."""
        # Normalize the name
        name_upper = name.upper()
        name_lower = name.lower()

        # Special handling for colors section - map COLOR_XXX to xxx
        if self._prefix == "colors" and name_upper.startswith("COLOR_"):
            color_name = name_lower.replace("color_", "")
            # Try direct lookup first
            if color_name in self._data:
                return self._data[color_name]
            # Try matching with snake_case conversion
            for key in self._data:
                if key.lower() == color_name:
                    return self._data[key]
                # Handle special cases like no_response -> no_response (already snake_case)
                if key.lower().replace("_", "") == color_name.replace("_", ""):
                    return self._data[key]

        # First, try exact match in current data
        if name in self._data:
            value = self._data[name]
            return ConfigProxy(value, f"{self._prefix}.{name}") if isinstance(value, dict) else value

        # Try uppercase/lowercase variants
        for key in self._data:
            if key.upper() == name_upper:
                value = self._data[key]
                return ConfigProxy(value, f"{self._prefix}.{key}") if isinstance(value, dict) else value
            if key.lower() == name_lower:
                value = self._data[key]
                return ConfigProxy(value, f"{self._prefix}.{key}") if isinstance(value, dict) else value

        # For nested structures, try flat access pattern (ESCAPE_VMAX_THRESHOLD -> escape.vmax_threshold)
        flattened = self._flatten_dict(self._data)
        for key, value in flattened.items():
            # Try exact match
            if key == name:
                return value
            # Try uppercase match
            if key.upper() == name_upper:
                return value
            # Try section_key format (escape_vmax_threshold)
            if key.replace(".", "_").upper() == name_upper:
                return value

        # Try nested access: split by underscore and traverse
        parts = name.lower().split("_")
        if len(parts) >= 2:
            current = self._data
            for part in parts:
                found = False
                for key in list(current.keys()):
                    if key.lower() == part:
                        current = current[key]
                        found = True
                        break
                if not found:
                    break
            else:
                # Successfully traversed all parts
                return ConfigProxy(current, f"{self._prefix}.{name}") if isinstance(current, dict) else current

        raise AttributeError(f"Config key '{name}' not found in '{self._prefix}' section")

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key (dict-style access)."""
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self._data or key.upper() in [k.upper() for k in self._data] or key.lower() in [k.lower() for k in self._data]

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying dictionary."""
        return self._data.copy()

    def keys(self):
        """Return configuration keys."""
        return self._data.keys()

    def values(self):
        """Return configuration values."""
        return self._data.values()

    def items(self):
        """Return configuration items."""
        return self._data.items()

    def __repr__(self) -> str:
        return f"ConfigProxy({self._prefix or 'root'})"


class DynamicConfigProxy:
    """
    A dynamic proxy that reloads configuration on access if changed.
    Used for the main config sections that should auto-reload.
    """

    def __init__(self, manager: "ConfigManager", section: str):
        self._manager = manager
        self._section = section

    def __getattr__(self, name: str) -> Any:
        """Get attribute, reloading if necessary."""
        if self._manager.auto_reload:
            self._manager.check_and_reload()

        section_data = self._manager._data.get(self._section, {})
        proxy = ConfigProxy(section_data, self._section)
        return getattr(proxy, name)

    def __getitem__(self, key: str) -> Any:
        """Get value by key."""
        if self._manager.auto_reload:
            self._manager.check_and_reload()

        section_data = self._manager._data.get(self._section, {})
        return section_data[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        section_data = self._manager._data.get(self._section, {})
        return key in section_data

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Return section as dictionary."""
        return self._manager._data.get(self._section, {}).copy()


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
    user_config_path: Optional[Path] = None

    # Behavior settings
    auto_reload: bool = True
    cache_validated: bool = True
    _watch_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _stop_watching: threading.Event = field(default_factory=threading.Event, repr=False)

    # Internal state
    _data: Dict[str, Any] = field(default_factory=dict, repr=False)
    _file_mtimes: Dict[Path, float] = field(default_factory=dict, repr=False)
    _validated_cache: Dict[str, Any] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _initialized: bool = field(default=False, repr=False)

    # Proxies for attribute access
    _thresholds_proxy: Optional[DynamicConfigProxy] = field(default=None, repr=False)
    _geometry_proxy: Optional[DynamicConfigProxy] = field(default=None, repr=False)
    _colors_proxy: Optional[DynamicConfigProxy] = field(default=None, repr=False)
    _escape_proxy: Optional[DynamicConfigProxy] = field(default=None, repr=False)
    _trajectory_proxy: Optional[DynamicConfigProxy] = field(default=None, repr=False)
    _visualization_proxy: Optional[DynamicConfigProxy] = field(default=None, repr=False)

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
            # Start with empty config
            merged: Dict[str, Any] = {}

            # 1. Load defaults first
            merged = self._load_defaults(merged)

            # 2. Load user config (overrides defaults)
            if self.user_config_path.exists():
                merged = self._load_yaml(self.user_config_path, merged)
                self._file_mtimes[self.user_config_path] = self.user_config_path.stat().st_mtime

            self._data = merged
            self._validated_cache.clear()

            log.info(f"Configuration loaded: {len(self._data)} sections")

    def _load_defaults(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Load all default configuration files."""
        if not self.defaults_dir.exists():
            log.warning(f"Defaults directory not found: {self.defaults_dir}")
            return base

        merged = base.copy()

        for yaml_file in sorted(self.defaults_dir.glob("*.yaml")):
            section_name = yaml_file.stem
            data = self._load_yaml_file(yaml_file)

            # Merge into the section
            if section_name in merged:
                merged[section_name].update(data)
            else:
                merged[section_name] = data

            # Track file modification time
            self._file_mtimes[yaml_file] = yaml_file.stat().st_mtime
            log.debug(f"Loaded defaults: {yaml_file.name}")

        return merged

    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        """Load a single YAML file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if data else {}
        except yaml.YAMLError as e:
            log.error(f"Failed to parse YAML file {path}: {e}")
            return {}
        except Exception as e:
            log.error(f"Failed to load file {path}: {e}")
            return {}

    def _load_yaml(self, path: Path, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load YAML file and merge with base configuration."""
        data = self._load_yaml_file(path)

        if base is None:
            return data

        # Deep merge
        merged = base.copy()
        for key, value in data.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value

        return merged

    def _setup_proxies(self) -> None:
        """Set up proxy objects for attribute access."""
        self._thresholds_proxy = DynamicConfigProxy(self, "thresholds")
        self._geometry_proxy = DynamicConfigProxy(self, "geometry")
        self._colors_proxy = DynamicConfigProxy(self, "colors")
        self._escape_proxy = DynamicConfigProxy(self, "escape")
        self._trajectory_proxy = DynamicConfigProxy(self, "trajectory")
        self._visualization_proxy = DynamicConfigProxy(self, "visualization")

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
            self._stop_watching.wait(timeout=1.0)  # Check every second

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

    @property
    def thresholds(self) -> DynamicConfigProxy:
        """Access threshold configuration."""
        return self._thresholds_proxy

    @property
    def geometry(self) -> DynamicConfigProxy:
        """Access geometry configuration."""
        return self._geometry_proxy

    @property
    def colors(self) -> DynamicConfigProxy:
        """Access color configuration."""
        return self._colors_proxy

    @property
    def escape(self) -> DynamicConfigProxy:
        """Access escape detection configuration."""
        return self._escape_proxy

    @property
    def trajectory(self) -> DynamicConfigProxy:
        """Access trajectory configuration."""
        return self._trajectory_proxy

    @property
    def visualization(self) -> DynamicConfigProxy:
        """Access visualization configuration."""
        return self._visualization_proxy

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
        """Get configuration value by dotted path (e.g., 'thresholds.ESCAPE_VMAX_THRESHOLD')."""
        if self.auto_reload:
            self.check_and_reload()

        parts = key.split(".")

        # Get the section data
        section = parts[0]
        section_data = None
        for k in self._data:
            if k.lower() == section.lower():
                section_data = self._data[k]
                break

        if section_data is None:
            return default

        # If only one part, return the section proxy
        if len(parts) == 1:
            return ConfigProxy(section_data, section)

        # For nested access, create a proxy and use attribute lookup
        proxy = ConfigProxy(section_data, section)
        attr_name = ".".join(parts[1:])

        try:
            # Navigate through nested attributes
            current = proxy
            for attr in parts[1:]:
                current = getattr(current, attr)
            return current
        except AttributeError:
            return default

    def get_validated(self, section: str, model_class: Optional[type] = None) -> Optional[BaseModel]:
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

    def to_dict(self) -> Dict[str, Any]:
        """Return complete configuration as dictionary."""
        return self._data.copy()

    def __getitem__(self, key: str) -> Any:
        """Dict-style access to top-level sections."""
        if self.auto_reload:
            self.check_and_reload()
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """Check if section exists."""
        return key in self._data

    def __repr__(self) -> str:
        return f"ConfigManager(sections={list(self._data.keys())}, auto_reload={self.auto_reload})"


# ═══════════════════════════════════════════════════════════════════════
# Module-Level Singleton
# ═══════════════════════════════════════════════════════════════════════

# Global singleton instance
_config_manager: Optional[ConfigManager] = None
_config_lock = threading.Lock()


def get_config(
    project_root: Optional[Union[str, Path]] = None,
    auto_reload: bool = True
) -> ConfigManager:
    """
    Get or create the global configuration manager instance.

    Args:
        project_root: Override project root directory
        auto_reload: Enable automatic reload on file changes

    Returns:
        ConfigManager singleton instance
    """
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
    cfg = get_config()
    cfg.reload()


# Create the module-level singleton
config: ConfigManager = get_config()


# ═══════════════════════════════════════════════════════════════════════
# Convenience Functions for Constants Module
# ═══════════════════════════════════════════════════════════════════════

def get_thresholds() -> DynamicConfigProxy:
    """Get thresholds configuration proxy.

    Returns:
        DynamicConfigProxy for thresholds section

    Example:
        from cercus.config import get_thresholds
        vmax = get_thresholds().ESCAPE_VMAX_THRESHOLD
    """
    return config.thresholds


def get_geometry() -> DynamicConfigProxy:
    """Get geometry configuration proxy.

    Returns:
        DynamicConfigProxy for geometry section

    Example:
        from cercus.config import get_geometry
        radius = get_geometry().RADIUS_MM
    """
    return config.geometry


def get_colors() -> DynamicConfigProxy:
    """Get colors configuration proxy.

    Returns:
        DynamicConfigProxy for colors section

    Example:
        from cercus.config import get_colors
        escape_color = get_colors().COLOR_ESCAPE
    """
    return config.colors


# ═══════════════════════════════════════════════════════════════════════
# Legacy Exports (for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════

from cercus.config.settings import BarLabelStyle, TrajectoryConfig

__all__ = [
    # Main config object
    "config",
    "get_config",
    "reload_config",

    # Config classes
    "ConfigManager",
    "ConfigProxy",
    "DynamicConfigProxy",

    # Pydantic models
    "ThresholdsConfig",
    "GeometryConfig",
    "ColorsConfig",
    "EscapeConfig",
    "TrajectoryConfigModel",
    "VisualizationConfig",

    # Convenience functions
    "get_thresholds",
    "get_geometry",
    "get_colors",

    # Legacy
    "BarLabelStyle",
    "TrajectoryConfig",
]
