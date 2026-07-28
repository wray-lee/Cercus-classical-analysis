"""
Cercus Framework — Publication-Grade Visualization
===================================================
*** DEPRECATED *** — Import from ``cercus.visualization`` instead.
All functions are re-exported for backward compatibility.
"""

from __future__ import annotations

import logging
import warnings

log = logging.getLogger(__name__)

warnings.warn(
    "Import from 'pipeline.visualization' is deprecated. "
    "Use 'from cercus.visualization import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all public functions from the new location
from cercus.visualization import *  # noqa: F401, F403, E402

# Internal helpers (kept for backward compatibility)
from cercus.core.kinematics.trajectory_integration import (  # noqa: F401, E402
    body_to_traj as _body_to_traj,
    integrate_body_trajectory as _integrate_body_trajectory,
    build_angular_peak_dz_mask as _build_angular_peak_dz_mask,
)
from cercus.visualization._core import (  # noqa: F401, E402
    _add_threshold_lines,
    _draw_oscilloscope_channels,
    _draw_side_arrows,
    _draw_standardized_grid,
)
from cercus.visualization._circstats import (  # noqa: F401, E402
    rayleigh_p as _rayleigh_p,
    watson_williams_test,
)
