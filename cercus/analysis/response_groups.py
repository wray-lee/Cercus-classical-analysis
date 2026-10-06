"""
Cercus Framework — Raw-to-Derived Response Grouping
===================================================
The raw classifier emits four labels (``response_type``); the merged main view
folds raw ``PreWalk`` into ``Escape`` while keeping ``PreEscape`` and
``NoResponse`` separate. ``response_group`` is *derived* — the raw
``response_type`` column and every raw registry stay untouched.

Toggle: ``analysis.response_grouping.merge_prewalk`` (default true). When false
``response_group`` mirrors ``response_type`` exactly, and the effective main
class registry/order reverts to the raw ``RESPONSE_TYPES`` (split ``PreWalk``
kept). Every helper takes an optional explicit ``merge_prewalk`` override for
tests; the effective class list/order is always dynamic in the flag, never a
static always-merged registry.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cercus.constants.response_types import RESPONSE_COLORS, RESPONSE_TYPES

#: Raw labels that fold into the derived ``Escape`` group when merging is on.
_PREWALK = "PreWalk"
_ESCAPE = "Escape"


def read_merge_prewalk() -> bool:
    """Effective merge flag from YAML; a non-bool value is a hard config error."""
    from cercus.config import get_config

    value: Any = get_config().analysis.response_grouping.merge_prewalk
    if not isinstance(value, bool):
        raise TypeError(
            "analysis.response_grouping.merge_prewalk must be a boolean, "
            f"got {type(value).__name__} ({value!r})"
        )
    return value


def resolve_merge_prewalk(merge_prewalk: bool | None = None) -> bool:
    """Explicit override for tests (must be a real bool), else the YAML value."""
    if merge_prewalk is None:
        return read_merge_prewalk()
    if not isinstance(merge_prewalk, bool):
        raise TypeError(f"merge_prewalk override must be a boolean, got {merge_prewalk!r}")
    return merge_prewalk


def derive_response_group(
    labels: pd.Series,
    merge_prewalk: bool | None = None,
) -> pd.Series:
    """Map raw ``response_type`` values to their derived group label.

    Unknown/missing labels are preserved verbatim — never coerced to
    ``NoResponse``. Merging only relabels raw ``PreWalk`` → ``Escape``.
    """
    if not resolve_merge_prewalk(merge_prewalk):
        return labels.copy()
    return labels.replace({_PREWALK: _ESCAPE})


def response_group_series(
    df: pd.DataFrame,
    merge_prewalk: bool | None = None,
) -> pd.Series:
    """``response_group`` for *df*: derived from raw ``response_type``.

    Deriving from the raw column keeps a stale cached group from surviving a
    toggle flip, and is the common selector used by every plot helper.
    """
    if "response_type" not in df.columns:
        if "response_group" in df.columns:
            return df["response_group"]
        raise KeyError("DataFrame has neither 'response_type' nor 'response_group'")
    return derive_response_group(df["response_type"], merge_prewalk)


def with_response_group(
    df: pd.DataFrame,
    merge_prewalk: bool | None = None,
) -> pd.DataFrame:
    """Return *df* with an up-to-date derived ``response_group`` column.

    Additive and non-mutating: the raw ``response_type`` column is preserved.
    The group is always re-derived from ``response_type`` so a stale column
    (e.g. after a ``merge_prewalk`` toggle) can never survive.
    """
    out = df.copy()
    out["response_group"] = response_group_series(df, merge_prewalk)
    return out


def effective_main_types(merge_prewalk: bool | None = None) -> tuple[str, ...]:
    """Display/priority order of the main view's classes for the current flag.

    Merged: raw registry minus ``PreWalk`` (it folded into ``Escape``).
    Split: the raw ``RESPONSE_TYPES`` unchanged.
    """
    if not resolve_merge_prewalk(merge_prewalk):
        return tuple(RESPONSE_TYPES)
    return tuple(rt for rt in RESPONSE_TYPES if rt != _PREWALK)


def effective_main_colors(merge_prewalk: bool | None = None) -> dict[str, str]:
    """Color map for :func:`effective_main_types`."""
    return {rt: RESPONSE_COLORS[rt] for rt in effective_main_types(merge_prewalk)}


def main_classes_present(df: pd.DataFrame, merge_prewalk: bool | None = None) -> list[str]:
    """Effective main registry order, restricted to groups actually observed."""
    groups = set(response_group_series(df, merge_prewalk).dropna().unique())
    return [rt for rt in effective_main_types(merge_prewalk) if rt in groups]


__all__ = [
    "read_merge_prewalk",
    "resolve_merge_prewalk",
    "derive_response_group",
    "response_group_series",
    "with_response_group",
    "effective_main_types",
    "effective_main_colors",
    "main_classes_present",
]
