"""Derived response grouping: merge/split, unknown preservation, conservation."""
from __future__ import annotations

import pandas as pd
import pytest

from cercus.analysis.response_groups import (
    derive_response_group,
    effective_main_types,
    response_group_series,
    resolve_merge_prewalk,
    with_response_group,
)
from cercus.constants.response_types import RESPONSE_TYPES


def _labels():
    return pd.Series(["Escape", "PreEscape", "PreWalk", "NoResponse"])


def test_merge_folds_prewalk_only():
    grouped = derive_response_group(_labels(), merge_prewalk=True)
    assert list(grouped) == ["Escape", "PreEscape", "Escape", "NoResponse"]


def test_split_is_identity():
    grouped = derive_response_group(_labels(), merge_prewalk=False)
    assert list(grouped) == list(_labels())


def test_unknown_and_missing_preserved():
    labels = pd.Series(["Escape", "Weird", None])
    for flag in (True, False):
        out = derive_response_group(labels, merge_prewalk=flag)
        assert out.iloc[1] == "Weird"
        assert pd.isna(out.iloc[2])


def test_effective_registry_dynamic():
    assert "PreWalk" in effective_main_types(merge_prewalk=False)
    assert "PreWalk" not in effective_main_types(merge_prewalk=True)
    assert set(effective_main_types(merge_prewalk=False)) == set(RESPONSE_TYPES)


def test_merge_conserves_counts():
    df = pd.DataFrame({"response_type": ["Escape", "PreWalk", "PreWalk", "NoResponse"]})
    grouped = response_group_series(df, merge_prewalk=True)
    assert (grouped == "Escape").sum() == 3
    assert len(grouped) == len(df)


def test_with_response_group_keeps_raw_column():
    df = pd.DataFrame({"response_type": ["PreWalk"]})
    out = with_response_group(df, merge_prewalk=True)
    assert out["response_type"].iloc[0] == "PreWalk"  # raw untouched
    assert out["response_group"].iloc[0] == "Escape"
    assert "response_group" not in df.columns  # non-mutating


def test_override_must_be_bool():
    with pytest.raises(TypeError):
        resolve_merge_prewalk("false")
