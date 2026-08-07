"""Small shared DataFrame factories for offline tests."""

from __future__ import annotations

import pandas as pd

from dms_parser._wildtype import WildTypeResolution, set_wt_resolution


def standardized_table() -> pd.DataFrame:
    """Return the standard columns required by dataset accounting."""
    table = pd.DataFrame(
        {"status": ["OK"], "is_wildtype": [False], "is_synthetic": [False]}
    )
    set_wt_resolution(
        table,
        WildTypeResolution(
            sequence="MKT",
            score=None,
            sequence_provenance="test_fixture",
            score_provenance=None,
            observed_wildtype_row=False,
            synthetic_wildtype_inserted=False,
            score_unavailable_reason=(
                "no_observed_or_metadata_wt_score_and_no_fallback"
            ),
        ),
    )
    return table
