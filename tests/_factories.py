"""Small shared DataFrame factories for offline tests."""

from __future__ import annotations

import pandas as pd


def standardized_table() -> pd.DataFrame:
    """Return the standard columns required by dataset accounting."""
    return pd.DataFrame(
        {"status": ["OK"], "is_wildtype": [False], "is_synthetic": [False]}
    )
