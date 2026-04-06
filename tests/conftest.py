from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def wt_sequence() -> str:
    return "MKTAYIAKQRQISFVKSHFSRQDILDLWQ"


@pytest.fixture
def simple_variant_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": ["WT", "M1A", "K2R", "T3Y"],
            "score_raw": [1.0, 0.8, 1.2, 0.1],
            "is_wildtype": [True, False, False, False],
            "n_mutations": [0, 1, 1, 1],
            "status": ["OK", "OK", "OK", "OK"],
        }
    )


@pytest.fixture
def proteingym_like_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": ["WT", "M1A", "K2R", "T3Y"],
            "DMS_score": [1.0, 0.8, 1.2, 0.1],
        }
    )


@pytest.fixture
def mavedb_like_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hgvs_pro": ["p.Met1Ala", "p.Lys2Arg", "p.Thr3Tyr"],
            "score": [0.8, 1.2, 0.1],
        }
    )


@pytest.fixture
def standardized_dataset(wt_sequence: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": ["toy"] * 4,
            "source": ["proteingym"] * 4,
            "protein_id": ["P1"] * 4,
            "gene": ["GENE1"] * 4,
            "uniprot_id": ["U1"] * 4,
            "wt_sequence": [wt_sequence] * 4,
            "variant": ["", "M1A", "K2R", "T3Y"],
            "mutated_sequence": [
                wt_sequence,
                "AKTAYIAKQRQISFVKSHFSRQDILDLWQ",
                "MRTAYIAKQRQISFVKSHFSRQDILDLWQ",
                "MKYAYIAKQRQISFVKSHFSRQDILDLWQ",
            ],
            "n_mutations": [0, 1, 1, 1],
            "is_wildtype": [True, False, False, False],
            "score_raw": [1.0, 0.8, 1.2, 0.1],
            "status": ["OK", "OK", "OK", "OK"],
            "error": ["", "", "", ""],
        }
    )