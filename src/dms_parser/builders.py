import pandas as pd


def build_mavedb_dataset(*args, **kwargs) -> pd.DataFrame:
    """Download, parse, validate, and standardize a MaveDB dataset."""
    raise NotImplementedError


def build_proteingym_dataset(*args, **kwargs) -> pd.DataFrame:
    """Download, parse, validate, and standardize a ProteinGym dataset."""
    raise NotImplementedError