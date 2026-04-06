from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dms_parser.exceptions import FileFormatError
from dms_parser.io import (
    ensure_local_copy,
    infer_filename_from_url,
    read_table,
    write_table,
)


def test_infer_filename_from_url():
    name = infer_filename_from_url("https://example.com/data.csv?download=1")
    assert "data.csv" in name


def test_read_and_write_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "test.csv"

    write_table(df, path)
    loaded = read_table(path)

    pd.testing.assert_frame_equal(df, loaded)


def test_read_and_write_tsv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "test.tsv"

    write_table(df, path)
    loaded = read_table(path)

    pd.testing.assert_frame_equal(df, loaded)


def test_read_and_write_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "test.parquet"

    write_table(df, path)
    loaded = read_table(path)

    pd.testing.assert_frame_equal(df, loaded)


def test_read_table_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "missing.csv")


def test_read_table_unsupported_format_raises(tmp_path):
    path = tmp_path / "bad.xlsx"
    path.write_text("dummy", encoding="utf-8")

    with pytest.raises(FileFormatError):
        read_table(path)


def test_write_table_unsupported_format_raises(tmp_path):
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(FileFormatError):
        write_table(df, tmp_path / "bad.xlsx")


def test_ensure_local_copy_with_existing_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    result = ensure_local_copy(path, output_dir=tmp_path)
    assert result == path