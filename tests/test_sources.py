from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import dmsroute.sources as sources_module
import dmsroute.sources._dataset_adapter as adapter_module
import dmsroute.sources.mavedb as mavedb_module
import dmsroute.sources.proteingym as proteingym_module
from dmsroute.acquisition.cache import FilesystemCache
from dmsroute.core.exceptions import InvalidCacheEntryError

SOURCE_ADAPTERS = (
    (
        mavedb_module,
        "mavedb",
        "MaveDB",
        "mavedb_dataset.csv",
        "./data/mavedb",
    ),
    (
        proteingym_module,
        "proteingym",
        "ProteinGym",
        "proteingym_dataset.csv",
        "./data/proteingym",
    ),
)


def test_public_source_adapter_functions_remain_importable() -> None:
    names = (
        ("download_mavedb_dataset", mavedb_module.download_mavedb_dataset),
        ("load_mavedb_dataset", mavedb_module.load_mavedb_dataset),
        ("load_mavedb_from_url", mavedb_module.load_mavedb_from_url),
        ("ensure_local_mavedb_copy", mavedb_module.ensure_local_copy),
        (
            "download_proteingym_dataset",
            proteingym_module.download_proteingym_dataset,
        ),
        ("load_proteingym_dataset", proteingym_module.load_proteingym_dataset),
        ("load_proteingym_from_url", proteingym_module.load_proteingym_from_url),
        ("ensure_local_proteingym_copy", proteingym_module.ensure_local_copy),
    )
    for name, implementation in names:
        assert getattr(sources_module, name) is implementation


@pytest.mark.parametrize(
    ("source_module", "source", "source_label", "default_filename", "output_dir"),
    SOURCE_ADAPTERS,
)
def test_download_wrapper_forwards_source_identity_defaults_and_options(
    source_module: Any,
    source: str,
    source_label: str,
    default_filename: str,
    output_dir: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    expected = tmp_path / "artifact.csv"

    def download(*args: Any, **kwargs: Any) -> Path:
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(source_module, "download_source_dataset", download)
    public_download = getattr(source_module, f"download_{source}_dataset")

    assert public_download("https://example.test/data.csv") == expected
    cache = FilesystemCache(tmp_path / "cache")
    assert public_download(
        "https://example.test/data.csv",
        output_dir=tmp_path,
        filename="chosen.csv",
        cache=cache,
        dataset_id="dataset-1",
        refresh=True,
    ) == expected

    assert calls[0][0] == ("https://example.test/data.csv", output_dir, None)
    assert calls[0][1] == {
        "overwrite": False,
        "cache": None,
        "dataset_id": None,
        "refresh": False,
        "source": source,
        "source_label": source_label,
        "default_filename": default_filename,
        "logger": source_module.logger,
    }
    assert calls[1][0] == (
        "https://example.test/data.csv",
        tmp_path,
        "chosen.csv",
    )
    assert calls[1][1]["cache"] is cache
    assert calls[1][1]["dataset_id"] == "dataset-1"
    assert calls[1][1]["refresh"] is True


def test_shared_download_dispatches_direct_and_cached_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    direct_path = tmp_path / "chosen.csv"
    cached_path = tmp_path / "cached.csv"

    monkeypatch.setattr(
        adapter_module,
        "download_file",
        lambda *args, **kwargs: calls.append(("direct", args, kwargs))
        or direct_path,
    )
    monkeypatch.setattr(
        adapter_module,
        "fetch_to_cache",
        lambda *args, **kwargs: calls.append(("cached", args, kwargs))
        or cached_path,
    )
    common = {
        "dataset_id": None,
        "refresh": False,
        "source": "mavedb",
        "source_label": "MaveDB",
        "default_filename": "mavedb_dataset.csv",
        "logger": mavedb_module.logger,
    }

    assert adapter_module.download_source_dataset(
        "https://example.test/data.csv",
        tmp_path,
        "chosen.csv",
        overwrite=True,
        cache=None,
        **common,
    ) == direct_path
    cache = FilesystemCache(tmp_path / "cache")
    assert adapter_module.download_source_dataset(
        "https://example.test/data.csv",
        tmp_path,
        None,
        overwrite=False,
        cache=cache,
        **{**common, "dataset_id": "dataset-1"},
    ) == cached_path

    assert [kind for kind, _, _ in calls] == ["direct", "cached"]
    assert calls[0][1] == ("https://example.test/data.csv", direct_path)
    assert calls[0][2] == {"overwrite": True}
    assert calls[1][2] == {
        "source": "mavedb",
        "dataset_id": "dataset-1",
        "cache": cache,
        "refresh": False,
    }


@pytest.mark.parametrize(
    ("cache", "dataset_id", "overwrite", "refresh", "message"),
    [
        ("cache", None, False, False, "dataset_id is required"),
        ("cache", "dataset-1", True, False, "overwrite cannot"),
        (None, None, False, True, "refresh requires a cache"),
    ],
)
def test_shared_download_rejects_incompatible_options(
    cache: str | None,
    dataset_id: str | None,
    overwrite: bool,
    refresh: bool,
    message: str,
    tmp_path: Path,
) -> None:
    resolved_cache = FilesystemCache(tmp_path / "cache") if cache else None
    with pytest.raises(InvalidCacheEntryError, match=message):
        adapter_module.download_source_dataset(
            "https://example.test/data.csv",
            tmp_path,
            None,
            overwrite=overwrite,
            cache=resolved_cache,
            dataset_id=dataset_id,
            refresh=refresh,
            source="mavedb",
            source_label="MaveDB",
            default_filename="mavedb_dataset.csv",
            logger=mavedb_module.logger,
        )


@pytest.mark.parametrize(
    ("source_module", "source", "_label", "default_filename", "output_dir"),
    SOURCE_ADAPTERS,
)
def test_load_and_local_copy_wrappers_preserve_source_defaults(
    source_module: Any,
    source: str,
    _label: str,
    default_filename: str,
    output_dir: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pd.DataFrame({"value": [1]})
    calls: dict[str, Any] = {}

    def read(path: str | Path, *, sep: str | None = None, **kwargs: Any):
        calls["read"] = (path, sep, kwargs)
        return table

    def ensure(path_or_url: str | Path, **kwargs: Any) -> Path:
        calls["ensure"] = (path_or_url, kwargs)
        return tmp_path / "local.csv"

    monkeypatch.setattr(source_module, "read_table", read)
    monkeypatch.setattr(source_module, "ensure_local_dataset_copy", ensure)

    assert getattr(source_module, f"load_{source}_dataset")(
        "table.tsv", sep="\t", dtype=str
    ) is table
    assert source_module.ensure_local_copy("https://example.test/data.csv") == (
        tmp_path / "local.csv"
    )
    assert calls["read"] == ("table.tsv", "\t", {"dtype": str})
    assert calls["ensure"] == (
        "https://example.test/data.csv",
        {
            "output_dir": output_dir,
            "default_name": default_filename,
            "overwrite": False,
        },
    )

    monkeypatch.setattr(
        source_module,
        f"download_{source}_dataset",
        lambda **kwargs: tmp_path / "downloaded.csv",
    )
    monkeypatch.setattr(
        source_module,
        f"load_{source}_dataset",
        lambda path, **kwargs: (path, kwargs),
    )
    loaded = getattr(source_module, f"load_{source}_from_url")(
        "https://example.test/data.csv", sep=";", dtype=str
    )
    assert loaded == (tmp_path / "downloaded.csv", {"sep": ";", "dtype": str})
