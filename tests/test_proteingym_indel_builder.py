from __future__ import annotations

from inspect import signature

import pandas as pd
import pytest

import dmsroute
import dmsroute.builders as builders_module
from dmsroute.core._wildtype import get_wt_resolution
from dmsroute.builders import (
    build_mavedb_dataset,
    build_proteingym_dataset,
    build_proteingym_indel_dataset,
)
from dmsroute.core.exceptions import (
    InvalidDatasetError,
    MissingWildTypeError,
    SequenceValidationError,
    WildTypeConflictError,
)
from dmsroute.sources.proteingym_resources import get_proteingym_resource


def _write_indel_table(tmp_path, table: pd.DataFrame, name: str = "indels.csv"):
    path = tmp_path / name
    table.to_csv(path, index=False)
    return path


def test_proteingym_indel_builder_is_public_with_expected_signature():
    assert dmsroute.build_proteingym_indel_dataset is (
        build_proteingym_indel_dataset
    )
    parameters = signature(build_proteingym_indel_dataset).parameters
    assert parameters["mutated_sequence_col"].default == "mutated_sequence"
    assert parameters["target_sequence_col"].default == "target_seq"
    assert parameters["add_relative_score"].default is False
    assert parameters["add_binary_label"].default is False
    assert parameters["add_wildtype_row"].default is False
    assert {
        "input_path",
        "score_col",
        "dataset_id",
        "protein_id",
        "gene",
        "uniprot_id",
        "wt_sequence",
        "wt_fasta_path",
        "drop_failed",
        "validate_output",
        "require_wt_for_transforms",
        "wt_score",
    }.issubset(parameters)
    assert get_proteingym_resource("dms_indels").processing_supported is True


def test_proteingym_indel_builder_uses_authoritative_sequences(tmp_path):
    source = pd.DataFrame(
        {
            "target_seq": ["MAAAA"] * 6,
            "mutated_sequence": [
                "MAAA",
                "MAAAAA",
                "MAAAAAAA",
                "MA",
                "MCAAA",
                "MAAAA",
            ],
            "DMS_score": [-0.5, 0.2, -1.2, -2.0, 0.4, 1.0],
            "DMS_score_bin": [0, 1, 0, 0, 1, 1],
            "DMS_id": ["assay_indels"] * 6,
            "mutant": [None] * 6,
        }
    )
    path = _write_indel_table(tmp_path, source)

    result = build_proteingym_indel_dataset(path, score_col="DMS_score")

    assert result["mutated_sequence"].tolist() == source[
        "mutated_sequence"
    ].tolist()
    assert result["variant"].isna().all()
    assert result.loc[:4, "n_mutations"].isna().all()
    assert result.loc[5, "n_mutations"] == 0
    assert result["is_wildtype"].tolist() == [False] * 5 + [True]
    assert result["is_synthetic"].tolist() == [False] * 6
    assert result["status"].tolist() == ["OK"] * 6
    assert result["error"].tolist() == [""] * 6
    assert result["score_raw"].tolist() == source["DMS_score"].tolist()
    assert result["DMS_score_bin"].tolist() == source[
        "DMS_score_bin"
    ].tolist()
    assert result["mutant"].isna().all()
    assert "score_binary_like" not in result.columns
    resolution = result.attrs["dmsroute_wt_resolution"]
    assert resolution["sequence"] == "MAAAA"
    assert resolution["sequence_provenance"] == "proteingym_target_seq"


def test_repeated_residues_do_not_create_positional_notation(tmp_path):
    path = _write_indel_table(
        tmp_path,
        pd.DataFrame(
            {
                "target_seq": ["AAAA"],
                "mutated_sequence": ["AAA"],
                "DMS_score": [0.5],
            }
        ),
    )

    result = build_proteingym_indel_dataset(path, score_col="DMS_score")

    assert result.loc[0, "mutated_sequence"] == "AAA"
    assert pd.isna(result.loc[0, "variant"])
    assert pd.isna(result.loc[0, "n_mutations"])


def test_mutant_column_may_be_absent(tmp_path):
    path = _write_indel_table(
        tmp_path,
        pd.DataFrame(
            {
                "target_seq": ["MKT"],
                "mutated_sequence": ["MK"],
                "DMS_score": [0.25],
            }
        ),
    )

    result = build_proteingym_indel_dataset(path, score_col="DMS_score")

    assert "mutant" not in result.columns
    assert result.loc[0, "status"] == "OK"


def test_proteingym_indel_builder_applies_opt_in_relative_score(tmp_path):
    path = _write_indel_table(
        tmp_path,
        pd.DataFrame(
            {
                "target_seq": ["MKT", "MKT"],
                "mutated_sequence": ["MKT", "MT"],
                "DMS_score": [1.0, 0.25],
                "DMS_score_bin": [1, 0],
            }
        ),
    )

    result = build_proteingym_indel_dataset(
        path,
        score_col="DMS_score",
        add_relative_score=True,
        relative_method="difference",
        relative_output_col="score_delta",
    )

    assert result["score_raw"].tolist() == [1.0, 0.25]
    assert result["score_delta"].tolist() == [0.0, -0.75]
    assert result["DMS_score_bin"].tolist() == [1, 0]


def test_proteingym_indel_builder_uses_explicit_wt_score_for_transform(tmp_path):
    path = _write_indel_table(
        tmp_path,
        pd.DataFrame(
            {
                "target_seq": ["MKT"],
                "mutated_sequence": ["MT"],
                "DMS_score": [0.25],
            }
        ),
    )

    result = build_proteingym_indel_dataset(
        path,
        score_col="DMS_score",
        wt_score=-0.25,
        add_relative_score=True,
        relative_method="difference",
        relative_output_col="score_delta",
        require_wt_for_transforms=True,
    )

    assert result["score_raw"].tolist() == [0.25]
    assert result["score_delta"].tolist() == [0.5]
    assert get_wt_resolution(result).score_provenance == "user_fallback"


def test_raw_indel_build_preserves_ambiguous_observed_wt_scores(tmp_path):
    path = _write_indel_table(
        tmp_path,
        pd.DataFrame(
            {
                "target_seq": ["MKT", "MKT", "MKT"],
                "mutated_sequence": ["MKT", "MKT", "MT"],
                "DMS_score": [1.0, 2.0, 0.25],
            }
        ),
    )

    result = build_proteingym_indel_dataset(path, score_col="DMS_score")
    resolution = get_wt_resolution(result)

    assert result["score_raw"].tolist() == [1.0, 2.0, 0.25]
    assert resolution.score is None
    assert resolution.score_unavailable_reason == "conflicting_observed_wildtype_scores"


def test_proteingym_indel_builder_adds_distinguishable_synthetic_wt(tmp_path):
    path = _write_indel_table(
        tmp_path,
        pd.DataFrame(
            {
                "target_seq": ["MKT"],
                "mutated_sequence": ["MT"],
                "DMS_score": [0.25],
            }
        ),
    )

    result = build_proteingym_indel_dataset(
        path,
        score_col="DMS_score",
        add_wildtype_row=True,
    )

    assert result["is_synthetic"].tolist() == [True, False]
    assert result["is_wildtype"].tolist() == [True, False]
    assert result["mutated_sequence"].tolist() == ["MKT", "MT"]
    assert result["variant"].isna().all()
    assert result.loc[0, "n_mutations"] == 0
    assert pd.isna(result.loc[0, "score_raw"])


def test_invalid_mutated_sequences_are_row_errors_and_drop_failed(tmp_path):
    source = pd.DataFrame(
        {
            "target_seq": ["MKT"] * 3,
            "mutated_sequence": [None, "MKJ", "MT"],
            "DMS_score": [0.1, 0.2, 0.3],
        }
    )
    path = _write_indel_table(tmp_path, source)

    result = build_proteingym_indel_dataset(path, score_col="DMS_score")
    dropped = build_proteingym_indel_dataset(
        path,
        score_col="DMS_score",
        drop_failed=True,
    )

    assert result["status"].tolist() == ["Error", "Error", "OK"]
    assert result.loc[:1, "mutated_sequence"].isna().all()
    assert result["score_raw"].tolist() == [0.1, 0.2, 0.3]
    assert dropped["status"].tolist() == ["OK"]
    assert dropped["mutated_sequence"].tolist() == ["MT"]


def test_expected_mutant_sequence_validation_error_is_row_error(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = _write_indel_table(
        tmp_path,
        pd.DataFrame(
            {
                "target_seq": ["MKT"],
                "mutated_sequence": ["MT"],
                "DMS_score": [0.25],
            }
        ),
    )
    monkeypatch.setattr(
        builders_module,
        "validate_wt_sequence",
        lambda sequence: (_ for _ in ()).throw(
            SequenceValidationError("invalid mutant sequence")
        ),
    )

    result = build_proteingym_indel_dataset(path, score_col="DMS_score")

    assert result.loc[0, "status"] == "Error"
    assert result.loc[0, "error"] == "invalid mutant sequence"


def test_unexpected_mutant_sequence_validation_error_propagates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = _write_indel_table(
        tmp_path,
        pd.DataFrame(
            {
                "target_seq": ["MKT"],
                "mutated_sequence": ["MT"],
                "DMS_score": [0.25],
            }
        ),
    )
    monkeypatch.setattr(
        builders_module,
        "validate_wt_sequence",
        lambda sequence: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )

    with pytest.raises(RuntimeError, match="unexpected"):
        build_proteingym_indel_dataset(path, score_col="DMS_score")


def test_proteingym_indel_builder_rejects_conflicting_target_sequences(tmp_path):
    path = _write_indel_table(
        tmp_path,
        pd.DataFrame(
            {
                "target_seq": ["MKT", "MAT", "MAT"],
                "mutated_sequence": ["MT", "MT", "MT"],
                "DMS_score": [0.1, 0.2, 0.3],
            }
        ),
    )

    with pytest.raises(WildTypeConflictError) as error:
        build_proteingym_indel_dataset(path, score_col="DMS_score")

    assert str(error.value).count("proteingym_target_seq") == 1


def test_source_target_sequence_rejects_conflicting_fallback(tmp_path):
    path = _write_indel_table(
        tmp_path,
        pd.DataFrame(
            {
                "target_seq": ["MKT"],
                "mutated_sequence": ["MT"],
                "DMS_score": [0.25],
            }
        ),
    )

    with pytest.raises(WildTypeConflictError, match="fallback conflicts"):
        build_proteingym_indel_dataset(
            path,
            score_col="DMS_score",
            wt_sequence="MAT",
        )


def test_proteingym_indel_builder_uses_fallback_without_target_evidence(tmp_path):
    path = _write_indel_table(
        tmp_path,
        pd.DataFrame(
            {
                "mutated_sequence": ["MT"],
                "DMS_score": [0.25],
            }
        ),
    )

    result = build_proteingym_indel_dataset(
        path,
        score_col="DMS_score",
        wt_sequence="MKT",
    )

    assert result["wt_sequence"].tolist() == ["MKT"]
    assert (
        result.attrs["dmsroute_wt_resolution"]["sequence_provenance"]
        == "user_fallback"
    )

    with pytest.raises(MissingWildTypeError):
        build_proteingym_indel_dataset(path, score_col="DMS_score")


@pytest.mark.parametrize("target_sequence", ["MKJ", 123])
def test_invalid_target_sequence_is_dataset_error(tmp_path, target_sequence):
    path = _write_indel_table(
        tmp_path,
        pd.DataFrame(
            {
                "target_seq": [target_sequence],
                "mutated_sequence": ["MKT"],
                "DMS_score": [0.25],
            }
        ),
    )

    with pytest.raises(InvalidDatasetError, match="WT sequence"):
        build_proteingym_indel_dataset(path, score_col="DMS_score")


def test_existing_substitution_and_mavedb_builders_keep_notation(tmp_path):
    protein_path = tmp_path / "substitutions.csv"
    pd.DataFrame({"mutant": ["M1A"], "DMS_score": [0.5]}).to_csv(
        protein_path,
        index=False,
    )
    mavedb_path = tmp_path / "mavedb.csv"
    pd.DataFrame({"hgvs_pro": ["p.Met1Ala"], "score": [0.5]}).to_csv(
        mavedb_path,
        index=False,
    )

    protein = build_proteingym_dataset(
        protein_path,
        score_col="DMS_score",
        variant_col="mutant",
        wt_sequence="MKT",
    )
    mavedb = build_mavedb_dataset(
        mavedb_path,
        score_col="score",
        wt_sequence="MKT",
    )

    assert protein["variant"].tolist() == ["M1A"]
    assert mavedb["variant"].tolist() == ["M1A"]
