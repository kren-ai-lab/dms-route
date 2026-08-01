from __future__ import annotations

import pandas as pd
import pytest

from dms_parser.exceptions import (
    InvalidHGVSVariantError,
    InvalidVariantError,
    MutationApplicationError,
    SequenceValidationError,
    UnsupportedVariantError,
)
from dms_parser.parsing import (
    apply_mutations,
    count_mutations,
    explode_mutations,
    extract_mutation_tokens,
    extract_positions,
    hgvs_pro_is_indel,
    hgvs_to_sequence,
    is_wildtype_variant,
    parse_hgvs_pro,
    parse_mavedb_hgvs_series,
    parse_variant,
    parse_variant_series,
    parse_variant_token,
    read_fasta_one,
    translate_dna,
    variant_to_sequence,
)


def test_is_wildtype_variant_detects_common_tokens():
    assert is_wildtype_variant("WT") is True
    assert is_wildtype_variant("wildtype") is True
    assert is_wildtype_variant("wild_type") is True
    assert is_wildtype_variant("") is True
    assert is_wildtype_variant(None) is True
    assert is_wildtype_variant("M1A") is False


def test_parse_variant_token_single_mutation():
    result = parse_variant_token("A23V")

    assert result["token"] == "A23V"
    assert result["wt_aa"] == "A"
    assert result["position"] == 23
    assert result["mut_aa"] == "V"


def test_parse_variant_token_invalid_raises():
    with pytest.raises(InvalidVariantError):
        parse_variant_token("foo")


def test_parse_variant_token_non_strict_returns_null_fields():
    result = parse_variant_token("foo", strict=False)

    assert result["token"] == "foo"
    assert result["wt_aa"] is None
    assert result["position"] is None
    assert result["mut_aa"] is None


def test_parse_variant_single():
    result = parse_variant("A23V")

    assert result["variant"] == "A23V"
    assert result["is_wildtype"] is False
    assert result["n_mutations"] == 1
    assert result["position"] == 23
    assert result["wt_aa"] == "A"
    assert result["mut_aa"] == "V"
    assert len(result["mutations"]) == 1


def test_parse_variant_multi():
    result = parse_variant("A23V;G45D")

    assert result["is_wildtype"] is False
    assert result["n_mutations"] == 2
    assert result["position"] is None
    assert result["wt_aa"] is None
    assert result["mut_aa"] is None
    assert result["mutations"][0]["token"] == "A23V"
    assert result["mutations"][1]["token"] == "G45D"


def test_parse_variant_wildtype():
    result = parse_variant("WT")

    assert result["is_wildtype"] is True
    assert result["n_mutations"] == 0
    assert result["mutations"] == []


def test_count_mutations():
    assert count_mutations("WT") == 0
    assert count_mutations("A23V") == 1
    assert count_mutations("A23V;G45D") == 2


def test_parse_variant_series():
    series = pd.Series(["WT", "A23V", "G45D;L10P"])
    result = parse_variant_series(series, prefix="parsed_")

    assert "parsed_variant" in result.columns
    assert "parsed_is_wildtype" in result.columns
    assert "parsed_n_mutations" in result.columns
    assert result.loc[0, "parsed_is_wildtype"] == True
    assert result.loc[1, "parsed_n_mutations"] == 1
    assert result.loc[2, "parsed_n_mutations"] == 2


def test_explode_mutations():
    series = pd.Series(["WT", "A23V;G45D"])
    parsed = parse_variant_series(series)
    exploded = explode_mutations(parsed, mutations_col="mutations")

    assert "token" in exploded.columns
    assert "position" in exploded.columns
    assert exploded["token"].dropna().tolist() == ["A23V", "G45D"]


def test_extract_positions():
    assert extract_positions("WT") == []
    assert extract_positions("A23V") == [23]
    assert extract_positions("A23V;G45D") == [23, 45]


def test_extract_mutation_tokens():
    assert extract_mutation_tokens("WT") == []
    assert extract_mutation_tokens("A23V;G45D") == ["A23V", "G45D"]


def test_hgvs_pro_is_indel():
    assert hgvs_pro_is_indel("p.Gly10del") is True
    assert hgvs_pro_is_indel("p.Met1Ala") is False


def test_parse_hgvs_pro_single():
    result = parse_hgvs_pro("p.Met1Ala")
    assert result == [("M", 1, "A")]


def test_parse_hgvs_pro_multi():
    result = parse_hgvs_pro("p.[Met1Ala;Lys2Arg]")
    assert result == [("M", 1, "A"), ("K", 2, "R")]


def test_parse_hgvs_pro_synonymous():
    result = parse_hgvs_pro("p.Met1=")
    assert result == [("M", 1, "M")]


def test_parse_hgvs_complete_identity_is_wildtype(wt_sequence: str):
    assert parse_hgvs_pro("p.=") == []
    assert hgvs_to_sequence(wt_sequence, "p.=") == (wt_sequence, "")
    assert is_wildtype_variant(hgvs_to_sequence(wt_sequence, "p.=")[1]) is True


def test_hgvs_population_synonymous_is_not_complete_wildtype():
    assert is_wildtype_variant("p.(=)") is False
    with pytest.raises(InvalidHGVSVariantError):
        parse_hgvs_pro("p.(=)")


def test_hgvs_position_equality_is_not_complete_wildtype(wt_sequence: str):
    mutated, variant = hgvs_to_sequence(wt_sequence, "p.Lys2=")

    assert mutated == wt_sequence
    assert variant == "K2K"
    assert is_wildtype_variant(variant) is False


def test_parse_hgvs_pro_invalid_raises():
    with pytest.raises(InvalidHGVSVariantError):
        parse_hgvs_pro("p.invalid")


def test_parse_hgvs_pro_unsupported_raises():
    with pytest.raises(UnsupportedVariantError):
        parse_hgvs_pro("p.Gly10del")


def test_apply_mutations(wt_sequence: str):
    mutated, variant = apply_mutations(wt_sequence, [("M", 1, "A"), ("K", 2, "R")])

    assert mutated.startswith("ART")
    assert variant == "M1A;K2R"


def test_apply_mutations_duplicate_position_raises(wt_sequence: str):
    with pytest.raises(MutationApplicationError):
        apply_mutations(wt_sequence, [("M", 1, "A"), ("M", 1, "V")])


def test_apply_mutations_out_of_range_raises(wt_sequence: str):
    with pytest.raises(MutationApplicationError):
        apply_mutations(wt_sequence, [("M", 999, "A")])


def test_apply_mutations_wt_mismatch_raises(wt_sequence: str):
    with pytest.raises(MutationApplicationError):
        apply_mutations(wt_sequence, [("A", 1, "V")])


def test_variant_to_sequence_single(wt_sequence: str):
    mutated, variant = variant_to_sequence(wt_sequence, "M1A")

    assert mutated[0] == "A"
    assert variant == "M1A"


def test_variant_to_sequence_wildtype_returns_same_sequence(wt_sequence: str):
    mutated, variant = variant_to_sequence(wt_sequence, "WT")

    assert mutated == wt_sequence
    assert variant == ""


def test_hgvs_to_sequence(wt_sequence: str):
    mutated, variant = hgvs_to_sequence(wt_sequence, "p.Met1Ala")

    assert mutated[0] == "A"
    assert variant == "M1A"


def test_parse_mavedb_hgvs_series():
    series = pd.Series(["p.Met1Ala", "p.Gly10del", "p.invalid"])
    result = parse_mavedb_hgvs_series(series, strict=False)

    assert list(result["status"]) == ["OK", "Unsupported", "Error"]


def test_translate_dna_basic():
    result = translate_dna("ATGAAAACC")
    assert result == "MKT"


def test_translate_dna_invalid_frame_raises():
    with pytest.raises(SequenceValidationError):
        translate_dna("ATGAAAACC", frame=4)


def test_read_fasta_one(tmp_path):
    fasta = tmp_path / "wt.fasta"
    fasta.write_text(">seq1\nMKTAYI\n", encoding="utf-8")

    header, seq = read_fasta_one(str(fasta))

    assert header == "seq1"
    assert seq == "MKTAYI"


def test_read_fasta_one_multiple_records_raises(tmp_path):
    fasta = tmp_path / "wt.fasta"
    fasta.write_text(">seq1\nAAA\n>seq2\nBBB\n", encoding="utf-8")

    with pytest.raises(SequenceValidationError):
        read_fasta_one(str(fasta))
