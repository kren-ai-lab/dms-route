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
from dms_parser.types import ProteinDeletionEdit, ProteinInsertionEdit


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
    assert hgvs_pro_is_indel("p.Cys2del") is False
    assert hgvs_pro_is_indel("p.Asp1_Ala2insLys") is False
    assert hgvs_pro_is_indel("p.Cys2_Asp3del") is True
    assert hgvs_pro_is_indel("p.[Cys2del]") is True
    assert hgvs_pro_is_indel("p.Met1Ala") is False


def test_parse_hgvs_pro_single():
    result = parse_hgvs_pro("p.Met1Ala")
    assert result == [("M", 1, "A")]


def test_parse_hgvs_pro_multi():
    result = parse_hgvs_pro("p.[Met1Ala;Lys2Arg]")
    assert result == [("M", 1, "A"), ("K", 2, "R")]


def test_parse_hgvs_pro_bounded_indels_are_structured():
    assert parse_hgvs_pro("p.Cys2del") == [ProteinDeletionEdit("C", 2)]
    assert parse_hgvs_pro("p.Asp1_Ala2insLys") == [
        ProteinInsertionEdit("D", 1, "A", 2, "K")
    ]


@pytest.mark.parametrize(
    ("hgvs_pro", "expected"),
    [
        ("p.[=]", []),
        ("p.[=;=]", []),
        ("p.[Asp1Glu;=]", [("D", 1, "E")]),
        ("p.[=;Gln31His]", [("Q", 31, "H")]),
        (
            "p.[Asp1Glu;=;Gln31His]",
            [("D", 1, "E"), ("Q", 31, "H")],
        ),
    ],
)
def test_parse_hgvs_pro_ignores_bracketed_equality_components(
    hgvs_pro: str,
    expected: list[tuple[str, int, str]],
) -> None:
    assert parse_hgvs_pro(hgvs_pro) == expected


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


@pytest.mark.parametrize("hgvs_pro", ["p.invalid", "p.[=;invalid]"])
def test_parse_hgvs_pro_invalid_raises(hgvs_pro: str) -> None:
    with pytest.raises(InvalidHGVSVariantError):
        parse_hgvs_pro(hgvs_pro)


@pytest.mark.parametrize(
    "hgvs_pro",
    [
        "p.Cys2_Asp3del",
        "p.Asp1_Ala2insLysArg",
        "p.Cys2dup",
        "p.Cys2delinsLys",
        "p.Cys2fs",
        "p.Cys2extTer4",
        "p.(Cys2del)",
        "p.Cys2del?",
        "p.Cys2insLys",
        "p.[Cys2del]",
        "p.[=;Cys2del]",
        "p.[Cys2del;Asp3Glu]",
    ],
)
def test_parse_hgvs_pro_unsupported_raises(hgvs_pro: str) -> None:
    with pytest.raises(UnsupportedVariantError):
        parse_hgvs_pro(hgvs_pro)


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


def test_hgvs_to_sequence_reconstructs_bounded_indels():
    assert hgvs_to_sequence("ACD", "p.Cys2del") == ("AD", "C2del")
    assert hgvs_to_sequence("DA", "p.Asp1_Ala2insLys") == (
        "DKA",
        "D1_A2insK",
    )


@pytest.mark.parametrize(
    ("wt_sequence", "hgvs_pro", "expected_sequence", "expected_variant"),
    [
        ("ACD", "p.Ala1del", "CD", "A1del"),
        ("ACD", "p.Asp3del", "AC", "D3del"),
    ],
)
def test_hgvs_to_sequence_deletes_first_or_last_residue(
    wt_sequence: str,
    hgvs_pro: str,
    expected_sequence: str,
    expected_variant: str,
) -> None:
    assert hgvs_to_sequence(wt_sequence, hgvs_pro) == (
        expected_sequence,
        expected_variant,
    )


@pytest.mark.parametrize(
    ("wt_sequence", "hgvs_pro", "error"),
    [
        ("ACD", "p.Asp2del", "WT mismatch"),
        ("ACD", "p.Cys4del", "out of range"),
        ("DCA", "p.Asp1_Ala3insLys", "adjacent"),
        ("DCA", "p.Ala3_Cys4insLys", "out of range"),
        ("DCA", "p.Ala1_Cys2insLys", "WT mismatch"),
        ("DCA", "p.Asp1_Ala2insLys", "WT mismatch"),
    ],
)
def test_hgvs_to_sequence_rejects_invalid_indel_references(
    wt_sequence: str,
    hgvs_pro: str,
    error: str,
) -> None:
    with pytest.raises(MutationApplicationError, match=error):
        hgvs_to_sequence(wt_sequence, hgvs_pro)


@pytest.mark.parametrize(
    "hgvs_pro",
    ["p.Foo2del", "p.Asp1_Ala2insFoo", "p.Asp1_Foo2insLys"],
)
def test_parse_hgvs_pro_rejects_invalid_indel_residue_codes(hgvs_pro: str) -> None:
    with pytest.raises(InvalidHGVSVariantError, match="Unknown amino acid code"):
        parse_hgvs_pro(hgvs_pro)


@pytest.mark.parametrize(
    ("hgvs_pro", "expected_sequence", "expected_variant"),
    [
        ("p.[=;=]", "DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQ", ""),
        ("p.[Asp1Glu;=]", "EAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQ", "D1E"),
        ("p.[=;Gln31His]", "DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH", "Q31H"),
    ],
)
def test_hgvs_to_sequence_ignores_bracketed_equality_components(
    hgvs_pro: str,
    expected_sequence: str,
    expected_variant: str,
) -> None:
    wt_sequence = "DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQ"

    assert hgvs_to_sequence(wt_sequence, hgvs_pro) == (
        expected_sequence,
        expected_variant,
    )


def test_parse_mavedb_hgvs_series():
    series = pd.Series(["p.Met1Ala", "p.[Gly10del]", "p.invalid"])
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
