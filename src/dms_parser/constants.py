"""Shared constants for the dms_parser package."""

from __future__ import annotations

import re

WILDTYPE_TOKENS = {"wt", "wildtype", "wild_type", "native", ""}
NEUTRAL_LABEL = 999
DEFAULT_EPSILON = 1e-8

AA3_TO_AA1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Sec": "U",
    "Pyl": "O",
    "Asx": "B",
    "Glx": "Z",
    "Xaa": "X",
    "Ter": "*",
}

DNA_CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY*X")

SINGLE_VARIANT_PATTERN = re.compile(
    r"^(?:p\.)?(?P<wt>[A-Za-z\*])(?P<position>\d+)(?P<mut>[A-Za-z\*])$"
)
MULTI_VARIANT_SPLIT_PATTERN = re.compile(r"[:;,/]\s*")

HGVS_BRACKET_RE = re.compile(r"^p\.\[(?P<body>.+)\]$")
HGVS_SINGLE_RE = re.compile(r"^p\.(?P<single>.+)$")
HGVS_MUT_RE = re.compile(
    r"^(?P<wt>[A-Z][a-z]{2})(?P<pos>\d+)(?P<mut>[A-Z][a-z]{2}|=|Ter|\*)$"
)

INDEL_TOKENS_PRO = ("delins", "del", "ins", "dup", "fs", "ext")
