STANDARD_COLUMNS = [
    "dataset_id",
    "source",
    "protein_id",
    "gene",
    "uniprot_id",
    "wt_sequence",
    "variant",
    "n_mutations",
    "is_wildtype",
    "score_raw",
    "score_normalized",
    "score_log_ratio",
    "score_binary_like",
]

WILDTYPE_TOKENS = {"wt", "wildtype", "wild_type", "native", ""}
NEUTRAL_LABEL = 999
DEFAULT_EPSILON = 1e-8