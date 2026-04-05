# dms-parser

Internal lightweight utilities for downloading, parsing, standardizing, and transforming deep mutational scanning datasets from MaveDB and ProteinGym.

## Scope

This package is intentionally small and focused on:

- dataset download
- raw table loading
- schema homogenization
- variant parsing
- wild-type relative score transformation
- pseudo-binary labeling

## Installation

```bash
pip install -e .