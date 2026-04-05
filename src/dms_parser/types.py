from typing import Optional, TypedDict


class ParsedVariant(TypedDict, total=False):
    wt_aa: Optional[str]
    position: Optional[int]
    mut_aa: Optional[str]
    token: str