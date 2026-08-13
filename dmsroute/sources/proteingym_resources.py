"""Official ProteinGym resource definitions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from dmsroute.core.exceptions import (
    UnknownSourceResourceError,
    UnsupportedSourceResourceError,
)

ProteinGymCollection = Literal["dms", "clinical"]
ProteinGymVariantType = Literal["substitutions", "indels"]


@dataclass(frozen=True)
class ProteinGymResource:
    """Describe one official ProteinGym metadata and data resource."""

    resource_id: str
    collection: ProteinGymCollection
    variant_type: ProteinGymVariantType
    metadata_url: str
    data_url: str
    metadata_filename: str
    data_filename: str
    processing_supported: bool


_RESOURCE_LIST = (
    ProteinGymResource(
        resource_id="dms_substitutions",
        collection="dms",
        variant_type="substitutions",
        metadata_url=(
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
            "reference_files/DMS_substitutions.csv"
        ),
        data_url=(
            "https://proteingym.s3.us-east-2.amazonaws.com/"
            "DMS_substitutions.parquet"
        ),
        metadata_filename="DMS_substitutions.csv",
        data_filename="DMS_substitutions.parquet",
        processing_supported=True,
    ),
    ProteinGymResource(
        resource_id="dms_indels",
        collection="dms",
        variant_type="indels",
        metadata_url=(
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
            "reference_files/DMS_indels.csv"
        ),
        data_url=(
            "https://proteingym.s3.us-east-2.amazonaws.com/DMS_indels.parquet"
        ),
        metadata_filename="DMS_indels.csv",
        data_filename="DMS_indels.parquet",
        processing_supported=True,
    ),
    ProteinGymResource(
        resource_id="clinical_substitutions",
        collection="clinical",
        variant_type="substitutions",
        metadata_url=(
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
            "reference_files/clinical_substitutions.csv"
        ),
        data_url=(
            "https://proteingym.s3.us-east-2.amazonaws.com/"
            "clinical_substitutions.parquet"
        ),
        metadata_filename="clinical_substitutions.csv",
        data_filename="clinical_substitutions.parquet",
        processing_supported=False,
    ),
    ProteinGymResource(
        resource_id="clinical_indels",
        collection="clinical",
        variant_type="indels",
        metadata_url=(
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
            "reference_files/clinical_indels.csv"
        ),
        data_url=(
            "https://proteingym.s3.us-east-2.amazonaws.com/"
            "clinical_indels.parquet"
        ),
        metadata_filename="clinical_indels.csv",
        data_filename="clinical_indels.parquet",
        processing_supported=False,
    ),
)

PROTEINGYM_RESOURCES: Mapping[str, ProteinGymResource] = MappingProxyType(
    {resource.resource_id: resource for resource in _RESOURCE_LIST}
)


def list_proteingym_resources() -> tuple[ProteinGymResource, ...]:
    """Return all official ProteinGym resources in deterministic order."""
    return _RESOURCE_LIST


def get_proteingym_resource(
    resource_id: str,
    *,
    require_processing: bool = False,
) -> ProteinGymResource:
    """Return one resource, optionally requiring current processing support."""
    if not isinstance(resource_id, str) or not resource_id.strip():
        raise UnknownSourceResourceError(
            "ProteinGym resource must be a non-empty string."
        )
    try:
        resource = PROTEINGYM_RESOURCES[resource_id]
    except KeyError as exc:
        raise UnknownSourceResourceError(
            f"Unknown ProteinGym resource {resource_id!r}."
        ) from exc

    if require_processing and not resource.processing_supported:
        raise UnsupportedSourceResourceError(
            f"ProteinGym resource {resource_id!r} is registered but is not "
            "currently supported for processing."
        )
    return resource
