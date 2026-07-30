"""Source-independent filesystem caching for downloaded dataset artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlparse

from dms_parser.exceptions import CorruptCacheManifestError, InvalidCacheEntryError

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "manifest.json"
_SAFE_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CacheManifest:
    """Metadata describing one artifact stored in the filesystem cache."""

    source: str
    dataset_id: str
    original_url: str
    downloaded_at: str
    file_size: int
    sha256: str
    artifact_filename: str

    @classmethod
    def from_dict(cls, values: object) -> CacheManifest:
        """Validate and construct a manifest from decoded JSON data."""
        if not isinstance(values, dict):
            raise CorruptCacheManifestError(
                "Cache manifest must contain a JSON object."
            )

        required_fields = {
            "source",
            "dataset_id",
            "original_url",
            "downloaded_at",
            "file_size",
            "sha256",
            "artifact_filename",
        }
        missing_fields = required_fields.difference(values)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise CorruptCacheManifestError(
                f"Cache manifest is missing required fields: {missing}."
            )

        string_fields = required_fields.difference({"file_size"})
        for field in string_fields:
            if not isinstance(values[field], str) or not values[field].strip():
                raise CorruptCacheManifestError(
                    f"Cache manifest field {field!r} must be a non-empty string."
                )

        file_size = values["file_size"]
        if (
            isinstance(file_size, bool)
            or not isinstance(file_size, int)
            or file_size < 0
        ):
            raise CorruptCacheManifestError(
                "Cache manifest field 'file_size' must be a non-negative integer."
            )

        checksum = values["sha256"]
        if _SHA256_PATTERN.fullmatch(checksum) is None:
            raise CorruptCacheManifestError(
                "Cache manifest field 'sha256' must be a lowercase SHA-256 digest."
            )

        artifact_filename = values["artifact_filename"]
        if (
            Path(artifact_filename).name != artifact_filename
            or artifact_filename in {".", ".."}
            or "/" in artifact_filename
            or "\\" in artifact_filename
        ):
            raise CorruptCacheManifestError(
                "Cache manifest contains an unsafe artifact filename."
            )

        try:
            downloaded_at = datetime.fromisoformat(
                values["downloaded_at"].replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise CorruptCacheManifestError(
                "Cache manifest field 'downloaded_at' is not a valid ISO timestamp."
            ) from exc
        if downloaded_at.tzinfo is None:
            raise CorruptCacheManifestError(
                "Cache manifest field 'downloaded_at' must include a timezone."
            )

        return cls(
            source=values["source"],
            dataset_id=values["dataset_id"],
            original_url=values["original_url"],
            downloaded_at=values["downloaded_at"],
            file_size=file_size,
            sha256=checksum,
            artifact_filename=artifact_filename,
        )


class FilesystemCache:
    """Store and validate artifacts keyed by source and dataset identifier.

    Cache lookups never create directories. A normal lookup returns a validated
    local artifact path on a hit and ``None`` on a miss. Passing ``refresh=True``
    deliberately treats an existing entry as a miss so callers can fetch and
    replace it.
    """

    def __init__(self, root: str | Path) -> None:
        """Initialize a cache rooted at ``root`` without creating it."""
        self.root = Path(root)
        logger.debug("Initialized filesystem cache root=%s", self.root)

    def entry_path(self, source: str, dataset_id: str) -> Path:
        """Return the deterministic directory for a cache key."""
        return (
            self.root
            / _safe_component(source, "source")
            / _safe_component(dataset_id, "dataset_id")
        )

    def manifest_path(self, source: str, dataset_id: str) -> Path:
        """Return the manifest path for a cache key without creating it."""
        return self.entry_path(source, dataset_id) / _MANIFEST_FILENAME

    def exists(
        self,
        source: str,
        dataset_id: str,
        *,
        validate_checksum: bool = True,
    ) -> bool:
        """Return whether a valid cache entry exists for a cache key.

        Invalid artifacts and malformed manifests raise package-level cache
        exceptions instead of being silently treated as misses.
        """
        return (
            self.resolve(
                source,
                dataset_id,
                validate_checksum=validate_checksum,
            )
            is not None
        )

    def resolve(
        self,
        source: str,
        dataset_id: str,
        *,
        refresh: bool = False,
        validate_checksum: bool = True,
    ) -> Path | None:
        """Resolve a valid artifact path, or return ``None`` for a cache miss.

        When ``refresh`` is true, the method always returns ``None`` and does
        not inspect the current entry. This gives callers an explicit way to
        bypass a cache hit before obtaining fresh data.
        """
        entry_path = self.entry_path(source, dataset_id)
        logger.debug(
            "Resolving cache entry source=%s dataset_id=%s path=%s "
            "validate_checksum=%s",
            source,
            dataset_id,
            entry_path,
            validate_checksum,
        )
        if refresh:
            logger.debug(
                "Skipping cache lookup for explicit refresh source=%s dataset_id=%s",
                source,
                dataset_id,
            )
            return None

        manifest_path = entry_path / _MANIFEST_FILENAME
        if not manifest_path.exists():
            logger.debug(
                "Cache miss reason=manifest_missing source=%s dataset_id=%s path=%s",
                source,
                dataset_id,
                manifest_path,
            )
            return None

        manifest = self.load_manifest(source, dataset_id)
        artifact_path = entry_path / manifest.artifact_filename
        if not artifact_path.is_file():
            raise InvalidCacheEntryError(
                f"Cached artifact is missing for {source!r}/{dataset_id!r}."
            )

        actual_size = artifact_path.stat().st_size
        if actual_size != manifest.file_size:
            raise InvalidCacheEntryError(
                f"Cached artifact size does not match its manifest for "
                f"{source!r}/{dataset_id!r}."
            )

        if validate_checksum:
            actual_checksum = _sha256_file(artifact_path)
            if actual_checksum != manifest.sha256:
                raise InvalidCacheEntryError(
                    f"Cached artifact checksum does not match its manifest for "
                    f"{source!r}/{dataset_id!r}."
                )

        logger.debug(
            "Validated cache entry source=%s dataset_id=%s path=%s "
            "size=%d checksum=%s",
            source,
            dataset_id,
            artifact_path,
            manifest.file_size,
            manifest.sha256,
        )
        return artifact_path

    def load_manifest(self, source: str, dataset_id: str) -> CacheManifest:
        """Load and validate the manifest associated with a cache key."""
        manifest_path = self.manifest_path(source, dataset_id)
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                values = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CorruptCacheManifestError(
                f"Could not load cache manifest at {manifest_path!s}: {exc}"
            ) from exc

        manifest = CacheManifest.from_dict(values)
        if manifest.source != source or manifest.dataset_id != dataset_id:
            raise CorruptCacheManifestError(
                "Cache manifest key does not match its filesystem location."
            )
        logger.debug(
            "Validated cache manifest source=%s dataset_id=%s path=%s",
            source,
            dataset_id,
            manifest_path,
        )
        return manifest

    def store_file(
        self,
        source: str,
        dataset_id: str,
        original_url: str,
        source_path: str | Path,
        *,
        refresh: bool = False,
        downloaded_at: datetime | None = None,
    ) -> Path:
        """Copy a local artifact into the cache and write its manifest.

        An existing valid entry is returned unchanged unless ``refresh`` is
        true. The source file is streamed, so this operation does not require
        loading an entire dataset into memory.
        """
        source_path = Path(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Artifact file not found: {source_path}")

        return self._store_stream(
            source,
            dataset_id,
            original_url,
            source_path.open("rb"),
            refresh=refresh,
            downloaded_at=downloaded_at,
        )

    def store_bytes(
        self,
        source: str,
        dataset_id: str,
        original_url: str,
        content: bytes,
        *,
        refresh: bool = False,
        downloaded_at: datetime | None = None,
    ) -> Path:
        """Store in-memory artifact content and write its manifest."""
        from io import BytesIO

        return self._store_stream(
            source,
            dataset_id,
            original_url,
            BytesIO(content),
            refresh=refresh,
            downloaded_at=downloaded_at,
        )

    def _store_stream(
        self,
        source: str,
        dataset_id: str,
        original_url: str,
        stream: BinaryIO,
        *,
        refresh: bool,
        downloaded_at: datetime | None,
    ) -> Path:
        """Store a binary stream after applying hit and refresh semantics."""
        entry_path: Path | None = None
        candidate_path: Path | None = None
        candidate_published = False
        previous_artifact_path: Path | None = None
        try:
            _validate_non_empty(original_url, "original_url")
            if downloaded_at is not None and downloaded_at.tzinfo is None:
                raise ValueError("downloaded_at must include timezone information.")

            if not refresh:
                cached_path = self.resolve(source, dataset_id)
                if cached_path is not None:
                    logger.info(
                        "Cache publication reused existing entry source=%s "
                        "dataset_id=%s",
                        source,
                        dataset_id,
                    )
                    return cached_path

            entry_path = self.entry_path(source, dataset_id)
            previous_artifact_path = self._previous_artifact_path(
                source,
                dataset_id,
            )
            entry_path.mkdir(parents=True, exist_ok=True)

            checksum = hashlib.sha256()
            size = 0
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=entry_path,
                    prefix=".artifact-",
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    while chunk := stream.read(1024 * 1024):
                        temporary_file.write(chunk)
                        checksum.update(chunk)
                        size += len(chunk)

                artifact_filename = _artifact_filename(
                    original_url,
                    checksum.hexdigest(),
                )
                candidate_path = entry_path / artifact_filename
                if candidate_path == previous_artifact_path:
                    temporary_path.unlink()
                else:
                    os.replace(temporary_path, candidate_path)
                    candidate_published = True
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()

            timestamp = downloaded_at or datetime.now(timezone.utc)
            manifest = CacheManifest(
                source=source,
                dataset_id=dataset_id,
                original_url=original_url,
                downloaded_at=timestamp.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                file_size=size,
                sha256=checksum.hexdigest(),
                artifact_filename=artifact_filename,
            )
            self._write_manifest(entry_path / _MANIFEST_FILENAME, manifest)
            candidate_published = False
            logger.info(
                "Published cache artifact source=%s dataset_id=%s size=%d",
                source,
                dataset_id,
                size,
            )
            logger.debug(
                "Cache publication details source=%s dataset_id=%s path=%s "
                "checksum=%s",
                source,
                dataset_id,
                candidate_path,
                manifest.sha256,
            )

            if (
                previous_artifact_path is not None
                and previous_artifact_path != candidate_path
                and previous_artifact_path.is_file()
            ):
                previous_artifact_path.unlink()

            return candidate_path
        except Exception:
            if (
                candidate_published
                and candidate_path is not None
                and candidate_path.exists()
            ):
                candidate_path.unlink()
            raise
        finally:
            stream.close()

    def _previous_artifact_path(
        self,
        source: str,
        dataset_id: str,
    ) -> Path | None:
        """Return the currently published artifact path when its manifest is valid."""
        manifest_path = self.manifest_path(source, dataset_id)
        if not manifest_path.exists():
            return None

        try:
            manifest = self.load_manifest(source, dataset_id)
        except CorruptCacheManifestError:
            logger.warning(
                "Ignoring invalid previous cache manifest source=%s dataset_id=%s",
                source,
                dataset_id,
            )
            return None
        return manifest_path.parent / manifest.artifact_filename

    @staticmethod
    def _write_manifest(path: Path, manifest: CacheManifest) -> None:
        """Write a manifest atomically."""
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=".manifest-",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(asdict(manifest), temporary_file, indent=2, sort_keys=True)
                temporary_file.write("\n")
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def _safe_component(value: str, field: str) -> str:
    """Convert a cache-key value into a safe deterministic path component."""
    _validate_non_empty(value, field)
    slug = _SAFE_COMPONENT_PATTERN.sub("-", value).strip("._-")
    if not slug:
        slug = field
    slug = slug[:48].rstrip("._-") or field
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{slug}-{digest}"


def _validate_non_empty(value: object, field: str) -> None:
    """Reject empty or non-string cache metadata values."""
    if not isinstance(value, str) or not value.strip():
        raise InvalidCacheEntryError(f"{field} must be a non-empty string.")


def _artifact_filename(original_url: str, checksum: str) -> str:
    """Build a content-addressed filename while retaining useful URL suffixes."""
    parsed_name = Path(unquote(urlparse(original_url).path)).name
    suffixes = Path(parsed_name).suffixes
    safe_suffixes = [
        suffix.lower()
        for suffix in suffixes[-2:]
        if re.fullmatch(r"\.[A-Za-z0-9]{1,10}", suffix)
    ]
    url_digest = hashlib.sha256(original_url.encode("utf-8")).hexdigest()[:16]
    return f"artifact-{url_digest}-{checksum[:16]}{''.join(safe_suffixes)}"


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for a file."""
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()
