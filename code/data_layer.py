"""
data_layer.py — Dataset I/O, image normalization, and on-disk caching.

Responsibilities (no model calls here):
  * Load the four dataset CSVs and resolve every referenced image to an
    existing file under dataset/.
  * Normalize each image to a model-ready payload: detect the TRUE format by
    content (files named .jpg are actually a mix of JPEG/PNG/WEBP/AVIF),
    transcode to RGB PNG with longest side <= 1024 px, return base64.
  * Surface UNTRUSTED EXIF text (ImageDescription / UserComment) for the
    reconciliation risk layer to *record* — never to act on.
  * Provide a content hash (sha256 of normalized bytes) and two on-disk caches:
      - normalization cache, keyed by SOURCE content hash alone;
      - perception cache, keyed by (normalized-content-hash, provider, model) so
        a Sonnet result is never reused when Opus/Gemini was requested.

Self-test:  python -m code.data_layer
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image, ExifTags

# Best-effort AVIF support. Pillow >= 11.3 decodes AVIF natively; the plugin is
# a belt-and-suspenders fallback for older Pillow builds. Neither import being
# present is only a problem if an AVIF actually needs decoding.
try:  # pragma: no cover - environment dependent
    import pillow_avif  # noqa: F401  (registers the AVIF plugin on import)
except Exception:  # pragma: no cover
    pass

# Import config whether loaded as a package (`code.data_layer`) or top-level.
try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore


# Allow large-but-reasonable images; guard against decompression-bomb surprises.
Image.MAX_IMAGE_PIXELS = 64_000_000

MAX_LONG_SIDE = 1024  # longest side after downscale, in pixels

# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------
CACHE_DIR = config.CODE_DIR / ".cache"
NORM_CACHE_DIR = CACHE_DIR / "norm"          # keyed by SOURCE content hash
PERCEPTION_CACHE_DIR = CACHE_DIR / "perception"  # keyed by (hash, provider, model)


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImageRef:
    """A single submitted image, resolved to a real file under dataset/."""
    image_id: str          # filename without extension, e.g. "img_1"
    rel_path: str          # path as written in the CSV, relative to dataset/
    abs_path: Path         # resolved absolute path (asserted to exist)


@dataclass
class Claim:
    """One row of claims.csv / sample_claims.csv."""
    user_id: str
    image_paths: str       # raw, semicolon-joined (verbatim input field)
    user_claim: str
    claim_object: str
    images: list[ImageRef] = field(default_factory=list)
    # Present only for the labeled sample set; empty for the test set.
    expected: dict[str, str] = field(default_factory=dict)


@dataclass
class NormalizedImage:
    """A model-ready, normalized image payload."""
    image_id: str
    source_rel_path: str
    true_format: str               # JPEG / PNG / WEBP / AVIF / ... (by content)
    width: int                     # after downscale
    height: int
    png_base64: str                # RGB PNG, longest side <= 1024
    content_hash: str              # sha256 of the normalized PNG bytes
    exif_text: Optional[str]       # UNTRUSTED ImageDescription/UserComment, or None
    media_type: str = "image/png"


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------
_LABELED_FIELDS = (
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
)


def parse_image_paths(raw: str, base_dir: Optional[Path] = None) -> list[ImageRef]:
    """Split image_paths on ';', resolve each relative to `base_dir`, assert exists.

    `base_dir` defaults to dataset/ (config.DATASET_DIR); the CLI passes --images.
    The image_id is the filename without extension (problem_statement.md).
    """
    root = base_dir or config.DATASET_DIR
    refs: list[ImageRef] = []
    for part in (raw or "").split(";"):
        rel = part.strip()
        if not rel:
            continue
        abs_path = (root / rel).resolve()
        assert abs_path.exists(), f"referenced image does not exist: {abs_path}"
        refs.append(ImageRef(image_id=Path(rel).stem, rel_path=rel, abs_path=abs_path))
    return refs


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_claims(
    path: Path = config.CLAIMS_CSV,
    labeled: bool = False,
    base_dir: Optional[Path] = None,
) -> list[Claim]:
    """Load claims.csv (test) or sample_claims.csv (labeled).

    `base_dir` (the CLI --images root) is used to resolve image_paths; defaults
    to dataset/.
    """
    claims: list[Claim] = []
    for row in _read_rows(path):
        claim = Claim(
            user_id=row["user_id"],
            image_paths=row["image_paths"],
            user_claim=row["user_claim"],
            claim_object=row["claim_object"],
            images=parse_image_paths(row["image_paths"], base_dir),
        )
        if labeled:
            claim.expected = {k: row.get(k, "") for k in _LABELED_FIELDS}
        claims.append(claim)
    return claims


def load_sample_claims(path: Path = config.SAMPLE_CLAIMS_CSV) -> list[Claim]:
    """Load the labeled development set (inputs + expected outputs)."""
    return load_claims(path, labeled=True)


def load_user_history(path: Path = config.USER_HISTORY_CSV) -> dict[str, dict[str, str]]:
    """Return {user_id: history_row_dict}."""
    return {row["user_id"]: row for row in _read_rows(path)}


def load_evidence_requirements(
    path: Path = config.EVIDENCE_REQUIREMENTS_CSV,
) -> list[dict[str, str]]:
    """Return the minimum-evidence rules as a list of row dicts."""
    return _read_rows(path)


# ---------------------------------------------------------------------------
# EXIF text extraction (UNTRUSTED — recorded, never acted upon)
# ---------------------------------------------------------------------------
_USER_COMMENT_PREFIXES = {
    b"ASCII\x00\x00\x00": "ascii",
    b"UNICODE\x00": "utf-16",
    b"JIS\x00\x00\x00\x00\x00": "shift_jis",
    b"\x00\x00\x00\x00\x00\x00\x00\x00": "latin-1",  # undefined → best effort
}
_TAG_IMAGE_DESCRIPTION = 270           # 0x010E
_TAG_EXIF_IFD = 0x8769
_TAG_USER_COMMENT = 0x9286


def _decode_user_comment(raw: bytes) -> str:
    for prefix, enc in _USER_COMMENT_PREFIXES.items():
        if raw.startswith(prefix):
            return raw[len(prefix):].decode(enc, "replace").strip("\x00").strip()
    return raw.decode("utf-8", "replace").strip("\x00").strip()


def _extract_exif_text(im: Image.Image) -> Optional[str]:
    """Pull ImageDescription and UserComment into a single string, or None."""
    parts: list[str] = []
    try:
        exif = im.getexif()
    except Exception:
        return None
    if not exif:
        return None

    desc = exif.get(_TAG_IMAGE_DESCRIPTION)
    if isinstance(desc, bytes):
        desc = desc.decode("utf-8", "replace")
    if desc and str(desc).strip():
        parts.append(str(desc).strip())

    try:
        exif_ifd = exif.get_ifd(_TAG_EXIF_IFD)
        uc = exif_ifd.get(_TAG_USER_COMMENT)
        if isinstance(uc, bytes):
            uc = _decode_user_comment(uc)
        if uc and str(uc).strip():
            parts.append(str(uc).strip())
    except Exception:
        pass

    # De-dup while preserving order (ImageDescription and UserComment often match).
    seen: list[str] = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    return " | ".join(seen) if seen else None


# ---------------------------------------------------------------------------
# Normalization cache (keyed by SOURCE content hash alone)
# ---------------------------------------------------------------------------
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _norm_cache_path(source_hash: str) -> Path:
    return NORM_CACHE_DIR / f"{source_hash}.json"


def _read_norm_cache(source_hash: str) -> Optional[dict]:
    p = _norm_cache_path(source_hash)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_norm_cache(source_hash: str, payload: dict) -> None:
    NORM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _norm_cache_path(source_hash).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def normalize_image(ref: ImageRef, use_cache: bool = True) -> NormalizedImage:
    """Detect true format, transcode to RGB PNG (longest side <= 1024), base64.

    The transcode result is cached by the SOURCE file's content hash, so an
    identical input file is never re-decoded. (Model-independent — no provider
    or model id in this key.)
    """
    source_bytes = ref.abs_path.read_bytes()
    source_hash = _sha256_bytes(source_bytes)

    if use_cache:
        cached = _read_norm_cache(source_hash)
        if cached is not None:
            return NormalizedImage(
                image_id=ref.image_id,
                source_rel_path=ref.rel_path,
                true_format=cached["true_format"],
                width=cached["width"],
                height=cached["height"],
                png_base64=cached["png_base64"],
                content_hash=cached["content_hash"],
                exif_text=cached["exif_text"],
            )

    with Image.open(io.BytesIO(source_bytes)) as im:
        true_format = im.format or "UNKNOWN"
        exif_text = _extract_exif_text(im)
        im.load()
        rgb = im.convert("RGB")
        rgb.thumbnail((MAX_LONG_SIDE, MAX_LONG_SIDE), Image.LANCZOS)  # only shrinks
        buf = io.BytesIO()
        rgb.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()
        width, height = rgb.size

    content_hash = _sha256_bytes(png_bytes)
    png_base64 = base64.b64encode(png_bytes).decode("ascii")

    if use_cache:
        _write_norm_cache(source_hash, {
            "true_format": true_format,
            "width": width,
            "height": height,
            "png_base64": png_base64,
            "content_hash": content_hash,
            "exif_text": exif_text,
        })

    return NormalizedImage(
        image_id=ref.image_id,
        source_rel_path=ref.rel_path,
        true_format=true_format,
        width=width,
        height=height,
        png_base64=png_base64,
        content_hash=content_hash,
        exif_text=exif_text,
    )


# ---------------------------------------------------------------------------
# Perception cache (keyed by normalized-content-hash + provider + model)
# ---------------------------------------------------------------------------
def perception_cache_key(content_hash: str, provider: str, model: str) -> str:
    """Composite key — never reuse a Sonnet result when Opus/Gemini was asked."""
    safe = lambda s: "".join(c if c.isalnum() or c in "-._" else "_" for c in s)
    return f"{content_hash}__{safe(provider)}__{safe(model)}"


def _perception_cache_path(content_hash: str, provider: str, model: str) -> Path:
    return PERCEPTION_CACHE_DIR / f"{perception_cache_key(content_hash, provider, model)}.json"


def get_cached_perception(content_hash: str, provider: str, model: str) -> Optional[dict]:
    p = _perception_cache_path(content_hash, provider, model)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def set_cached_perception(content_hash: str, provider: str, model: str, result: dict) -> None:
    PERCEPTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _perception_cache_path(content_hash, provider, model).write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Self-test:  python -m code.data_layer
# ---------------------------------------------------------------------------
def _self_test() -> None:
    from collections import Counter

    claims = load_claims(config.CLAIMS_CSV)
    # Unique images across the whole test set (de-dup by resolved path).
    refs: dict[str, ImageRef] = {}
    for c in claims:
        for r in c.images:
            refs[str(r.abs_path)] = r

    fmt_hist: Counter[str] = Counter()
    exif_count = 0
    exif_examples: list[tuple[str, str]] = []
    for r in refs.values():
        norm = normalize_image(r)
        fmt_hist[norm.true_format] += 1
        if norm.exif_text:
            exif_count += 1
            if len(exif_examples) < 3:
                exif_examples.append((norm.image_id, norm.exif_text))

    print(f"Claims (claims.csv):        {len(claims)}")
    print(f"Unique test images:         {len(refs)}")
    print("True-format histogram (by content, all named .jpg):")
    for fmt, n in fmt_hist.most_common():
        print(f"    {fmt:<8} {n}")
    print(f"Images carrying EXIF text:  {exif_count} / {len(refs)}")
    if exif_examples:
        print("EXIF text examples (UNTRUSTED — recorded only):")
        for img_id, text in exif_examples:
            snippet = text if len(text) <= 80 else text[:77] + "..."
            print(f"    {img_id}: {snippet!r}")


if __name__ == "__main__":
    _self_test()
