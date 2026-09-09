"""SatQuery AI v2 — upload validation and hardening.

Validates user-supplied files BEFORE any processing:

* extension + MIME allow-lists
* file-size cap
* magic-byte sniffing (don't trust the filename)
* PIL open/verify + dimension + pixel-count caps (decompression-bomb guard)
* safe server-side filenames (never trust user-provided names)

Never executes uploaded files; never serves them back as anything but data.
"""
from __future__ import annotations

import io
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

from . import config

# Pillow's own bomb guard, tightened to our ops limit.
Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS

_MAGIC = {
    b"\x89PNG\r\n\x1a\n": ".png",
    b"\xff\xd8\xff": ".jpg",
    b"II*\x00": ".tif",
    b"MM\x00*": ".tif",
    b"II+\x00": ".tif",   # BigTIFF little-endian
    b"MM+\x00": ".tif",   # BigTIFF big-endian
    b"RIFF": ".webp",     # also .bmp-ish RIFF; refined by PIL open
    b"BM": ".bmp",
}


def sniff_extension(content: bytes) -> Optional[str]:
    for magic, ext in _MAGIC.items():
        if content.startswith(magic):
            return ext
    return None


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    mime: str = ""
    ext: str = ""
    size_bytes: int = 0
    width: int = 0
    height: int = 0
    mode: str = ""
    safe_name: str = ""


def safe_filename(original: str, ext: str) -> str:
    """Generate a server-side filename; the user name is never reused."""
    keep = "".join(c for c in os.path.splitext(original or "scene")[0].lower()
                   if c.isalnum() or c in ("-", "_"))[:40] or "scene"
    return f"{keep}_{uuid.uuid4().hex[:8]}{ext}"


def validate_upload(filename: str, content: bytes,
                    content_type: str = "") -> ValidationResult:
    res = ValidationResult(ok=False)
    res.size_bytes = len(content)
    res.mime = (content_type or "").split(";")[0].strip().lower()

    if not content:
        res.errors.append("Empty file.")
        return res
    if len(content) > config.MAX_UPLOAD_BYTES:
        res.errors.append(
            f"File too large ({len(content)/1e6:.1f} MB). "
            f"Limit is {config.MAX_UPLOAD_MB} MB.")
        return res

    claimed = os.path.splitext(filename or "")[1].lower()
    if claimed and claimed not in config.ALLOWED_EXTENSIONS:
        res.errors.append(f"Extension '{claimed}' is not supported. "
                          f"Allowed: {sorted(config.ALLOWED_EXTENSIONS)}.")
        return res
    if res.mime and res.mime not in config.ALLOWED_MIMES:
        res.errors.append(f"MIME type '{res.mime}' is not accepted.")
        return res

    sniffed = sniff_extension(content)
    if sniffed is None:
        res.errors.append("File is not a recognised image (magic-byte check failed).")
        return res
    if claimed and sniffed != claimed and not (
            claimed in (".jpg", ".jpeg") and sniffed == ".jpg"):
        res.warnings.append(
            f"Extension '{claimed}' does not match file signature ({sniffed}); "
            f"treating as {sniffed}.")

    try:
        with Image.open(io.BytesIO(content)) as im:
            im.load()  # force decode; catches truncated/malicious payloads
            w, h = im.size
            res.width, res.height, res.mode = w, h, im.mode
    except Exception as e:
        res.errors.append(f"Image could not be decoded: {type(e).__name__}.")
        return res

    if max(w, h) > config.MAX_IMAGE_DIM:
        res.errors.append(
            f"Image dimensions {w}x{h} exceed the {config.MAX_IMAGE_DIM}px limit.")
        return res
    if w * h > config.MAX_IMAGE_PIXELS:
        res.errors.append("Image pixel count exceeds the processing limit.")
        return res

    res.ext = sniffed if sniffed != ".jpg" else (claimed or ".jpg")
    if res.ext == ".jpg" and claimed == ".jpeg":
        res.ext = ".jpeg"
    res.safe_name = safe_filename(filename, res.ext)
    res.ok = True
    return res
