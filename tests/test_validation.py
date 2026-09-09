"""Unit tests: upload validation + hardening."""
import io

from PIL import Image

from core import config
from core.validation import safe_filename, sniff_extension, validate_upload


def _png_bytes(w=64, h=64):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 200, 30)).save(buf, "PNG")
    return buf.getvalue()


def test_valid_png_passes():
    v = validate_upload("scene.png", _png_bytes(), "image/png")
    assert v.ok and v.errors == []
    assert (v.width, v.height) == (64, 64)
    assert v.safe_name.endswith(".png") and v.safe_name != "scene.png"


def test_path_traversal_filename_never_reused():
    v = validate_upload("../../etc/evil.png", _png_bytes(), "image/png")
    assert v.ok
    assert ".." not in v.safe_name and "/" not in v.safe_name


def test_bad_magic_rejected():
    v = validate_upload("scene.png", b"not an image at all....", "image/png")
    assert not v.ok and any("magic" in e.lower() for e in v.errors)


def test_bad_extension_rejected():
    v = validate_upload("run.exe", _png_bytes(), "application/octet-stream")
    assert not v.ok


def test_oversize_rejected():
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (config.MAX_UPLOAD_BYTES + 1)
    v = validate_upload("big.png", big, "image/png")
    assert not v.ok and any("too large" in e.lower() for e in v.errors)


def test_empty_rejected():
    assert not validate_upload("x.png", b"", "image/png").ok


def test_sniff_and_safe_name():
    assert sniff_extension(_png_bytes()) == ".png"
    assert sniff_extension(b"junk") is None
    assert safe_filename("My Scene 01.JPG", ".jpg").endswith(".jpg")
