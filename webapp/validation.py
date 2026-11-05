"""Task 4.2 — upload validation. Three independent gates:
  1. extension allow-list      -> 415 Unsupported Media Type
  2. size limit                -> 413 (Flask's MAX_CONTENT_LENGTH, plus an explicit check here)
  3. real decode with Pillow   -> 422 when bytes look like an image but are not decodable
The file name is sanitised with werkzeug's secure_filename (strips paths like
../../etc/passwd, control characters and spaces) and only stored as text.
"""
import hashlib
from dataclasses import dataclass

from werkzeug.utils import secure_filename

from visionguard.preprocess import ImageDecodeError, decode_image


class UploadError(Exception):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.message, self.status = message, status


@dataclass
class ValidUpload:
    raw: bytes
    filename: str
    sha256: str
    width: int
    height: int


def validate_upload(file_storage, allowed_ext: set[str], max_bytes: int) -> ValidUpload:
    if file_storage is None or not file_storage.filename:
        raise UploadError("No file was selected. Choose an image to analyse.", 400)
    name = secure_filename(file_storage.filename) or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in allowed_ext:
        raise UploadError(f"'.{ext or '?'}' files are not accepted. Allowed: "
                          + ", ".join(sorted(allowed_ext)), 415)
    raw = file_storage.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise UploadError(f"File is larger than the {max_bytes / 1024 / 1024:.0f} MB limit.", 413)
    if not raw:
        raise UploadError("The file is empty.", 400)
    try:
        img = decode_image(raw)
    except ImageDecodeError:
        raise UploadError("The file has an image extension but could not be decoded as an image. "
                          "It may be corrupt or a different file type renamed.", 422) from None
    return ValidUpload(raw=raw, filename=name[:255], sha256=hashlib.sha256(raw).hexdigest(),
                       width=img.width, height=img.height)
