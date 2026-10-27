"""Stream a raw request body to a private temp file, enforcing the size limit as bytes arrive."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import Request

from onconova.errors import UploadRejected


async def body_to_tempfile(request: Request, tmp_dir: Path, max_bytes: int) -> Path:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise UploadRejected("file_too_large", f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit.", status_code=413)
    fd, name = tempfile.mkstemp(prefix="upload-", dir=tmp_dir)
    path = Path(name)
    size = 0
    try:
        with os.fdopen(fd, "wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > max_bytes:
                    raise UploadRejected("file_too_large", f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                                         status_code=413)
                fh.write(chunk)
        if size == 0:
            raise UploadRejected("empty_upload", "The upload is empty.")
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise
