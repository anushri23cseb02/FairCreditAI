"""
Basic security helpers.

Currently provides safe_filename() for turning an untrusted uploaded
filename into a path-traversal-safe one. Not yet wired into an
endpoint (batch prediction reads the uploaded CSV in-memory via pandas
and never writes the original filename to disk), but kept here as the
one place this kind of helper belongs if a future endpoint needs it.
"""
import re
import uuid


def safe_filename(original_filename: str) -> str:
    """
    Produce a filesystem-safe filename that cannot be used for path
    traversal, while keeping the original extension.
    """
    original_filename = original_filename.strip().replace("\\", "/").split("/")[-1]
    name, _, ext = original_filename.rpartition(".")
    ext = re.sub(r"[^a-zA-Z0-9]", "", ext)[:10]
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:80] or "file"
    unique_suffix = uuid.uuid4().hex[:8]
    if ext:
        return f"{safe_stem}_{unique_suffix}.{ext}"
    return f"{safe_stem}_{unique_suffix}"
