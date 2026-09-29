"""
Basic security helpers.

Provides safe_filename() for turning an untrusted uploaded filename
into a path-traversal-safe one (not yet wired into an endpoint --
batch prediction reads the uploaded CSV in-memory via pandas and never
writes the original filename to disk -- but kept here as the one place
this kind of helper belongs if a future endpoint needs it), and
BasicAuthMiddleware for optional public-deployment auth.
"""
import base64
import re
import secrets
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """
    Opt-in HTTP Basic Auth for the whole API. Only added when both
    APP_BASIC_AUTH_USER and APP_BASIC_AUTH_PASSWORD are set (see
    backend/main.py) -- local/dev usage without those variables is
    unaffected. /health is always left open so a platform health
    check (which sends no credentials) keeps working.
    """

    def __init__(self, app, username: str, password: str):
        super().__init__(app)
        self._username = username
        self._password = password

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        supplied_user, supplied_pass = "", ""
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
                supplied_user, _, supplied_pass = decoded.partition(":")
            except Exception:
                pass

        if secrets.compare_digest(supplied_user, self._username) and secrets.compare_digest(
            supplied_pass, self._password
        ):
            return await call_next(request)

        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="FairCredit AI"'})


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
