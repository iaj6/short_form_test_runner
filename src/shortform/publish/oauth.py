"""One-time Google OAuth flow to obtain a YouTube refresh token.

Run once via `shortform publish-auth`. Opens the browser to Google's consent
page, catches the callback on localhost, exchanges the code for tokens, and
writes YOUTUBE_REFRESH_TOKEN to the project .env. Every later upload trades
that refresh token for a short-lived access token with no user interaction.

Deliberately stdlib-only — no google-api-python-client or google-auth-oauthlib.
The whole flow is two HTTP calls and a one-shot local server, and the Google
SDKs would add a large transitive dependency tree to a project whose runtime
deps are otherwise slim. Same reasoning as the rest of this codebase preferring
a subprocess to a heavyweight client.

TWO TRAPS, both caused by the consent screen being in "Testing" status.
Console: APIs & Services -> OAuth consent screen (newer UI: Google Auth
Platform -> Audience). Clicking PUBLISH APP fixes both at once.

1. "Error 403: access_denied ... can only be accessed by developer-approved
   testers", at the consent page, before you get anywhere. The account you're
   signing in with isn't on the Test users list. Either publish the app, or add
   the account under Test users.

2. invalid_grant on an upload roughly a week later. While in Testing, Google
   expires refresh tokens after 7 days, so a flow that worked on Monday fails
   on the following Tuesday. Only publishing fixes this one — adding a test
   user does not.

Publishing does NOT require passing Google verification. `youtube.upload` is a
sensitive scope, so the console will say verification is needed for public use;
an unverified production app still works, showing a warning screen on the
consent page (Advanced -> Go to <app> (unsafe)) with a 100-user cap, which is
99 more than this needs.
"""

from __future__ import annotations

import http.server
import json
import logging
import secrets
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from typing import Any

from shortform.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

ENV_FILE = PROJECT_ROOT / ".env"

# Google "Desktop app" clients register redirect_uris as ["http://localhost"]
# and accept any loopback port at auth time. 8890 is registered in
# ~/.ports.json under `shortform` (8888/8889 belong to liu-spotify/liu-youtube).
PORT = 8890
REDIRECT_URI = f"http://localhost:{PORT}"

# youtube.upload is the narrow scope for inserting videos. The broader
# `youtube` scope would also permit reading and deleting; an uploader has no
# business with either.
SCOPES = "https://www.googleapis.com/auth/youtube.upload"

TOKEN_URL = "https://oauth2.googleapis.com/token"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"

_state: dict[str, str | None] = {"code": None, "error": None, "expected_state": None}


class MissingCredentialsError(RuntimeError):
    """Raised when the client id/secret or refresh token isn't configured."""


def read_env(name: str) -> str:
    """Read a var straight from .env.

    Reads the file rather than os.environ because `publish-auth` WRITES the
    refresh token here, and a value written this run must be readable by the
    next command without re-sourcing the environment.
    """
    if not ENV_FILE.exists():
        raise MissingCredentialsError(f"{ENV_FILE} does not exist")
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise MissingCredentialsError(f"{name} not found in {ENV_FILE}")


def write_env(name: str, value: str) -> None:
    """Add or update a single var in .env, preserving every other line."""
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    new_line = f"{name}={value}"
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{name}="):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    ENV_FILE.write_text("\n".join(lines) + "\n")


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass  # the server's own logging would interleave with ours

    def do_GET(self) -> None:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if not params:
            # Google redirects to the bare registered URI, so accept any path
            # rather than pinning /callback — but an empty query is a favicon
            # request or similar, not the callback.
            self.send_response(404)
            self.end_headers()
            return

        if params.get("state", [None])[0] != _state["expected_state"]:
            _state["error"] = "state mismatch (possible CSRF)"
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"State mismatch. Auth aborted.")
            return

        if "error" in params:
            _state["error"] = params["error"][0]
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Auth failed.")
            return

        _state["code"] = params.get("code", [None])[0]
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body style='font-family: monospace; padding: 40px;'>"
            b"<h1>Auth complete.</h1>"
            b"<p>You can close this tab and return to the terminal.</p>"
            b"</body></html>"
        )


def run_consent_flow() -> str:
    """Run the browser consent flow and persist the refresh token.

    Returns the refresh token. Raises MissingCredentialsError or RuntimeError.
    """
    client_id = read_env("YOUTUBE_CLIENT_ID")
    client_secret = read_env("YOUTUBE_CLIENT_SECRET")

    _state["expected_state"] = secrets.token_urlsafe(16)
    _state["code"] = None
    _state["error"] = None

    auth_url = f"{AUTH_URL}?" + urllib.parse.urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "state": _state["expected_state"],
            # access_type=offline + prompt=consent is what actually returns a
            # refresh token. Without prompt=consent Google omits it on re-auth,
            # and you get an access token that dies in an hour with no way back.
            "access_type": "offline",
            "prompt": "consent",
        }
    )

    print(f"Starting local callback server on {REDIRECT_URI}")
    print("Opening browser for Google consent...")
    print("If you see an 'unverified app' warning: Advanced -> Go to (unsafe).")
    print(f"If the browser does not open, visit:\n  {auth_url}\n")

    server = http.server.HTTPServer(("127.0.0.1", PORT), _CallbackHandler)
    webbrowser.open(auth_url)
    server.handle_request()  # exactly one request: the callback
    server.server_close()

    if _state["error"]:
        raise RuntimeError(f"Auth failed: {_state['error']}")
    if not _state["code"]:
        raise RuntimeError("No code returned from callback")

    tokens = _post_form(
        TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "code": _state["code"],
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": REDIRECT_URI,
        },
    )

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise RuntimeError(
            "No refresh token in response. Revoke this app's access at "
            "https://myaccount.google.com/permissions and run publish-auth again."
        )

    write_env("YOUTUBE_REFRESH_TOKEN", refresh_token)
    return str(refresh_token)


def get_access_token() -> str:
    """Trade the long-lived refresh token for a short-lived access token."""
    try:
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": read_env("YOUTUBE_REFRESH_TOKEN"),
            "client_id": read_env("YOUTUBE_CLIENT_ID"),
            "client_secret": read_env("YOUTUBE_CLIENT_SECRET"),
        }
    except MissingCredentialsError as e:
        raise MissingCredentialsError(
            f"{e}. Run `shortform publish-auth` first."
        ) from e

    return str(_post_form(TOKEN_URL, payload)["access_token"])


def _post_form(url: str, payload: dict[str, str]) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(payload).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            parsed: dict[str, Any] = json.loads(r.read())
            return parsed
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        if "invalid_grant" in body:
            raise RuntimeError(
                f"Token refresh failed: HTTP {e.code}: {body}\n\n"
                "invalid_grant usually means the refresh token expired. Google "
                "kills refresh tokens after 7 days while the OAuth consent "
                "screen is in 'Testing' status. Set it to 'In production' in "
                "Google Cloud Console, then re-run `shortform publish-auth`."
            ) from e
        raise RuntimeError(f"OAuth request failed: HTTP {e.code}: {body}") from e
