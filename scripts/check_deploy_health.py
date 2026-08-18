"""POST-deploy smoke check: GET /health on a deployed base URL."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: uv run python scripts/check_deploy_health.py BASE_URL\n"
            "Example: uv run python scripts/check_deploy_health.py "
            "https://your-app.up.railway.app",
            file=sys.stderr,
        )
        sys.exit(2)

    base = sys.argv[1].rstrip("/")
    url = f"{base}/health"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        print(f"Error: HTTP {exc.code} from {url}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Error: could not reach {url}\nDetail: {exc.reason}", file=sys.stderr)
        sys.exit(1)

    if status != 200:
        print(f"Error: expected HTTP 200, got {status}", file=sys.stderr)
        sys.exit(1)

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        print(f"Error: non-JSON response: {body!r}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {url}")
    print(json.dumps(payload, indent=2))

    if payload.get("status") != "ok":
        print("Error: status is not 'ok'", file=sys.stderr)
        sys.exit(1)

    db = payload.get("db")
    if db == "connected":
        print("PostgreSQL: connected")
    elif db == "disabled":
        print(
            "Warning: database disabled (DATABASE_URL not set on host).",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(
            f"Error: database status is {db!r} (expected 'connected').",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
