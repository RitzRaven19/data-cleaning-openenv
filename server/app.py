"""ASGI app entrypoint expected by multi-mode validators."""

from __future__ import annotations

import os

import uvicorn

from app import app


def main() -> None:
    """Run the API server via the `server` console script entrypoint."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()

