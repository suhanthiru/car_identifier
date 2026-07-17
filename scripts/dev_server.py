"""Dev server entry: 3D bridge enabled, port from PORT env (default 8010)."""
import os
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("EYES_ENABLE_3D", "1")

from server.api import create_app  # noqa: E402 — after env is set

if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1",
                port=int(os.environ.get("PORT", "8010")), log_level="warning")
