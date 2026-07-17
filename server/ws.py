"""WebSocket fan-out for the live console."""
from __future__ import annotations

import json
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """Tracks console connections; broadcast failures drop the client."""

    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)

    @property
    def count(self) -> int:
        return len(self._clients)

    async def broadcast(self, message: dict[str, Any]) -> None:
        text = json.dumps(message, default=str)
        dead: list[WebSocket] = []
        for ws in self._clients:
            try:
                await ws.send_text(text)
            except Exception:
                # Client vanished mid-send; prune it rather than crash the feed.
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)
