"""Uvicorn entrypoint: `uvicorn server.main:app`."""
from server.api import create_app

app = create_app()
