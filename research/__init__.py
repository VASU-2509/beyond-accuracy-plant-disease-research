"""Offline research pipeline; never invoked by public API requests."""
from .storage import configure_runtime

configure_runtime()
