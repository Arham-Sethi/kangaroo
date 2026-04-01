"""Ingestion connectors -- API proxy and manual import.

Connectors are the INPUT side that feeds data into the pipeline:
    - ManualImportConnector: paste/upload conversations
    - ProxyCapture: transparent API request/response capture
"""

from app.core.connectors.api_proxy import CapturedExchange, ProxyCapture
from app.core.connectors.manual_import import ImportResult, ManualImportConnector

__all__ = [
    "CapturedExchange",
    "ImportResult",
    "ManualImportConnector",
    "ProxyCapture",
]
