"""Compatibility shim for ragas + newer langchain-community.

ragas still imports ``langchain_community.chat_models.vertexai.ChatVertexAI``,
which was removed from langchain-community 0.4+. We only need the symbol to
exist for import-time isinstance checks.
"""

from __future__ import annotations

import sys
import types


def ensure_vertexai_shim() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    try:
        from langchain_google_vertexai import ChatVertexAI  # type: ignore
    except Exception:

        class ChatVertexAI:  # noqa: N801 - match expected symbol name
            """Placeholder so ragas can import without Vertex AI installed."""

            pass

    module = types.ModuleType(module_name)
    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module

    # Also expose on parent package if already imported
    parent_name = "langchain_community.chat_models"
    parent = sys.modules.get(parent_name)
    if parent is not None and not hasattr(parent, "vertexai"):
        parent.vertexai = module
