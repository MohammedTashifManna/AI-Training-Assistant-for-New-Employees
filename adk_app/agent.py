"""Entry point for ADK's own tooling (``adk web`` and ``adk run``).

ADK discovers agents by importing a package and reading its ``root_agent``.
Exposing the same graph here means the project can be driven three ways without
duplicating any wiring:

* ``streamlit run ui/streamlit_app.py`` - the product UI, with citations.
* ``python scripts/ask.py`` - the CLI, with a routing trace.
* ``adk web`` from the project root - ADK's built-in developer UI, which shows
  the raw event stream, every transfer and every tool call.

Launch ADK's UI with::

    adk web

then pick ``adk_app`` in the agent dropdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from training_assistant.agents.graph import build_agent_graph  # noqa: E402
from training_assistant.logging_setup import quiet_third_party  # noqa: E402

quiet_third_party()

#: The name ADK looks for.
root_agent = build_agent_graph()

__all__ = ["root_agent"]
