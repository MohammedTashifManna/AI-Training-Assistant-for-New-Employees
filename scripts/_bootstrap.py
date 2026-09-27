"""Put ``src/`` on ``sys.path`` so the scripts run without an installation step.

Importing this module is the first thing every entry point does. It keeps the
project runnable straight from a clone with plain ``python scripts/....py``,
which matters for reproducibility when someone is grading or reviewing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"

if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
