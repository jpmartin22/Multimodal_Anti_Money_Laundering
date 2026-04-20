"""JSON read/write helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: Path) -> None:
    """Write ``obj`` to ``path`` as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    """Read a JSON file and return the parsed object."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
