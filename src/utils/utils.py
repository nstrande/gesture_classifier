from __future__ import annotations

import json
from typing import Any
from typing import Dict


def save_as_json(data: Dict[str, Any], outpath: str) -> None:
    with open(outpath, "w") as f:
        json.dump(data, f)
