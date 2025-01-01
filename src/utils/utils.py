from __future__ import annotations

import json


def save_as_json(data: dict[str, any], outpath: str) -> None:
    with open(outpath, "w") as f:
        json.dump(data, f)
