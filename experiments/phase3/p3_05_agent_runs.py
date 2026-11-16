"""Full-agent runs on the fixed 24-question subset, one arm at a time (resumable).

usage: p3_05_agent_runs.py <tier> <window:on|off> [more arms as tier:window ...]
Records go to results/phase3/agent_<tier>_<window>.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.benchmark.llm import DailyLimitError
from src.benchmark.run_agent import run

OUT = Path("results/phase3")
import os
N_SUBSET = int(os.environ.get("P3_AGENT_N", "24"))


def subset() -> list[dict]:
    qs = [q for q in (json.loads(l) for l in (OUT / "questions.jsonl").read_text().splitlines())
          if q["split"] == "test"]
    qs.sort(key=lambda q: hashlib.sha256(("agent:" + q["id"]).encode()).hexdigest())
    return qs[:N_SUBSET]


def main(arms: list[str]) -> None:
    qs = subset()
    for arm in arms:
        tier, win = arm.split(":")
        path = OUT / f"agent_{tier}_{win}.jsonl"
        done = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
        with path.open("a") as f:
            for q in qs:
                if q["id"] in done:
                    continue
                try:
                    r = run(q["question"], tier=tier, context_window=(win == "on"))
                except DailyLimitError as e:
                    print("DAILY LIMIT", arm, q["id"], str(e)[:200])
                    return
                r["id"] = q["id"]
                f.write(json.dumps(r) + "\n")
                f.flush()
        print("finished", arm)


if __name__ == "__main__":
    main(sys.argv[1:])
