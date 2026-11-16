"""Human-readable citation labels for indexed chunks.

Every SEC EDGAR full-submission file is named `full-submission.txt`, so the
file name cannot tell filings apart. The ticker and accession number sit in the
path (`.../NVDA/10-K/0001045810-26-000021/...`), so labels are derived from it.
"""

import re

_PATH_RE = re.compile(r"(?P<tkr>[A-Z]{1,5})[\\/](?P<form>10-[KQ]|8-K)[\\/](?P<acc>[\d-]+)")


def source_label(metadata: dict) -> str:
    """'NVDA 10-K 0001045810-26-000021' when the path allows, else the file name."""
    path = str(metadata.get("source_path") or metadata.get("source") or "unknown")
    m = _PATH_RE.search(path)
    if m:
        return f"{m['tkr']} {m['form']} {m['acc']}"
    return path.replace("\\", "/").rsplit("/", 1)[-1]
