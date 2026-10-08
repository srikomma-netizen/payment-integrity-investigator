"""Policy ingestion: front matter + heading-aware chunking.

Each chunk carries its policy metadata and section position; retrieval filtering,
source validation and context expansion all depend on it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"
_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{2,3})\s+(.*?)\s*$")   # only ## and ###; the # title line is part of the preamble
# "2. Upcoding Indicators" -> ("2", ...), "Appendix A. Recognized Panel Pairs" -> ("A", ...)
_SECTION_NUM = re.compile(r"^(?:Appendix\s+)?([A-Z]|\d+(?:\.\d+)*)\.?\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    policy_id: str
    policy_title: str
    version: int
    status: str
    effective_date: str
    access_level: str
    claim_types: list[str]
    section_id: str
    section_title: str
    text: str
    part: int = 0
    parent_section: str | None = None
    neighbors: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"{self.policy_id} v{self.version} §{self.section_id} {self.section_title}"


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    # Deliberately tiny YAML subset (scalars and [a, b] lists) to avoid a PyYAML dependency.
    m = _FRONT.match(raw)
    if not m:
        return {}, raw
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)   # split once so values like titles can contain colons
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            meta[k.strip()] = [x.strip() for x in v[1:-1].split(",") if x.strip()]
        else:
            meta[k.strip()] = v
    return meta, raw[m.end():]


def _split_words(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    parts, start = [], 0
    while start < len(words):
        parts.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break   # without this the last window would be emitted again as a pure-overlap tail
        start += max_words - overlap   # assumes overlap < max_words, or this never advances
    return parts


def chunk_policy(path: Path, *, max_words: int = 160, overlap: int = 30) -> list[Chunk]:
    """Split one policy file into section chunks; long sections become overlapping parts."""
    meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    current_id, current_title, parent, buf = "0", "Preamble", None, []
    h2_id = None

    # closure over the loop state below; reads current_id/title/parent at call time
    def flush():
        text = "\n".join(buf).strip()
        if not text or current_id == "0":   # skip the preamble; it's title/boilerplate, not policy text
            return
        for i, part in enumerate(_split_words(text, max_words, overlap)):
            chunks.append(Chunk(
                # version is in the id so v1 and v2 of the same section can coexist in the index
                chunk_id=f"{meta['policy_id']}:v{meta['version']}:{current_id}:{i}",
                policy_id=meta["policy_id"], policy_title=meta.get("title", ""),
                version=int(meta.get("version", 0)), status=meta.get("status", "unknown"),
                effective_date=meta.get("effective_date", "1970-01-01"),
                access_level=meta.get("access_level", "standard"),
                claim_types=list(meta.get("claim_types", [])),
                section_id=current_id, section_title=current_title, text=part, part=i, parent_section=parent,
            ))

    for line in body.splitlines():
        m = _HEADING.match(line)
        if not m:
            buf.append(line)
            continue
        flush()
        buf = []
        level, heading = len(m.group(1)), m.group(2)
        num = _SECTION_NUM.match(heading)
        # unnumbered headings fall back to a truncated title as the id; good enough, but not guaranteed unique
        current_id, current_title = (num.group(1), num.group(2)) if num else (heading[:12], heading)
        if level == 2:
            h2_id, parent = current_id, None
        else:
            parent = h2_id
    flush()   # last section has no following heading to trigger it

    # neighbours within the same policy for context expansion
    for i, c in enumerate(chunks):
        c.neighbors = [chunks[j].chunk_id for j in (i - 1, i + 1) if 0 <= j < len(chunks)]
    return chunks


def ingest(policy_dir: Path = POLICY_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(policy_dir.glob("*.md")):
        chunks.extend(chunk_policy(path))
    return chunks
