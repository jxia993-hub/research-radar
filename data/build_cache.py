"""Rebuild data/sample_papers.json — the offline cache of *real* arXiv papers.

The cache lets the agent run fully offline and lets a grader reproduce results without
network. We never fabricate arXiv ids.

Source: the arXiv **RSS feeds** at ``rss.arxiv.org`` (one request per category). These are
reliable and far more permissive than the keyword-search API at ``export.arxiv.org`` (which
rate-limits aggressively — see the runtime fetcher, which uses that API for live ``--query``
search and falls back to this cache). We interleave a few categories so the cache spans the
taxonomy (ML, NLP, robotics, vision). Re-run any time:

    python data/build_cache.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_radar.perception.arxiv_source import Paper  # noqa: E402

DC = "{http://purl.org/dc/elements/1.1/}"
CATEGORIES = ["cs.LG", "cs.CL", "cs.RO", "cs.CV"]  # ML, NLP, robotics, vision -> topic spread
PER_CATEGORY = 60
CAP = 40
MIN_OK = 12
OUT = Path(__file__).resolve().parent / "sample_papers.json"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def parse_rss(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for item in root.findall(".//item"):
        title = _clean(item.findtext("title") or "")
        link = (item.findtext("link") or "").strip()
        desc = item.findtext("description") or ""
        creator = item.findtext(f"{DC}creator") or ""
        cats = [c.text for c in item.findall("category") if c.text]

        m = re.search(r"arXiv:(\d+\.\d+(?:v\d+)?)", desc)
        arxiv_id = m.group(1) if m else link.rsplit("/", 1)[-1]
        if "Abstract:" in desc:
            abstract = _clean(desc.split("Abstract:", 1)[1])
        else:
            abstract = _clean(re.sub(r"^arXiv:\S+\s+Announce Type:\s+\w+\s*", "", desc))
        authors = [a.strip() for a in re.split(r"[,;]", creator) if a.strip()]
        if not arxiv_id or not abstract:
            continue
        papers.append(Paper(
            arxiv_id=arxiv_id, title=title, abstract=abstract, authors=authors,
            categories=cats, published="", abs_url=link,
            pdf_url=link.replace("/abs/", "/pdf/"),
        ))
    return papers


def fetch_category(cat: str) -> list[Paper]:
    url = f"https://rss.arxiv.org/rss/{cat}"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "research-radar/0.4"})
    resp.raise_for_status()
    return parse_rss(resp.text)


def main() -> int:
    per_cat: dict[str, list[Paper]] = {}
    for cat in CATEGORIES:
        try:
            per_cat[cat] = fetch_category(cat)[:PER_CATEGORY]
            print(f"  {cat}: {len(per_cat[cat])} papers")
        except (requests.RequestException, ET.ParseError) as exc:
            print(f"  {cat}: fetch failed ({exc})")
            per_cat[cat] = []
        time.sleep(3)  # be polite between feeds

    # Round-robin interleave so the cache is topically diverse, not all one field.
    seen, out, idx = set(), [], 0
    while len(out) < CAP and any(idx < len(per_cat[c]) for c in CATEGORIES):
        for cat in CATEGORIES:
            if idx < len(per_cat[cat]):
                p = per_cat[cat][idx]
                if p.arxiv_id not in seen:
                    seen.add(p.arxiv_id)
                    out.append(p.to_dict())
                    if len(out) >= CAP:
                        break
        idx += 1

    if len(out) < MIN_OK:
        print(f"ERROR: only {len(out)} papers (< {MIN_OK}); NOT overwriting {OUT.name}.")
        return 1
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(out)} papers -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
