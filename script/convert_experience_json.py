#!/usr/bin/env python3
"""
Convert ART resume data JSON into a provenance-friendly format.

What it does
- Adds deterministic job_id for each experience (company + primary role)
- Adds deterministic project_id for each project (project name)
- Converts bullets from plain strings into objects:
    {"id": "b01", "text_latex": "..."}
- Preserves existing bullet ids when present
- Allows reordering without changing ids (when text matches exactly)
- Assigns ids only to new bullets

Usage
  python convert_experience_json.py --input data/raw_experience_data.json --output data/my_experience.json

Notes
- Bullets are assumed to be LaTeX-ready. This script does NOT sanitize LaTeX.
- If you want ids to remain stable even when you edit bullet text, edit the my_experience.json file
  and keep the existing bullet "id" fields. The script will preserve them.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROLE_SPLIT_TOKEN = "$|$"

_slug_non_alnum = re.compile(r"[^a-z0-9]+")
_slug_multi_us = re.compile(r"_+")
logger = logging.getLogger(__name__)


def slugify(s: str) -> str:
    """Slugify.

    Args:
        s: The s value.

    Returns:
        String result.
    """
    if s is None:
        return "unknown"
    s = s.replace(ROLE_SPLIT_TOKEN, " ")
    s = s.strip().lower()
    s = _slug_non_alnum.sub("_", s)
    s = _slug_multi_us.sub("_", s).strip("_")
    return s or "unknown"


def primary_role(role: str) -> str:
    """Primary role.

    Args:
        role: Role title.

    Returns:
        String result.
    """
    if not role:
        return "unknown"
    parts = role.split(ROLE_SPLIT_TOKEN, 1)
    return parts[0].strip() or "unknown"


def make_job_id(company: str, role: str) -> str:
    """Make job id.

    Args:
        company: Company name.
        role: Role title.

    Returns:
        String result.
    """
    return f"{slugify(company)}__{slugify(primary_role(role))}"


def make_project_id(name: str) -> str:
    """Make project id.

    Args:
        name: Name value.

    Returns:
        String result.
    """
    return slugify(name)


def _parse_bullet_num(bid: str) -> Optional[int]:
    """Parse bullet num.

    Args:
        bid: Bullet identifier.

    Returns:
        Integer result.
    """
    m = re.fullmatch(r"b(\d+)", (bid or "").strip().lower())
    if not m:
        return None
    return int(m.group(1))


def _next_bullet_id(existing_ids: List[str]) -> str:
    """Next bullet ID.

    Args:
        existing_ids: Existing bullet identifiers.

    Returns:
        String result.
    """
    nums = [_parse_bullet_num(x) for x in existing_ids]
    nums = [n for n in nums if n is not None]
    nxt = (max(nums) + 1) if nums else 1
    width = 2 if nxt < 100 else 3
    return f"b{nxt:0{width}d}"


def normalize_bullets(bullets: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Normalize bullets.

    Args:
        bullets: The bullets value.

    Returns:
        Tuple of results.
    """
    warnings: List[str] = []
    if bullets is None:
        return [], warnings

    raw_items: List[Any] = bullets if isinstance(bullets, list) else [bullets]

    text_to_id: Dict[str, str] = {}
    existing_ids: List[str] = []

    for item in raw_items:
        if isinstance(item, dict):
            bid = item.get("id")
            txt = item.get("text_latex") if "text_latex" in item else item.get("text")
            if isinstance(bid, str) and isinstance(txt, str):
                bid_norm = bid.strip()
                text_to_id[txt] = bid_norm
                existing_ids.append(bid_norm)

    normalized: List[Dict[str, Any]] = []
    used_ids: set[str] = set()

    def allocate_id(for_text: str) -> str:
        """Allocate ID.

        Args:
            for_text: The for text value.

        Returns:
            String result.
        """
        if for_text in text_to_id and text_to_id[for_text] not in used_ids:
            return text_to_id[for_text]
        return _next_bullet_id(existing_ids + list(used_ids))

    for item in raw_items:
        if isinstance(item, str):
            txt = item
            bid = allocate_id(txt)
            used_ids.add(bid)
            normalized.append({"id": bid, "text_latex": txt})
        elif isinstance(item, dict):
            if "id" in item and ("text_latex" in item or "text" in item):
                bid = str(item.get("id")).strip()
                txt = item.get("text_latex") if "text_latex" in item else item.get("text")
                if not isinstance(txt, str):
                    warnings.append(f"Bullet with id {bid} has non-string text, skipped")
                    continue
                if not bid:
                    bid = allocate_id(txt)
                if bid in used_ids:
                    warnings.append(f"Duplicate bullet id {bid} detected, reassigned")
                    bid = allocate_id(txt)
                used_ids.add(bid)
                normalized.append({"id": bid, "text_latex": txt})
            else:
                warnings.append("Found bullet dict not in expected shape, skipped")
        else:
            warnings.append("Found bullet item not string or dict, skipped")

    return normalized, warnings


def convert(input_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """Convert.

    Args:
        input_path: Filesystem path for input.

    Returns:
        Tuple of results.
    """
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {input_path}") from exc
    warnings: List[str] = []

    data.setdefault("schema_version", "my_experience_v2")

    exps = data.get("experiences", [])
    if not isinstance(exps, list):
        raise ValueError("'experiences' must be a list")

    for i, exp in enumerate(exps):
        if not isinstance(exp, dict):
            warnings.append(f"experiences[{i}] is not an object, skipped")
            continue
        company = exp.get("company", "")
        role = exp.get("role", "")
        exp.setdefault("job_id", make_job_id(str(company), str(role)))

        bullets, w = normalize_bullets(exp.get("bullets", []))
        warnings.extend([f"experiences[{i}]: {msg}" for msg in w])
        exp["bullets"] = bullets

    projs = data.get("projects", [])
    if not isinstance(projs, list):
        raise ValueError("'projects' must be a list")

    for i, proj in enumerate(projs):
        if not isinstance(proj, dict):
            warnings.append(f"projects[{i}] is not an object, skipped")
            continue
        name = proj.get("name", "")
        proj.setdefault("project_id", make_project_id(str(name)))

        bullets, w = normalize_bullets(proj.get("bullets", []))
        warnings.extend([f"projects[{i}]: {msg}" for msg in w])
        proj["bullets"] = bullets

    return data, warnings


def main() -> None:
    """Main.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, help="Path to input JSON file")
    p.add_argument("--output", type=str, help="Path to write converted JSON file")
    p.add_argument("--in-place", type=str, help="Convert and overwrite this JSON file")
    p.add_argument("--print-warnings", action="store_true", help="Print warnings to stderr")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if bool(args.in_place) == bool(args.input):
        raise SystemExit("Provide exactly one of --in-place or --input/--output")

    if args.in_place:
        in_path = Path(args.in_place)
        out_path = in_path
    else:
        if not args.output:
            raise SystemExit("When using --input, you must provide --output")
        in_path = Path(args.input)
        out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    converted, warnings = convert(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.print_warnings and warnings:
        for w in warnings:
            print(w, file=sys.stderr)
    elif warnings:
        logger.info("Conversion completed with %s warnings.", len(warnings))


if __name__ == "__main__":
    main()
