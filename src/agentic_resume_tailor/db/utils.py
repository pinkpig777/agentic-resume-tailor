from __future__ import annotations

import re
from typing import Iterable, Optional

ROLE_SPLIT_TOKEN = "$|$"

_slug_non_alnum = re.compile(r"[^a-z0-9]+")
_slug_multi_us = re.compile(r"_+")


def slugify(value: str | None) -> str:
    if value is None:
        return "unknown"
    value = value.replace(ROLE_SPLIT_TOKEN, " ")
    value = value.strip().lower()
    value = _slug_non_alnum.sub("_", value)
    value = _slug_multi_us.sub("_", value).strip("_")
    return value or "unknown"


def primary_role(role: str | None) -> str:
    if not role:
        return "unknown"
    parts = role.split(ROLE_SPLIT_TOKEN, 1)
    primary = parts[0].strip()
    return primary or "unknown"


def make_job_id(company: str | None, role: str | None) -> str:
    return f"{slugify(company)}__{slugify(primary_role(role))}"


def make_project_id(name: str | None) -> str:
    return slugify(name)


def ensure_unique_slug(base: str, existing: Iterable[str]) -> str:
    existing_set = {s for s in existing if s}
    if base not in existing_set:
        return base
    suffix = 2
    while True:
        candidate = f"{base}__{suffix}"
        if candidate not in existing_set:
            return candidate
        suffix += 1


def _parse_bullet_num(bid: str | None) -> Optional[int]:
    if not bid:
        return None
    match = re.fullmatch(r"b(\d+)", bid.strip().lower())
    if not match:
        return None
    return int(match.group(1))


def next_bullet_id(existing_ids: Iterable[str]) -> str:
    nums = [_parse_bullet_num(bid) for bid in existing_ids]
    nums = [n for n in nums if n is not None]
    nxt = (max(nums) + 1) if nums else 1
    width = 2 if nxt < 100 else 3
    return f"b{nxt:0{width}d}"


def next_sort_order(existing_orders: Iterable[int | None]) -> int:
    nums = [n for n in existing_orders if isinstance(n, int)]
    return (max(nums) + 1) if nums else 1
