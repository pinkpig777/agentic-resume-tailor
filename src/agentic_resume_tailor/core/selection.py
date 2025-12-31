from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class SelectionDecision:
    bullet_id: str
    action: str  # "selected" | "skipped"
    reason: str


def select_topk(
    candidates: List[Any], max_bullets: int = 16
) -> Tuple[List[str], List[SelectionDecision]]:
    """Select top bullet ids from ranked candidates."""
    selected: List[str] = []
    seen = set()
    decisions: List[SelectionDecision] = []

    for c in candidates:
        if len(selected) >= max_bullets:
            break

        bid = getattr(c, "bullet_id", None) or getattr(c, "id", None)
        if not bid:
            continue

        if bid in seen:
            decisions.append(
                SelectionDecision(bullet_id=bid, action="skipped", reason="duplicate_bullet_id")
            )
            continue

        seen.add(bid)
        selected.append(bid)
        decisions.append(
            SelectionDecision(bullet_id=bid, action="selected", reason="topk_relevance")
        )

    return selected, decisions
