from types import SimpleNamespace

from agentic_resume_tailor.core.selection import select_topk


def test_select_topk_dedupes_and_limits() -> None:
    """Test select topk dedupes and limits."""
    candidates = [
        SimpleNamespace(bullet_id="a"),
        SimpleNamespace(bullet_id="a"),
        SimpleNamespace(bullet_id="b"),
        SimpleNamespace(bullet_id="c"),
    ]

    selected, decisions = select_topk(candidates, max_bullets=2)

    assert selected == ["a", "b"]
    assert len(decisions) == 3
    assert decisions[1].action == "skipped"
    assert decisions[1].reason == "duplicate_bullet_id"
