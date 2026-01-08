from agentic_resume_tailor.core.agents.rewrite_validation import validate_rewrite


def test_validate_rewrite_blocks_new_numbers() -> None:
    original = "Improved latency by 20% using Python services."
    rewritten = "Improved latency by 30% using Python services."
    result = validate_rewrite(original, rewritten, {"python"})
    assert not result.ok
    assert "new_numbers" in result.violations


def test_validate_rewrite_blocks_new_tools() -> None:
    original = "Built APIs in Python."
    rewritten = "Built APIs in Python and C++."
    result = validate_rewrite(original, rewritten, {"python"})
    assert not result.ok
    assert "new_tools" in result.violations
