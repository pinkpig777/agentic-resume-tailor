from agentic_resume_tailor.core.retrieval import _compute_quant_bonus


def test_quant_bonus_for_numbers() -> None:
    """Test quant bonus for numbers."""
    bonus = _compute_quant_bonus("Improved latency by 45% and cut costs by 2x")
    assert bonus >= 0.05


def test_quant_bonus_absent() -> None:
    """Test quant bonus absent."""
    bonus = _compute_quant_bonus("Built a scalable service for users")
    assert bonus == 0.0


def test_quant_bonus_caps() -> None:
    """Test quant bonus caps at configured max."""
    text = (
        "Improved latency by 45% from 200 ms to 50 ms, cut costs by 2x for 10 users, "
        "processed 1gb data, auc 0.91"
    )
    bonus = _compute_quant_bonus(text)
    assert bonus <= 0.20
    assert bonus == 0.20
