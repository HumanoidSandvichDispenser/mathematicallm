from server import find_stable_matching


def test_find_stable_matching_reports_partial_matching() -> None:
    result = find_stable_matching(
        proposer_preferences={
            "A": ["X"],
            "B": ["X"],
            "C": [],
        },
        receiver_preferences={
            "X": ["B", "A"],
            "Y": [],
        },
    )

    assert "Deferred acceptance result:" in result
    assert "A -> unmatched" in result
    assert "B -> X" in result
    assert "C -> unmatched" in result
    assert "Unmatched proposers: A, C" in result
    assert "Unmatched receivers: Y" in result


def test_find_stable_matching_reports_validation_errors() -> None:
    result = find_stable_matching(
        proposer_preferences={"A": ["X"]},
        receiver_preferences={"Y": ["A"]},
    )

    assert result.startswith("Error:")
    assert "unknown receiver" in result
