import pytest

from zermelo.analysis.matching import (
    DeferredAcceptanceResult,
    deferred_acceptance,
)


def _assert_stable(
    result: DeferredAcceptanceResult[str, str],
    proposer_preferences: dict[str, list[str]],
    receiver_preferences: dict[str, list[str]],
) -> None:
    receiver_rankings = {
        receiver: {proposer: rank for rank, proposer in enumerate(preferences)}
        for receiver, preferences in receiver_preferences.items()
    }

    for proposer, preferences in proposer_preferences.items():
        current_receiver = result.proposer_matches[proposer]
        if current_receiver is None:
            preferred_receivers = preferences
        else:
            cutoff = preferences.index(current_receiver)
            preferred_receivers = preferences[:cutoff]

        for receiver in preferred_receivers:
            ranking = receiver_rankings[receiver]
            if proposer not in ranking:
                continue

            current_proposer = result.receiver_matches[receiver]
            if current_proposer is None:
                raise AssertionError(
                    f"Blocking pair found: {proposer!r} and unmatched receiver {receiver!r}."
                )

            if ranking[proposer] < ranking[current_proposer]:
                raise AssertionError(
                    f"Blocking pair found: {proposer!r} and {receiver!r}."
                )


class TestDeferredAcceptance:
    def test_finds_complete_matching_when_all_are_acceptable(self):
        proposer_preferences = {
            "A": ["X", "Y", "Z"],
            "B": ["Y", "X", "Z"],
            "C": ["Y", "Z", "X"],
        }
        receiver_preferences = {
            "X": ["B", "A", "C"],
            "Y": ["A", "C", "B"],
            "Z": ["A", "B", "C"],
        }

        result = deferred_acceptance(proposer_preferences, receiver_preferences)

        assert result.proposer_matches == {"A": "Y", "B": "X", "C": "Z"}
        assert result.receiver_matches == {"X": "B", "Y": "A", "Z": "C"}
        assert result.unmatched_proposers == []
        assert result.unmatched_receivers == []
        _assert_stable(result, proposer_preferences, receiver_preferences)

    def test_leaves_people_unmatched_when_preferences_are_incomplete(self):
        proposer_preferences = {
            "A": ["X"],
            "B": ["X"],
            "C": [],
        }
        receiver_preferences = {
            "X": ["B", "A"],
            "Y": [],
        }

        result = deferred_acceptance(proposer_preferences, receiver_preferences)

        assert result.proposer_matches == {"A": None, "B": "X", "C": None}
        assert result.receiver_matches == {"X": "B", "Y": None}
        assert result.unmatched_proposers == ["A", "C"]
        assert result.unmatched_receivers == ["Y"]
        _assert_stable(result, proposer_preferences, receiver_preferences)

    def test_receiver_replaces_tentative_match_with_preferred_proposer(self):
        proposer_preferences = {
            "A": ["X", "Y"],
            "B": ["X", "Y"],
        }
        receiver_preferences = {
            "X": ["B", "A"],
            "Y": ["A", "B"],
        }

        result = deferred_acceptance(proposer_preferences, receiver_preferences)

        assert result.proposer_matches == {"A": "Y", "B": "X"}
        assert result.receiver_matches == {"X": "B", "Y": "A"}
        _assert_stable(result, proposer_preferences, receiver_preferences)

    def test_unacceptable_proposal_is_treated_as_rejection(self):
        proposer_preferences = {
            "A": ["X", "Y"],
            "B": ["X"],
        }
        receiver_preferences = {
            "X": ["B"],
            "Y": ["A"],
        }

        result = deferred_acceptance(proposer_preferences, receiver_preferences)

        assert result.proposer_matches == {"A": "Y", "B": "X"}
        assert result.receiver_matches == {"X": "B", "Y": "A"}
        _assert_stable(result, proposer_preferences, receiver_preferences)

    def test_rejects_unknown_participants_and_duplicates(self):
        with pytest.raises(ValueError, match="unknown receiver"):
            deferred_acceptance({"A": ["X"]}, {"Y": ["A"]})

        with pytest.raises(ValueError, match="unknown proposer"):
            deferred_acceptance({"A": ["X"]}, {"X": ["B"]})

        with pytest.raises(ValueError, match="more than once"):
            deferred_acceptance({"A": ["X", "X"]}, {"X": ["A"]})

        with pytest.raises(ValueError, match="more than once"):
            deferred_acceptance({"A": ["X"]}, {"X": ["A", "A"]})
