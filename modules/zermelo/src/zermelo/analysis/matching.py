from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar


ProposerT = TypeVar("ProposerT", bound=Hashable)
ReceiverT = TypeVar("ReceiverT", bound=Hashable)


@dataclass(frozen=True)
class DeferredAcceptanceResult(Generic[ProposerT, ReceiverT]):
    proposer_matches: dict[ProposerT, ReceiverT | None]
    receiver_matches: dict[ReceiverT, ProposerT | None]

    @property
    def matched_pairs(self) -> list[tuple[ProposerT, ReceiverT]]:
        return [
            (proposer, receiver)
            for proposer, receiver in self.proposer_matches.items()
            if receiver is not None
        ]

    @property
    def unmatched_proposers(self) -> list[ProposerT]:
        return [
            proposer
            for proposer, receiver in self.proposer_matches.items()
            if receiver is None
        ]

    @property
    def unmatched_receivers(self) -> list[ReceiverT]:
        return [
            receiver
            for receiver, proposer in self.receiver_matches.items()
            if proposer is None
        ]


def deferred_acceptance(
    proposer_preferences: Mapping[ProposerT, Sequence[ReceiverT]],
    receiver_preferences: Mapping[ReceiverT, Sequence[ProposerT]],
) -> DeferredAcceptanceResult[ProposerT, ReceiverT]:
    """
    Compute a stable matching using deferred acceptance with proposers on the
    left side and receivers on the right side.

    Preference lists may be incomplete. A missing partner is treated as
    unacceptable, so the result may be a partial matching.
    """
    proposers = list(proposer_preferences)
    receivers = list(receiver_preferences)

    _validate_preferences(proposer_preferences, receiver_preferences)

    receiver_rankings = {
        receiver: {proposer: rank for rank, proposer in enumerate(preferences)}
        for receiver, preferences in receiver_preferences.items()
    }

    proposer_matches: dict[ProposerT, ReceiverT | None] = {
        proposer: None for proposer in proposers
    }
    receiver_matches: dict[ReceiverT, ProposerT | None] = {
        receiver: None for receiver in receivers
    }
    next_choice_index = {proposer: 0 for proposer in proposers}
    unmatched_proposers = deque(proposers)

    while unmatched_proposers:
        proposer = unmatched_proposers.popleft()
        preferences = proposer_preferences[proposer]

        while proposer_matches[proposer] is None and next_choice_index[proposer] < len(
            preferences
        ):
            receiver = preferences[next_choice_index[proposer]]
            next_choice_index[proposer] += 1

            ranking = receiver_rankings[receiver]
            if proposer not in ranking:
                continue

            current_match = receiver_matches[receiver]
            if current_match is None:
                proposer_matches[proposer] = receiver
                receiver_matches[receiver] = proposer
                break

            if ranking[proposer] < ranking[current_match]:
                proposer_matches[current_match] = None
                unmatched_proposers.append(current_match)
                proposer_matches[proposer] = receiver
                receiver_matches[receiver] = proposer
                break

    return DeferredAcceptanceResult(
        proposer_matches=proposer_matches,
        receiver_matches=receiver_matches,
    )


def _validate_preferences(
    proposer_preferences: Mapping[ProposerT, Sequence[ReceiverT]],
    receiver_preferences: Mapping[ReceiverT, Sequence[ProposerT]],
) -> None:
    proposers = set(proposer_preferences)
    receivers = set(receiver_preferences)

    for proposer, preferences in proposer_preferences.items():
        seen_receivers: set[ReceiverT] = set()
        for receiver in preferences:
            if receiver not in receivers:
                raise ValueError(
                    f"Proposer {proposer!r} ranks unknown receiver {receiver!r}."
                )
            if receiver in seen_receivers:
                raise ValueError(
                    f"Proposer {proposer!r} ranks receiver {receiver!r} more than once."
                )
            seen_receivers.add(receiver)

    for receiver, preferences in receiver_preferences.items():
        seen_proposers: set[ProposerT] = set()
        for proposer in preferences:
            if proposer not in proposers:
                raise ValueError(
                    f"Receiver {receiver!r} ranks unknown proposer {proposer!r}."
                )
            if proposer in seen_proposers:
                raise ValueError(
                    f"Receiver {receiver!r} ranks proposer {proposer!r} more than once."
                )
            seen_proposers.add(proposer)
