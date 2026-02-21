from collections.abc import Mapping
from sympy import Matrix


class Strategy(Mapping):
    def __init__(self, decisions: dict[str, str]):
        self._decisions: dict[str, str] = dict(decisions)

    def __getitem__(self, info_set: str) -> str:
        return self._decisions[info_set]

    def __iter__(self):
        return iter(self._decisions)

    def __len__(self):
        return len(self._decisions)

    def __repr__(self):
        items = ", ".join(f"{k}: {v}" for k, v in sorted(self._decisions.items()))
        return f"Strategy({items})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Strategy):
            return NotImplemented
        return self._decisions == other._decisions

    def __hash__(self) -> int:
        return hash(frozenset(self._decisions.items()))

    def concat(self, strategy: "Strategy") -> "Strategy":
        """
        Returns a new strategy that combines the decisions of this strategy and
        the given strategy.
        """
        combined_decisions = {**self._decisions, **strategy._decisions}
        return Strategy(combined_decisions)
