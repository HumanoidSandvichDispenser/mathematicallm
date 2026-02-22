from collections.abc import Mapping
from .strategy import Strategy
from sympy import Expr


class MixedStrategy(Mapping):
    """
    A mixed strategy is a probability distribution over pure strategies. It is
    represented as a mapping from pure strategies to their probabilities.
    """

    def __init__(self, strategy_probs: dict[Strategy, Expr]):
        self._strategy_probs: dict[Strategy, Expr] = dict(strategy_probs)

    def __getitem__(self, strategy: Strategy) -> Expr:
        return self._strategy_probs[strategy]

    def __iter__(self):
        return iter(self._strategy_probs)

    def __len__(self):
        return len(self._strategy_probs)

    def __str__(self):
        items = ", ".join(
            f"{s}: {p}"
            for s, p in sorted(self._strategy_probs.items(), key=lambda x: str(x[0]))
        )
        return f"{{{items}}}"

    def __repr__(self):
        items = ", ".join(
            f"{s}: {p}"
            for s, p in sorted(self._strategy_probs.items(), key=lambda x: str(x[0]))
        )
        return f"MixedStrategy({items})"
