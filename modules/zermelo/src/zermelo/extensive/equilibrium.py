"""
Equilibrium path representation for game trees.

This module defines classes to represent subgame perfect Nash equilibria,
including support for multiple equilibria when players are indifferent.
"""

from dataclasses import dataclass
from sympy import Expr  # type: ignore
from ..extensive.strategy import Strategy


@dataclass
class EquilibriumPath:
    """
    Represents one subgame perfect Nash equilibrium path through the game tree.

    When there are multiple equilibria (due to ties in payoffs), each equilibrium
    is represented as a separate EquilibriumPath object.

    Attributes:
        payoffs: Equilibrium payoff tuple (one value per player)
        actions: Map from decision node ID -> chosen child ID along this equilibrium path
    """

    payoffs: tuple[Expr, ...]
    actions: dict[str, str]  # node_id -> child_id

    def __str__(self) -> str:
        """String representation of the equilibrium."""
        payoff_strs = [str(p) for p in self.payoffs]
        return f"Payoffs: ({', '.join(payoff_strs)}), Actions: {self.actions}"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"EquilibriumPath(payoffs={self.payoffs}, actions={self.actions})"


@dataclass
class SubgamePerfectEquilibrium:
    """
    Represents a subgame perfect Nash equilibrium, including both the equilibrium
    path and the complete strategies each player follows.

    The strategies specify what each player would do at ALL their information sets,
    not just those on the equilibrium path. This is important because if a player
    deviates from the equilibrium path, the other players' strategies still specify
    their optimal responses at every subgame.

    Attributes:
        payoffs: Equilibrium payoff tuple (one value per player)
        path: Map from decision node ID -> chosen child ID along this equilibrium path
        strategies: List of Strategy objects, one per player
    """

    payoffs: tuple[Expr, ...]
    path: dict[str, str]
    strategies: list[Strategy]

    def __str__(self) -> str:
        """String representation of the equilibrium."""
        payoff_strs = [str(p) for p in self.payoffs]
        lines = [
            f"Payoffs: ({', '.join(payoff_strs)})",
            f"Path: {self.path}",
        ]
        for i, strategy in enumerate(self.strategies):
            lines.append(f"Player {i} strategy: {strategy}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Developer representation."""
        return f"SubgamePerfectEquilibrium(payoffs={self.payoffs}, path={self.path}, strategies={self.strategies})"
