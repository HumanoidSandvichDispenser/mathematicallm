"""
Equilibrium path representation for game trees.

This module defines classes to represent subgame perfect Nash equilibria,
including support for multiple equilibria when players are indifferent.
"""

from dataclasses import dataclass
from sympy import Expr  # type: ignore


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
