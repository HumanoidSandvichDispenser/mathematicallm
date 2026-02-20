"""
Data payload types for game tree nodes.

Uses composable dataclass mixins to represent different node types.
Node type is determined by isinstance checks rather than discriminated unions.
"""

from dataclasses import dataclass, field
from typing import Optional
from sympy import Expr, sympify


@dataclass
class NodeData:
    """
    Base class for all node types.

    The probability field is set only on children of chance nodes, representing
    the probability of the edge leading INTO this node.

    Attributes:
        probability: Probability of the edge leading to this node (for chance outcomes)
    """

    probability: Optional[Expr] = None

    def __post_init__(self):
        if self.probability is not None:
            self.probability = sympify(self.probability)


@dataclass
class BIValue:
    """
    Mixin for nodes that receive a backed-up value from backward induction.

    This mixin provides a bi_value field that is mutated during backward induction
    to store the computed value at this node. Only DecisionNodeData and ChanceNodeData
    inherit this mixin; TerminalNodeData implements bi_value as a property.

    Attributes:
        bi_value: Tuple of symbolic expressions representing each player's expected payoff
        optimal_children: List of child node IDs that achieve optimal payoff (for decision nodes with ties)
    """

    bi_value: Optional[tuple[Expr, ...]] = None
    optimal_children: list[str] = field(default_factory=list)


@dataclass
class DecisionNodeData(BIValue, NodeData):
    """
    A node where a player makes a choice.

    Attributes:
        player: Zero-indexed player number who makes the decision
        information_set: Identifier for the information set this node belongs to.
            If None, defaults to the node's identifier (single-node info set).
            Nodes in the same information set must have the same player and
            the same set of available actions.
        bi_value: Computed value from backward induction (inherited from BIValue)
        probability: Edge probability (inherited from NodeData)
    """

    player: int = 0
    information_set: Optional[str] = None


@dataclass
class ChanceNodeData(BIValue, NodeData):
    """
    A node where nature moves randomly.

    Probabilities live on the children via NodeData.probability, representing
    the probability of each outcome. Probabilities do not need to sum to 1.

    Attributes:
        bi_value: Computed expected value from backward induction (inherited from BIValue)
        probability: Edge probability (inherited from NodeData)
    """

    pass


@dataclass
class TerminalNodeData(NodeData):
    """
    A leaf node with payoffs.

    The bi_value is implemented as a property that returns payoffs, rather than
    as a mutable field. This keeps terminal nodes immutable after construction.

    Attributes:
        payoffs: Tuple of symbolic expressions, one for each player
        probability: Edge probability (inherited from NodeData)
    """

    __payoffs: tuple[Expr, ...] = field(default_factory=tuple, init=False, repr=False)

    def __init__(
        self,
        payoffs: tuple[Expr | int, ...],
        probability: Optional[Expr | int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.payoffs = self.__sympify_tuple(payoffs)

        if probability is not None:
            self.probability = self.__coalesce_sympy(probability)

    def __coalesce_sympy(self, value: Expr | int) -> Expr:
        return value if isinstance(value, Expr) else sympify(value)

    def __sympify_tuple(self, values: tuple[Expr | int, ...]) -> tuple[Expr, ...]:
        return tuple(self.__coalesce_sympy(v) for v in values)

    def __post_init__(self):
        super().__post_init__()
        #self.payoffs = tuple(sympify(p) for p in self.payoffs)

    @property
    def bi_value(self) -> tuple[Expr, ...]:
        """Returns the payoffs as the backward induction value."""
        return self.payoffs
