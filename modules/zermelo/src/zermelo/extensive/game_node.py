"""
Game tree node class that works seamlessly with treelib.

GameNode is a thin wrapper around treelib.Node that ensures the data field
contains NodeData instances (DecisionNodeData, ChanceNodeData, or TerminalNodeData).
"""

from treelib.node import Node  # type: ignore
from typing import Optional
from .node_data import NodeData, DecisionNodeData, ChanceNodeData, TerminalNodeData


class GameNode(Node):  # type: ignore
    """
    A node in an extensive-form game tree.

    The node's .data field always contains a NodeData instance (DecisionNodeData,
    ChanceNodeData, or TerminalNodeData). This design keeps the node class uniform and
    compatible with treelib's tree operations while providing type-safe node semantics.

    Usage:
        # Create a decision node
        node = GameNode(
            tag="Player 0 decision",
            identifier="n1",
            data=DecisionNodeData(player=0)
        )

        # Create a terminal node
        node = GameNode(
            tag="Outcome",
            identifier="t1",
            data=TerminalNodeData(payoffs=(3, 3))
        )

        # Create a chance outcome (child of chance node)
        node = GameNode(
            tag="Heads",
            identifier="heads",
            data=ChanceNodeData(probability=0.5)
        )
    """

    def __init__(
        self,
        tag: Optional[str] = None,
        identifier: Optional[str] = None,
        expanded: bool = True,
        data: Optional[NodeData] = None,
    ) -> None:
        """
        Initialize a GameNode.

        Args:
            tag: Human-readable label for the node
            identifier: Unique identifier (auto-generated if None)
            expanded: Whether node is expanded in tree display
            data: NodeData instance (DecisionNodeData, ChanceNodeData, or TerminalNodeData)
        """
        super().__init__(tag=tag, identifier=identifier, expanded=expanded, data=data)

    @property
    def is_decision(self) -> bool:
        """True if this is a decision node."""
        return isinstance(self.data, DecisionNodeData)

    @property
    def is_chance(self) -> bool:
        """True if this is a chance node."""
        return isinstance(self.data, ChanceNodeData)

    @property
    def is_terminal(self) -> bool:
        """True if this is a terminal node."""
        return isinstance(self.data, TerminalNodeData)

    def __str__(self) -> str:
        """
        Custom string representation for tree display.

        Shows: [name] (id) [probability] [BI result]
        """
        parts = []

        # Name (tag)
        if self.tag:
            parts.append(str(self.tag))

        # ID
        parts.append(f"[{self.identifier}]")

        # Node type and relevant info
        if self.is_decision:
            parts.append(f"P{self.data.player}")
        elif self.is_chance:
            parts.append("CHANCE")
        elif self.is_terminal:
            payoffs_str = ", ".join(str(p) for p in self.data.payoffs)
            parts.append(f"({payoffs_str})")

        # Probability (if set)
        if self.data and self.data.probability is not None:
            parts.append(f"p={self.data.probability}")

        # Backward induction result (if available)
        if hasattr(self.data, 'bi_value') and self.data.bi_value is not None:
            bi_str = ", ".join(str(v) for v in self.data.bi_value)
            parts.append(f"BI=({bi_str})")
        elif self.is_terminal:
            # Terminal nodes have bi_value as property
            bi_str = ", ".join(str(v) for v in self.data.bi_value)
            parts.append(f"BI=({bi_str})")

        return " ".join(parts)
