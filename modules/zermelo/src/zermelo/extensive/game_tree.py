"""
Extensive-form game tree implementation.

Provides a GameTree class extending treelib.Tree with game-theoretic algorithms
like backward induction and utilities for building, inspecting, and solving
extensive-form games.
"""

import treelib.tree  # type: ignore
from typing import Optional
import sympy as sp
from sympy import Expr
from .node_data import (
    NodeData,
    BIValue,
    DecisionNodeData,
    ChanceNodeData,
    TerminalNodeData,
)
from .game_node import GameNode
from .equilibrium import EquilibriumPath, SubgamePerfectEquilibrium


class GameTree(treelib.tree.Tree):  # type: ignore
    """
    An extensive-form game tree.

    Extends treelib.Tree to support game-theoretic operations on trees with
    decision nodes (players), chance nodes (nature), and terminal nodes (payoffs).

    Example:
        # Build a simple game tree
        tree = GameTree(num_players=2)
        tree.create_node(
            tag="Root",
            identifier="root",
            data=DecisionNodeData(player=0)
        )
        tree.create_node(
            tag="Left outcome",
            identifier="t1",
            parent="root",
            data=TerminalNodeData(payoffs=(1, 0))
        )

        # Solve with backward induction
        tree.backward_induction(mutate=True)
        result = tree.get_node("root").data.bi_value
    """

    def __init__(
        self, num_players: int = 2, tree=None, deep=False, node_class=GameNode, **kwargs
    ):
        """
        Initialize a GameTree.

        Args:
            num_players: Number of players in the game (default: 2)
            tree: Optional tree to copy from
            deep: Whether to deep-copy the tree
            node_class: Node class to use (defaults to GameNode)
            **kwargs: Additional arguments passed to treelib.Tree
        """
        self.num_players = num_players
        super().__init__(tree=tree, deep=deep, node_class=node_class, **kwargs)

    def get_information_sets(self, player: int) -> set[str]:
        """
        Get all information set IDs for a given player.

        Args:
            player: Zero-indexed player number

        Returns:
            Set of information set IDs belonging to this player
        """
        info_sets = set()
        for node in self.all_nodes():
            if node.is_decision and node.data.player == player:
                info_set_id = node.data.information_set
                if info_set_id is None:
                    info_set_id = node.identifier
                info_sets.add(info_set_id)
        return info_sets

    def get_nodes_in_information_set(self, info_set_id: str) -> list[GameNode]:
        """
        Get all nodes belonging to a given information set.

        Args:
            info_set_id: The information set identifier

        Returns:
            List of GameNodes in this information set
        """
        nodes = []
        for node in self.all_nodes():
            if node.is_decision:
                node_info_set = node.data.information_set
                if node_info_set is None:
                    node_info_set = node.identifier
                if node_info_set == info_set_id:
                    nodes.append(node)
        return nodes

    def is_perfect_information(self) -> bool:
        """
        Check if this is a perfect information game.

        A game is perfect information if every information set contains exactly
        one node. If any player has an information set with multiple nodes,
        the game is imperfect information.

        Returns:
            True if the game is perfect information, False otherwise
        """
        for node in self.all_nodes():
            if node.is_decision:
                info_set_id = node.data.information_set
                if info_set_id is None:
                    info_set_id = node.identifier
                nodes_in_set = self.get_nodes_in_information_set(info_set_id)
                if len(nodes_in_set) > 1:
                    return False
        return True

    def get_node(self, nid: str) -> GameNode | None:
        """
        Get a node by its identifier, typed as GameNode.
        """
        if nid is None or not self.contains(nid):
            return None
        return self._nodes[nid]

    def all_nodes(self) -> list[GameNode]:
        """
        Get a list of all nodes in the tree, typed as GameNode.
        """
        return list(self._nodes.values())

    def to_dict(self, include_bi_values: bool = True) -> dict:
        """
        Serialize the game tree to a JSON-compatible dictionary.

        Args:
            include_bi_values: Whether to include backward induction values

        Returns:
            Dictionary with 'nodes' list and tree metadata
        """
        nodes_data = []
        for node in self.all_nodes():
            parent_node = self.parent(node.identifier)
            node_dict = {
                "id": node.identifier,
                "tag": node.tag,
                "parent": parent_node.identifier if parent_node else None,
            }

            # Serialize node data based on type
            if isinstance(node.data, DecisionNodeData):
                node_dict["data"] = {
                    "type": "decision",
                    "player": node.data.player,
                    "information_set": node.data.information_set,
                    "probability": sp.srepr(node.data.probability)
                    if node.data.probability
                    else None,
                }
                if include_bi_values and node.data.bi_value is not None:
                    node_dict["data"]["bi_value"] = tuple(
                        sp.srepr(v) for v in node.data.bi_value
                    )
            elif isinstance(node.data, ChanceNodeData):
                node_dict["data"] = {
                    "type": "chance",
                    "probability": sp.srepr(node.data.probability)
                    if node.data.probability
                    else None,
                }
                if include_bi_values and node.data.bi_value is not None:
                    node_dict["data"]["bi_value"] = tuple(
                        sp.srepr(v) for v in node.data.bi_value
                    )
            elif isinstance(node.data, TerminalNodeData):
                node_dict["data"] = {
                    "type": "terminal",
                    "probability": sp.srepr(node.data.probability)
                    if node.data.probability
                    else None,
                    "payoffs": tuple(sp.srepr(p) for p in node.data.payoffs),
                }
            else:
                # Fallback for nodes without typed data
                node_dict["data"] = None

            nodes_data.append(node_dict)

        return {
            "num_players": self.num_players,
            "nodes": nodes_data,
            "root": self.root,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GameTree":
        """
        Deserialize a game tree from a dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed GameTree instance
        """
        num_players = data.get("num_players", 2)
        tree = cls(num_players=num_players)

        # Build node ID to data mapping
        nodes_by_id = {node["id"]: node for node in data["nodes"]}

        # Create nodes in parent-first order (BFS from root)
        root_id = data["root"]
        queue = [root_id]
        visited = set()

        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)

            node_data = nodes_by_id[node_id]
            parent_id = node_data["parent"]

            # Deserialize the data payload
            data_payload = None
            if node_data["data"]:
                dtype = node_data["data"]["type"]
                raw_prob = node_data["data"].get("probability")
                prob = sp.sympify(raw_prob) if raw_prob is not None else None

                if dtype == "decision":
                    bi_val = None
                    if (
                        "bi_value" in node_data["data"]
                        and node_data["data"]["bi_value"]
                    ):
                        bi_val = tuple(
                            sp.sympify(v) for v in node_data["data"]["bi_value"]
                        )
                    data_payload = DecisionNodeData(
                        player=node_data["data"]["player"],
                        information_set=node_data["data"].get("information_set"),
                        probability=prob,
                        bi_value=bi_val,
                    )
                elif dtype == "chance":
                    bi_val = None
                    if (
                        "bi_value" in node_data["data"]
                        and node_data["data"]["bi_value"]
                    ):
                        bi_val = tuple(
                            sp.sympify(v) for v in node_data["data"]["bi_value"]
                        )
                    data_payload = ChanceNodeData(
                        probability=prob,
                        bi_value=bi_val,
                    )
                elif dtype == "terminal":
                    data_payload = TerminalNodeData(
                        probability=prob,
                        payoffs=tuple(
                            sp.sympify(p) for p in node_data["data"]["payoffs"]
                        ),
                    )

            # Create the node
            tree.create_node(
                tag=node_data.get("tag") or node_id,
                identifier=node_id,
                parent=parent_id,
                data=data_payload,
            )

            # Add children to queue
            for child_node in data["nodes"]:
                if child_node["parent"] == node_id and child_node["id"] not in visited:
                    queue.append(child_node["id"])

        return tree

    def show(
        self,
        nid: Optional[str] = None,
        level: int = 0,
        idhidden: bool = True,
        filter=None,
        key=None,
        reverse: bool = False,
        line_type: str = "ascii-ex",
        data_property: Optional[str] = None,
        stdout: bool = True,
        sorting: bool = True,
        show_id: bool = True,
        show_probability: bool = True,
        show_bi_value: bool = True,
    ):
        """
        Display the game tree with rich node information.

        This extends treelib's show() to include game-specific attributes like
        player info, probabilities, and backward induction values.

        Args:
            nid: Starting node (defaults to root)
            level: Starting level for display
            idhidden: Whether to hide node IDs
            filter: Optional filter function
            key: Optional sorting key function
            reverse: Reverse sort order
            line_type: Tree line style ('ascii', 'ascii-ex', 'ascii-em')
            data_property: Specific data property to show
            stdout: If True, print to stdout; if False, return as string
            sorting: Whether to sort children
            show_id: Show node identifiers (default: True)
            show_probability: Show edge probabilities (default: True)
            show_bi_value: Show backward induction values (default: True)

        Returns:
            String representation if stdout=False, otherwise None
        """
        # Save original tags
        original_tags = {}

        # Temporarily update tags with rich display info
        for node in self.all_nodes():
            assert isinstance(node, GameNode)

            original_tags[node.identifier] = node.tag

            parts = []

            # Name (original tag)
            if node.tag:
                parts.append(str(node.tag))

            # ID
            if show_id:
                parts.append(f"[{node.identifier}]")

            # Node type and relevant info
            if node.is_decision:
                parts.append(f"P{node.data.player}")
            elif node.is_chance:
                parts.append("CHANCE")
            elif node.is_terminal:
                payoffs_str = ", ".join(str(p) for p in node.data.payoffs)
                parts.append(f"({payoffs_str})")

            # Probability (if set)
            if show_probability and node.data and node.data.probability is not None:
                parts.append(f"p={node.data.probability}")

            # Backward induction result (if available)
            if show_bi_value:
                if hasattr(node.data, "bi_value") and node.data.bi_value is not None:
                    bi_str = ", ".join(str(v) for v in node.data.bi_value)
                    parts.append(f"BI=({bi_str})")
                elif node.is_terminal:
                    # Terminal nodes have bi_value as property
                    bi_str = ", ".join(str(v) for v in node.data.bi_value)
                    parts.append(f"BI=({bi_str})")

            # Update the tag
            node.tag = " ".join(parts)

        # Call parent's show method
        try:
            result = super().show(
                nid=nid,
                level=level,
                idhidden=idhidden,
                filter=filter,
                key=key,
                reverse=reverse,
                line_type=line_type,
                data_property=data_property,
                stdout=stdout,
                sorting=sorting,
            )
        finally:
            # Restore original tags
            for node in self.all_nodes():
                node.tag = original_tags[node.identifier]

        return result
