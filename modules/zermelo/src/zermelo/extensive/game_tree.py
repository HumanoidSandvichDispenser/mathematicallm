"""
Extensive-form game tree implementation.

Provides a GameTree class extending treelib.Tree with game-theoretic algorithms
like backward induction and utilities for building, inspecting, and solving
extensive-form games.
"""

import treelib.tree  # type: ignore
from typing import Optional, Sequence
import sympy as sp
from sympy import Expr, sympify, simplify
from .node_data import (
    NodeData,
    BIValue,
    DecisionNodeData,
    ChanceNodeData,
    TerminalNodeData,
)
from .game_node import GameNode
from .equilibrium import EquilibriumPath


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

    def __init__(self, num_players: int = 2, tree=None, deep=False, node_class=GameNode, **kwargs):
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

    def all_nodes(self) -> list[GameNode]:
        """
        Get a list of all nodes in the tree, typed as GameNode.
        """
        return list(self._nodes.values())

    def backward_induction(
        self,
        node_id: Optional[str] = None,
        mutate: bool = False
    ) -> tuple[Expr, ...]:
        """
        Compute backward induction solution from the given node.

        For each node, computes the expected payoff vector by working backwards
        from terminal nodes:
        - Terminal nodes: return their payoff tuple
        - Chance nodes: return probability-weighted sum of children
        - Decision nodes: return the child that maximizes the current player's payoff

        The bi_value field is mutated on all BIValue nodes during computation.

        Args:
            node_id: Starting node (defaults to root)
            mutate: If True, modifies tree in-place; if False, works on a copy

        Returns:
            Tuple of expressions representing each player's expected payoffs

        Raises:
            ValueError: If node_id not found in tree
        """
        if node_id is None:
            if self.root is None:
                raise ValueError("Tree is empty (no root node)")
            node_id = self.root

        if not mutate:
            # Work on a copy to avoid modifying the original tree
            subtree_copy = self.subtree(node_id)
            if subtree_copy.root is None:
                raise ValueError(f"Node with id {node_id} not found in the tree.")
            tree = GameTree(num_players=self.num_players, tree=subtree_copy, deep=True)
            if tree.root is None:
                raise ValueError(f"Subtree root is None")
            return tree.backward_induction(node_id=tree.root, mutate=True)

        # Get the node
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node with id {node_id} not found in the tree.")

        # Base case: terminal node
        if isinstance(node.data, TerminalNodeData):
            return node.data.bi_value

        # Recursive case: process all children first
        children = self.children(node_id)
        if not children:
            raise ValueError(
                f"Non-terminal node {node_id} has no children. "
                f"Node type: {type(node.data).__name__}"
            )

        child_payoffs = [
            self.backward_induction(child.identifier, mutate=True)
            for child in children
        ]

        # Chance node: expected value (probability-weighted sum)
        if isinstance(node.data, ChanceNodeData):
            result = self._compute_expected_value(children, child_payoffs)
            node.data.bi_value = result
            return result

        # Decision node: maximize current player's payoff
        if isinstance(node.data, DecisionNodeData):
            result, optimal_indices = self._compute_optimal_choice(node.data.player, children, child_payoffs)
            node.data.bi_value = result
            # Store the IDs of all optimal children
            node.data.optimal_children = [children[idx].identifier for idx in optimal_indices]
            return result

        raise ValueError(
            f"Unknown node type at {node_id}: {type(node.data).__name__}"
        )

    def get_all_equilibria(self, node_id: Optional[str] = None) -> list[EquilibriumPath]:
        """
        Enumerate all subgame perfect Nash equilibria.

        This method must be called AFTER backward_induction() has been run with mutate=True.
        It enumerates all equilibrium paths by recursively exploring all optimal choices
        at decision nodes.

        When a decision node has multiple optimal children (ties), this creates multiple
        equilibrium paths - one for each combination of choices across all such nodes.

        Args:
            node_id: Starting node (defaults to root)

        Returns:
            List of EquilibriumPath objects, one per equilibrium

        Raises:
            ValueError: If backward_induction hasn't been run yet (bi_value not set)

        Example:
            tree.backward_induction(mutate=True)
            equilibria = tree.get_all_equilibria()
            for eq in equilibria:
                print(f"Payoffs: {eq.payoffs}, Actions: {eq.actions}")
        """
        if node_id is None:
            if self.root is None:
                raise ValueError("Tree is empty (no root node)")
            node_id = self.root

        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node with id {node_id} not found")

        # Check that BI has been run
        if hasattr(node.data, 'bi_value'):
            if node.data.bi_value is None:
                raise ValueError(
                    "backward_induction has not been run yet. "
                    "Call tree.backward_induction(mutate=True) first."
                )

        # Helper function to recursively enumerate equilibria
        def enumerate_paths(current_node_id: str, current_actions: dict[str, str]) -> list[dict[str, str]]:
            """
            Recursively enumerate all equilibrium action profiles from this node down.

            Returns list of action dictionaries, each mapping decision_node_id -> child_id
            """
            current_node = self.get_node(current_node_id)

            assert isinstance(current_node, GameNode)

            # Base case: terminal node
            if current_node.is_terminal:
                return [current_actions.copy()]

            # Chance node: follow all children (nature doesn't choose strategically)
            if current_node.is_chance:
                all_paths = []
                for child in self.children(current_node_id):
                    child_paths = enumerate_paths(child.identifier, current_actions)
                    all_paths.extend(child_paths)
                return all_paths if all_paths else [current_actions.copy()]

            # Decision node: follow all optimal children
            if current_node.is_decision:
                if not current_node.data.optimal_children:
                    raise ValueError(
                        f"Decision node {current_node_id} has no optimal children. "
                        "Ensure backward_induction was run with mutate=True."
                    )

                all_paths = []
                for optimal_child_id in current_node.data.optimal_children:
                    # Record this action
                    new_actions = current_actions.copy()
                    new_actions[current_node_id] = optimal_child_id

                    # Recurse into this child
                    child_paths = enumerate_paths(optimal_child_id, new_actions)
                    all_paths.extend(child_paths)

                return all_paths

            # Unknown node type
            raise ValueError(f"Unknown node type at {current_node_id}")

        # Enumerate all action profiles
        action_profiles = enumerate_paths(node_id, {})  # type: ignore

        # Convert to EquilibriumPath objects
        root_node = self.get_node(node_id)
        assert isinstance(root_node, GameNode)
        payoffs = root_node.data.bi_value if hasattr(root_node.data, 'bi_value') else root_node.data.payoffs

        equilibria = [
            EquilibriumPath(payoffs=payoffs, actions=actions)
            for actions in action_profiles
        ]

        return equilibria


    def _compute_expected_value(
        self,
        children: Sequence,  # type: ignore
        child_payoffs: list[tuple[Expr, ...]]
    ) -> tuple[Expr, ...]:
        """
        Compute expected value across chance node children.

        Each child should have a probability set (via child.data.probability).
        Probabilities do not need to sum to 1.

        Args:
            children: List of child nodes (must have probability set)
            child_payoffs: Corresponding payoff tuples for each child

        Returns:
            Tuple of expected payoffs, one for each player
        """
        if not child_payoffs:
            raise ValueError("Cannot compute expected value with no children")

        # Determine number of players from first child
        num_players = len(child_payoffs[0])

        # Compute weighted sum for each player
        expected = []
        for player_idx in range(num_players):
            weighted_sum = sp.Integer(0)
            for child, payoffs in zip(children, child_payoffs):
                prob = child.data.probability if child.data.probability is not None else sp.Integer(1)
                player_payoff = payoffs[player_idx]
                weighted_sum += prob * player_payoff
            expected.append(simplify(weighted_sum))

        return tuple(expected)

    def _compute_optimal_choice(
        self,
        player: int,
        children: Sequence,  # type: ignore
        child_payoffs: list[tuple[Expr, ...]]
    ) -> tuple[tuple[Expr, ...], list[int]]:
        """
        Select ALL children that maximize the decision-maker's payoff.

        When multiple children yield the same optimal payoff (ties), all are returned.
        This allows detecting multiple perfect Nash equilibria.

        Args:
            player: Zero-indexed player number who owns this decision node
            children: List of child nodes
            child_payoffs: Corresponding payoff tuples for each child

        Returns:
            Tuple of (optimal_payoff, list_of_optimal_child_indices)

        Note:
            For symbolic payoffs, this uses sympy's simplify and considers payoffs
            equal if their difference simplifies to zero. For indeterminate comparisons,
            assumes they could be equal.
        """
        if not child_payoffs:
            raise ValueError("No children to choose from")

        best_value = child_payoffs[0][player]
        best_indices = [0]

        for idx, payoffs in enumerate(child_payoffs[1:], start=1):
            current_value = payoffs[player]

            # Try to determine relationship between current and best
            diff = simplify(current_value - best_value)

            # Check if current > best
            is_better = False
            if diff.is_number:
                is_better = diff > 0
            elif diff.is_positive:
                is_better = True

            # Check if current == best (tie)
            is_equal = False
            if diff.is_number:
                is_equal = (diff == 0)
            elif diff.is_zero:
                is_equal = True

            if is_better:
                # Found strictly better option - reset best
                best_value = current_value
                best_indices = [idx]
            elif is_equal:
                # Found a tie - add to list
                best_indices.append(idx)

        return (child_payoffs[best_indices[0]], best_indices)

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
                    "probability": sp.srepr(node.data.probability) if node.data.probability else None,
                }
                if include_bi_values and node.data.bi_value is not None:
                    node_dict["data"]["bi_value"] = tuple(sp.srepr(v) for v in node.data.bi_value)
            elif isinstance(node.data, ChanceNodeData):
                node_dict["data"] = {
                    "type": "chance",
                    "probability": sp.srepr(node.data.probability) if node.data.probability else None,
                }
                if include_bi_values and node.data.bi_value is not None:
                    node_dict["data"]["bi_value"] = tuple(sp.srepr(v) for v in node.data.bi_value)
            elif isinstance(node.data, TerminalNodeData):
                node_dict["data"] = {
                    "type": "terminal",
                    "probability": sp.srepr(node.data.probability) if node.data.probability else None,
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
                prob = sp.sympify(node_data["data"]["probability"]) if node_data["data"]["probability"] else None

                if dtype == "decision":
                    bi_val = None
                    if "bi_value" in node_data["data"] and node_data["data"]["bi_value"]:
                        bi_val = tuple(sp.sympify(v) for v in node_data["data"]["bi_value"])
                    data_payload = DecisionNodeData(
                        player=node_data["data"]["player"],
                        probability=prob,
                        bi_value=bi_val,
                    )
                elif dtype == "chance":
                    bi_val = None
                    if "bi_value" in node_data["data"] and node_data["data"]["bi_value"]:
                        bi_val = tuple(sp.sympify(v) for v in node_data["data"]["bi_value"])
                    data_payload = ChanceNodeData(
                        probability=prob,
                        bi_value=bi_val,
                    )
                elif dtype == "terminal":
                    data_payload = TerminalNodeData(
                        probability=prob,
                        payoffs=tuple(sp.sympify(p) for p in node_data["data"]["payoffs"]),
                    )

            # Create the node
            tree.create_node(
                tag=node_data["tag"],
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
        line_type: str = 'ascii-ex',
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
                if hasattr(node.data, 'bi_value') and node.data.bi_value is not None:
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
