"""
Solver services for extensive-form games.

Provides dedicated solvers for computing game-theoretic solutions like
backward induction and subgame perfect Nash equilibria.
"""

from __future__ import annotations
from typing import Optional, Sequence, TYPE_CHECKING
import sympy as sp
from sympy import Expr, simplify
from ..extensive.game_node import GameNode
from ..extensive.node_data import (
    DecisionNodeData,
    ChanceNodeData,
    TerminalNodeData,
)
from ..extensive.equilibrium import EquilibriumPath, SubgamePerfectEquilibrium
from ..extensive.strategy import Strategy

if TYPE_CHECKING:
    from ..extensive.game_tree import GameTree


def backward_induction(
    tree: "GameTree",
    node_id: Optional[str] = None,
    mutate: bool = False,
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
        tree: The game tree to solve
        node_id: Starting node (defaults to root)
        mutate: If True, modifies tree in-place; if False, works on a copy

    Returns:
        Tuple of expressions representing each player's expected payoffs

    Raises:
        ValueError: If node_id not found in tree
    """
    if node_id is None:
        if tree.root is None:
            raise ValueError("Tree is empty (no root node)")
        node_id = tree.root

    if not tree.is_perfect_information():
        raise NotImplementedError(
            "This game has imperfect information (multiple nodes per information set). "
            "Backward induction requires perfect information. "
            "Use algorithms like Counterfactual Regret Minimization (CFR) for imperfect information games."
        )

    if not mutate:
        subtree_copy = tree.subtree(node_id)
        if subtree_copy.root is None:
            raise ValueError(f"Node with id {node_id} not found in the tree.")
        from ..extensive.game_tree import GameTree

        tree_copy = GameTree(num_players=tree.num_players, tree=subtree_copy, deep=True)
        if tree_copy.root is None:
            raise ValueError(f"Subtree root is None")
        return backward_induction(tree_copy, node_id=tree_copy.root, mutate=True)

    node = tree.get_node(node_id)
    if node is None:
        raise ValueError(f"Node with id {node_id} not found in the tree.")

    if isinstance(node.data, TerminalNodeData):
        return node.data.bi_value

    children = tree.children(node_id)
    if not children:
        raise ValueError(
            f"Non-terminal node {node_id} has no children. "
            f"Node type: {type(node.data).__name__}"
        )

    child_payoffs = [
        backward_induction(tree, child.identifier, mutate=True) for child in children
    ]

    if isinstance(node.data, ChanceNodeData):
        result = _compute_expected_value(children, child_payoffs)
        node.data.bi_value = result
        return result

    if isinstance(node.data, DecisionNodeData):
        result, optimal_indices = _compute_optimal_choice(
            node.data.player, children, child_payoffs
        )
        node.data.bi_value = result
        node.data.optimal_children = [
            children[idx].identifier for idx in optimal_indices
        ]
        return result

    raise ValueError(f"Unknown node type at {node_id}: {type(node.data).__name__}")


def get_all_equilibria(
    tree: "GameTree",
    node_id: Optional[str] = None,
) -> list[EquilibriumPath]:
    """
    Enumerate all subgame perfect Nash equilibria.

    This method must be called AFTER backward_induction() has been run with mutate=True.
    It enumerates all equilibrium paths by recursively exploring all optimal choices
    at decision nodes.

    When a decision node has multiple optimal children (ties), this creates multiple
    equilibrium paths - one for each combination of choices across all such nodes.

    Args:
        tree: The game tree to solve
        node_id: Starting node (defaults to root)

    Returns:
        List of EquilibriumPath objects, one per equilibrium

    Raises:
        ValueError: If backward_induction hasn't been run yet (bi_value not set)

    Example:
        backward_induction(tree, mutate=True)
        equilibria = get_all_equilibria(tree)
        for eq in equilibria:
            print(f"Payoffs: {eq.payoffs}, Actions: {eq.actions}")
    """
    if node_id is None:
        if tree.root is None:
            raise ValueError("Tree is empty (no root node)")
        node_id = tree.root

    node = tree.get_node(node_id)
    if node is None:
        raise ValueError(f"Node with id {node_id} not found")

    if hasattr(node.data, "bi_value"):
        if node.data.bi_value is None:
            raise ValueError(
                "backward_induction has not been run yet. "
                "Call backward_induction(tree, mutate=True) first."
            )

    def enumerate_paths(
        current_node_id: str, current_actions: dict[str, str]
    ) -> list[dict[str, str]]:
        current_node = tree.get_node(current_node_id)

        assert isinstance(current_node, GameNode)

        if current_node.is_terminal:
            return [current_actions.copy()]

        if current_node.is_chance:
            all_paths = []
            for child in tree.children(current_node_id):
                child_paths = enumerate_paths(child.identifier, current_actions)
                all_paths.extend(child_paths)
            return all_paths if all_paths else [current_actions.copy()]

        if current_node.is_decision:
            if not current_node.data.optimal_children:
                raise ValueError(
                    f"Decision node {current_node_id} has no optimal children. "
                    "Ensure backward_induction was run with mutate=True."
                )

            all_paths = []
            for optimal_child_id in current_node.data.optimal_children:
                new_actions = current_actions.copy()
                new_actions[current_node_id] = optimal_child_id

                child_paths = enumerate_paths(optimal_child_id, new_actions)
                all_paths.extend(child_paths)

            return all_paths

        raise ValueError(f"Unknown node type at {current_node_id}")

    action_profiles = enumerate_paths(node_id, {})

    root_node = tree.get_node(node_id)
    assert isinstance(root_node, GameNode)
    payoffs = (
        root_node.data.bi_value
        if hasattr(root_node.data, "bi_value")
        else root_node.data.payoffs
    )

    equilibria = [
        EquilibriumPath(payoffs=payoffs, actions=actions) for actions in action_profiles
    ]

    return equilibria


def get_all_spne(
    tree: "GameTree",
    node_id: Optional[str] = None,
) -> list[SubgamePerfectEquilibrium]:
    """
    Enumerate all subgame perfect Nash equilibria with complete strategies.

    This method must be called AFTER backward_induction() has been run with mutate=True.
    It returns not just the equilibrium path but also the complete strategy for each
    player at ALL their information sets, not just those on the equilibrium path.

    This is important because if a player deviates from the equilibrium path, the other
    players' strategies still specify their optimal responses at every subgame.

    Args:
        tree: The game tree to solve
        node_id: Starting node (defaults to root)

    Returns:
        List of SubgamePerfectEquilibrium objects, one per equilibrium

    Raises:
        ValueError: If backward_induction hasn't been run yet (bi_value not set)
    """
    equilibria_paths = get_all_equilibria(tree, node_id)

    if node_id is None:
        if tree.root is None:
            raise ValueError("Tree is empty (no root node)")
        node_id = tree.root

    player_strategies = _build_spne_strategies(tree, node_id)

    return [
        SubgamePerfectEquilibrium(
            payoffs=eq.payoffs,
            path=eq.actions,
            strategies=player_strategies,
        )
        for eq in equilibria_paths
    ]


def _build_spne_strategies(
    tree: "GameTree",
    node_id: str,
) -> list[Strategy]:
    """
    Build complete SPNE strategies for all players.

    For each player, iterates through all their information sets and determines
    the optimal action at each based on backward induction values.
    """
    num_players = tree.num_players
    all_strategies: list[Strategy] = []

    for player in range(num_players):
        player_info_sets = tree.get_information_sets(player)
        decisions: dict[str, str] = {}

        for info_set_id in player_info_sets:
            nodes = tree.get_nodes_in_information_set(info_set_id)
            if not nodes:
                continue

            node = nodes[0]
            if (
                not hasattr(node.data, "optimal_children")
                or not node.data.optimal_children
            ):
                continue

            chosen_child = node.data.optimal_children[0]
            decisions[info_set_id] = chosen_child

        all_strategies.append(Strategy(decisions))

    return all_strategies


def _compute_expected_value(
    children: Sequence,
    child_payoffs: list[tuple[Expr, ...]],
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

    num_players = len(child_payoffs[0])

    expected = []
    for player_idx in range(num_players):
        weighted_sum = sp.Integer(0)
        for child, payoffs in zip(children, child_payoffs):
            prob = (
                child.data.probability
                if child.data.probability is not None
                else sp.Integer(1)
            )
            player_payoff = payoffs[player_idx]
            weighted_sum += prob * player_payoff
        expected.append(simplify(weighted_sum))

    return tuple(expected)


def _compute_optimal_choice(
    player: int,
    children: Sequence,
    child_payoffs: list[tuple[Expr, ...]],
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

        diff = simplify(current_value - best_value)

        is_better = False
        if diff.is_number:
            is_better = diff > 0
        elif diff.is_positive:
            is_better = True

        is_equal = False
        if diff.is_number:
            is_equal = diff == 0
        elif diff.is_zero:
            is_equal = True

        if is_better:
            best_value = current_value
            best_indices = [idx]
        elif is_equal:
            best_indices.append(idx)

    return (child_payoffs[best_indices[0]], best_indices)
