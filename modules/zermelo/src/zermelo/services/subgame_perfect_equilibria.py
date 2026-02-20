"""
Solver services for extensive-form games.

Provides dedicated solvers for computing game-theoretic solutions like
backward induction and subgame perfect Nash equilibria.

Two clearly separated phases:

1. Backward induction (Phase 1): Traverses the tree bottom-up, computing
   backed-up payoff values at every node and recording which children are
   optimal.  Mutates ``BIValue`` fields on each node.

2. SPNE enumeration (Phase 2): A second pass that uses the results of
   Phase 1 to enumerate all subgame perfect Nash equilibria as lists of
   ``Strategy`` objects (one per player).

Key design note — the ``optimal_children`` fix
----------------------------------------------
The naive approach for decision nodes takes ``max(child.bi_value)`` as the
representative payoff for each child and picks the globally-best child.
This is incorrect when a child has multiple backed-up payoffs (due to ties
lower in the tree), because a different resolution of that child's subgame
could make a *different* parent-level action optimal.

The fix: iterate over the full Cartesian product of each child's possible
payoff tuples.  For every joint resolution, record which children are
optimal.  ``optimal_children`` is the *union* of all children that are
optimal in at least one resolution; ``bi_value`` collects only the payoff
tuples that are actually achieved as the equilibrium outcome in some
resolution (non-optimal payoffs are discarded).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product as cart_product
from typing import Optional

import sympy as sp
from sympy import Expr, simplify

from ..extensive.equilibrium import EquilibriumPath, SubgamePerfectEquilibrium
from ..extensive.game_node import GameNode
from ..extensive.game_tree import GameTree
from ..extensive.node_data import ChanceNodeData, DecisionNodeData, TerminalNodeData
from ..extensive.strategy import Strategy

# ---------------------------------------------------------------------------
# Phase 1 — Backward Induction
# ---------------------------------------------------------------------------


def backward_induction(
    tree: GameTree,
    node_id: Optional[str] = None,
    mutate: bool = False,
) -> list[tuple[Expr, ...]]:
    """
    Compute backward induction solution from the given node.

    For each node, computes the set of possible equilibrium payoff vectors by
    working backwards from terminal nodes:

    - Terminal nodes: return ``[payoffs]`` (single-element list).
    - Chance nodes: return a list of all possible expected-value tuples,
      one per combination of SPNE resolutions in the children.
    - Decision nodes: for each joint resolution of children's payoffs
      (Cartesian product), record which children are optimal.  The node's
      ``bi_value`` is the list of payoffs that are the equilibrium outcome
      in *some* resolution; ``optimal_children`` is the union of all
      children that are optimal in at least one resolution.

    The ``bi_value``, ``optimal_children``, and ``child_bi_values`` fields
    are mutated on all ``BIValue`` nodes during computation.

    Args:
        tree: The game tree to solve.
        node_id: Starting node (defaults to root).
        mutate: If ``True``, modifies tree in-place; if ``False``, works on a
            deep copy and leaves the original untouched.

    Returns:
        List of payoff tuples representing all possible equilibrium outcomes
        reachable from ``node_id``.

    Raises:
        ValueError: If ``node_id`` is not found in the tree, or a
            non-terminal node has no children.
        NotImplementedError: If the game has imperfect information.
    """
    if node_id is None:
        if tree.root is None:
            raise ValueError("Tree is empty (no root node)")
        node_id = tree.root

    if not tree.is_perfect_information():
        raise NotImplementedError(
            "This game has imperfect information (multiple nodes per information set). "
            "Backward induction requires perfect information. "
            "Use algorithms like Counterfactual Regret Minimization (CFR) for "
            "imperfect information games."
        )

    if not mutate:
        subtree_copy = tree.subtree(node_id)
        if subtree_copy.root is None:
            raise ValueError(f"Node with id {node_id} not found in the tree.")
        tree_copy = GameTree(num_players=tree.num_players, tree=subtree_copy, deep=True)
        if tree_copy.root is None:
            raise ValueError("Subtree root is None")
        return backward_induction(tree_copy, node_id=tree_copy.root, mutate=True)

    node = tree.get_node(node_id)
    if node is None:
        raise ValueError(f"Node with id {node_id} not found in the tree.")

    # --- Terminal ---
    if isinstance(node.data, TerminalNodeData):
        return [node.data.payoffs]

    children = tree.children(node_id)
    if not children:
        raise ValueError(
            f"Non-terminal node {node_id} has no children. "
            f"Node type: {type(node.data).__name__}"
        )

    child_ids: list[str] = [c.identifier for c in children]

    # Recurse on every child first
    child_bi: dict[str, list[tuple[Expr, ...]]] = {
        cid: backward_induction(tree, cid, mutate=True) for cid in child_ids
    }
    node.data.child_bi_values = child_bi

    # --- Chance node ---
    if isinstance(node.data, ChanceNodeData):
        child_probs = {
            c.identifier: (
                c.data.probability if c.data.probability is not None else sp.Integer(1)
            )
            for c in children
        }
        num_players = len(child_bi[child_ids[0]][0])

        possible_evs: list[tuple[Expr, ...]] = []
        for combo in cart_product(*[child_bi[cid] for cid in child_ids]):
            ev = tuple(
                simplify(
                    sum(
                        child_probs[child_ids[j]] * combo[j][p]
                        for j in range(len(child_ids))
                    )
                )
                for p in range(num_players)
            )
            possible_evs.append(ev)

        unique_evs = _deduplicate_payoffs(possible_evs)
        node.data.bi_value = unique_evs
        return unique_evs

    # --- Decision node ---
    if isinstance(node.data, DecisionNodeData):
        player = node.data.player
        opt_set: set[str] = set()
        bi_list: list[tuple[Expr, ...]] = []

        for combo in cart_product(*[child_bi[cid] for cid in child_ids]):
            # combo[j] is one payoff tuple chosen from child j's bi_value list.
            # For this joint resolution, find which child(ren) maximise the
            # deciding player's payoff.
            player_vals: list[Expr] = [combo[j][player] for j in range(len(child_ids))]
            best_val = _max_payoff(player_vals)

            for j, cid in enumerate(child_ids):
                if _payoff_geq_or_indeterminate(combo[j][player], best_val):
                    opt_set.add(cid)
                    bi_list.append(combo[j])

        node.data.optimal_children = list(opt_set)
        node.data.bi_value = bi_list
        return bi_list

    raise ValueError(f"Unknown node type at {node_id}: {type(node.data).__name__}")


# ---------------------------------------------------------------------------
# Phase 2 — SPNE Enumeration
# ---------------------------------------------------------------------------


def get_all_equilibria(
    tree: GameTree,
    node_id: Optional[str] = None,
) -> list[EquilibriumPath]:
    """
    Enumerate all subgame perfect Nash equilibria (on-path actions only).

    Must be called **after** ``backward_induction(mutate=True)``.

    When a decision node has multiple optimal children (ties), each
    combination of tied choices across the whole tree produces a distinct
    equilibrium path.

    Args:
        tree: The game tree.
        node_id: Starting node (defaults to root).

    Returns:
        List of ``EquilibriumPath`` objects, one per equilibrium.

    Raises:
        ValueError: If ``backward_induction`` has not been run yet.
    """
    if node_id is None:
        if tree.root is None:
            raise ValueError("Tree is empty (no root node)")
        node_id = tree.root

    _check_bi_run(tree, node_id)

    raw = _enumerate_spne(tree, node_id)

    result: list[EquilibriumPath] = []
    seen: set[tuple] = set()
    for complete_profile, payoff in raw:
        on_path = _get_on_path_actions(tree, node_id, complete_profile)
        key = (frozenset(on_path.items()), payoff)
        if key not in seen:
            seen.add(key)
            result.append(EquilibriumPath(payoffs=payoff, actions=on_path))

    return result


def get_all_spne(
    tree: GameTree,
    node_id: Optional[str] = None,
) -> list[SubgamePerfectEquilibrium]:
    """
    Enumerate all subgame perfect Nash equilibria with complete strategy profiles.

    Must be called **after** ``backward_induction(mutate=True)``.

    Returns not just the equilibrium path but also the complete strategy for
    each player at **all** their information sets, including those that are
    off the equilibrium path.  This is necessary because a strategy in an
    extensive-form game must specify behaviour at every information set even
    if that information set is never reached in equilibrium.

    When there are ties (a player is indifferent between multiple actions),
    all combinations that constitute distinct SPNEs are enumerated.

    Args:
        tree: The game tree.
        node_id: Starting node (defaults to root).

    Returns:
        List of ``SubgamePerfectEquilibrium`` objects, one per equilibrium.

    Raises:
        ValueError: If ``backward_induction`` has not been run yet.
    """
    if node_id is None:
        if tree.root is None:
            raise ValueError("Tree is empty (no root node)")
        node_id = tree.root

    _check_bi_run(tree, node_id)

    raw = _enumerate_spne(tree, node_id)

    result: list[SubgamePerfectEquilibrium] = []
    seen: set[tuple] = set()
    for complete_profile, payoff in raw:
        key = (frozenset(complete_profile.items()), payoff)
        if key not in seen:
            seen.add(key)
            on_path = _get_on_path_actions(tree, node_id, complete_profile)
            strategies = _build_strategies_from_profile(tree, node_id, complete_profile)
            result.append(
                SubgamePerfectEquilibrium(
                    payoffs=payoff,
                    path=on_path,
                    strategies=strategies,
                )
            )

    return result


# ---------------------------------------------------------------------------
# Core enumeration  (internal)
# ---------------------------------------------------------------------------


def _enumerate_spne(
    tree: GameTree,
    node_id: str,
) -> list[tuple[dict[str, str], tuple[Expr, ...]]]:
    """
    Recursively enumerate all SPNE-valid (complete_profile, payoff) pairs.

    ``complete_profile`` maps **every** decision node in the subtree rooted
    at ``node_id`` to a chosen child, including nodes that are off the
    equilibrium path.

    Algorithm (decision node):
        For each joint combination of sub-profiles across *all* children
        (Cartesian product), the deciding player's optimal action is
        determined from that combination's concrete payoffs.  Only profiles
        in which the node's recorded action is genuinely optimal for that
        player are kept — invalid combinations are naturally discarded rather
        than generated-and-filtered.

    This correctly handles the case where a child subgame has multiple
    tied payoffs that, when resolved differently, change which action is
    optimal at the parent.
    """
    node = tree.get_node(node_id)
    assert isinstance(node, GameNode)

    # --- Terminal ---
    if node.is_terminal:
        return [({}, node.data.payoffs)]

    children = tree.children(node_id)
    if not children:
        raise ValueError(
            f"Non-terminal node {node_id} has no children. "
            f"Node type: {type(node.data).__name__}"
        )

    child_ids: list[str] = [c.identifier for c in children]

    # Enumerate every child's SPNE sub-profiles
    child_enums: dict[str, list[tuple[dict[str, str], tuple[Expr, ...]]]] = {
        cid: _enumerate_spne(tree, cid) for cid in child_ids
    }

    # --- Chance node ---
    if node.is_chance:
        child_probs = {
            c.identifier: (
                c.data.probability if c.data.probability is not None else sp.Integer(1)
            )
            for c in children
        }
        num_players = len(child_enums[child_ids[0]][0][1])

        result: list[tuple[dict[str, str], tuple[Expr, ...]]] = []
        for combo in cart_product(*[child_enums[cid] for cid in child_ids]):
            # combo[j] = (profile_j, payoff_j) for child j
            merged: dict[str, str] = {}
            for profile, _ in combo:
                merged.update(profile)

            ev = tuple(
                simplify(
                    sum(
                        child_probs[child_ids[j]] * combo[j][1][p]
                        for j in range(len(child_ids))
                    )
                )
                for p in range(num_players)
            )
            result.append((merged, ev))

        return _deduplicate_enumerations(result)

    # --- Decision node ---
    if node.is_decision:
        player = node.data.player

        result = []
        for combo in cart_product(*[child_enums[cid] for cid in child_ids]):
            # Concrete payoff each child delivers in this joint resolution
            child_payoffs = {child_ids[j]: combo[j][1] for j in range(len(child_ids))}

            player_vals: list[Expr] = [child_payoffs[cid][player] for cid in child_ids]
            best_val = _max_payoff(player_vals)

            # Merge all children's strategy choices (on- and off-path)
            merged = {}
            for profile, _ in combo:
                merged.update(profile)

            for j, cid in enumerate(child_ids):
                if _payoff_geq_or_indeterminate(child_payoffs[cid][player], best_val):
                    profile = dict(merged)
                    profile[node_id] = cid
                    result.append((profile, child_payoffs[cid]))

        return _deduplicate_enumerations(result)

    raise ValueError(f"Unknown node type at {node_id}: {type(node.data).__name__}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_bi_run(tree: GameTree, node_id: str) -> None:
    """Raise ValueError if backward induction has not been run on this tree."""
    node = tree.get_node(node_id)
    if node is None:
        raise ValueError(f"Node with id {node_id} not found")
    if hasattr(node.data, "bi_value") and node.data.bi_value is None:
        raise ValueError(
            "backward_induction has not been run yet. "
            "Call backward_induction(tree, mutate=True) first."
        )


def _get_on_path_actions(
    tree: GameTree,
    root_id: str,
    complete_profile: dict[str, str],
) -> dict[str, str]:
    """
    Walk the tree from *root_id* following *complete_profile* and return only
    the decision node → chosen child pairs that are actually visited.

    Chance nodes do not contribute to the action map; all their children are
    visited (each branch can be reached with positive probability).
    """
    on_path: dict[str, str] = {}
    to_visit: list[str] = [root_id]

    while to_visit:
        current_id = to_visit.pop(0)
        node = tree.get_node(current_id)
        assert isinstance(node, GameNode)

        if node.is_terminal:
            continue

        if node.is_decision:
            action = complete_profile.get(current_id)
            if action is not None:
                on_path[current_id] = action
                to_visit.append(action)

        elif node.is_chance:
            for child in tree.children(current_id):
                to_visit.append(child.identifier)

    return on_path


def _build_strategies_from_profile(
    tree: GameTree,
    root_id: str,
    complete_profile: dict[str, str],
) -> list[Strategy]:
    """
    Convert a complete node→action profile into one ``Strategy`` per player.

    ``Strategy.decisions`` maps **information set IDs** to chosen child IDs.
    For perfect-information games each node is its own information set (id =
    node id), so this is a direct projection.
    """
    per_player: dict[int, dict[str, str]] = defaultdict(dict)

    for node_id, chosen_child in complete_profile.items():
        node = tree.get_node(node_id)
        assert isinstance(node, GameNode)
        if not node.is_decision:
            continue
        info_set_id = node.data.information_set or node_id
        per_player[node.data.player][info_set_id] = chosen_child

    num_players = tree.num_players
    return [Strategy(per_player.get(p, {})) for p in range(num_players)]


def _max_payoff(values: list[Expr]) -> Expr:
    """
    Return the maximum of a list of sympy expressions.

    For symbolic expressions that cannot be compared definitively, returns
    the first value that cannot be shown to be strictly less than everything
    else (conservative: may over-include candidates).
    """
    best = values[0]
    for v in values[1:]:
        diff = simplify(v - best)
        is_better: bool
        if diff.is_number:
            is_better = bool(diff > 0)
        elif diff.is_positive:
            is_better = True
        else:
            is_better = False
        if is_better:
            best = v
    return best


def _payoff_geq_or_indeterminate(value: Expr, best: Expr) -> bool:
    """
    Return ``True`` if *value* >= *best*, or if the comparison is indeterminate
    (symbolic expressions that cannot be resolved).  Returns ``False`` only
    when the difference is definitively negative.
    """
    diff = simplify(value - best)
    if diff.is_number:
        return bool(diff >= 0)
    if diff.is_negative:
        return False
    # Non-negative or indeterminate — include as potentially optimal
    return True


def _deduplicate_payoffs(
    payoffs: list[tuple[Expr, ...]],
) -> list[tuple[Expr, ...]]:
    """Remove duplicate payoff tuples while preserving order."""
    seen: set[tuple[Expr, ...]] = set()
    result: list[tuple[Expr, ...]] = []
    for p in payoffs:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _deduplicate_enumerations(
    enumerations: list[tuple[dict[str, str], tuple[Expr, ...]]],
) -> list[tuple[dict[str, str], tuple[Expr, ...]]]:
    """Remove duplicate (profile, payoff) pairs while preserving order."""
    seen: set[tuple] = set()
    result: list[tuple[dict[str, str], tuple[Expr, ...]]] = []
    for profile, payoff in enumerations:
        key = (frozenset(profile.items()), payoff)
        if key not in seen:
            seen.add(key)
            result.append((profile, payoff))
    return result


# ---------------------------------------------------------------------------
# Legacy helpers kept for backward compatibility
# ---------------------------------------------------------------------------


def _compute_expected_value(
    children,
    child_payoffs: list[tuple[Expr, ...]],
) -> tuple[Expr, ...]:
    """
    Compute expected value across chance node children.

    Each child should have a probability set (via ``child.data.probability``).
    Probabilities do not need to sum to 1.
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
            weighted_sum += prob * payoffs[player_idx]
        expected.append(simplify(weighted_sum))
    return tuple(expected)
