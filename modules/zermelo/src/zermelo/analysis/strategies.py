from collections.abc import Generator
from functools import reduce
from itertools import product
from sympy import Matrix
from sympy.matrices.dense import Matrix as Mat
from sympy.tensor.array.ndim_array import NDimArray
from zermelo.trees.node import (
    Node,
    DecisionNode,
    ChanceNode,
    TerminalNode,
)
from zermelo.trees.strategy import Strategy


def find_full_pure_strategies(root: Node, player: int) -> list[Strategy]:
    """
    Finds all full pure strategies for the game tree rooted at `root` for the
    specified `player`. A full pure strategy is a strategy that specifies an
    action for every decision node in the game tree. The returned strategies
    are represented as dictionaries mapping information set labels to actions.
    """
    if isinstance(root, TerminalNode):
        return [Strategy({})]

    child_strategies = [
        find_full_pure_strategies(child, player) for child in root.children.values()
    ]

    if isinstance(root, DecisionNode) and root.player == player:
        info_set = root.information_set.label
        results = []
        for action in root.actions:
            for combo in product(*child_strategies):
                combined = {info_set: action}
                for s in combo:
                    combined.update(s)
                results.append(Strategy(combined))
        return results

    results = []
    for combo in product(*child_strategies):
        combined = {}
        for s in combo:
            combined.update(s)
        results.append(Strategy(combined))

    seen = set()
    deduplicated = []
    for s in results:
        key = tuple(sorted(s.items()))
        if key not in seen:
            seen.add(key)
            deduplicated.append(s)

    return deduplicated


def find_reduced_pure_strategies(root: Node, player: int) -> list[Strategy]:
    """
    Finds all reduced pure strategies for the game tree rooted at `root` for
    the specified `player`. A reduced pure strategy is a strategy that
    specifies an action for every decision node that is reachable under some
    strategy profile.

    The algorithm mirrors find_full_pure_strategies with one key difference:
    at the player's own decision node, for each action we only recurse into
    the subtrees reachable via that action (across all nodes in the same
    information set), rather than taking the product over all children.
    """
    if isinstance(root, TerminalNode):
        return [Strategy({})]

    if isinstance(root, DecisionNode) and root.player == player:
        info_set = root.information_set
        info_set_label = info_set.label

        results = []
        for action in root.actions:
            # Collect subgame strategies reachable via `action` from every
            # node in this information set (we must handle all of them since
            # we cannot distinguish which node we are at).
            per_node_strats: list[list[Strategy]] = []
            for node in info_set.nodes:
                child = node.children[action]
                per_node_strats.append(find_reduced_pure_strategies(child, player))

            # Take the Cartesian product across all nodes in the info set,
            # then merge the resulting strategy dicts.
            for combo in product(*per_node_strats):
                combined: dict = {info_set_label: action}
                for s in combo:
                    combined.update(s)
                results.append(Strategy(combined))

        seen = set()
        deduplicated = []
        for s in results:
            key = tuple(sorted(s.items()))
            if key not in seen:
                seen.add(key)
                deduplicated.append(s)

        return deduplicated

    # Chance node or other player's decision node: all children are potentially
    # reachable. However, if multiple children are player-owned nodes in the
    # same information set, we must not recurse into each independently —
    # the info set handler already visits all its nodes when called on any
    # one representative. Recursing into each sibling separately would
    # multiply the strategies from each sibling's sub-tree as if they were
    # independent decisions, when in fact they share one info set entry.
    #
    # Solution: group children by (player-owned info set label), deduplicate
    # so each info set appears only once, then take the Cartesian product of
    # the deduplicated groups.
    seen_info_sets: set[str] = set()
    representative_children: list[Node] = []
    for child in root.children.values():
        if isinstance(child, DecisionNode) and child.player == player:
            label = child.information_set.label
            if label in seen_info_sets:
                continue  # already have a representative for this info set
            seen_info_sets.add(label)
        representative_children.append(child)

    child_strategies = [
        find_reduced_pure_strategies(child, player) for child in representative_children
    ]

    results = []
    for combo in product(*child_strategies):
        combined = {}
        for s in combo:
            combined.update(s)
        results.append(Strategy(combined))

    seen = set()
    deduplicated = []
    for s in results:
        key = tuple(sorted(s.items()))
        if key not in seen:
            seen.add(key)
            deduplicated.append(s)

    return deduplicated


def create_payoff_array(root: Node, profiles: list[tuple[Strategy]]) -> NDimArray:
    """
    Creates a payoff array for the game tree rooted at `root` based on
    the provided strategy profiles. The payoff array has one dimension for each
    player, and the size of each dimension corresponds to the number of
    strategies available to that player in the provided profiles. Each entry
    is a tuple of payoffs for each player under that strategy profile.
    """
    num_profiles = len(profiles)
    strategy_counts = [len(p) for p in profiles]

    sample_payoff = root.apply_strategy([p[0] for p in profiles])
    actual_players = len(sample_payoff)
    shape = tuple(strategy_counts) + (actual_players,)

    entries = []
    for index in product(*[range(c) for c in strategy_counts]):
        strategies = [profiles[i][index[i]] for i in range(num_profiles)]
        payoff = root.apply_strategy(strategies)
        entries.append(payoff)

    return NDimArray(entries, shape)
