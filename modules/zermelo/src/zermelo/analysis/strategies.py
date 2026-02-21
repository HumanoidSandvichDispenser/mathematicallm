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
    return results


def find_reduced_pure_strategies(root: Node, player: int) -> list[Strategy]:
    """
    Finds all reduced pure strategies for the game tree rooted at `root` for
    the specified `player`. A reduced pure strategy is a strategy that
    specifies an action for every decision node in the game tree that is
    reachable under some strategy profile. The returned strategies are
    represented as dictionaries mapping information set labels to actions.
    """
    if isinstance(root, TerminalNode):
        return [Strategy({})]

    if isinstance(root, DecisionNode) and root.player == player:
        info_set = root.information_set.label
        results = []
        for action, child in root.children.items():
            child_strategies = find_reduced_pure_strategies(child, player)
            if not child_strategies:
                child_strategies = [Strategy({})]
            for s in child_strategies:
                combined = {info_set: action}
                combined.update(s)
                results.append(Strategy(combined))
        return results

    results = []
    seen = set()
    for child in root.children.values():
        for s in find_reduced_pure_strategies(child, player):
            if len(s) == 0:
                continue
            key = tuple(sorted(s.items()))
            if key not in seen:
                results.append(s)
                seen.add(key)

    if not results:
        return [Strategy({})]

    return results


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
