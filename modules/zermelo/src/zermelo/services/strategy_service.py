from itertools import product
from zermelo.extensive.game_tree import GameTree
from zermelo.extensive.strategy import Strategy


def find_full_pure_strategies(game: GameTree, player: int) -> set[Strategy]:
    """
    Find all full pure strategies for the given extensive-form game. A full
    pure strategy is a strategy that specifies an action for every information
    set in the game, regardless of whether that information set is reachable
    under the strategy or not.

    Args:
        game: An extensive-form game represented as a GameTree.
        player: The player for whom to find the full pure strategies.

    Returns:
        A set of Strategy objects, each representing a full pure strategy for the
        game.
    """
    info_sets = game.get_information_sets(player)

    if not info_sets:
        return {Strategy({})}

    info_set_actions = []
    for info_set_id in sorted(info_sets):
        nodes = game.get_nodes_in_information_set(info_set_id)
        if not nodes:
            continue
        node = nodes[0]
        children = game.children(node.identifier)
        action_ids = [child.identifier for child in children]
        if action_ids:
            info_set_actions.append(action_ids)

    if not info_set_actions:
        return {Strategy({})}

    strategies: set[Strategy] = set()
    for action_combo in product(*info_set_actions):
        decisions = {}
        for info_set_id, action_id in zip(sorted(info_sets), action_combo):
            decisions[info_set_id] = action_id
        strategies.add(Strategy(decisions))

    return strategies


def find_reduced_pure_strategies(game: GameTree, player):
    """
    Find all reduced pure strategies for the given extensive-form game. A
    reduced pure strategy is a strategy that specifies an action for every
    information set that is reachable under the strategy. All decisions that
    are not reachable because of the player's earlier decisions are not
    included in a reduced pure strategy.

    Args:
        game: An extensive-form game represented as a GameTree.
        player: The player for whom to find the reduced pure strategies.

    Returns:
        A set of Strategy objects, each representing a reduced pure strategy for
        the game.
    """

    # how this can be implemented is that on every decision point, we can
    # recursively find the reduced pure strategies for the subgame that starts
    # at that decision point, and then combine those with the decisions that
    # lead to that decision point. This way, we only include the decisions that
    # are reachable under the strategy, and we exclude the decisions that are
    # not reachable because of the player's earlier decisions.

    # note that this should respect the information sets of the player, so we
    # need to make sure that all actions that are in the same information set
    # are treated as indistinguishable, and we need to make sure that we only
    # include one action for each information set in the strategy.

    strategies: set[Strategy] = set()
    return strategies
