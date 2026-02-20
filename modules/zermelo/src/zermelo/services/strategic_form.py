"""
Strategic form conversion for extensive-form games.

Converts an extensive-form game tree into strategic (normal) form,
producing a payoff tensor where each cell contains the payoff for
each player given the strategy profile.
"""

import sympy as sp
from itertools import product
from sympy import MutableDenseNDimArray, Expr, simplify
from zermelo.extensive.game_node import GameNode
from zermelo.extensive.game_tree import GameTree
from zermelo.extensive.strategy import Strategy
from zermelo.services.strategy_service import find_full_pure_strategies


def _execute_from_node(
    game: GameTree, node_id: str, profile: dict[int, Strategy]
) -> tuple[Expr, ...]:
    """
    Execute strategy profile starting from a given node.

    Recursively handles:
    - Terminal nodes: return their payoff
    - Chance nodes: compute probability-weighted expected payoff
    - Decision nodes: follow the player's strategy
    """
    node = game.get_node(node_id)
    assert isinstance(node, GameNode)

    if node.is_terminal:
        return node.data.payoffs

    if node.is_chance:
        children = game.children(node_id)
        if not children:
            raise ValueError(f"Chance node {node_id} has no children")

        child_payoffs = [
            _execute_from_node(game, child.identifier, profile) for child in children
        ]

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

    if node.is_decision:
        player = node.data.player
        info_set_id = node.data.information_set or node_id

        if player not in profile:
            raise ValueError(f"No strategy specified for player {player} at {node_id}")

        strategy = profile[player]
        if info_set_id not in strategy.decisions:
            raise ValueError(
                f"Player {player} strategy does not specify action for "
                f"info set {info_set_id}"
            )

        chosen_action_id = strategy.decisions[info_set_id]

        children = game.children(node_id)
        child_map = {c.identifier: c.identifier for c in children}

        if chosen_action_id not in child_map:
            raise ValueError(
                f"Action {chosen_action_id} is not a valid child of {node_id}. "
                f"Valid children: {list(child_map.keys())}"
            )

        return _execute_from_node(game, chosen_action_id, profile)

    raise ValueError(f"Unknown node type at {node_id}")


def execute_strategy_profile(
    game: GameTree, profile: dict[int, Strategy]
) -> tuple[Expr, ...]:
    """
    Execute a strategy profile and return the resulting payoff tuple.

    Given a game tree and a mapping from player index to their Strategy,
    walks through the tree from the root:
    - At decision nodes, looks up the action from the relevant player's
      strategy using the info set ID
    - At chance nodes, computes probability-weighted expected payoff
    - Returns the terminal payoff tuple

    Args:
        game: An extensive-form game represented as a GameTree.
        profile: A dictionary mapping player index (0, 1, ...) to their
            Strategy (full pure strategy).

    Returns:
        A tuple of payoffs, one value per player.

    Raises:
        ValueError: If a decision node is encountered with no strategy
            specified for its player, or if the specified action is not
            a valid child.
    """
    if game.root is None:
        raise ValueError("Game tree is empty")

    return _execute_from_node(game, game.root, profile)


def extensive_to_strategic(
    game: GameTree,
) -> tuple[list[list[Strategy]], MutableDenseNDimArray]:
    """
    Convert an extensive-form game to strategic (normal) form.

    Produces a payoff tensor where entry [i0, i1, ..., ik, p] contains
    player p's payoff when player 0 plays strategy i0, player 1 plays
    strategy i1, etc.

    Args:
        game: An extensive-form game represented as a GameTree.

    Returns:
        A tuple (strategies, payoffs) where:
        - strategies[player] is an ordered list of that player's full pure
          strategies
        - payoffs is a MutableDenseNDimArray of shape
          (n0, n1, ..., nk, num_players) where nk is the number of
          strategies for player k, and the final dimension contains
          each player's payoff.

    Example:
        >>> strategies, payoffs = extensive_to_strategic(game)
        >>> # Player 0's strategy 2 vs Player 1's strategy 1:
        >>> payoffs[2, 1, 0]  # player 0's payoff
        >>> payoffs[2, 1, 1]  # player 1's payoff
    """
    num_players = game.num_players

    strategies: list[list[Strategy]] = []
    for player in range(num_players):
        player_strategies = find_full_pure_strategies(game, player)
        strategies.append(sorted(player_strategies, key=repr))

    num_strategies_per_player = [len(s) for s in strategies]

    shape = tuple(num_strategies_per_player + [num_players])
    payoffs = MutableDenseNDimArray.zeros(*shape)

    for idx in product(*[range(n) for n in num_strategies_per_player]):
        profile = {p: strategies[p][idx[p]] for p in range(num_players)}
        profile_payoffs = execute_strategy_profile(game, profile)
        payoff_idx = idx + tuple(range(num_players))
        for p, payoff in enumerate(profile_payoffs):
            payoffs[payoff_idx[:-num_players] + (p,)] = payoff

    return strategies, payoffs
