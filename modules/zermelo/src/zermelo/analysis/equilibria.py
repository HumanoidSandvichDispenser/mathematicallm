from itertools import product

from sympy.tensor.array.ndim_array import NDimArray
from zermelo.trees.node import Node
from zermelo.trees.strategy import Strategy


def find_pure_nash_equilibria(
    profiles: list[list[Strategy]],
    array: NDimArray,
) -> list[tuple[Strategy, ...]]:
    """
    Finds all pure Nash equilibria using a payoff array.

    Args:
        profiles: A list of strategy lists, one per player.
        array: An NDimArray of payoffs with shape (n_p0, n_p1, ..., n_pk, num_players).
            Each entry is a tuple of payoffs for each player.

    Returns:
        A list of pure Nash equilibria, where each equilibrium is a tuple of
        Strategy objects (one per player).
    """
    num_players = len(profiles)
    strategy_counts = [len(p) for p in profiles]
    num_strat_dims = len(strategy_counts)

    equilibria = []

    for index in product(*[range(c) for c in strategy_counts]):
        payoff = array[index]
        is_equilibrium = True

        for player in range(num_players):
            current_payoff = payoff[player]

            player_deviated = False
            for deviation in range(strategy_counts[player]):
                if deviation == index[player]:
                    continue

                deviated_index = list(index)
                deviated_index[player] = deviation
                deviated_payoff = array[tuple(deviated_index)][player]

                if deviated_payoff > current_payoff:
                    player_deviated = True
                    break

            if player_deviated:
                is_equilibrium = False
                break

        if is_equilibrium:
            equilibrium_strategies = tuple(
                profiles[p][index[p]] for p in range(num_players)
            )
            equilibria.append(equilibrium_strategies)

    return equilibria


def find_pure_subgame_perfect_equilibria(root: "Node") -> list[Strategy]:
    """
    Finds all pure subgame perfect equilibria using backward induction.
    """
    raise NotImplementedError()
