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
        array: An NDimArray of payoffs with shape (n_p0, n_p1, ..., n_pk, k)
            for k players, where n_pi is the number of strategies for player i.
            The last dimension corresponds to the payoff scalar for each
            player; thus if you were to index the array (n_p0, n_p1, ...,
            n_pk), you would get the payoff vector of dimensions k for that
            strategy profile.

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


def backwards_induction(root: "Node") -> list[Strategy]:
    """
    Performs backwards induction on a game tree to find the subgame perfect
    equilibrium strategy profile. This function assumes that all players are
    rational and will choose their best response at each decision node, given
    the strategies of the other players.
    """
    raise NotImplementedError()
