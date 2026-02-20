"""
Nash equilibrium computation for strategic-form games.

Given a game in strategic form (strategies + payoff tensor), finds all
pure strategy Nash equilibria.
"""

from itertools import product
from sympy import MutableDenseNDimArray, Expr, simplify
from zermelo.extensive.strategy import Strategy


PayoffVector = tuple[Expr, ...]

def find_pure_nash_equilibria(
    strategies: list[list[Strategy]], payoffs: MutableDenseNDimArray
) -> list[tuple[tuple[int, ...], PayoffVector]]:
    """
    Find all pure strategy Nash equilibria in a strategic-form game.

    A pure strategy Nash equilibrium is a strategy profile where no player
    can unilaterally deviate to improve their own payoff.

    Args:
        strategies: strategies[player] is an ordered list of that player's
            strategies.
        payoffs: A tensor of shape (n0, n1, ..., nk, num_players) where
            payoffs[i0, i1, ..., ik, p] is player p's payoff when player 0
            plays strategy i0, player 1 plays strategy i1, etc.

    Returns:
        A list of tuples (profile_indices, payoff_tuple) where:
        - profile_indices is a tuple (i0, i1, ...) indicating which strategy
          each player plays
        - payoff_tuple is the resulting payoff for each player

    Example:
        >>> strategies, payoffs = extensive_to_strategic(game)
        >>> equilibria = find_pure_nash_equilibria(strategies, payoffs)
        >>> for idx, pay in equilibria:
        ...     print(f"Profile {idx}: {pay}")
    """
    num_players = len(strategies)
    num_strategies = [len(s) for s in strategies]

    equilibria: list[tuple[tuple[int, ...], tuple[Expr, ...]]] = []

    for profile in product(*[range(n) for n in num_strategies]):
        is_equilibrium = True

        for player in range(num_players):
            current_payoff = payoffs[profile + (player,)]

            can_improve = False
            for alt_idx in range(num_strategies[player]):
                if alt_idx == profile[player]:
                    continue

                alt_profile = list(profile)
                alt_profile[player] = alt_idx
                alt_profile = tuple(alt_profile)

                alt_payoff = payoffs[alt_profile + (player,)]

                diff = simplify(alt_payoff - current_payoff)

                is_better = False
                if diff.is_number:
                    is_better = diff > 0
                elif diff.is_positive:
                    is_better = True

                if is_better:
                    can_improve = True
                    break

            if can_improve:
                is_equilibrium = False
                break

        if is_equilibrium:
            payoff_tuple = tuple(payoffs[profile + (p,)] for p in range(num_players))
            equilibria.append((profile, payoff_tuple))

    return equilibria
