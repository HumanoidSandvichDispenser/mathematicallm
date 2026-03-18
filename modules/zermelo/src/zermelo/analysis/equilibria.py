from itertools import product, combinations
from dataclasses import dataclass

from sympy import Rational, Matrix, Expr
from sympy.tensor.array.ndim_array import NDimArray
from zermelo.trees.node import Node, DecisionNode, TerminalNode
from zermelo.trees.mixed_strategy import MixedStrategy
from zermelo.trees.strategy import Strategy


@dataclass(frozen=True)
class PureMMSolution:
    """Pure maximin solution for a single player."""

    strategies: list[Strategy]
    value: Expr


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


def find_mixed_nash_equilibria(
    profiles: list[list[Strategy]],
    array: NDimArray,
) -> list[tuple[MixedStrategy, MixedStrategy]]:
    """
    Find all mixed Nash Equilibria via support enumeration.

    Args:
        profiles: list of strategy lists, one per player. profiles[0] = row
                  player strategies, profiles[1] = col player strategies.
        array:    NDimArray of shape (m, n, 2) — array[i, j] = (v_row, v_col)

    Returns:
        A list of equilibria. Each equilibrium is a tuple of two MixedStrategy
        objects (row player, col player).
    """
    if len(profiles) != 2:
        raise ValueError("This algorithm only supports two-player games.")

    m = len(profiles[0])
    n = len(profiles[1])

    A = Matrix(m, n, lambda i, j: array[i, j][0])
    B = Matrix(m, n, lambda i, j: array[i, j][1])

    equilibria = []
    seen = set()

    for k in range(1, min(m, n) + 1):
        for I in combinations(range(m), k):
            for J in combinations(range(n), k):
                M_q: Matrix = Matrix.zeros(k, k)
                rhs_q = Matrix.zeros(k, 1)

                for row_idx in range(k - 1):
                    i0 = I[0]
                    i1 = I[row_idx + 1]
                    for col_idx, j in enumerate(J):
                        M_q[row_idx, col_idx] = A[i0, j] - A[i1, j]  # type: ignore
                    rhs_q[row_idx] = 0

                for col_idx in range(k):
                    M_q[k - 1, col_idx] = 1
                rhs_q[k - 1] = 1

                if M_q.det() == 0:
                    continue

                q_vec = M_q.solve(rhs_q)

                if any(q_vec[t] < 0 for t in range(k)):
                    continue

                q_full = [Rational(0)] * n
                for t, j in enumerate(J):
                    q_full[j] = q_vec[t]

                v1 = sum(A[I[0], j] * q_full[j] for j in range(n))

                M_p = Matrix.zeros(k, k)
                rhs_p = Matrix.zeros(k, 1)

                for col_idx in range(k - 1):
                    j0 = J[0]
                    j1 = J[col_idx + 1]
                    for row_idx, i in enumerate(I):
                        M_p[col_idx, row_idx] = B[i, j0] - B[i, j1]  # type: ignore
                    rhs_p[col_idx] = 0

                for row_idx in range(k):
                    M_p[k - 1, row_idx] = 1
                rhs_p[k - 1] = 1

                if M_p.det() == 0:
                    continue

                p_vec = M_p.solve(rhs_p)

                if any(p_vec[t] < 0 for t in range(k)):
                    continue

                p_full = [Rational(0)] * m
                for t, i in enumerate(I):
                    p_full[i] = p_vec[t]

                v2 = sum(p_full[i] * B[i, J[0]] for i in range(m))

                row_ok = all(
                    sum(A[i, j] * q_full[j] for j in range(n)) <= v1
                    for i in range(m)
                    if i not in I
                )

                col_ok = all(
                    sum(p_full[i] * B[i, j] for i in range(m)) <= v2
                    for j in range(n)
                    if j not in J
                )

                if not (row_ok and col_ok):
                    continue

                key = (tuple(p_full), tuple(q_full))
                if key in seen:
                    continue
                seen.add(key)

                row_mix = MixedStrategy(
                    {profiles[0][i]: p_full[i] for i in range(m) if p_full[i] > 0}
                )
                col_mix = MixedStrategy(
                    {profiles[1][j]: q_full[j] for j in range(n) if q_full[j] > 0}
                )

                equilibria.append((row_mix, col_mix))

    return equilibria


def find_pure_mm_solutions(
    profiles: list[list[Strategy]],
    array: NDimArray,
) -> list[PureMMSolution]:
    """
    Find pure maximin solutions for each player in a normal-form game.

    For player i, the algorithm computes the worst-case payoff for each pure
    strategy (min over all opponent strategy profiles), then selects all
    strategies that maximize this minimum.

    Args:
        profiles: A list of strategy lists, one per player.
        array: An NDimArray of payoffs with shape (n_p0, n_p1, ..., n_pk, k)
            for k players, where n_pi is the number of strategies for player i.

    Returns:
        A list of PureMMSolution objects, one per player, in player-index order.
        Each object contains the player's maximin-optimal pure strategies and
        the shared maximin value among those strategies.
    """
    num_players = len(profiles)
    strategy_counts = [len(p) for p in profiles]

    expected_shape = tuple(strategy_counts) + (num_players,)
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"Payoff array shape {tuple(array.shape)} does not match expected "
            f"shape {expected_shape} for the provided strategy profiles."
        )

    solutions: list[PureMMSolution] = []

    for player in range(num_players):
        opponent_ranges = [
            range(strategy_counts[i]) for i in range(num_players) if i != player
        ]

        worst_case_payoffs: list[Expr] = []
        for own_strategy_idx in range(strategy_counts[player]):
            worst_case_payoff = None

            for opponent_index in product(*opponent_ranges):
                profile_index = []
                opponent_cursor = 0
                for i in range(num_players):
                    if i == player:
                        profile_index.append(own_strategy_idx)
                    else:
                        profile_index.append(opponent_index[opponent_cursor])
                        opponent_cursor += 1

                payoff = array[tuple(profile_index)][player]
                if worst_case_payoff is None or payoff < worst_case_payoff:
                    worst_case_payoff = payoff

            if worst_case_payoff is None:
                raise ValueError("Each player must have at least one strategy.")

            worst_case_payoffs.append(worst_case_payoff)

        maximin_value = max(worst_case_payoffs)
        maximin_strategies = [
            profiles[player][i]
            for i, payoff in enumerate(worst_case_payoffs)
            if payoff == maximin_value
        ]

        solutions.append(
            PureMMSolution(
                strategies=maximin_strategies,
                value=maximin_value,
            )
        )

    return solutions
