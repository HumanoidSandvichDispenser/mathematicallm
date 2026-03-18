"""
Tests for equilibria.py - finding Nash equilibria.
"""

import pytest
from sympy import S, Rational
from sympy.tensor.array.ndim_array import NDimArray
from zermelo.trees.strategy import Strategy
from zermelo.trees.mixed_strategy import MixedStrategy
from zermelo.analysis.equilibria import (
    find_pure_nash_equilibria,
    find_mixed_nash_equilibria,
    find_pure_mm_solutions,
    PureMMSolution,
)


class TestFindPureNashEquilibria:
    def test_single_player_single_strategy(self):
        """Single player with single strategy is always a NE."""
        p0_strats = [Strategy({})]
        array = NDimArray([(S(5),)], (1, 1))

        ne = find_pure_nash_equilibria([p0_strats], array)

        assert len(ne) == 1
        assert ne[0] == (p0_strats[0],)

    def test_two_by_two_game_one_ne(self):
        """2x2 game with one pure NE at (0, 0)."""
        p0_strats = [Strategy({"A": "a"}), Strategy({"A": "b"})]
        p1_strats = [Strategy({"B": "x"}), Strategy({"B": "y"})]

        array = NDimArray(
            [(S(3), S(3)), (S(0), S(5)), (S(5), S(0)), (S(1), S(1))],
            (2, 2, 2),
        )

        ne = find_pure_nash_equilibria([p0_strats, p1_strats], array)

        assert len(ne) == 1
        assert ne[0][0] == p0_strats[1]
        assert ne[0][1] == p1_strats[1]

    def test_two_by_two_game_two_ne(self):
        """2x2 coordination game with two pure NE at (0, 0) and (1, 1)."""
        p0_strats = [Strategy({"A": "a"}), Strategy({"A": "b"})]
        p1_strats = [Strategy({"B": "x"}), Strategy({"B": "y"})]

        array = NDimArray(
            [(S(3), S(3)), (S(0), S(0)), (S(0), S(0)), (S(1), S(1))],
            (2, 2, 2),
        )

        ne = find_pure_nash_equilibria([p0_strats, p1_strats], array)

        assert len(ne) == 2
        assert (p0_strats[0], p1_strats[0]) in ne
        assert (p0_strats[1], p1_strats[1]) in ne

    def test_no_pure_ne(self):
        """2x2 matching pennies has no pure NE."""
        p0_strats = [Strategy({"A": "a"}), Strategy({"A": "b"})]
        p1_strats = [Strategy({"B": "x"}), Strategy({"B": "y"})]

        array = NDimArray(
            [(S(1), S(-1)), (S(-1), S(1)), (S(-1), S(1)), (S(1), S(-1))],
            (2, 2, 2),
        )

        ne = find_pure_nash_equilibria([p0_strats, p1_strats], array)

        assert len(ne) == 0

    def test_three_player_game(self):
        """Three player game with one NE."""
        p0_strats = [Strategy({})]
        p1_strats = [Strategy({})]
        p2_strats = [Strategy({}), Strategy({})]

        array = NDimArray(
            [(S(3), S(3), S(3)), (S(1), S(1), S(1))],
            (1, 1, 2, 3),
        )

        ne = find_pure_nash_equilibria([p0_strats, p1_strats, p2_strats], array)

        assert len(ne) == 1

    def test_player_cannot_improve_unilaterally(self):
        """NE where each player cannot improve by deviating."""
        p0_strats = [Strategy({"root": "L"}), Strategy({"root": "R"})]
        p1_strats = [Strategy({"I1": "U"}), Strategy({"I1": "D"})]

        array = NDimArray(
            [(S(2), S(1)), (S(0), S(0)), (S(0), S(0)), (S(1), S(2))],
            (2, 2, 2),
        )

        ne = find_pure_nash_equilibria([p0_strats, p1_strats], array)

        assert len(ne) == 2
        assert (p0_strats[0], p1_strats[0]) in ne
        assert (p0_strats[1], p1_strats[1]) in ne


class TestFindMixedNashEquilibria:
    def test_single_strategy_each_player(self):
        """Single strategy each player is trivially a NE."""
        p0_strats = [Strategy({})]
        p1_strats = [Strategy({})]

        array = NDimArray([(S(5), S(3))], (1, 1, 2))

        ne = find_mixed_nash_equilibria([p0_strats, p1_strats], array)

        assert len(ne) == 1
        row_mix, col_mix = ne[0]
        assert isinstance(row_mix, MixedStrategy)
        assert isinstance(col_mix, MixedStrategy)
        assert row_mix[p0_strats[0]] == 1
        assert col_mix[p1_strats[0]] == 1

    def test_matching_pennies(self):
        """Matching pennies has unique mixed NE with 50/50 for each player."""
        p0_strats = [Strategy({"A": "a"}), Strategy({"A": "b"})]
        p1_strats = [Strategy({"B": "x"}), Strategy({"B": "y"})]

        array = NDimArray(
            [(S(1), S(-1)), (S(-1), S(1)), (S(-1), S(1)), (S(1), S(-1))],
            (2, 2, 2),
        )

        ne = find_mixed_nash_equilibria([p0_strats, p1_strats], array)

        assert len(ne) == 1
        row_mix, col_mix = ne[0]
        assert row_mix[p0_strats[0]] == Rational(1, 2)
        assert row_mix[p0_strats[1]] == Rational(1, 2)
        assert col_mix[p1_strats[0]] == Rational(1, 2)
        assert col_mix[p1_strats[1]] == Rational(1, 2)

    def test_pure_ne_also_in_mixed(self):
        """Pure NE should also appear in mixed NE results."""
        p0_strats = [Strategy({"A": "a"}), Strategy({"A": "b"})]
        p1_strats = [Strategy({"B": "x"}), Strategy({"B": "y"})]

        array = NDimArray(
            [(S(3), S(3)), (S(0), S(0)), (S(0), S(0)), (S(1), S(1))],
            (2, 2, 2),
        )

        ne = find_mixed_nash_equilibria([p0_strats, p1_strats], array)

        assert len(ne) == 3
        pure_ne_found = False
        for row_mix, col_mix in ne:
            if row_mix[p0_strats[0]] == 1 and col_mix[p1_strats[0]] == 1:
                pure_ne_found = True
                break
        assert pure_ne_found

    def test_non_2_player_raises_error(self):
        """Should raise ValueError for non-2-player games."""
        p0_strats = [Strategy({})]

        array = NDimArray([(S(5),)], (1, 1))

        with pytest.raises(ValueError, match="two-player"):
            find_mixed_nash_equilibria([p0_strats], array)


class TestFindPureMMSolutions:
    def test_single_player_single_strategy(self):
        """Single player with one strategy has itself as maximin."""
        p0_strats = [Strategy({})]
        array = NDimArray([(S(5),)], (1, 1))

        mm = find_pure_mm_solutions([p0_strats], array)

        assert len(mm) == 1
        assert isinstance(mm[0], PureMMSolution)
        assert mm[0].strategies == [p0_strats[0]]
        assert mm[0].value == S(5)

    def test_two_player_unique_maximin_each(self):
        """Each player has a unique maximin strategy."""
        p0_strats = [Strategy({"A": "a"}), Strategy({"A": "b"})]
        p1_strats = [Strategy({"B": "x"}), Strategy({"B": "y"})]

        array = NDimArray(
            [(S(3), S(2)), (S(1), S(5)), (S(2), S(1)), (S(2), S(3))],
            (2, 2, 2),
        )

        mm = find_pure_mm_solutions([p0_strats, p1_strats], array)

        assert len(mm) == 2
        assert mm[0].strategies == [p0_strats[1]]
        assert mm[0].value == S(2)
        assert mm[1].strategies == [p1_strats[1]]
        assert mm[1].value == S(3)

    def test_two_player_ties(self):
        """Ties in maximin include all tied strategies."""
        p0_strats = [Strategy({"A": "a"}), Strategy({"A": "b"})]
        p1_strats = [Strategy({"B": "x"}), Strategy({"B": "y"})]

        array = NDimArray(
            [(S(1), S(2)), (S(2), S(2)), (S(1), S(2)), (S(3), S(2))],
            (2, 2, 2),
        )

        mm = find_pure_mm_solutions([p0_strats, p1_strats], array)

        assert len(mm) == 2
        assert mm[0].strategies == [p0_strats[0], p0_strats[1]]
        assert mm[0].value == S(1)
        assert mm[1].strategies == [p1_strats[0], p1_strats[1]]
        assert mm[1].value == S(2)

    def test_three_player_support(self):
        """Pure maximin works for n-player games."""
        p0_strats = [Strategy({"A": "a"}), Strategy({"A": "b"})]
        p1_strats = [Strategy({"B": "x"}), Strategy({"B": "y"})]
        p2_strats = [Strategy({"C": "l"}), Strategy({"C": "r"})]

        array = NDimArray(
            [
                (S(2), S(2), S(2)),
                (S(2), S(2), S(2)),
                (S(2), S(2), S(2)),
                (S(2), S(2), S(2)),
                (S(3), S(1), S(1)),
                (S(3), S(1), S(1)),
                (S(3), S(1), S(1)),
                (S(3), S(1), S(1)),
            ],
            (2, 2, 2, 3),
        )

        mm = find_pure_mm_solutions([p0_strats, p1_strats, p2_strats], array)

        assert len(mm) == 3
        assert mm[0].strategies == [p0_strats[1]]
        assert mm[0].value == S(3)
        assert mm[1].strategies == [p1_strats[0], p1_strats[1]]
        assert mm[1].value == S(1)
        assert mm[2].strategies == [p2_strats[0], p2_strats[1]]
        assert mm[2].value == S(1)

    def test_shape_mismatch_raises_error(self):
        """Array shape must match strategy profile dimensions."""
        p0_strats = [Strategy({}), Strategy({})]
        p1_strats = [Strategy({})]

        array = NDimArray([(S(1), S(1)), (S(2), S(2))], (2, 2, 2))

        with pytest.raises(ValueError, match="shape"):
            find_pure_mm_solutions([p0_strats, p1_strats], array)
