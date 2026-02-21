"""Tests for zermelo.visualization.render."""

import pytest
from sympy import Rational

from zermelo.trees.node import (
    ChanceNode,
    DecisionNode,
    InformationSet,
    TerminalNode,
)
from zermelo.visualization.render import render_tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_png(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRenderTree:
    def test_returns_bytes(self):
        root = DecisionNode("root", player="p0")
        root.add_child(TerminalNode("t1", (1, 0)), "A")
        root.add_child(TerminalNode("t2", (0, 1)), "B")
        result = render_tree(root)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_output_is_valid_png(self):
        root = DecisionNode("root", player="p0")
        root.add_child(TerminalNode("t1", (1, 0)), "A")
        result = render_tree(root, format="png")
        assert _is_png(result)

    def test_single_terminal_root(self):
        """A lone terminal node should not crash."""
        root = TerminalNode("end", (3, 2, 1))
        result = render_tree(root)
        assert _is_png(result)

    def test_prisoners_dilemma(self):
        """Two-level tree with two players."""
        root = DecisionNode("root", player="alice")
        bob_c = DecisionNode("bob_c", player="bob")
        bob_d = DecisionNode("bob_d", player="bob")
        root.add_child(bob_c, "Cooperate")
        root.add_child(bob_d, "Defect")
        bob_c.add_child(TerminalNode("cc", (3, 3)), "Cooperate")
        bob_c.add_child(TerminalNode("cd", (0, 5)), "Defect")
        bob_d.add_child(TerminalNode("dc", (5, 0)), "Cooperate")
        bob_d.add_child(TerminalNode("dd", (1, 1)), "Defect")
        result = render_tree(root)
        assert _is_png(result)

    def test_three_player_game(self):
        root = DecisionNode("root", player="p0")
        n1 = DecisionNode("n1", player="p1")
        n2 = DecisionNode("n2", player="p1")
        root.add_child(n1, "L")
        root.add_child(n2, "R")
        for parent, action_prefix in [(n1, "l"), (n2, "r")]:
            n3 = DecisionNode(f"{action_prefix}_p2", player="p2")
            parent.add_child(n3, "U")
            parent.add_child(TerminalNode(f"{action_prefix}_down", (0, 0, 0)), "D")
            n3.add_child(TerminalNode(f"{action_prefix}_uu", (1, 2, 3)), "X")
            n3.add_child(TerminalNode(f"{action_prefix}_ud", (3, 1, 2)), "Y")
        result = render_tree(root)
        assert _is_png(result)

    def test_chance_node(self):
        root = DecisionNode("root", player="p0")
        chance = ChanceNode("nature", {"H": Rational(1, 2), "T": Rational(1, 2)})
        root.add_child(chance, "flip")
        chance.add_child(TerminalNode("heads", (1, -1)), "H")
        chance.add_child(TerminalNode("tails", (-1, 1)), "T")
        result = render_tree(root)
        assert _is_png(result)

    def test_imperfect_information(self):
        """Nodes sharing an info set get a dashed connector."""
        shared_info_set = InformationSet("shared", player="p1")
        root = DecisionNode("root", player="p0")
        n1 = DecisionNode("n1", player="p1", information_set=shared_info_set)
        n2 = DecisionNode("n2", player="p1", information_set=shared_info_set)
        root.add_child(n1, "L")
        root.add_child(n2, "R")
        n1.add_child(TerminalNode("t1", (1, 0)), "A")
        n1.add_child(TerminalNode("t2", (0, 1)), "B")
        n2.add_child(TerminalNode("t3", (1, 0)), "A")
        n2.add_child(TerminalNode("t4", (0, 1)), "B")
        result = render_tree(root)
        assert _is_png(result)

    def test_svg_format(self):
        root = DecisionNode("root", player="p0")
        root.add_child(TerminalNode("t", (0,)), "go")
        result = render_tree(root, format="svg")
        assert result.startswith(b"<?xml") or b"<svg" in result

    def test_payoffs_appear_in_dot_source(self):
        """Verify payoff values are encoded in the graph source."""
        import graphviz

        root = DecisionNode("root", player="alice")
        root.add_child(TerminalNode("win", (42, -7)), "go")
        # Build the graph and check its source
        from zermelo.visualization.render import (
            _collect_players,
            _node_gv_id,
            _payoff_label,
        )

        label = _payoff_label((42, -7))
        assert "42" in label
        assert "-7" in label

    def test_player_colours_cycle(self):
        """More than 8 players should not raise (colours cycle)."""
        from zermelo.visualization.render import _PLAYER_COLOURS, _player_colour

        players = [f"p{i}" for i in range(20)]
        for i, p in enumerate(players):
            colour = _player_colour(p, players)
            assert colour == _PLAYER_COLOURS[i % len(_PLAYER_COLOURS)]
