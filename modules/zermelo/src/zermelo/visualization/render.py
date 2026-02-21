"""Graphical rendering of game trees using Graphviz.

Layout convention:
  - Tree grows left to right (rankdir=LR)
  - Decision nodes: circle, labelled with the player name above and node label below
  - Chance nodes:   diamond, labelled "chance"
  - Terminal nodes: rectangle showing the payoff vector
  - Edge labels:    action name along each edge
  - Information sets: dotted arc (no arrow) connecting nodes that share an info set
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import graphviz  # type: ignore

if TYPE_CHECKING:
    from zermelo.trees.node import Node

from zermelo.trees.node import ChanceNode, DecisionNode, InformationSet, TerminalNode


# ---------------------------------------------------------------------------
# Colour palette (player index → fill colour)
# ---------------------------------------------------------------------------
_PLAYER_COLOURS = [
    "#AED6F1",  # light blue  (player 0 / first player)
    "#A9DFBF",  # light green (player 1)
    "#F9E79F",  # light yellow
    "#F5CBA7",  # light orange
    "#D2B4DE",  # light purple
    "#F1948A",  # light red
    "#A8D8EA",  # sky blue
    "#B2BABB",  # grey
]

_CHANCE_COLOUR = "#FDFEFE"
_TERMINAL_COLOUR = "#EAECEE"
_MUTED_LABEL_COLOUR = "#00000055"  # transparent black


def _player_colour(player: str, player_order: list[str]) -> str:
    try:
        idx = player_order.index(player)
    except ValueError:
        idx = len(player_order)
    return _PLAYER_COLOURS[idx % len(_PLAYER_COLOURS)]


def _collect_info_sets(root: Node) -> dict[str, list[DecisionNode]]:
    """Return a mapping of info-set label → list of DecisionNodes in that set."""
    info_sets: dict[str, list[DecisionNode]] = {}
    for node in root.traverse_preorder():
        if isinstance(node, DecisionNode):
            lbl = node.information_set.label
            info_sets.setdefault(lbl, []).append(node)
    return info_sets


def _collect_players(root: Node) -> list[str]:
    """Return players in stable order (first encountered, pre-order)."""
    seen: list[str] = []
    for node in root.traverse_preorder():
        if isinstance(node, DecisionNode) and node.player not in seen:
            seen.append(node.player)
    return seen


def _node_gv_id(node: Node) -> str:
    """Stable graphviz node id (using Python object id to avoid label clashes)."""
    return f"n{id(node)}"


def _payoff_label(payoffs: tuple) -> str:
    """Format a payoff tuple as a compact string like (1, 2, 3)."""
    parts = ", ".join(str(p) for p in payoffs)
    return f"({parts})"


def render_tree(
    root: Node,
    *,
    format: str = "png",
    engine: str = "dot",
) -> bytes:
    """Render a game tree to an image and return the raw bytes.

    Parameters
    ----------
    root:
        Root node of the game tree.
    format:
        Output format passed to Graphviz (e.g. ``"png"``, ``"svg"``, ``"pdf"``).
    engine:
        Graphviz layout engine. ``"dot"`` gives the cleanest left-to-right
        hierarchical layout.

    Returns
    -------
    bytes
        Raw image bytes in the requested format.
    """
    player_order = _collect_players(root)

    g = graphviz.Digraph(engine=engine, format=format)
    g.attr(rankdir="LR", bgcolor="white", fontname="Helvetica")
    g.attr("node", fontname="Helvetica", fontsize="11")
    g.attr("edge", fontname="Helvetica", fontsize="10", color="#555555")

    # ------------------------------------------------------------------
    # First pass: add all nodes
    # ------------------------------------------------------------------
    for node in root.traverse_preorder():
        nid = _node_gv_id(node)

        if isinstance(node, TerminalNode):
            payoff_str = _payoff_label(node.payoffs)
            html = (
                f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
                f"<TR>"
                f'<TD><FONT POINT-SIZE="8" COLOR="{_MUTED_LABEL_COLOUR}" FACE="Courier">{node.label}</FONT></TD>'
                f"<TD>{payoff_str}</TD>"
                f"</TR>"
                f"</TABLE>>"
            )
            g.node(
                nid,
                label=html,
                shape="plaintext",
                style="filled",
                fillcolor=_TERMINAL_COLOUR,
                color="#888888",
                height="0.35",
            )

        elif isinstance(node, ChanceNode):
            html = (
                f"<chance<BR/>"
                f'<FONT POINT-SIZE="8" COLOR="{_MUTED_LABEL_COLOUR}" FACE="Courier">'
                f"{node.label}</FONT>>"
            )
            g.node(
                nid,
                label=html,
                shape="diamond",
                style="filled",
                fillcolor=_CHANCE_COLOUR,
                color="#888888",
                width="0.8",
                height="0.8",
            )

        elif isinstance(node, DecisionNode):
            fill = _player_colour(node.player, player_order)
            # Player name (normal) above, node label (small, muted, mono) below
            html = (
                f"<{node.player}<BR/>"
                f'<FONT POINT-SIZE="8" COLOR="{_MUTED_LABEL_COLOUR}" FACE="Courier">'
                f"{node.label}</FONT>>"
            )
            g.node(
                nid,
                label=html,
                shape="circle",
                style="filled",
                fillcolor=fill,
                color="#444444",
                width="0.75",
                height="0.75",
                fixedsize="true",
            )

    # ------------------------------------------------------------------
    # Second pass: add edges with action labels
    # ------------------------------------------------------------------
    for node in root.traverse_preorder():
        nid = _node_gv_id(node)

        if isinstance(node, ChanceNode):
            for action, child in node.children.items():
                prob = node.probability_map.get(action, "?")
                edge_label = f"{action}\n[{prob}]"
                g.edge(nid, _node_gv_id(child), label=edge_label)
        else:
            for action, child in node.children.items():
                g.edge(nid, _node_gv_id(child), label=action)

    # ------------------------------------------------------------------
    # Third pass: draw information set groupings with dotted arc
    # ------------------------------------------------------------------
    info_sets = _collect_info_sets(root)
    for is_label, nodes in info_sets.items():
        if len(nodes) < 2:
            continue  # singleton info sets need no annotation
        # Dotted arc between consecutive members — no arrow, no label
        for a, b in itertools.pairwise(nodes):
            g.edge(
                _node_gv_id(a),
                _node_gv_id(b),
                style="dotted",
                color="#00000066",
                penwidth="1.5",
                constraint="false",
                arrowhead="none",
                arrowtail="none",
            )

    return g.pipe()
