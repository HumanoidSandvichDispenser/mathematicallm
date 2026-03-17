"""MCP server for game tree analysis operations."""

import os

from mcp.server.fastmcp import FastMCP, Image
from zermelo.trees.node import (
    DecisionNode,
    ChanceNode,
    TerminalNode,
    InformationSet,
    Node,
)
from zermelo.trees.strategy import Strategy
from zermelo.analysis.strategies import (
    find_full_pure_strategies,
    find_reduced_pure_strategies,
    create_payoff_array,
)
from zermelo.analysis.equilibria import (
    find_pure_nash_equilibria,
    find_mixed_nash_equilibria,
)
from zermelo.trees.mixed_strategy import MixedStrategy
from zermelo.parsers.yaml import load_game_from_yaml
from zermelo.visualization.render import render_tree as _render_tree


def _fastmcp_host() -> str:
    return os.getenv("FASTMCP_HOST", "127.0.0.1")


def _fastmcp_port() -> int:
    return int(os.getenv("FASTMCP_PORT", "8000"))


HOST = _fastmcp_host()
PORT = _fastmcp_port()

mcp = FastMCP("zermelo-mcp", host=HOST, port=PORT)


_games: dict[str, Node] = {}
_counter = 0


def _new_id(prefix: str) -> str:
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


@mcp.tool()
def create_game(name: str | None = None) -> str:
    """
    Create a new game tree with a root decision node.

    Args:
        name: Optional name for the game (auto-generated if not provided)

    Returns:
        game_id: Unique identifier for the created game
    """
    game_id = _new_id("game") if name is None else name
    root = DecisionNode("root", player="p0")
    _games[game_id] = root
    return f"Created game '{game_id}'. Root node is decision node for player 'p0'."


@mcp.tool()
def add_decision_node(
    game_id: str,
    node_id: str,
    parent_id: str,
    player: str,
    actions: list[str] | None = None,
    information_set: str | None = None,
) -> str:
    """
    Add a decision node to the game tree.

    Args:
        game_id: ID of the game tree
        node_id: Unique identifier for the new node
        parent_id: ID of the parent node
        player: Player identifier (e.g., "p0", "p1", "alice", "bob")
        actions: Optional list of action labels (e.g., ["up", "down"]) for this decision.
            These become child keys, and also serve as action names in strategies.
        information_set: Optional identifier for the information set (for imperfect information)

    Returns:
        Success message
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    root = _games[game_id]
    parent = _find_node(root, parent_id)
    if parent is None:
        return f"Error: Parent node '{parent_id}' not found"

    if _find_node(root, node_id) is not None:
        return f"Error: Node '{node_id}' already exists"

    info_set_obj = None
    if information_set:
        info_set_obj = InformationSet(information_set, player)

    new_node = DecisionNode(node_id, player, information_set=info_set_obj)

    if actions:
        for action in actions:
            new_node.add_child(TerminalNode(f"{node_id}_{action}", (0,)), action)
    else:
        new_node.add_child(TerminalNode(f"{node_id}_child", (0,)), "default")

    parent.add_child(new_node, node_id)

    info_set_info = (
        f" in information set '{information_set}'" if information_set else ""
    )
    actions_info = f" with actions {actions}" if actions else ""
    return f"Added decision node '{node_id}' for player {player} under '{parent_id}'{info_set_info}{actions_info}"


@mcp.tool()
def add_chance_node(
    game_id: str,
    node_id: str,
    parent_id: str,
    probabilities: dict[str, str | int] | None = None,
) -> str:
    """
    Add a chance node to the game tree.

    Args:
        game_id: ID of the game tree
        node_id: Unique identifier for the new node
        parent_id: ID of the parent node
        probabilities: Optional dict of {action: probability} for the children

    Returns:
        Success message
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    root = _games[game_id]
    parent = _find_node(root, parent_id)
    if parent is None:
        return f"Error: Parent node '{parent_id}' not found"

    if _find_node(root, node_id) is not None:
        return f"Error: Node '{node_id}' already exists"

    from sympy import Basic, sympify

    def _parse_probability(value: str | int | Basic) -> Basic:
        if isinstance(value, Basic):
            return value
        if isinstance(value, int):
            return sympify(value)
        if isinstance(value, str):
            return sympify(value)
        raise TypeError(
            "Chance node probabilities must be ints or SymPy expressions encoded as strings"
        )

    default_probs = {"default": "1"}
    probs_input = probabilities or default_probs
    probs = {k: _parse_probability(v) for k, v in probs_input.items()}
    new_node = ChanceNode(node_id, probs)

    for action in probs:
        new_node.add_child(TerminalNode(f"{node_id}_{action}", (0,)), action)

    parent.add_child(new_node, node_id)

    prob_info = f" with probabilities {probabilities}" if probabilities else ""
    return f"Added chance node '{node_id}' under '{parent_id}'{prob_info}"


@mcp.tool()
def add_terminal_node(
    game_id: str,
    node_id: str,
    parent_id: str,
    payoffs: list[str | int],
    action: str | None = None,
) -> str:
    """
    Add a terminal node to the game tree.

    Args:
        game_id: ID of the game tree
        node_id: Unique identifier for the new node
        parent_id: ID of the parent node
        payoffs: List of payoffs for each player expressed as strings or ints
        action: The action key under parent to connect this terminal node (defaults to node_id)

    Returns:
        Success message
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    root = _games[game_id]
    parent = _find_node(root, parent_id)
    if parent is None:
        return f"Error: Parent node '{parent_id}' not found"

    from sympy import Basic, sympify

    def _parse_payoff(value: str | int | Basic) -> Basic:
        if isinstance(value, Basic):
            return value
        if isinstance(value, int):
            return sympify(value)
        if isinstance(value, str):
            return sympify(value)
        raise TypeError(
            "Terminal payoffs must be ints or strings representing sympy expressions"
        )

    terminal = TerminalNode(node_id, tuple(_parse_payoff(value) for value in payoffs))
    action_key = action if action else node_id
    parent.add_child(terminal, action_key)

    return f"Added terminal node '{node_id}' under '{parent_id}' with payoffs {payoffs}"


def _find_node(root: Node, node_id: str) -> Node | None:
    """Find a node by label in the tree."""
    if root.label == node_id:
        return root
    for child in root.children.values():
        result = _find_node(child, node_id)
        if result:
            return result
    return None


def _count_nodes(root: Node) -> int:
    """Count nodes in tree."""
    count = 1
    for child in root.children.values():
        count += _count_nodes(child)
    return count


def _describe_node(root: Node, indent: int = 0) -> list[str]:
    """Generate a text description of the tree."""
    prefix = "  " * indent
    lines = []

    if isinstance(root, TerminalNode):
        lines.append(f"{prefix}{root.label} (terminal): {root.payoffs}")
    elif isinstance(root, ChanceNode):
        lines.append(f"{prefix}{root.label} (chance): {root.probability_map}")
        for action, child in root.children.items():
            lines.extend(_describe_node(child, indent + 1))
    elif isinstance(root, DecisionNode):
        lines.append(
            f"{prefix}{root.label} (P{root.player} @{root.information_set.label}): {list(root.children.keys())}"
        )
        for action, child in root.children.items():
            lines.extend(_describe_node(child, indent + 1))

    return lines


def _get_players(root: Node) -> list[str]:
    """Get all unique players in the game tree."""
    players = set()
    for node in root.traverse_preorder():
        if isinstance(node, DecisionNode):
            players.add(node.player)
    return sorted(players, key=str)


@mcp.tool(title="Show game tree")
def show_tree(game_id: str) -> str:
    """
    Display the game tree structure in text format.

    Args:
        game_id: ID of the game tree

    Returns:
        Tree visualization
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    root = _games[game_id]
    lines = [f"Game tree '{game_id}':", f"Root: {root.label}"]
    lines.extend(_describe_node(root, 1))
    return "\n".join(lines)


@mcp.tool(title="Find pure strategies for a player")
def find_player_strategies(
    game_id: str, players: list[str], reduced: bool = True
) -> str:
    """
    Find all pure strategies for the specified players in an extensive-form game.

    Args:
        game_id: ID of the game tree
        player: List of player identifiers to find strategies for (e.g., ["alice", "bob"])
        reduced: If False, find all full pure strategies. If True (default),
            find only reduced pure strategies.

    Returns:
        A description of all strategies for the given players.
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    root = _games[game_id]

    output = []

    for player in players:
        try:
            if reduced:
                strategies = find_reduced_pure_strategies(root, player)
                strategy_type = "reduced"
            else:
                strategies = find_full_pure_strategies(root, player)
                strategy_type = "full"

            player_out = [
                f"Found {len(strategies)} {strategy_type} pure strategies "
                f"for player {player} in '{game_id}':"
            ]

            for i, strategy in enumerate(sorted(strategies, key=str), 1):
                if strategy._decisions:
                    player_out.append(f"  Strategy {i}: {dict(strategy._decisions)}")
                else:
                    player_out.append(f"  Strategy {i}: (empty)")

            output.append("\n".join(player_out))
        except Exception as e:
            return f"Error finding strategies: {e}"

    return "\n\n".join(output)


@mcp.tool(title="Compute strategic form")
def compute_strategic_form(game_id: str) -> str:
    """
    Convert an extensive-form game to strategic (normal) form. This also finds
    the reduced pure strategies for each player, so the resulting strategic
    form is based on the reduced strategy sets.

    Produces a payoff tensor where entry [i0, i1, ..., ik, p] contains
    player p's payoff when player p0 plays strategy i0, player p1 plays
    strategy i1, etc.

    Args:
        game_id: ID of the game tree

    Returns:
        A description of the strategic form with all strategy profiles and payoffs
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    root = _games[game_id]

    try:
        players = root.get_players()
        profiles = {p: find_reduced_pure_strategies(root, p) for p in players}

        array, player_order = create_payoff_array(root, profiles)

        output = [
            f"Strategic form for '{game_id}':",
            f"Players: {player_order}",
            str(array),
        ]

        for player, strats in profiles.items():
            output.append(f"\nPlayer '{player}' strategies ({len(strats)}):")
            for i, strategy in enumerate(strats):
                output.append(f"  {i}: {dict(strategy._decisions)}")

        output.append(f"\nPayoff tensor shape: {array.shape}")

        total_profiles = 1
        for n in array.shape[:-1]:
            total_profiles *= n
        output.append(f"Total strategy profiles: {total_profiles}")

        return "\n".join(output)
    except Exception as e:
        return f"Error computing strategic form: {e}"


@mcp.tool(title="Execute strategy profile")
def execute_profile(game_id: str, profile_json: str) -> str:
    """
    Execute a strategy profile and return the resulting payoff.

    Args:
        game_id: ID of the game tree
        profile_json: JSON string mapping player identifiers to their strategy decisions.
            Format: {"p0": {"root": "left"}, "p1": {}}

    Returns:
        The resulting payoff tuple
    """
    import json

    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    root = _games[game_id]

    try:
        profile_data = json.loads(profile_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON — {e}"

    try:
        strategies = {
            player: Strategy(decisions) for player, decisions in profile_data.items()
        }

        payoff = root.apply_strategy(strategies)

        payoff_strs = [str(p) for p in payoff]
        return f"Payoff: ({', '.join(payoff_strs)})"
    except Exception as e:
        return f"Error executing profile: {e}"


@mcp.tool(title="Load game from YAML")
def load_game_yaml(yaml_content: str, name: str | None = None) -> str:
    """
    Create a game tree from a YAML description. This is the preferred way of
    creating games, as it allows you to specify the entire tree structure in
    one go.

    Args:
        yaml_content: YAML string describing the game tree
        name: Optional name for the game (auto-generated if not provided)

    Returns:
        Success message with game_id

    ## YAML Format

    ```yaml
    root:
        type: decision
        label: root
        player: 0
        children:
            left:
                type: terminal
                label: left
                payoffs: [1, 2]
            right:
                type: decision
                label: n1
                player: 1
                children:
                    up:
                        type: terminal
                        label: up
                        payoffs: [3, 4]

    information_sets:
        shared:
            player: 1
    ```
    """
    try:
        root = load_game_from_yaml(yaml_content)
    except Exception as e:
        return f"Error loading YAML: {e}"

    game_id = _new_id("game") if name is None else name
    _games[game_id] = root

    return f"Loaded game '{game_id}' with root node '{root.label}'."


@mcp.tool()
def list_games() -> str:
    """List all active game sessions."""
    if not _games:
        return "No active games"

    output = ["Active games:"]
    for game_id, root in _games.items():
        output.append(f"  {game_id}: root='{root.label}' ({_count_nodes(root)} nodes)")

    return "\n".join(output)


@mcp.tool()
def delete_game(game_id: str) -> str:
    """Delete a game from the session store."""
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    del _games[game_id]
    return f"Deleted game '{game_id}'"


@mcp.tool(title="Render game tree as image")
def render_game_tree(game_id: str) -> Image:
    """
    Render the game tree as a PNG image.

    The image is laid out left to right. Decision nodes are circles coloured
    by player. Chance nodes are diamonds. Terminal nodes are rectangles showing
    the payoff vector. Action names are shown on edges. Nodes that share an
    information set are connected by a dashed line.

    Args:
        game_id: ID of the game tree

    Returns:
        PNG image of the game tree
    """
    if game_id not in _games:
        raise ValueError(f"Game '{game_id}' not found")

    root = _games[game_id]
    png_bytes = _render_tree(root, format="png")
    return Image(data=png_bytes, format="png")


@mcp.tool(title="Find mixed Nash equilibria")
def find_mixed_ne(game_id: str) -> str:
    """
    Find all mixed Nash equilibria via support enumeration.

    This function computes mixed Nash equilibria for 2-player games. It returns
    both pure and mixed strategy equilibria. For games with more than 2 players,
    an error is returned.

    Args:
        game_id: ID of the game tree

    Returns:
        A description of all mixed Nash equilibria found
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    root = _games[game_id]

    try:
        players = root.get_players()
        if len(players) != 2:
            return f"Error: Mixed Nash equilibrium only supports 2-player games. Found {len(players)} players: {players}"

        profiles = {p: find_reduced_pure_strategies(root, p) for p in players}
        array, player_order = create_payoff_array(root, profiles)
        profiles_list = [profiles[p] for p in player_order]

        mixed_ne = find_mixed_nash_equilibria(profiles_list, array)

        if not mixed_ne:
            return f"No mixed Nash equilibria found for '{game_id}'"

        output = [f"Mixed Nash equilibria for '{game_id}' ({len(mixed_ne)} found):"]

        for i, (row_mix, col_mix) in enumerate(mixed_ne, 1):
            output.append(f"\n--- Equilibrium {i} ---")

            for player_idx, (mix, player) in enumerate(
                [(row_mix, player_order[0]), (col_mix, player_order[1])]
            ):
                strat_list = [f"{dict(s._decisions)}: {p}" for s, p in mix.items()]
                output.append(f"  Player '{player}': {', '.join(strat_list)}")

        return "\n".join(output)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error finding mixed Nash equilibria: {e}"


@mcp.tool(title="Find Nash equilibria from payoff matrix")
def find_equilibria_from_matrix(
    row_payoffs: list[list[str]],
    col_payoffs: list[list[str]],
    row_strategy_names: list[str] | None = None,
    col_strategy_names: list[str] | None = None,
) -> str:
    """
    Find Nash equilibria directly from a payoff matrix (normal form game).

    This is a convenience function that lets you analyze 2-player games
    without having to build a game tree. Simply provide the payoff matrices
    for each player.

    Args:
        row_payoffs: 2D list of row player payoffs described as strings.
            Shape: (m, n) where m is number of row strategies, n is number of
            column strategies.
        col_payoffs: 2D list of column player payoffs described as strings.
            Shape: (m, n).
        row_strategy_names: Optional list of names for row strategies.
            Defaults to ["r0", "r1", ...].
        col_strategy_names: Optional list of names for column strategies.
            Defaults to ["c0", "c1", ...].

    Returns:
        A description of all pure and mixed Nash equilibria found.

    ## Example

    Matching pennies:
    ```
    row_payoffs = [["1", "-1"], ["-1", "1"]]
    col_payoffs = [["-1", "1"], ["1", "-1"]]
    ```
    """
    from sympy import Basic, Rational, sympify
    from sympy.tensor.array.ndim_array import NDimArray

    m = len(row_payoffs)
    n = len(row_payoffs[0])

    if len(col_payoffs) != m or any(len(row) != n for row in col_payoffs):
        return "Error: Payoff matrices must have the same dimensions"

    r_names = row_strategy_names or [f"r{i}" for i in range(m)]
    c_names = col_strategy_names or [f"c{j}" for j in range(n)]

    if len(r_names) != m:
        return f"Error: Expected {m} row strategy names, got {len(r_names)}"
    if len(c_names) != n:
        return f"Error: Expected {n} column strategy names, got {len(c_names)}"

    def _parse_entry(value: str | Basic) -> Basic:
        if isinstance(value, Basic):
            return value
        if not isinstance(value, str):
            raise TypeError("Payoff entries must be strings when using sympy symbols")
        return sympify(value)

    row_values = [[_parse_entry(entry) for entry in row] for row in row_payoffs]
    col_values = [[_parse_entry(entry) for entry in row] for row in col_payoffs]

    row_display = [[str(entry) for entry in row] for row in row_payoffs]
    col_display = [[str(entry) for entry in row] for row in col_payoffs]

    row_strategies = [Strategy({r_names[i]: r_names[i]}) for i in range(m)]
    col_strategies = [Strategy({c_names[j]: c_names[j]}) for j in range(n)]

    entries = []
    for i in range(m):
        for j in range(n):
            entries.append((row_values[i][j], col_values[i][j]))

    shape = (m, n, 2)
    array = NDimArray(entries, shape)

    profiles = [row_strategies, col_strategies]

    pure_ne = find_pure_nash_equilibria(profiles, array)
    mixed_ne = find_mixed_nash_equilibria(profiles, array)

    output = [
        f"2-player game with {m} row strategies x {n} column strategies:",
        f"Row strategies: {r_names}",
        f"Column strategies: {c_names}",
        "",
        "Payoff matrix (row, col):",
    ]

    header = "       " + "".join(f"{c:>8}" for c in c_names)
    output.append(header)
    for i in range(m):
        row_str = f"{r_names[i]:>6}" + "".join(
            f"({row_display[i][j]},{col_display[i][j]})" for j in range(n)
        )
        output.append(row_str)

    output.append("")

    if pure_ne:
        output.append(f"Pure Nash Equilibria ({len(pure_ne)} found):")
        for i, eq in enumerate(pure_ne, 1):
            row_strat, col_strat = eq
            r_idx = row_strategies.index(row_strat)
            c_idx = col_strategies.index(col_strat)
            output.append(f"  {i}. Row: {r_names[r_idx]}, Col: {c_names[c_idx]}")
    else:
        output.append("Pure Nash Equilibria: None")

    output.append("")

    if mixed_ne:
        output.append(f"Mixed Nash Equilibria ({len(mixed_ne)} found):")
        for i, (row_mix, col_mix) in enumerate(mixed_ne, 1):
            output.append(f"  --- Equilibrium {i} ---")

            r_probs = []
            for strat, prob in row_mix.items():
                idx = row_strategies.index(strat)
                r_probs.append(f"{r_names[idx]}:{prob}")
            output.append(f"    Row: {', '.join(r_probs)}")

            c_probs = []
            for strat, prob in col_mix.items():
                idx = col_strategies.index(strat)
                c_probs.append(f"{c_names[idx]}:{prob}")
            output.append(f"    Col: {', '.join(c_probs)}")
    else:
        output.append("Mixed Nash Equilibria: None")

    return "\n".join(output)


if __name__ == "__main__":
    transport = os.getenv("FASTMCP_TRANSPORT", "streamable-http")
    mcp.run(transport=transport)
