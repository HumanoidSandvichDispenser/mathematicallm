"""MCP server for game tree analysis operations."""

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
from zermelo.analysis.equilibria import find_pure_nash_equilibria
from zermelo.parsers.yaml import load_game_from_yaml
from zermelo.visualization.render import render_tree as _render_tree

mcp = FastMCP("zermelo-mcp")

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
    probabilities: dict[str, float] | None = None,
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

    from sympy import Rational

    probs = {k: Rational(v) for k, v in (probabilities or {"default": 1.0}).items()}
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
    payoffs: list[float],
    action: str | None = None,
) -> str:
    """
    Add a terminal node to the game tree.

    Args:
        game_id: ID of the game tree
        node_id: Unique identifier for the new node
        parent_id: ID of the parent node
        payoffs: List of payoffs for each player (e.g., [3, 1] for 2 players)
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

    terminal = TerminalNode(node_id, tuple(payoffs))
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
def find_player_strategies(game_id: str, players: list[str], reduced: bool = True) -> str:
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


if __name__ == "__main__":
    mcp.run()
