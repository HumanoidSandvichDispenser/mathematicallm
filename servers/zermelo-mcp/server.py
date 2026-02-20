"""MCP server for game tree analysis operations."""

from mcp.server.fastmcp import FastMCP
from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    ChanceNodeData,
    TerminalNodeData,
)
from zermelo.extensive.strategy import Strategy
from zermelo.services import subgame_perfect_equilibria as sge
from zermelo.services.strategic_form import (
    execute_strategy_profile as exec_profile,
    extensive_to_strategic,
)
from zermelo.services.strategy import (
    find_full_pure_strategies,
    find_reduced_pure_strategies,
)

mcp = FastMCP("zermelo-mcp")

# Session store for game tree objects
_games: dict[str, GameTree] = {}
_counter = 0


def _new_id(prefix: str) -> str:
    """Generate a new unique ID for storing objects."""
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


@mcp.tool()
def create_game(name: str | None = None, num_players: int = 2) -> str:
    """
    Create a new game tree with a root decision node.

    Args:
        name: Optional name for the game (auto-generated if not provided)
        num_players: Number of players in the game (default: 2)

    Returns:
        game_id: Unique identifier for the created game
    """
    game_id = _new_id("game")
    tree = GameTree(num_players=num_players)
    tree.create_node(
        tag="root",
        identifier="root",
        data=DecisionNodeData(player=0),
    )
    _games[game_id] = tree
    return f"Created game '{game_id}' with {num_players} players. Root node is decision node for player 0."


@mcp.tool()
def add_decision_node(
    game_id: str,
    node_id: str,
    parent_id: str,
    player: int,
    probability: str | None = None,
    information_set: str | None = None,
    actions: list[str] | None = None,
) -> str:
    """
    Add a decision node to the game tree.

    Args:
        game_id: ID of the game tree
        node_id: Unique identifier for the new node
        parent_id: ID of the parent node
        player: Player index (0-based) who makes the decision at this node
        probability: Optional probability expression for the edge (for chance node children)
        information_set: Optional identifier for the information set (for imperfect information)
        actions: Optional list of action labels (e.g., ["up", "down"]) for this decision.
            If provided, these are used as action names in strategies instead of child node IDs.

    Returns:
        Success message
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    tree = _games[game_id]

    if tree.contains(node_id):
        return f"Error: Node '{node_id}' already exists"

    if not tree.contains(parent_id):
        return f"Error: Parent node '{parent_id}' not found"

    tree.create_node(
        tag=node_id,
        identifier=node_id,
        parent=parent_id,
        data=DecisionNodeData(
            player=player,
            probability=probability,
            information_set=information_set,
            actions=actions,
        ),
    )

    prob_info = f" with edge probability {probability}" if probability else ""
    info_set_info = (
        f" in information set '{information_set}'" if information_set else ""
    )
    actions_info = f" with actions {actions}" if actions else ""
    return f"Added decision node '{node_id}' for player {player} under '{parent_id}'{prob_info}{info_set_info}{actions_info}"


@mcp.tool()
def add_chance_node(
    game_id: str,
    node_id: str,
    parent_id: str,
    probability: str | None = None,
) -> str:
    """
    Add a chance node to the game tree.

    Args:
        game_id: ID of the game tree
        node_id: Unique identifier for the new node
        parent_id: ID of the parent node
        probability: Optional probability expression for the edge (for chance node children)

    Returns:
        Success message
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    tree = _games[game_id]

    if tree.contains(node_id):
        return f"Error: Node '{node_id}' already exists"

    if not tree.contains(parent_id):
        return f"Error: Parent node '{parent_id}' not found"

    tree.create_node(
        tag=node_id,
        identifier=node_id,
        parent=parent_id,
        data=ChanceNodeData(probability=probability),
    )

    prob_info = f" with edge probability {probability}" if probability else ""
    return f"Added chance node '{node_id}' under '{parent_id}'{prob_info}"


@mcp.tool()
def add_terminal_node(
    game_id: str,
    node_id: str,
    parent_id: str,
    payoffs: str,
    probability: str | None = None,
) -> str:
    """
    Add a terminal node to the game tree.

    Args:
        game_id: ID of the game tree
        node_id: Unique identifier for the new node
        parent_id: ID of the parent node
        payoffs: Comma-separated payoffs for each player (e.g., "3,1" for 2 players)
        probability: Optional probability expression for the edge (for chance node children)

    Returns:
        Success message
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    tree = _games[game_id]

    if tree.contains(node_id):
        return f"Error: Node '{node_id}' already exists"

    if not tree.contains(parent_id):
        return f"Error: Parent node '{parent_id}' not found"

    # Parse payoffs
    try:
        payoff_values = tuple(p.strip() for p in payoffs.split(","))
        if len(payoff_values) != tree.num_players:
            return (
                f"Error: Expected {tree.num_players} payoffs, got {len(payoff_values)}"
            )
    except Exception as e:
        return f"Error parsing payoffs: {e}"

    tree.create_node(
        tag=node_id,
        identifier=node_id,
        parent=parent_id,
        data=TerminalNodeData(payoffs=payoff_values, probability=probability),
    )

    prob_info = f" with edge probability {probability}" if probability else ""
    return f"Added terminal node '{node_id}' under '{parent_id}' with payoffs {payoff_values}{prob_info}"


@mcp.tool(title="Solve perfect game with backward induction")
def solve_perfect_game(game_id: str) -> str:
    """
    Solve the game using backward induction.

    Args:
        game_id: ID of the game tree

    Returns:
        The equilibrium payoffs and full solution details
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    tree = _games[game_id]

    try:
        result = sge.backward_induction(tree, mutate=True)

        # Format result
        payoff_strs = [str(p) for p in result]
        output = [f"Backward induction solution for '{game_id}':"]
        output.append(f"Equilibrium payoffs: ({', '.join(payoff_strs)})")
        output.append("\nNode values:")

        # Show BI values for all nodes
        for node_id in tree.expand_tree():
            node = tree.get_node(node_id)
            if node.is_terminal:
                output.append(f"  {node_id} (terminal): {node.data.bi_value}")
            elif hasattr(node.data, "bi_value") and node.data.bi_value is not None:
                bi_val = node.data.bi_value
                bi_strs = [str(v) for v in bi_val]
                if node.is_decision:
                    # Show optimal children for decision nodes
                    optimal_str = (
                        f", optimal: {node.data.optimal_children}"
                        if node.data.optimal_children
                        else ""
                    )
                    output.append(
                        f"  {node_id} (player {node.data.player}): ({', '.join(bi_strs)}){optimal_str}"
                    )
                else:
                    output.append(f"  {node_id} (chance): ({', '.join(bi_strs)})")

        return "\n".join(output)
    except Exception as e:
        return f"Error solving game: {e}"


@mcp.tool(title="Get all SPNEs")
def get_all_spne(game_id: str) -> str:
    """
    Get all subgame perfect Nash equilibria with complete strategy profiles.

    Must be called after solve_game(). Returns the full strategy for every
    player in every equilibrium — specifying what each player would do at
    ALL their information sets (not just those on the equilibrium path).
    When there are ties, all equilibria are enumerated.

    Use get_all_equilibria() if you only need the on-path actions and payoffs.

    Args:
        game_id: ID of the game tree

    Returns:
        All SPNEs with payoffs, equilibrium path actions, and per-player strategies
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    tree = _games[game_id]

    try:
        spne_list = sge.get_all_spne(tree)

        output = [f"Subgame perfect Nash equilibria for '{game_id}':"]
        output.append(f"Found {len(spne_list)} equilibrium/equilibria\n")

        for i, spne in enumerate(spne_list, 1):
            payoff_strs = [str(p) for p in spne.payoffs]
            output.append(f"Equilibrium {i}:")
            output.append(f"  Payoffs: ({', '.join(payoff_strs)})")

            if spne.path:
                output.append("  Equilibrium path:")
                for node_id, child_id in sorted(spne.path.items()):
                    output.append(f"    {node_id} → {child_id}")
            else:
                output.append(
                    "  Equilibrium path: (none - tree is just a terminal node)"
                )

            output.append("  Strategies:")
            for player_idx, strategy in enumerate(spne.strategies):
                if strategy.decisions:
                    output.append(f"    Player {player_idx}:")
                    for info_set, action in sorted(strategy.decisions.items()):
                        output.append(f"      {info_set} → {action}")
                else:
                    output.append(f"    Player {player_idx}: (no decisions)")

            output.append("")

        return "\n".join(output)
    except Exception as e:
        return f"Error getting SPNE: {e}"


@mcp.tool(title="Get game state as JSON")
def get_game_state(game_id: str, include_solution: bool = False) -> str:
    """
    Get the current state of a game tree in JSON format.

    Args:
        game_id: ID of the game tree
        include_solution: Whether to include BI solution values (default: False)

    Returns:
        JSON representation of the game tree
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    tree = _games[game_id]

    try:
        import json

        state_dict = tree.to_dict(include_bi_values=include_solution)
        return json.dumps(state_dict, indent=2)
    except Exception as e:
        return f"Error serializing game: {e}"


@mcp.tool()
def list_games() -> str:
    """
    List all active game sessions.

    Returns:
        List of game IDs and their basic info
    """
    if not _games:
        return "No active games"

    output = ["Active games:"]
    for game_id, tree in _games.items():
        num_nodes = tree.size()
        output.append(f"  {game_id}: {tree.num_players} players, {num_nodes} nodes")

    return "\n".join(output)


@mcp.tool()
def delete_game(game_id: str) -> str:
    """
    Delete a game from the session store.

    Args:
        game_id: ID of the game tree to delete

    Returns:
        Success message
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    del _games[game_id]
    return f"Deleted game '{game_id}'"


@mcp.tool()
def show_tree(
    game_id: str,
    show_id: bool = True,
    show_probability: bool = True,
    show_bi_value: bool = True,
    line_type: str = "ascii-ex",
) -> str:
    """
    Display the game tree structure in text format.

    Args:
        game_id: ID of the game tree
        show_id: Show node identifiers (default: True)
        show_probability: Show edge probabilities (default: True)
        show_bi_value: Show backward induction values (default: True)
        line_type: Tree line style - 'ascii', 'ascii-ex', 'ascii-em' (default: 'ascii-ex')

    Returns:
        Tree visualization
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    tree = _games[game_id]

    try:
        # Build tree representation
        output = [f"Game tree '{game_id}':"]
        output.append(
            tree.show(
                stdout=False,
                show_id=show_id,
                show_probability=show_probability,
                show_bi_value=show_bi_value,
                line_type=line_type,
            )
        )
        return "\n".join(output)
    except Exception as e:
        return f"Error displaying tree: {e}"


@mcp.tool()
def load_game(game_json: str, name: str | None = None) -> str:
    """
    Create a game tree from a JSON description. This is the fastest way to
    define a game — write the entire tree in one shot instead of adding nodes
    one by one.

    Args:
        game_json: JSON string describing the game tree (see schema below)
        name: Optional name for the game (auto-generated if not provided)

    Returns:
        game_id that can be used with solve_game, show_tree, etc.

    ## JSON Schema

    ```json
    {
      "num_players": <int>,          // number of players (default: 2)
      "root": "<node_id>",           // id of the root node
      "nodes": [
        {
          "id": "<node_id>",         // unique identifier for this node
          "tag": "<label>",          // optional human-readable label (defaults to id)
          "parent": "<node_id>",     // parent node id; null or omitted for the root
          "data": { ... }            // node data (see types below)
        },
        ...
      ]
    }
    ```

    ### Node data types

    **Decision node** — a player makes a choice:
    ```json
    { "type": "decision", "player": <int> }
    ```
    Optional fields:
    - `"probability"`: edge probability (e.g., `"1/2"`)
    - `"information_set"`: string id grouping nodes in the same info set
    - `"actions"`: list of action labels (e.g., `["up", "down"]`) used as
      action names in strategies (this is very necessary for imperfect
      information games, otherwise child node ids are used as action names)

    **Chance node** — nature moves with given probabilities on its children:
    ```json
    { "type": "chance" }
    ```
    Probabilities belong on the **children** of a chance node, not on the chance node
    itself. Optional field: `"probability"` (edge probability for this node's own edge).

    **Terminal node** — end of the game with payoffs for each player:
    ```json
    { "type": "terminal", "payoffs": [<p1>, <p2>, ...] }
    ```
    Payoffs can be integers, floats, or fraction strings like `"1/2"`.
    Optional field: `"probability"` (edge probability).

    ## Example — Centipede game (2 players, 4 terminal nodes)

    ```json
    {
      "num_players": 2,
      "root": "p1a",
      "nodes": [
        {"id": "p1a",  "parent": null,  "data": {"type": "decision", "player": 0}},
        {"id": "stop1","parent": "p1a", "data": {"type": "terminal", "payoffs": [1, 0]}},
        {"id": "p2a",  "parent": "p1a", "data": {"type": "decision", "player": 1}},
        {"id": "stop2","parent": "p2a", "data": {"type": "terminal", "payoffs": [0, 3]}},
        {"id": "p1b",  "parent": "p2a", "data": {"type": "decision", "player": 0}},
        {"id": "stop3","parent": "p1b", "data": {"type": "terminal", "payoffs": [2, 1]}},
        {"id": "cont", "parent": "p1b", "data": {"type": "terminal", "payoffs": [1, 4]}}
      ]
    }
    ```
    """
    import json

    try:
        raw = json.loads(game_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON — {e}"

    # Fill in optional tag fields
    for node in raw.get("nodes", []):
        if "tag" not in node or node["tag"] is None:
            node["tag"] = node["id"]
        # Normalise payoffs: allow lists in addition to tuples (JSON has no tuples)
        data = node.get("data")
        if data and data.get("type") == "terminal":
            data["payoffs"] = list(data["payoffs"])

    try:
        tree = GameTree.from_dict(raw)
    except Exception as e:
        return f"Error building game tree: {e}"

    game_id = _new_id("game") if name is None else name
    if game_id in _games:
        return f"Error: A game named '{game_id}' already exists"
    _games[game_id] = tree

    num_nodes = tree.size()
    return (
        f"Loaded game '{game_id}' with {tree.num_players} players and {num_nodes} nodes. "
        f"Call solve_game('{game_id}') to run backward induction (only for perfect information)."
    )


@mcp.tool(title="Find pure strategies for a player")
def find_player_strategies(game_id: str, player: int, reduced: bool = False) -> str:
    """
    Find all pure strategies for a player in an extensive-form game.

    Args:
        game_id: ID of the game tree
        player: Player index (0-based)
        reduced: If True, find reduced pure strategies; otherwise full pure strategies (default: False)

    Returns:
        A description of all strategies for the player
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    tree = _games[game_id]

    try:
        if reduced:
            strategies = find_reduced_pure_strategies(tree, player)
            strategy_type = "reduced"
        else:
            strategies = find_full_pure_strategies(tree, player)
            strategy_type = "full"

        output = [
            f"Found {len(strategies)} {strategy_type} pure strategies for player {player} in '{game_id}':"
        ]

        for i, strategy in enumerate(sorted(strategies, key=repr), 1):
            if strategy.decisions:
                output.append(f"  Strategy {i}:")
                for info_set, action in sorted(strategy.decisions.items()):
                    output.append(f"    {info_set} → {action}")
            else:
                output.append(f"  Strategy {i}: (empty)")

        return "\n".join(output)
    except Exception as e:
        return f"Error finding strategies: {e}"


@mcp.tool(title="Compute strategic form")
def compute_strategic_form(game_id: str) -> str:
    """
    Convert an extensive-form game to strategic (normal) form.

    Produces a payoff tensor where entry [i0, i1, ..., ik, p] contains
    player p's payoff when player 0 plays strategy i0, player 1 plays
    strategy i1, etc.

    Args:
        game_id: ID of the game tree

    Returns:
        A description of the strategic form with all strategy profiles and payoffs
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    tree = _games[game_id]

    try:
        strategies, payoffs = extensive_to_strategic(tree)

        output = [
            f"Strategic form for '{game_id}':",
            f"Players: {tree.num_players}",
        ]

        for player_idx, player_strategies in enumerate(strategies):
            output.append(
                f"\nPlayer {player_idx} strategies ({len(player_strategies)}):"
            )
            for i, strategy in enumerate(player_strategies):
                if strategy.decisions:
                    decisions_str = ", ".join(
                        f"{info_set}→{action}"
                        for info_set, action in sorted(strategy.decisions.items())
                    )
                    output.append(f"  {i}: {decisions_str}")
                else:
                    output.append(f"  {i}: (empty)")

        output.append(f"\nPayoff tensor shape: {payoffs.shape}")

        total_profiles = 1
        for n in payoffs.shape[:-1]:
            total_profiles *= n
        output.append(f"Total strategy profiles: {total_profiles}")

        return "\n".join(output)
    except Exception as e:
        return f"Error computing strategic form: {e}"


@mcp.tool(title="Execute strategy profile")
def execute_profile(game_id: str, profile_json: str) -> str:
    """
    Execute a strategy profile and return the resulting payoff.

    Given a game tree and a mapping from player index to their Strategy,
    walks through the tree from the root and returns the terminal payoff.

    Args:
        game_id: ID of the game tree
        profile_json: JSON string mapping player indices to their strategy decisions.
            Format: {"<player_idx>": {"<info_set>": "<action_id>", ...}, ...}
            Example: {"0": {"root": "left"}, "1": {}}

    Returns:
        The resulting payoff tuple
    """
    import json

    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"

    tree = _games[game_id]

    try:
        profile_data = json.loads(profile_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON — {e}"

    try:
        profile: dict[int, Strategy] = {}
        for player_str, decisions in profile_data.items():
            player = int(player_str)
            profile[player] = Strategy(decisions)

        payoff = exec_profile(tree, profile)

        payoff_strs = [str(p) for p in payoff]
        return f"Payoff: ({', '.join(payoff_strs)})"
    except Exception as e:
        return f"Error executing profile: {e}"


if __name__ == "__main__":
    mcp.run()
