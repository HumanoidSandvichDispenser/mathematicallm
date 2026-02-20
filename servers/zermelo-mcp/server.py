"""MCP server for game tree analysis operations."""

from mcp.server.fastmcp import FastMCP
from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    ChanceNodeData,
    TerminalNodeData,
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
) -> str:
    """
    Add a decision node to the game tree.
    
    Args:
        game_id: ID of the game tree
        node_id: Unique identifier for the new node
        parent_id: ID of the parent node
        player: Player index (0-based) who makes the decision at this node
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
        data=DecisionNodeData(player=player, probability=probability),
    )
    
    prob_info = f" with edge probability {probability}" if probability else ""
    return f"Added decision node '{node_id}' for player {player} under '{parent_id}'{prob_info}"


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
            return f"Error: Expected {tree.num_players} payoffs, got {len(payoff_values)}"
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


@mcp.tool()
def solve_game(game_id: str) -> str:
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
        result = tree.backward_induction(mutate=True)
        
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
            elif hasattr(node.data, 'bi_value') and node.data.bi_value is not None:
                bi_val = node.data.bi_value
                bi_strs = [str(v) for v in bi_val]
                if node.is_decision:
                    # Show optimal children for decision nodes
                    optimal_str = f", optimal: {node.data.optimal_children}" if node.data.optimal_children else ""
                    output.append(f"  {node_id} (player {node.data.player}): ({', '.join(bi_strs)}){optimal_str}")
                else:
                    output.append(f"  {node_id} (chance): ({', '.join(bi_strs)})")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error solving game: {e}"


@mcp.tool()
def get_all_equilibria(game_id: str) -> str:
    """
    Get all subgame perfect Nash equilibria.
    
    Must be called after solve_game(). When there are ties (players indifferent
    between actions), this enumerates all possible equilibrium paths.
    
    Args:
        game_id: ID of the game tree
    
    Returns:
        All equilibrium paths with payoffs and action profiles
    """
    if game_id not in _games:
        return f"Error: Game '{game_id}' not found"
    
    tree = _games[game_id]
    
    try:
        equilibria = tree.get_all_equilibria()
        
        output = [f"Equilibria for '{game_id}':"]
        output.append(f"Found {len(equilibria)} equilibrium/equilibria\n")
        
        for i, eq in enumerate(equilibria, 1):
            payoff_strs = [str(p) for p in eq.payoffs]
            output.append(f"Equilibrium {i}:")
            output.append(f"  Payoffs: ({', '.join(payoff_strs)})")
            if eq.actions:
                output.append(f"  Actions:")
                for node_id, child_id in sorted(eq.actions.items()):
                    output.append(f"    {node_id} → {child_id}")
            else:
                output.append(f"  Actions: (none - tree is just a terminal node)")
            output.append("")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error getting equilibria: {e}"


@mcp.tool()
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


if __name__ == "__main__":
    mcp.run()
