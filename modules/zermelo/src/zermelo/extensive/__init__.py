"""
Extensive-form game analysis module.

This module provides tools for building, analyzing, and solving extensive-form
games (game trees). It uses treelib for tree structure and sympy for symbolic
payoff expressions.

Main components:
- GameTree: Tree structure for extensive-form games
- GameNode: Node wrapper compatible with treelib
- NodeData types: DecisionNodeData, ChanceNodeData, TerminalNodeData, BIValue

Example:
    from zermelo.extensive import GameTree, DecisionNodeData, TerminalNodeData
    
    # Build a simple game
    tree = GameTree()
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    tree.create_node(
        "Left", "left", parent="root",
        data=TerminalNodeData(payoffs=(1, 0))
    )
    tree.create_node(
        "Right", "right", parent="root",
        data=TerminalNodeData(payoffs=(0, 1))
    )
    
    # Solve
    tree.backward_induction(mutate=True)
    result = tree.get_node("root").data.bi_value
    print(result)  # (1, 0) — Player 0 chooses Left
"""

from .game_tree import GameTree
from .game_node import GameNode
from .node_data import NodeData, BIValue, DecisionNodeData, ChanceNodeData, TerminalNodeData
from .equilibrium import EquilibriumPath

__all__ = [
    "GameTree",
    "GameNode",
    "NodeData",
    "BIValue",
    "DecisionNodeData",
    "ChanceNodeData",
    "TerminalNodeData",
    "EquilibriumPath",
]
