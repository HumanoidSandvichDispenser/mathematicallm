#!/usr/bin/env python3
"""
Demonstrate the enhanced tree display with all node attributes.

This example shows how the tree display presents:
- Node names (tags)
- Node IDs
- Player information (P0, P1, etc.)
- Chance nodes (CHANCE)
- Terminal payoffs
- Edge probabilities
- Backward induction values
"""

from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    ChanceNodeData,
    TerminalNodeData,
)

print("=" * 70)
print("Enhanced Tree Display Demo")
print("=" * 70)

# Build a game with all node types
tree = GameTree(num_players=2)

# Root: Player 0 decides
tree.create_node(
    tag="Player 0's Choice",
    identifier="root",
    data=DecisionNodeData(player=0)
)

# P0 can choose "Risky" or "Safe"
tree.create_node(
    tag="Risky Path",
    identifier="risky",
    parent="root",
    data=ChanceNodeData()  # Nature decides
)

tree.create_node(
    tag="Safe Path",
    identifier="safe",
    parent="root",
    data=TerminalNodeData(payoffs=(2, 2))  # Safe outcome
)

# If risky, nature flips a coin
tree.create_node(
    tag="Lucky Outcome",
    identifier="lucky",
    parent="risky",
    data=DecisionNodeData(player=1, probability="0.6")  # 60% chance
)

tree.create_node(
    tag="Unlucky Outcome",
    identifier="unlucky",
    parent="risky",
    data=DecisionNodeData(player=1, probability="0.4")  # 40% chance
)

# Player 1 responds to lucky
tree.create_node(
    tag="Accept",
    identifier="lucky_accept",
    parent="lucky",
    data=TerminalNodeData(payoffs=(5, 3))
)

tree.create_node(
    tag="Reject",
    identifier="lucky_reject",
    parent="lucky",
    data=TerminalNodeData(payoffs=(1, 4))
)

# Player 1 responds to unlucky
tree.create_node(
    tag="Accept",
    identifier="unlucky_accept",
    parent="unlucky",
    data=TerminalNodeData(payoffs=(0, 1))
)

tree.create_node(
    tag="Reject",
    identifier="unlucky_reject",
    parent="unlucky",
    data=TerminalNodeData(payoffs=(3, 0))
)

print("\n" + "-" * 70)
print("BEFORE solving (no BI values yet)")
print("-" * 70)
tree.show()

print("\n" + "-" * 70)
print("Solving with backward induction...")
print("-" * 70)
result = tree.backward_induction(mutate=True)
print(f"Equilibrium payoffs: {result}")

print("\n" + "-" * 70)
print("AFTER solving (with BI values)")
print("-" * 70)
tree.show()

print("\n" + "-" * 70)
print("Display Options Demo")
print("-" * 70)

print("\nOption 1: Hide IDs")
tree.show(show_id=False)

print("\nOption 2: Hide probabilities")
tree.show(show_probability=False)

print("\nOption 3: Hide BI values (clean structure view)")
tree.show(show_bi_value=False)

print("\nOption 4: Minimal view (names and types only)")
tree.show(show_id=False, show_probability=False, show_bi_value=False)

print("\n" + "=" * 70)
print("The display shows:")
print("  • Node name (tag)")
print("  • [Node ID] in brackets")
print("  • P0, P1, etc. for players")
print("  • CHANCE for nature nodes")
print("  • (payoff1, payoff2) for terminals")
print("  • p=probability for edge probabilities")
print("  • BI=(value1, value2) for backward induction results")
print("=" * 70)
