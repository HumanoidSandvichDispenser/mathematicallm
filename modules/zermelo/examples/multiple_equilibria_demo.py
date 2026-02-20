#!/usr/bin/env python3
"""
Demonstration of multiple perfect Nash equilibria support.

This example shows how to detect and enumerate all subgame perfect equilibria
when players have ties (indifference) between actions.
"""

from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    TerminalNodeData,
    EquilibriumPath,
)

print("=" * 70)
print("Multiple Equilibria Demo")
print("=" * 70)

# Example 1: Simple tie at root
print("\n" + "-" * 70)
print("Example 1: Two equilibria from tie at root")
print("-" * 70)

tree1 = GameTree(num_players=2)
tree1.create_node("P0 chooses", "root", data=DecisionNodeData(player=0))
tree1.create_node("Left", "left", parent="root", data=TerminalNodeData(payoffs=(3, 1)))
tree1.create_node("Right", "right", parent="root", data=TerminalNodeData(payoffs=(3, 2)))

print("\nGame tree:")
tree1.show(show_bi_value=False)

tree1.backward_induction(mutate=True)
equilibria1 = tree1.get_all_equilibria()

print(f"\nFound {len(equilibria1)} equilibrium/equilibria:")
for i, eq in enumerate(equilibria1, 1):
    print(f"  Equilibrium {i}: {eq}")

# Example 2: 4 equilibria from independent ties
print("\n\n" + "-" * 70)
print("Example 2: Four equilibria from 2x2 independent ties")
print("-" * 70)

tree2 = GameTree(num_players=2)

# P0 has two equally good choices
tree2.create_node("P0 chooses", "root", data=DecisionNodeData(player=0))
tree2.create_node("Path A", "A", parent="root", data=DecisionNodeData(player=1))
tree2.create_node("Path B", "B", parent="root", data=DecisionNodeData(player=1))

# Under A, P1 also has a tie
tree2.create_node("A-Left", "A1", parent="A", data=TerminalNodeData(payoffs=(5, 3)))
tree2.create_node("A-Right", "A2", parent="A", data=TerminalNodeData(payoffs=(5, 3)))

# Under B, P1 also has a tie  
tree2.create_node("B-Left", "B1", parent="B", data=TerminalNodeData(payoffs=(5, 3)))
tree2.create_node("B-Right", "B2", parent="B", data=TerminalNodeData(payoffs=(5, 3)))

print("\nGame tree:")
tree2.show(show_bi_value=False)

tree2.backward_induction(mutate=True)
equilibria2 = tree2.get_all_equilibria()

print(f"\nFound {len(equilibria2)} equilibria:")
for i, eq in enumerate(equilibria2, 1):
    print(f"  Equilibrium {i}:")
    print(f"    Payoffs: {eq.payoffs}")
    print(f"    P0 chooses: {eq.actions.get('root', 'N/A')}")
    chosen_subgame = eq.actions.get('root', '')
    if chosen_subgame in eq.actions.values():
        for node, choice in eq.actions.items():
            if node != 'root':
                print(f"    P1 at {node} chooses: {choice}")

# Example 3: Unique equilibrium (no ties)
print("\n\n" + "-" * 70)
print("Example 3: Unique equilibrium (no ties)")
print("-" * 70)

tree3 = GameTree(num_players=2)
tree3.create_node("P0 chooses", "root", data=DecisionNodeData(player=0))
tree3.create_node("Left (P0 prefers)", "left", parent="root", data=TerminalNodeData(payoffs=(5, 1)))
tree3.create_node("Right", "right", parent="root", data=TerminalNodeData(payoffs=(2, 3)))

print("\nGame tree:")
tree3.show(show_bi_value=False)

tree3.backward_induction(mutate=True)
equilibria3 = tree3.get_all_equilibria()

print(f"\nFound {len(equilibria3)} equilibrium:")
for eq in equilibria3:
    print(f"  {eq}")

# Example 4: Tie only in off-equilibrium-path subgame
print("\n\n" + "-" * 70)
print("Example 4: Tie in subgame that's reached")
print("-" * 70)

tree4 = GameTree(num_players=2)
tree4.create_node("P0 chooses", "root", data=DecisionNodeData(player=0))
tree4.create_node("Go to subgame", "sub", parent="root", data=DecisionNodeData(player=1))
tree4.create_node("Take outside option", "out", parent="root", data=TerminalNodeData(payoffs=(2, 0)))

# P1 has tie in subgame
tree4.create_node("Subgame-A", "sub_a", parent="sub", data=TerminalNodeData(payoffs=(4, 2)))
tree4.create_node("Subgame-B", "sub_b", parent="sub", data=TerminalNodeData(payoffs=(4, 2)))

print("\nGame tree:")
tree4.show(show_bi_value=False)

tree4.backward_induction(mutate=True)

print("\nAfter solving (with BI values):")
tree4.show()

equilibria4 = tree4.get_all_equilibria()

print(f"\nFound {len(equilibria4)} equilibria:")
for i, eq in enumerate(equilibria4, 1):
    print(f"  Equilibrium {i}: {eq}")

print("\n" + "=" * 70)
print("Key Insights:")
print("  • backward_induction() computes optimal choices")
print("  • get_all_equilibria() enumerates all equilibrium paths")
print("  • Ties create multiple equilibria")
print("  • optimal_children field stores all optimal actions")
print("=" * 70)
