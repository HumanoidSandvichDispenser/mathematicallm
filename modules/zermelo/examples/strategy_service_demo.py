"""
Example demonstrating pure strategy generation with information sets.

This shows how to use find_full_pure_strategies with both perfect and
imperfect information games.
"""

from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    TerminalNodeData,
)
from zermelo.services.strategy_service import find_full_pure_strategies


def example_perfect_information():
    """
    A perfect information game where each decision node is its own info set.
    """
    print("=== Perfect Information Game ===\n")
    print("Structure:")
    print("  P0 at root -> P1 at left OR P1 at right")
    print("  Each node is its own information set\n")

    tree = GameTree(num_players=2)

    # Root: P0 decides
    tree.create_node("P0 decision", "root", data=DecisionNodeData(player=0))

    # P0 goes left -> P1 decides
    tree.create_node(
        "P1 after Left", "p1_left", parent="root", data=DecisionNodeData(player=1)
    )
    tree.create_node(
        "L1 outcome", "l1", parent="p1_left", data=TerminalNodeData(payoffs=(1, 1))
    )
    tree.create_node(
        "L2 outcome", "l2", parent="p1_left", data=TerminalNodeData(payoffs=(2, 0))
    )

    # P0 goes right -> P1 decides
    tree.create_node(
        "P1 after Right", "p1_right", parent="root", data=DecisionNodeData(player=1)
    )
    tree.create_node(
        "R1 outcome", "r1", parent="p1_right", data=TerminalNodeData(payoffs=(0, 2))
    )
    tree.create_node(
        "R2 outcome", "r2", parent="p1_right", data=TerminalNodeData(payoffs=(3, 3))
    )

    print("Game tree:")
    tree.show()
    print()

    # Find strategies for each player
    p0_strategies = find_full_pure_strategies(tree, player=0)
    p1_strategies = find_full_pure_strategies(tree, player=1)

    print(f"Player 0 has {len(p0_strategies)} full pure strategy(s):")
    for i, s in enumerate(p0_strategies, 1):
        print(f"  {i}. {s.decisions}")

    print(f"\nPlayer 1 has {len(p1_strategies)} full pure strategy(ies):")
    for i, s in enumerate(p1_strategies, 1):
        print(f"  {i}. {s.decisions}")

    print(f"\nIs perfect information: {tree.is_perfect_information()}")
    print()


def example_imperfect_information():
    """
    An imperfect information game where P1 has two nodes in the same info set.
    """
    print("=== Imperfect Information Game ===\n")
    print("Structure:")
    print("  P0 at root -> goes to P1 (left or right branch)")
    print("  P1 cannot distinguish which branch they're in (same info set)")
    print("  P1 sees same set of actions at both nodes\n")

    tree = GameTree(num_players=2)

    # Root: P0 decides
    tree.create_node("P0 decision", "root", data=DecisionNodeData(player=0))

    # Both branches lead to P1 in the SAME information set
    tree.create_node(
        "P1 at left",
        "p1_left",
        parent="root",
        data=DecisionNodeData(player=1, information_set="I1"),
    )
    tree.create_node(
        "P1 at right",
        "p1_right",
        parent="root",
        data=DecisionNodeData(player=1, information_set="I1"),
    )

    # Each P1 node has same actions: Up and Down
    tree.create_node(
        "Up outcome", "up_l", parent="p1_left", data=TerminalNodeData(payoffs=(1, 1))
    )
    tree.create_node(
        "Down outcome",
        "down_l",
        parent="p1_left",
        data=TerminalNodeData(payoffs=(2, 0)),
    )

    tree.create_node(
        "Up outcome", "up_r", parent="p1_right", data=TerminalNodeData(payoffs=(0, 2))
    )
    tree.create_node(
        "Down outcome",
        "down_r",
        parent="p1_right",
        data=TerminalNodeData(payoffs=(3, 3)),
    )

    print("Game tree:")
    tree.show()
    print()

    # Find strategies for each player
    p0_strategies = find_full_pure_strategies(tree, player=0)
    p1_strategies = find_full_pure_strategies(tree, player=1)

    print(f"Player 0 has {len(p0_strategies)} full pure strategy(s):")
    for i, s in enumerate(p0_strategies, 1):
        print(f"  {i}. {s.decisions}")

    print(f"\nPlayer 1 has {len(p1_strategies)} full pure strategy(ies):")
    for i, s in enumerate(p1_strategies, 1):
        print(f"  {i}. {s.decisions}")
    print("Note: P1's strategies map to info set 'I1', not individual nodes!")

    print(f"\nIs perfect information: {tree.is_perfect_information()}")
    print()


def example_multiple_info_sets():
    """
    A game with multiple information sets for the same player.
    """
    print("=== Multiple Information Sets ===\n")
    print("Structure:")
    print("  P0 decides at root")
    print("  After P0, P1 decides (info set 'I1')")
    print("  After that, P0 decides again (info set 'I2')")
    print()

    tree = GameTree(num_players=2)

    # Root: P0 decides
    tree.create_node("P0 first", "root", data=DecisionNodeData(player=0))

    # Both branches lead to P1 in the same info set
    tree.create_node(
        "P1 decision",
        "p1_a",
        parent="root",
        data=DecisionNodeData(player=1, information_set="I1"),
    )
    tree.create_node(
        "P1 decision",
        "p1_b",
        parent="root",
        data=DecisionNodeData(player=1, information_set="I1"),
    )

    # Each P1 node leads to P0 again (different info set)
    tree.create_node(
        "P0 second (a)",
        "p0_a",
        parent="p1_a",
        data=DecisionNodeData(player=0, information_set="I2"),
    )
    tree.create_node(
        "P0 second (b)",
        "p0_b",
        parent="p1_b",
        data=DecisionNodeData(player=0, information_set="I2"),
    )

    # Terminals
    tree.create_node("T1", "t1", parent="p0_a", data=TerminalNodeData(payoffs=(1, 1)))
    tree.create_node("T2", "t2", parent="p0_a", data=TerminalNodeData(payoffs=(2, 0)))
    tree.create_node("T3", "t3", parent="p0_b", data=TerminalNodeData(payoffs=(0, 2)))
    tree.create_node("T4", "t4", parent="p0_b", data=TerminalNodeData(payoffs=(3, 3)))

    print("Game tree:")
    tree.show()
    print()

    # Find strategies
    p0_strategies = find_full_pure_strategies(tree, player=0)
    p1_strategies = find_full_pure_strategies(tree, player=1)

    print(f"Player 0 has {len(p0_strategies)} full pure strategy(ies):")
    print("  (P0 has 2 info sets: 'root' and 'I2', each with 2 actions)")
    for i, s in enumerate(p0_strategies, 1):
        print(f"  {i}. {s.decisions}")

    print(f"\nPlayer 1 has {len(p1_strategies)} full pure strategy(ies):")
    print("  (P1 has 1 info set 'I1' with 2 actions)")
    for i, s in enumerate(p1_strategies, 1):
        print(f"  {i}. {s.decisions}")

    print(f"\nIs perfect information: {tree.is_perfect_information()}")
    print()


if __name__ == "__main__":
    example_perfect_information()
    print("-" * 50)
    example_imperfect_information()
    print("-" * 50)
    example_multiple_info_sets()
