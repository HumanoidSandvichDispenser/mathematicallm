"""
Example usage of the GameTree implementation with composable NodeData.

This demonstrates how to build, solve, and serialize game trees using the
new mixin-based NodeData architecture.
"""

import sympy as sp
from zermelo.extensive import (
    GameTree,
    DecisionNodeData,
    ChanceNodeData,
    TerminalNodeData,
)


def example_prisoners_dilemma():
    """
    Build and solve a sequential prisoners' dilemma.

    Player 0 moves first (Cooperate or Defect)
    Player 1 observes P0's choice and responds
    """
    print("=== Sequential Prisoners' Dilemma ===\n")

    tree = GameTree(num_players=2)

    # Root: Player 0 decides
    tree.create_node(
        tag="P0 decision",
        identifier="root",
        data=DecisionNodeData(player=0)
    )

    # Branch 1: P0 cooperates, P1 decides
    tree.create_node(
        tag="P1 after P0 cooperates",
        identifier="p0_coop",
        parent="root",
        data=DecisionNodeData(player=1)
    )

    tree.create_node(
        tag="Both cooperate",
        identifier="cc",
        parent="p0_coop",
        data=TerminalNodeData(payoffs=(3, 3))
    )

    tree.create_node(
        tag="P0 cooperates, P1 defects",
        identifier="cd",
        parent="p0_coop",
        data=TerminalNodeData(payoffs=(0, 5))
    )

    # Branch 2: P0 defects, P1 decides
    tree.create_node(
        tag="P1 after P0 defects",
        identifier="p0_def",
        parent="root",
        data=DecisionNodeData(player=1)
    )

    tree.create_node(
        tag="P0 defects, P1 cooperates",
        identifier="dc",
        parent="p0_def",
        data=TerminalNodeData(payoffs=(5, 0))
    )

    tree.create_node(
        tag="Both defect",
        identifier="dd",
        parent="p0_def",
        data=TerminalNodeData(payoffs=(1, 1))
    )

    # Display tree
    print("Game tree:")
    tree.show()
    print()

    # Solve
    result = tree.backward_induction(mutate=True)
    print(f"Backward induction result: {result}")
    print(f"P0 gets: {result[0]}, P1 gets: {result[1]}")

    # Show BI values at each node
    print("\nBI values at decision nodes:")
    print(f"Root (P0): {tree.get_node('root').data.bi_value}")
    print(f"After P0 cooperates (P1): {tree.get_node('p0_coop').data.bi_value}")
    print(f"After P0 defects (P1): {tree.get_node('p0_def').data.bi_value}")
    print()


def example_chance_node():
    """
    Build a game with a chance node (coin flip).
    """
    print("=== Game with Chance Node ===\n")

    tree = GameTree(num_players=2)

    # Root: Nature flips a coin
    tree.create_node(
        tag="Coin flip",
        identifier="root",
        data=ChanceNodeData()
    )

    # Heads (50%): P0 decides
    tree.create_node(
        tag="After heads",
        identifier="p0_heads",
        parent="root",
        data=DecisionNodeData(player=0, probability=sp.Rational(1, 2))
    )

    tree.create_node(
        tag="Take (after heads)",
        identifier="heads_take",
        parent="p0_heads",
        data=TerminalNodeData(payoffs=(10,))
    )

    tree.create_node(
        tag="Pass (after heads)",
        identifier="heads_pass",
        parent="p0_heads",
        data=TerminalNodeData(payoffs=(5,))
    )

    # Tails (50%): Different payoffs
    tree.create_node(
        tag="After tails",
        identifier="p0_tails",
        parent="root",
        data=DecisionNodeData(player=0, probability=sp.Rational(1, 2))
    )

    tree.create_node(
        tag="Take (after tails)",
        identifier="tails_take",
        parent="p0_tails",
        data=TerminalNodeData(payoffs=(2,))
    )

    tree.create_node(
        tag="Pass (after tails)",
        identifier="tails_pass",
        parent="p0_tails",
        data=TerminalNodeData(payoffs=(3,))
    )

    print("Game tree:")
    tree.show()
    print()

    result = tree.backward_induction(mutate=True)
    print(f"Expected value: {result}")
    print(f"Root BI value: {tree.get_node('root').data.bi_value}")
    print()


def example_symbolic_payoffs():
    """
    Demonstrate symbolic payoff expressions.
    """
    print("=== Symbolic Payoffs ===\n")

    x, y = sp.symbols('x y', real=True, positive=True)

    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))

    tree.create_node(
        "Option A",
        "a",
        parent="root",
        data=TerminalNodeData(payoffs=(x, y))
    )

    tree.create_node(
        "Option B",
        "b",
        parent="root",
        data=TerminalNodeData(payoffs=(2*x, y/2))
    )

    result = tree.backward_induction(mutate=True)
    print(f"P0 chooses to maximize their payoff")
    print(f"Result: {result}")
    print(f"P0 gets: {result[0]} (should be max(x, 2x) = 2x)")
    print(f"Stored BI value: {tree.get_node('root').data.bi_value}")
    print()


def example_serialization():
    """
    Demonstrate serialization and deserialization.
    """
    print("=== Serialization ===\n")

    # Build a simple tree
    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=DecisionNodeData(player=0))
    tree.create_node(
        "Left",
        "left",
        parent="root",
        data=TerminalNodeData(payoffs=(1,))
    )
    tree.create_node(
        "Right",
        "right",
        parent="root",
        data=TerminalNodeData(payoffs=(2,))
    )

    # Solve it
    tree.backward_induction(mutate=True)

    # Serialize
    data = tree.to_dict()
    print("Serialized data:")
    import json
    print(json.dumps(data, indent=2))
    print()

    # Deserialize
    tree2 = GameTree.from_dict(data)
    print("Deserialized tree:")
    tree2.show()
    print()

    # Verify BI values preserved
    print(f"Original root BI value: {tree.get_node('root').data.bi_value}")
    print(f"Restored root BI value: {tree2.get_node('root').data.bi_value}")
    print()


def example_automatic_sympify():
    """
    Demonstrate automatic sympification of probabilities and payoffs.
    """
    print("=== Automatic Sympification ===\n")

    tree = GameTree(num_players=2)
    tree.create_node("Root", "root", data=ChanceNodeData())

    # Pass different types - all get sympified
    tree.create_node(
        "A", "a", parent="root",
        data=TerminalNodeData(payoffs=(1, 2), probability=0.5)  # float
    )
    tree.create_node(
        "B", "b", parent="root",
        data=TerminalNodeData(payoffs=(3.5, "5/2"), probability="1/2")  # mixed types
    )

    print("Node A:")
    a = tree.get_node("a")
    print(f"  Payoffs: {a.data.payoffs} (types: {[type(p).__name__ for p in a.data.payoffs]})")
    print(f"  Probability: {a.data.probability} (type: {type(a.data.probability).__name__})")

    print("\nNode B:")
    b = tree.get_node("b")
    print(f"  Payoffs: {b.data.payoffs} (types: {[type(p).__name__ for p in b.data.payoffs]})")
    print(f"  Probability: {b.data.probability} (type: {type(b.data.probability).__name__})")

    # Solve and show result
    result = tree.backward_induction(mutate=True)
    print(f"\nExpected value: {result}")
    print()


if __name__ == "__main__":
    example_prisoners_dilemma()
    example_chance_node()
    example_symbolic_payoffs()
    example_serialization()
    example_automatic_sympify()
