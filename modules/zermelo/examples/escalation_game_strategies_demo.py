"""
Demo: pure strategies in the Escalation Game.

The Escalation Game (perfect information, 2 players):

    Player 1 [p1_root]
    ├── Accept ──────────────────── (0, 0)
    └── Threaten [p2_node]
        ├── Concede ─────────────── (1, -2)
        └── Escalate [p1_again]
            ├── Give Up ─────────── (-2, 1)
            └── War ─────────────── (-1, -1)

Full pure strategies specify an action at every information set the player
owns, regardless of reachability.

Reduced pure strategies only specify actions at information sets that are
actually reachable given the player's own earlier choices.
"""

from zermelo.extensive import GameTree, DecisionNodeData, TerminalNodeData
from zermelo.services.strategy_service import (
    find_full_pure_strategies,
    find_reduced_pure_strategies,
)


def build_escalation_game() -> GameTree:
    tree = GameTree(num_players=2)
    tree.create_node("Player 1", "p1_root", data=DecisionNodeData(player=0))
    tree.create_node(
        "Accept", "accept", parent="p1_root", data=TerminalNodeData(payoffs=(0, 0))
    )
    tree.create_node(
        "Player 2", "p2_node", parent="p1_root", data=DecisionNodeData(player=1)
    )
    tree.create_node(
        "Concede", "concede", parent="p2_node", data=TerminalNodeData(payoffs=(1, -2))
    )
    tree.create_node(
        "Player 1 again",
        "p1_again",
        parent="p2_node",
        data=DecisionNodeData(player=0),
    )
    tree.create_node(
        "Give Up", "give_up", parent="p1_again", data=TerminalNodeData(payoffs=(-2, 1))
    )
    tree.create_node(
        "War", "war", parent="p1_again", data=TerminalNodeData(payoffs=(-1, -1))
    )
    return tree


def print_strategies(label: str, strategies: set) -> None:
    print(f"{label} ({len(strategies)}):")
    for i, s in enumerate(sorted(strategies, key=repr), 1):
        print(f"  {i}. {s}")
    print()


def main() -> None:
    game = build_escalation_game()

    print("=== Escalation Game ===\n")
    game.show()
    print()

    # ------------------------------------------------------------------
    # Full pure strategies
    # ------------------------------------------------------------------
    print("--- Full Pure Strategies ---\n")
    print("Every information set must be assigned an action, even unreachable ones.\n")

    p1_full = find_full_pure_strategies(game, player=0)
    p2_full = find_full_pure_strategies(game, player=1)

    print_strategies("Player 1 (P0) full strategies", p1_full)
    print_strategies("Player 2 (P1) full strategies", p2_full)

    # ------------------------------------------------------------------
    # Reduced pure strategies
    # ------------------------------------------------------------------
    print("--- Reduced Pure Strategies ---\n")
    print(
        "Information sets that are cut off by the player's own earlier\n"
        "choices are omitted.\n"
    )

    p1_reduced = find_reduced_pure_strategies(game, player=0)
    p2_reduced = find_reduced_pure_strategies(game, player=1)

    print_strategies("Player 1 (P0) reduced strategies", p1_reduced)
    print_strategies("Player 2 (P1) reduced strategies", p2_reduced)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    print("--- Comparison ---\n")
    print(
        "Player 1 owns two information sets (p1_root and p1_again).\n"
        "If they choose Accept at p1_root, p1_again is never reached.\n"
        "Full strategies still require a choice there; reduced do not.\n"
    )
    print(f"  Player 1 full strategies:    {len(p1_full)}")
    print(f"  Player 1 reduced strategies: {len(p1_reduced)}")
    print()
    print(
        "Player 2 owns only p2_node, which is reached only when Player 1\n"
        "chooses Threaten. Both strategy types agree for Player 2.\n"
    )
    print(f"  Player 2 full strategies:    {len(p2_full)}")
    print(f"  Player 2 reduced strategies: {len(p2_reduced)}")


if __name__ == "__main__":
    main()
