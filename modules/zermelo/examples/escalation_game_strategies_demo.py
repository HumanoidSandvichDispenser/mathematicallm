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
from zermelo.services.strategy import (
    find_full_pure_strategies,
    find_reduced_pure_strategies,
)
from zermelo.services.strategic_form import extensive_to_strategic
from zermelo.services.nash import find_pure_nash_equilibria


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

    # ------------------------------------------------------------------
    # Strategic form
    # ------------------------------------------------------------------
    print("\n--- Strategic Form ---\n")
    print("Converting extensive form to strategic (normal) form...\n")

    strategies, payoffs = extensive_to_strategic(game)

    print(f"Player strategies in strategic form:")
    print(f"  Player 0: {len(strategies[0])} strategies")
    for i, s in enumerate(strategies[0]):
        print(f"    {i}: {s}")
    print(f"  Player 1: {len(strategies[1])} strategies")
    for i, s in enumerate(strategies[1]):
        print(f"    {i}: {s}")

    print(f"\nPayoff tensor shape: {payoffs.shape}")
    print("\nPayoff matrix (rows = P0 strategies, cols = P1 strategies):")
    print("           P1→")
    print("        ", end="")
    for j in range(len(strategies[1])):
        print(f"   {j:>4}", end="")
    print()
    for i in range(len(strategies[0])):
        print(f"  P0 {i}:", end="")
        for j in range(len(strategies[1])):
            p0_payoff = payoffs[i, j, 0]
            p1_payoff = payoffs[i, j, 1]
            print(f" ({str(p0_payoff):>2},{str(p1_payoff):>2})", end="")
        print()

    # ------------------------------------------------------------------
    # Pure Strategy Nash Equilibrium (PSNE)
    # ------------------------------------------------------------------
    print("\n--- Pure Strategy Nash Equilibria ---\n")

    equilibria = find_pure_nash_equilibria(strategies, payoffs)

    print(f"Found {len(equilibria)} pure strategy Nash equilibrium(ies):\n")
    for profile_idx, payoff_tuple in equilibria:
        p0_idx, p1_idx = profile_idx
        print(f"  Profile: P0={p0_idx}, P1={p1_idx}")
        print(f"    Strategies: {strategies[0][p0_idx]}, {strategies[1][p1_idx]}")
        print(f"    Payoffs: {payoff_tuple}")
        print()

    # ------------------------------------------------------------------
    # Subgame Perfect Nash Equilibrium (SPNE)
    # ------------------------------------------------------------------
    print("--- Subgame Perfect Nash Equilibrium (SPNE) ---\n")

    game_spne = build_escalation_game()
    game_spne.backward_induction(mutate=True)
    spne_equilibria = game_spne.get_all_equilibria()

    print(f"Found {len(spne_equilibria)} subgame perfect Nash equilibrium(s):\n")
    for eq in spne_equilibria:
        print(f"  Payoffs: {eq.payoffs}")
        print(f"  Path: {eq.actions}")
        print()

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    print("--- PSNE vs SPNE Comparison ---\n")
    print("The PSNE found:")
    print("  (P0=0, P1=1): Accept+GiveUp vs Escalate -> (0, 0)")
    print("  (P0=2, P1=1): Accept+War    vs Escalate -> (0, 0)")
    print()
    print(
        "Both PSNE have P0 choosing Accept, which ends the game immediately.\n"
        "P1's strategy specifies what to do at p2_node, but that's unreachable.\n"
        "So P1's payoff is (0,0) regardless of whether they 'choose' Concede\n"
        "or Escalate in their strategy.\n"
    )
    print(
        "This is why BOTH (Accept+GiveUp, Escalate) and (Accept+War, Escalate)\n"
        "are Nash equilibria - P1's action at p2_node doesn't affect the outcome!\n"
    )
    print("The SPNE from backward induction:")
    print(f"  {spne_equilibria[0].actions}")
    print(f"  Payoffs: {spne_equilibria[0].payoffs}")
    print()
    print(
        "SPNE is a REFINEMENT of Nash equilibrium. It eliminates equilibria\n"
        "that rely on non-credible threats by requiring rationality at every\n"
        "subgame. Since P0 choosing Accept ends the game before P1 moves,\n"
        "the SPNE path is just {p1_root: accept}.\n"
    )


if __name__ == "__main__":
    main()
