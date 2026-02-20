#!/usr/bin/env python3
"""Test script for MCP server - simulates tool calls."""

import sys
import json

# Import the server module to access tools directly
from server import (
    create_game,
    add_decision_node,
    add_terminal_node,
    add_chance_node,
    solve_game,
    show_tree,
    get_game_state,
    list_games,
)


def test_simple_game():
    """Test creating and solving a simple 2-player game."""
    print("=" * 60)
    print("TEST: Simple 2-player game (Prisoner's Dilemma variant)")
    print("=" * 60)
    
    # Create game
    result = create_game(name="prisoners_dilemma", num_players=2)
    print(f"\n1. {result}")
    game_id = "game_1"  # Based on _counter starting at 0
    
    # Add Player 2's decision nodes
    result = add_decision_node(game_id, "cooperate", "root", player=1)
    print(f"2. {result}")
    
    result = add_decision_node(game_id, "defect", "root", player=1)
    print(f"3. {result}")
    
    # Add terminal nodes under cooperate
    result = add_terminal_node(game_id, "cc", "cooperate", payoffs="3,3")
    print(f"4. {result}")
    
    result = add_terminal_node(game_id, "cd", "cooperate", payoffs="0,5")
    print(f"5. {result}")
    
    # Add terminal nodes under defect
    result = add_terminal_node(game_id, "dc", "defect", payoffs="5,0")
    print(f"6. {result}")
    
    result = add_terminal_node(game_id, "dd", "defect", payoffs="1,1")
    print(f"7. {result}")
    
    # Show tree structure
    print("\n" + "=" * 60)
    result = show_tree(game_id)
    print(result)
    
    # Solve game
    print("\n" + "=" * 60)
    result = solve_game(game_id)
    print(result)
    
    print("\n" + "=" * 60)
    print("TEST PASSED: Simple game")
    print("=" * 60)


def test_chance_node_game():
    """Test creating a game with chance nodes."""
    print("\n\n" + "=" * 60)
    print("TEST: Game with chance nodes")
    print("=" * 60)
    
    # Create game
    result = create_game(num_players=2)
    print(f"\n1. {result}")
    game_id = "game_2"
    
    # Add chance node
    result = add_chance_node(game_id, "nature", "root")
    print(f"2. {result}")
    
    # Add branches with probabilities
    result = add_decision_node(game_id, "lucky", "nature", player=0, probability="2/3")
    print(f"3. {result}")
    
    result = add_decision_node(game_id, "unlucky", "nature", player=0, probability="1/3")
    print(f"4. {result}")
    
    # Add terminal nodes
    result = add_terminal_node(game_id, "l_win", "lucky", payoffs="10,0")
    print(f"5. {result}")
    
    result = add_terminal_node(game_id, "l_lose", "lucky", payoffs="0,10")
    print(f"6. {result}")
    
    result = add_terminal_node(game_id, "u_win", "unlucky", payoffs="5,0")
    print(f"7. {result}")
    
    result = add_terminal_node(game_id, "u_lose", "unlucky", payoffs="0,5")
    print(f"8. {result}")
    
    # Show tree
    print("\n" + "=" * 60)
    result = show_tree(game_id)
    print(result)
    
    # Solve game
    print("\n" + "=" * 60)
    result = solve_game(game_id)
    print(result)
    
    print("\n" + "=" * 60)
    print("TEST PASSED: Chance node game")
    print("=" * 60)


def test_serialization():
    """Test game state serialization."""
    print("\n\n" + "=" * 60)
    print("TEST: Game serialization")
    print("=" * 60)
    
    game_id = "game_1"  # From first test
    
    # Get state without solution
    print("\nGame state (without solution):")
    result = get_game_state(game_id, include_solution=False)
    print(result)
    
    # Get state with solution
    print("\nGame state (with solution):")
    result = get_game_state(game_id, include_solution=True)
    print(result)
    
    print("\n" + "=" * 60)
    print("TEST PASSED: Serialization")
    print("=" * 60)


def test_list_games():
    """Test listing all games."""
    print("\n\n" + "=" * 60)
    print("TEST: List games")
    print("=" * 60)
    
    result = list_games()
    print(result)
    
    print("\n" + "=" * 60)
    print("TEST PASSED: List games")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_simple_game()
        test_chance_node_game()
        test_serialization()
        test_list_games()
        
        print("\n\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n\nTEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
