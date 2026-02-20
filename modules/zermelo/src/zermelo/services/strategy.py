from itertools import product
from zermelo.extensive.game_node import GameNode
from zermelo.extensive.game_tree import GameTree
from zermelo.extensive.strategy import Strategy


def find_full_pure_strategies(game: GameTree, player: int) -> set[Strategy]:
    """
    Find all full pure strategies for the given extensive-form game. A full
    pure strategy is a strategy that specifies an action for every information
    set in the game, regardless of whether that information set is reachable
    under the strategy or not.

    Args:
        game: An extensive-form game represented as a GameTree.
        player: The player for whom to find the full pure strategies.

    Returns:
        A set of Strategy objects, each representing a full pure strategy for the
        game.
    """
    info_sets = game.get_information_sets(player)

    if not info_sets:
        return {Strategy({})}

    info_set_actions = []
    for info_set_id in sorted(info_sets):
        nodes = game.get_nodes_in_information_set(info_set_id)
        if not nodes:
            continue
        node = nodes[0]
        if node.data.actions:
            action_ids = node.data.actions
        else:
            children = game.children(node.identifier)
            action_ids = [child.identifier for child in children]
        if action_ids:
            info_set_actions.append(action_ids)

    if not info_set_actions:
        return {Strategy({})}

    strategies: set[Strategy] = set()
    for action_combo in product(*info_set_actions):
        decisions = {}
        for info_set_id, action_id in zip(sorted(info_sets), action_combo):
            decisions[info_set_id] = action_id
        strategies.add(Strategy(decisions))

    return strategies


def find_reduced_pure_strategies(game: GameTree, player: int) -> set[Strategy]:
    """
    Find all reduced pure strategies for the given extensive-form game. A
    reduced pure strategy is a strategy that specifies an action for every
    information set that is reachable under the strategy. All decisions that
    are not reachable because of the player's earlier decisions are not
    included in a reduced pure strategy.

    Args:
        game: An extensive-form game represented as a GameTree.
        player: The player for whom to find the reduced pure strategies.

    Returns:
        A set of Strategy objects, each representing a reduced pure strategy for
        the game.
    """

    def traverse(node_id: str, decisions: dict[str, str]) -> list[dict[str, str]]:
        """
        Recursively walk the tree from node_id, accumulating decisions.

        Returns a list of completed decision dicts (one per reachable leaf
        combination from this node).
        """
        node = game.get_node(node_id)
        assert isinstance(node, GameNode)

        if node.is_terminal:
            return [decisions]

        children = game.children(node_id)

        if node.is_chance:
            # Follow all branches — nature's move, not the player's
            results = []
            for child in children:
                results.extend(traverse(child.identifier, decisions))
            return results

        if node.is_decision:
            if node.data.player != player:
                # Another player's decision — follow all branches
                results = []
                for child in children:
                    results.extend(traverse(child.identifier, decisions))
                return results

            # This player's decision node
            info_set_id = node.data.information_set or node_id
            first_node = game.get_nodes_in_information_set(info_set_id)[0]
            use_action_labels = first_node.data.actions is not None
            if use_action_labels:
                available_actions = list(first_node.data.actions)
                first_children = game.children(first_node.identifier)
            else:
                first_children = game.children(first_node.identifier)
                available_actions = [c.identifier for c in first_children]

            if info_set_id in decisions:
                # Already committed to an action for this info set —
                # follow the previously chosen child that belongs to this node.
                # In imperfect info, info set nodes may have different children,
                # so we use the child at the same index as the chosen action
                # within the first node of the info set.
                chosen_action = decisions[info_set_id]
                if use_action_labels:
                    chosen_index = available_actions.index(chosen_action)
                else:
                    chosen_index = next(
                        i
                        for i, c in enumerate(first_children)
                        if c.identifier == chosen_action
                    )
                this_children = game.children(node_id)
                if chosen_index < len(this_children):
                    return traverse(this_children[chosen_index].identifier, decisions)
                return [decisions]

            # Not yet decided — branch on every action.
            # Use the first node of the info set's children as canonical
            # action IDs so that all nodes in the info set share the same
            # action labels and de-duplicate correctly.
            results = []
            for action_index, action_label in enumerate(available_actions):
                new_decisions = {**decisions, info_set_id: action_label}
                # Follow this node's child at the same index
                if action_index < len(children):
                    results.extend(
                        traverse(children[action_index].identifier, new_decisions)
                    )
            return results

        return [decisions]

    if game.root is None:
        return {Strategy({})}

    all_decision_dicts = traverse(game.root, {})

    # Deduplicate
    unique_dicts: list[dict[str, str]] = []
    for d in all_decision_dicts:
        if d not in unique_dicts:
            unique_dicts.append(d)

    # Remove any dict that is a strict subset of another dict in the results.
    # These arise from paths where the player never reaches some of their info
    # sets (e.g. an opponent's move cuts off the subtree), producing a partial
    # decision record that is already subsumed by a more-complete one.
    def is_strict_subset(a: dict, b: dict) -> bool:
        return len(a) < len(b) and all(b.get(k) == v for k, v in a.items())

    maximal = [
        d
        for d in unique_dicts
        if not any(is_strict_subset(d, other) for other in unique_dicts)
    ]

    return {Strategy(d) for d in maximal}
