from typing import Generator
from abc import ABC, abstractmethod
from zermelo.trees.strategy import Strategy
from sympy import Expr, Rational


class Node(ABC):
    def __init__(self, label: str):
        self.label: str = label
        self.parent: "Node | None" = None
        self.children: dict[str, "Node"] = {}

    @property
    def is_perfect_information(self) -> bool:
        """
        Determine if the game tree rooted at this node is a perfect information
        game.
        """
        return all(c.is_perfect_information for c in self.children.values())

    def get_players(self) -> set[str]:
        players = set()
        for node in self.traverse_preorder():
            if isinstance(node, DecisionNode):
                players.add(node.player)
        return players

    def add_child(self, child: "Node", action: str):
        child.parent = self
        self.children[action] = child

    def rename_action(self, old_action: str, new_action: str):
        if old_action not in self.children:
            raise ValueError(f"Action '{old_action}' not found among children.")
        if new_action in self.children:
            raise ValueError(f"Action '{new_action}' already exists among children.")
        self.children[new_action] = self.children.pop(old_action)

    def traverse_postorder(self) -> Generator["Node", None, None]:
        for child in self.children.values():
            yield from child.traverse_postorder()
        yield self

    def traverse_preorder(self) -> Generator["Node", None, None]:
        yield self
        for child in self.children.values():
            yield from child.traverse_preorder()

    def visualize(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        result = prefix + connector + self._visualize_label() + "\n"

        children = list(self.children.items())
        for i, (action, child) in enumerate(children):
            is_last_child = i == len(children) - 1
            extension = prefix + ("    " if is_last else "│   ")
            result += child.visualize(extension, is_last_child)

        return result

    def _visualize_label(self) -> str:
        return self.label

    @abstractmethod
    def apply_strategy(self, strategies: dict[str, Strategy]) -> tuple[Expr, ...]:
        """
        Applies the given strategies from each player to the game tree,
        returning a payoff vector for the resulting game. If the game is not
        fully specified by the strategies, raises a ValueError. If the game
        contains chance nodes, the resulting payoff vector is a weighted average
        of the payoffs at the terminal nodes.

        Args:
            strategies: A dict mapping player (str) to their Strategy
        """
        pass


class DecisionNode(Node):
    def __init__(
        self, label: str, player: str, information_set: "InformationSet | None" = None
    ):
        super().__init__(label)
        self.player: str = player
        self.information_set = information_set or InformationSet(label, player)
        self.information_set.add_node(self)

    @property
    def actions(self):
        return list(self.children.keys())

    @property
    def is_perfect_information(self) -> bool:
        return len(self.information_set.nodes) == 1 and super().is_perfect_information

    def _visualize_label(self) -> str:
        return f"{self.label} [P{self.player} @{self.information_set.label}]"

    def apply_strategy(self, strategies: dict[str, Strategy]) -> tuple[Expr, ...]:
        strategy = strategies[self.player]

        if self.information_set.label not in strategy:
            raise ValueError(
                f"Strategy for player {self.player} does not specify an action for information set '{self.information_set.label}'. Strategy: {strategy._decisions}"
            )

        action = strategy[self.information_set.label]
        if action not in self.children:
            raise ValueError(
                f"Action '{action}' specified by strategy for player {self.player} is not available at node '{self.label}'."
            )

        return self.children[action].apply_strategy(strategies)


class ChanceNode(Node):
    def __init__(self, label: str, probability_map: dict[str, Expr]):
        super().__init__(label)
        self.probability_map: dict[str, Expr] = probability_map

    def set_probability(self, action: str, probability: Expr):
        if action not in self.children:
            raise ValueError(f"Action '{action}' not found among children.")
        self.probability_map[action] = probability

    def add_child(self, child: "Node", action: str, probability: Expr | None = None):
        super().add_child(child, action)
        if action not in self.probability_map:
            self.probability_map[action] = (
                probability if probability is not None else Rational(0)
            )

    def rename_action(self, old_action: str, new_action: str):
        super().rename_action(old_action, new_action)
        self.probability_map[new_action] = self.probability_map.pop(old_action)

    def apply_strategy(self, strategies: dict[str, Strategy]) -> tuple[Expr, ...]:
        total_payoff: tuple[Expr, ...] = tuple(
            Rational(0) for _ in range(100)
        )  # placeholder, will be replaced
        first = True
        for action, child in self.children.items():
            probability = self.probability_map.get(action)
            if probability is None:
                raise ValueError(
                    f"Probability for action '{action}' not specified in chance node '{self.label}'."
                )
            child_payoff = child.apply_strategy(strategies)
            if first:
                total_payoff = tuple(
                    probability * child_payoff[i] for i in range(len(child_payoff))
                )
                first = False
            else:
                total_payoff = tuple(
                    total_payoff[i] + probability * child_payoff[i]
                    for i in range(len(child_payoff))
                )
        return total_payoff


class TerminalNode(Node):
    def __init__(self, label: str, payoffs: tuple[Expr, ...]):
        super().__init__(label)
        self.payoffs: tuple[Expr, ...] = payoffs

    def add_child(self, **_):
        raise ValueError("Terminal nodes cannot have children.")

    def _visualize_label(self) -> str:
        return f"{self.label} → {self.payoffs}"

    def apply_strategy(self, strategies: dict[str, Strategy]) -> tuple[Expr, ...]:
        return self.payoffs


class InformationSet:
    def __init__(self, label: str, player: str, nodes: set[DecisionNode] | None = None):
        self.player: str = player
        self.label: str = label
        self.nodes: set[DecisionNode] = nodes or set()

    def add_node(self, node: DecisionNode):
        for existing_node in self.nodes:
            if existing_node.actions != node.actions:
                raise ValueError(
                    "All nodes in an information set must have the same actions."
                )
        self.nodes.add(node)

    def remove_node(self, node: DecisionNode):
        self.nodes.remove(node)

    @property
    def actions(self):
        if not self.nodes:
            return []
        for node in self.nodes:
            if node.actions:
                return node.actions
        return []
