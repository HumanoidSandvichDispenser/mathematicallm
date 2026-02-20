# TODO: Consider using a more structured representation for decisions, such as
# a named tuple or a dataclass, to improve readability and maintainability.
Decision = tuple

class Strategy:
    """
    Represents the strategy of a player in an extensive-form game. A strategy
    is a mapping from information sets to actions, specifying the action that
    the player will take at each information set. In an extensive-form game, a
    player's strategy must specify an action for every information set that the
    player may encounter during the game, even if some of those information
    sets are not reachable under the strategy itself.
    """

    def __init__(self, decisions: dict):
        """
        Initialize a Strategy object.

        Args:
            decisions: A dictionary mapping information sets to actions. Each key
                is an information set (which can be represented as a unique
                identifier), and each value is the action that the player will
                take at that information set.
        """
        self.decisions = decisions

    def __repr__(self):
        # the string representation of a strategy be in the format of:
        # Strategy(info_set_1: action_1, info_set_2: action_2, ...)
        return f"Strategy()"
