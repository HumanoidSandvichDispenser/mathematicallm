# zermelo — Extensive-Form Game Tree Implementation

## Architecture: Composable DataClass Mixins

The new design uses **composable dataclass mixins** instead of discriminated unions, providing cleaner type checking via `isinstance` and shared state where appropriate.

## Core Design Principles

1. **NodeData is composable via dataclass mixins**
2. **Node type determined by `isinstance` checks** (not discriminated unions)
3. **All numeric values use `sympy.Expr`** for exact symbolic arithmetic
4. **Probabilities do not need to sum to 1**
5. **Edges represented by child nodes** — `probability` on a node refers to the edge leading INTO it
6. **Automatic sympification** via `__post_init__` for probabilities and payoffs

## Type Hierarchy

```
NodeData (base)
├── probability: Expr | None

BIValue (mixin)
├── bi_value: tuple[Expr, ...] | None  (mutable, for BI algorithm)

DecisionNodeData(BIValue, NodeData)
├── player: int
├── bi_value (from BIValue)
├── probability (from NodeData)

ChanceNodeData(BIValue, NodeData)
├── bi_value (from BIValue)
├── probability (from NodeData)

TerminalNodeData(NodeData)
├── payoffs: tuple[Expr, ...]
├── probability (from NodeData)
├── bi_value → @property that returns payoffs
```

## Key Design Decisions

### Why BIValue Mixin is Legitimate

`BIValue` provides **shared mutable state** for `DecisionNodeData` and `ChanceNodeData` only:
- These nodes need their `bi_value` mutated during backward induction
- `TerminalNodeData` stays out — its `bi_value` is a `@property` derived from immutable `payoffs`
- Clean separation: mutable BI state vs. immutable terminal payoffs

### Automatic Sympification

Both `NodeData` and `TerminalNodeData` use `__post_init__` to sympify inputs:
```python
# Pass any type — int, float, str, or Expr
DecisionNodeData(player=0, probability=0.5)      # → probability becomes Rational(1, 2)
TerminalNodeData(payoffs=(1, "3/2", 1.5))        # → all sympified
```

### No Probability Validation

Probabilities on children of chance nodes **do not need to sum to 1**:
- Allows un-normalized distributions
- Useful for intermediate computations
- Validation can be added as a separate `validate()` method if needed

## isinstance Dispatch Pattern

```python
# Type checking
isinstance(node.data, DecisionNodeData)   # decision node
isinstance(node.data, ChanceNodeData)     # chance node  
isinstance(node.data, TerminalNodeData)   # terminal node
isinstance(node.data, BIValue)            # has bi_value field (decision or chance)

# Edge probability check
node.data.probability is not None         # child of a chance node
```

## Usage Examples

### Building a Game Tree

```python
from zermelo.extensive import GameTree, DecisionNodeData, TerminalNodeData
import sympy as sp

tree = GameTree()

# Root: Player 0 decides
tree.create_node("Root", "root", data=DecisionNodeData(player=0))

# Outcomes
tree.create_node(
    "Left", "left", parent="root",
    data=TerminalNodeData(payoffs=(1, 0))
)
tree.create_node(
    "Right", "right", parent="root",
    data=TerminalNodeData(payoffs=(0, 1))
)
```

### Solving with Backward Induction

```python
# Solve (mutates bi_value fields)
result = tree.backward_induction(mutate=True)
# result = (1, 0)  — Player 0 chooses Left

# Access BI value at root
root = tree.get_node("root")
print(root.data.bi_value)  # (1, 0)
```

### Chance Nodes

```python
tree.create_node("Nature", "root", data=ChanceNodeData())

# Children have probabilities on their edges
tree.create_node(
    "Heads", "heads", parent="root",
    data=TerminalNodeData(payoffs=(10,), probability=sp.Rational(1, 2))
)
tree.create_node(
    "Tails", "tails", parent="root",
    data=TerminalNodeData(payoffs=(5,), probability=sp.Rational(1, 2))
)

result = tree.backward_induction(mutate=True)
# result = (15/2,)  — expected value
```

### Symbolic Payoffs

```python
x = sp.Symbol('x', positive=True)

tree.create_node("Root", "root", data=DecisionNodeData(player=0))
tree.create_node("A", "a", parent="root", data=TerminalNodeData(payoffs=(x,)))
tree.create_node("B", "b", parent="root", data=TerminalNodeData(payoffs=(2*x,)))

result = tree.backward_induction(mutate=True)
# result = (2*x,)  — symbolic maximum
```

## Implementation

### GameTree.backward_induction()

```python
def backward_induction(self, node_id=None, mutate=False) -> tuple[Expr, ...]:
    """
    Computes backward induction solution.
    
    - Terminal nodes: return payoffs
    - Chance nodes: probability-weighted sum of children
    - Decision nodes: maximize current player's payoff
    
    Mutates bi_value field on all BIValue nodes.
    
    Returns:
        Tuple of expressions (one per player)
    """
```

**Algorithm**:
1. Post-order recursion (children first)
2. Terminal: return `node.data.payoffs`
3. Chance: compute weighted sum using `child.data.probability`
4. Decision: select child maximizing `payoffs[player]`
5. Store result in `node.data.bi_value` (for decision/chance nodes)

### Serialization

```python
tree.to_dict()       # → JSON-compatible dict
GameTree.from_dict() # ← reconstruct from dict
```

- Uses `sympy.srepr()` / `sympy.sympify()` for expressions
- Preserves `bi_value` fields (useful for caching solutions)
- Type field: `"decision"`, `"chance"`, or `"terminal"`

## Testing

8 comprehensive tests (all passing ✅):
1. Two-player sequential game (prisoners' dilemma logic)
2. Chance nodes with probability weighting  
3. Symbolic payoffs and symbolic reasoning
4. Serialization round-trip (structure preservation)
5. Serialization with BI values (cached solutions)
6. Node type checks (isinstance patterns)
7. Probability automatic sympification
8. Payoffs automatic sympification

Run: `uv run pytest modules/zermelo/tests/test_game_tree.py -v`

## Type Safety

Zero pyright errors:
```bash
$ pyright modules/zermelo/src/zermelo/extensive/
0 errors, 0 warnings, 0 informations
```

## Files

**Core implementation**:
- `node_data.py` — NodeData, BIValue, DecisionNodeData, ChanceNodeData, TerminalNodeData
- `game_node.py` — GameNode with convenience properties
- `game_tree.py` — GameTree with backward_induction and serialization
- `__init__.py` — Public API exports

**Tests & Examples**:
- `tests/test_game_tree.py` — 8 comprehensive tests
- `examples/game_tree_usage.py` — Working examples

## Comparison to Previous Design

**Old (discriminated union)**:
- Used `kind: Literal["decision" | "chance" | "terminal"]`
- Pattern matching on `node.data.kind`
- Separate fields for each variant
- More verbose serialization

**New (composable mixins)**:
- Uses `isinstance(node.data, DecisionNodeData)`
- Cleaner inheritance hierarchy
- Shared mixin (BIValue) for mutable state
- Property-based `bi_value` for terminals
- More Pythonic and type-safe

## Future Extensions

### Information Sets (Imperfect Information)
Add an optional `info_set: str | None` field to `DecisionNodeData`:
```python
@dataclass
class DecisionNodeData(BIValue, NodeData):
    player: int = 0
    info_set: str | None = None  # Add this
```

### Strategies
Add a strategy representation:
```python
@dataclass
class Strategy:
    player: int
    actions: dict[str, str]  # node_id → action_id
```

### Nash Equilibrium
For simultaneous-move subgames or normal-form conversions.

## Next Steps

If you want to integrate with MCP servers:
1. Update `servers/zermelo-mcp/server.py` with tools like:
   - `create_game() → game_id`
   - `add_decision_node(game_id, parent_id, player)`
   - `add_terminal_node(game_id, parent_id, payoffs)`
   - `set_probability(game_id, node_id, prob)`
   - `solve(game_id) → solution`
   - `get_bi_value(game_id, node_id) → payoffs`
