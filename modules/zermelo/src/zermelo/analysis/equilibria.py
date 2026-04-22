from itertools import product, combinations
from dataclasses import dataclass
from fractions import Fraction
from math import sqrt, cos, sin, pi

from sympy import Rational, Matrix, Expr
from sympy.tensor.array.ndim_array import NDimArray
from zermelo.trees.node import Node, DecisionNode, TerminalNode
from zermelo.trees.mixed_strategy import MixedStrategy
from zermelo.trees.strategy import Strategy


@dataclass(frozen=True)
class PureMMSolution:
    """Pure maximin solution for a single player."""

    strategies: list[Strategy]
    value: Expr


def find_pure_nash_equilibria(
    profiles: list[list[Strategy]],
    array: NDimArray,
) -> list[tuple[Strategy, ...]]:
    """
    Finds all pure Nash equilibria using a payoff array.

    Args:
        profiles: A list of strategy lists, one per player.
        array: An NDimArray of payoffs with shape (n_p0, n_p1, ..., n_pk, k)
            for k players, where n_pi is the number of strategies for player i.
            The last dimension corresponds to the payoff scalar for each
            player; thus if you were to index the array (n_p0, n_p1, ...,
            n_pk), you would get the payoff vector of dimensions k for that
            strategy profile.

    Returns:
        A list of pure Nash equilibria, where each equilibrium is a tuple of
        Strategy objects (one per player).
    """
    num_players = len(profiles)
    strategy_counts = [len(p) for p in profiles]
    num_strat_dims = len(strategy_counts)

    equilibria = []

    for index in product(*[range(c) for c in strategy_counts]):
        payoff = array[index]
        is_equilibrium = True

        for player in range(num_players):
            current_payoff = payoff[player]

            player_deviated = False
            for deviation in range(strategy_counts[player]):
                if deviation == index[player]:
                    continue

                deviated_index = list(index)
                deviated_index[player] = deviation
                deviated_payoff = array[tuple(deviated_index)][player]

                if deviated_payoff > current_payoff:
                    player_deviated = True
                    break

            if player_deviated:
                is_equilibrium = False
                break

        if is_equilibrium:
            equilibrium_strategies = tuple(
                profiles[p][index[p]] for p in range(num_players)
            )
            equilibria.append(equilibrium_strategies)

    return equilibria


def backwards_induction(root: "Node") -> list[Strategy]:
    """
    Performs backwards induction on a game tree to find the subgame perfect
    equilibrium strategy profile. This function assumes that all players are
    rational and will choose their best response at each decision node, given
    the strategies of the other players.
    """
    raise NotImplementedError()


def find_mixed_nash_equilibria(
    profiles: list[list[Strategy]],
    array: NDimArray,
) -> list[tuple[MixedStrategy, MixedStrategy]]:
    """
    Find all mixed Nash Equilibria via support enumeration.

    Args:
        profiles: list of strategy lists, one per player. profiles[0] = row
                  player strategies, profiles[1] = col player strategies.
        array:    NDimArray of shape (m, n, 2) — array[i, j] = (v_row, v_col)

    Returns:
        A list of equilibria. Each equilibrium is a tuple of two MixedStrategy
        objects (row player, col player).
    """
    if len(profiles) != 2:
        raise ValueError("This algorithm only supports two-player games.")

    m = len(profiles[0])
    n = len(profiles[1])

    A = Matrix(m, n, lambda i, j: array[i, j][0])
    B = Matrix(m, n, lambda i, j: array[i, j][1])

    equilibria = []
    seen = set()

    for k in range(1, min(m, n) + 1):
        for I in combinations(range(m), k):
            for J in combinations(range(n), k):
                M_q: Matrix = Matrix.zeros(k, k)
                rhs_q = Matrix.zeros(k, 1)

                for row_idx in range(k - 1):
                    i0 = I[0]
                    i1 = I[row_idx + 1]
                    for col_idx, j in enumerate(J):
                        M_q[row_idx, col_idx] = A[i0, j] - A[i1, j]  # type: ignore
                    rhs_q[row_idx] = 0

                for col_idx in range(k):
                    M_q[k - 1, col_idx] = 1
                rhs_q[k - 1] = 1

                if M_q.det() == 0:
                    continue

                q_vec = M_q.solve(rhs_q)

                if any(q_vec[t] < 0 for t in range(k)):
                    continue

                q_full = [Rational(0)] * n
                for t, j in enumerate(J):
                    q_full[j] = q_vec[t]

                v1 = sum(A[I[0], j] * q_full[j] for j in range(n))

                M_p = Matrix.zeros(k, k)
                rhs_p = Matrix.zeros(k, 1)

                for col_idx in range(k - 1):
                    j0 = J[0]
                    j1 = J[col_idx + 1]
                    for row_idx, i in enumerate(I):
                        M_p[col_idx, row_idx] = B[i, j0] - B[i, j1]  # type: ignore
                    rhs_p[col_idx] = 0

                for row_idx in range(k):
                    M_p[k - 1, row_idx] = 1
                rhs_p[k - 1] = 1

                if M_p.det() == 0:
                    continue

                p_vec = M_p.solve(rhs_p)

                if any(p_vec[t] < 0 for t in range(k)):
                    continue

                p_full = [Rational(0)] * m
                for t, i in enumerate(I):
                    p_full[i] = p_vec[t]

                v2 = sum(p_full[i] * B[i, J[0]] for i in range(m))

                row_ok = all(
                    sum(A[i, j] * q_full[j] for j in range(n)) <= v1
                    for i in range(m)
                    if i not in I
                )

                col_ok = all(
                    sum(p_full[i] * B[i, j] for i in range(m)) <= v2
                    for j in range(n)
                    if j not in J
                )

                if not (row_ok and col_ok):
                    continue

                key = (tuple(p_full), tuple(q_full))
                if key in seen:
                    continue
                seen.add(key)

                row_mix = MixedStrategy(
                    {profiles[0][i]: p_full[i] for i in range(m) if p_full[i] > 0}
                )
                col_mix = MixedStrategy(
                    {profiles[1][j]: q_full[j] for j in range(n) if q_full[j] > 0}
                )

                equilibria.append((row_mix, col_mix))

    return equilibria


def find_pure_mm_solutions(
    profiles: list[list[Strategy]],
    array: NDimArray,
) -> list[PureMMSolution]:
    """
    Find pure maximin solutions for each player in a normal-form game.

    For player i, the algorithm computes the worst-case payoff for each pure
    strategy (min over all opponent strategy profiles), then selects all
    strategies that maximize this minimum.

    Args:
        profiles: A list of strategy lists, one per player.
        array: An NDimArray of payoffs with shape (n_p0, n_p1, ..., n_pk, k)
            for k players, where n_pi is the number of strategies for player i.

    Returns:
        A list of PureMMSolution objects, one per player, in player-index order.
        Each object contains the player's maximin-optimal pure strategies and
        the shared maximin value among those strategies.
    """
    num_players = len(profiles)
    strategy_counts = [len(p) for p in profiles]

    expected_shape = tuple(strategy_counts) + (num_players,)
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"Payoff array shape {tuple(array.shape)} does not match expected "
            f"shape {expected_shape} for the provided strategy profiles."
        )

    solutions: list[PureMMSolution] = []

    for player in range(num_players):
        opponent_ranges = [
            range(strategy_counts[i]) for i in range(num_players) if i != player
        ]

        worst_case_payoffs: list[Expr] = []
        for own_strategy_idx in range(strategy_counts[player]):
            worst_case_payoff = None

            for opponent_index in product(*opponent_ranges):
                profile_index = []
                opponent_cursor = 0
                for i in range(num_players):
                    if i == player:
                        profile_index.append(own_strategy_idx)
                    else:
                        profile_index.append(opponent_index[opponent_cursor])
                        opponent_cursor += 1

                payoff = array[tuple(profile_index)][player]
                if worst_case_payoff is None or payoff < worst_case_payoff:
                    worst_case_payoff = payoff

            if worst_case_payoff is None:
                raise ValueError("Each player must have at least one strategy.")

            worst_case_payoffs.append(worst_case_payoff)

        maximin_value = max(worst_case_payoffs)
        maximin_strategies = [
            profiles[player][i]
            for i, payoff in enumerate(worst_case_payoffs)
            if payoff == maximin_value
        ]

        solutions.append(
            PureMMSolution(
                strategies=maximin_strategies,
                value=maximin_value,
            )
        )

    return solutions


def _coerce_3x3_matrix(name: str, matrix_like) -> list[list[float]]:
    matrix = [list(row) for row in matrix_like]
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError(f"{name} must be a 3x3 payoff matrix")
    return [[float(v) for v in row] for row in matrix]


def _coerce_probability_vector(name: str, vector_like) -> tuple[float, float, float]:
    values = [float(v) for v in vector_like]
    if len(values) != 3:
        raise ValueError(f"{name} must have exactly 3 coordinates")
    if any(v < -1e-9 for v in values):
        raise ValueError(f"{name} has negative coordinates")
    total = sum(values)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to 1")
    normalized = [max(0.0, min(1.0, v)) for v in values]
    return (normalized[0], normalized[1], normalized[2])


def _triangle_vertices(
    cx: float, cy: float, radius: float
) -> list[tuple[float, float]]:
    return [
        (cx, cy - radius),
        (cx - radius * sqrt(3) / 2.0, cy + radius / 2.0),
        (cx + radius * sqrt(3) / 2.0, cy + radius / 2.0),
    ]


def _barycentric_to_cartesian(
    bary: tuple[float, float, float],
    vertices: list[tuple[float, float]],
) -> tuple[float, float]:
    x = bary[0] * vertices[0][0] + bary[1] * vertices[1][0] + bary[2] * vertices[2][0]
    y = bary[0] * vertices[0][1] + bary[1] * vertices[1][1] + bary[2] * vertices[2][1]
    return x, y


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _indifference_anchors(
    d: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    eps = 1e-9
    candidates: list[tuple[float, float, float]] = []

    if abs(d[1] - d[2]) > eps:
        q1 = -d[2] / (d[1] - d[2])
        if -eps <= q1 <= 1 + eps:
            q1 = max(0.0, min(1.0, q1))
            candidates.append((0.0, q1, 1.0 - q1))

    if abs(d[0] - d[2]) > eps:
        q0 = -d[2] / (d[0] - d[2])
        if -eps <= q0 <= 1 + eps:
            q0 = max(0.0, min(1.0, q0))
            candidates.append((q0, 0.0, 1.0 - q0))

    if abs(d[0] - d[1]) > eps:
        q0 = -d[1] / (d[0] - d[1])
        if -eps <= q0 <= 1 + eps:
            q0 = max(0.0, min(1.0, q0))
            candidates.append((q0, 1.0 - q0, 0.0))

    unique: list[tuple[float, float, float]] = []
    seen = set()
    for c in candidates:
        key = tuple(round(v, 9) for v in c)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    return unique


def _fraction_label(values: tuple[float, float, float]) -> str:
    parts = []
    for value in values:
        frac = Fraction(value).limit_denominator(24)
        if frac.denominator == 1:
            parts.append(str(frac.numerator))
        else:
            parts.append(f"{frac.numerator}/{frac.denominator}")
    return f"({', '.join(parts)})"


def _coerce_strategy_labels(name: str, labels, defaults: list[str]) -> list[str]:
    if labels is None:
        return defaults
    items = [str(label) for label in labels]
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly 3 labels")
    return items


def _spread_overlapping_points(
    points: list[tuple[float, float]],
    *,
    key_precision: int = 3,
    spread_radius: float = 7.0,
) -> tuple[list[tuple[float, float]], list[bool]]:
    displayed = list(points)
    shifted = [False] * len(points)

    groups: dict[tuple[float, float], list[int]] = {}
    for idx, (x, y) in enumerate(points):
        key = (round(x, key_precision), round(y, key_precision))
        groups.setdefault(key, []).append(idx)

    for group in groups.values():
        if len(group) < 2:
            continue
        for slot, point_idx in enumerate(group):
            angle = -pi / 2.0 + (2.0 * pi * slot) / len(group)
            x, y = points[point_idx]
            displayed[point_idx] = (
                x + spread_radius * cos(angle),
                y + spread_radius * sin(angle),
            )
            shifted[point_idx] = True

    return displayed, shifted


def _rounded_key3(
    vec: tuple[float, float, float], precision: int = 6
) -> tuple[float, ...]:
    return tuple(round(v, precision) for v in vec)


def _clip_polygon_by_halfspace(
    polygon: list[tuple[float, float, float]],
    d: tuple[float, float, float],
    *,
    eps: float = 1e-9,
) -> list[tuple[float, float, float]]:
    if not polygon:
        return []

    def inside(point: tuple[float, float, float]) -> bool:
        return _dot3(d, point) >= -eps

    def intersect(
        p1: tuple[float, float, float], p2: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        v1 = _dot3(d, p1)
        v2 = _dot3(d, p2)
        t = v1 / (v1 - v2)
        return (
            p1[0] + t * (p2[0] - p1[0]),
            p1[1] + t * (p2[1] - p1[1]),
            p1[2] + t * (p2[2] - p1[2]),
        )

    output: list[tuple[float, float, float]] = []
    prev = polygon[-1]
    prev_inside = inside(prev)

    for curr in polygon:
        curr_inside = inside(curr)

        if curr_inside:
            if not prev_inside:
                output.append(intersect(prev, curr))
            output.append(curr)
        elif prev_inside:
            output.append(intersect(prev, curr))

        prev = curr
        prev_inside = curr_inside

    return output


def _best_response_region_polygon(
    strategy_idx: int,
    inequality_ds: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    poly: list[tuple[float, float, float]] = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    for other_idx, d in enumerate(inequality_ds):
        if other_idx == strategy_idx:
            continue
        poly = _clip_polygon_by_halfspace(poly, d)
        if len(poly) < 3:
            return []

    return poly


def draw_simplex_diagrams(
    p1_payoffs,
    p2_payoffs,
    equilibria,
    *,
    row_strategy_labels=None,
    col_strategy_labels=None,
    width: int = 980,
    height: int = 460,
    radius: float = 150.0,
) -> str:
    """
    Render a two-simplex SVG diagram for 3x3 two-player normal-form games.

    The left simplex (Q) is Player II's strategy space with indifference lines
    derived from Player I's payoff matrix. The right simplex (P) is Player I's
    strategy space with indifference lines derived from Player II's matrix.

    Args:
        p1_payoffs: 3x3 payoff matrix for Player I.
        p2_payoffs: 3x3 payoff matrix for Player II.
        equilibria: Iterable of equilibria; each equilibrium must be a pair
            ``(p, q)`` where both vectors are length-3 probability vectors.
        row_strategy_labels: Optional labels for Player I strategies.
            Defaults to ["1", "2", "3"].
        col_strategy_labels: Optional labels for Player II strategies.
            Defaults to ["A", "B", "C"].
        width: SVG width in pixels.
        height: SVG height in pixels.
        radius: Radius of each simplex triangle.

    Returns:
        SVG markup string.
    """
    p1 = _coerce_3x3_matrix("p1_payoffs", p1_payoffs)
    p2 = _coerce_3x3_matrix("p2_payoffs", p2_payoffs)
    row_labels = _coerce_strategy_labels(
        "row_strategy_labels", row_strategy_labels, ["1", "2", "3"]
    )
    col_labels = _coerce_strategy_labels(
        "col_strategy_labels", col_strategy_labels, ["A", "B", "C"]
    )

    parsed_equilibria: list[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ] = []
    for idx, equilibrium in enumerate(equilibria, start=1):
        if len(equilibrium) != 2:
            raise ValueError(f"equilibrium #{idx} must contain (p, q)")
        p_vec = _coerce_probability_vector(f"equilibrium #{idx} p", equilibrium[0])
        q_vec = _coerce_probability_vector(f"equilibrium #{idx} q", equilibrium[1])
        parsed_equilibria.append((p_vec, q_vec))

    q_center_x = width * 0.28
    p_center_x = width * 0.72
    center_y = height * 0.53

    q_vertices = _triangle_vertices(q_center_x, center_y, radius)
    p_vertices = _triangle_vertices(p_center_x, center_y, radius)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        "<style>",
        ".title { font: 700 18px sans-serif; fill: #222; }",
        ".subtitle { font: 600 13px sans-serif; fill: #4b5563; }",
        ".axis-label { font: 600 12px sans-serif; fill: #334155; }",
        ".line-label { font: 600 11px sans-serif; fill: #475569; }",
        ".eq-label { font: 500 11px monospace; fill: #1f2937; }",
        ".simplex { fill: #f8fafc; stroke: #0f172a; stroke-width: 1.6; }",
        ".br-region { opacity: 0.5; }",
        ".indiff { stroke: #1d4ed8; stroke-width: 1.3; stroke-dasharray: 6 4; fill: none; }",
        ".eq-component { stroke: #0f766e; stroke-width: 2.2; fill: none; stroke-linecap: round; }",
        ".br-label { font: 500 10px sans-serif; fill: #1e40af; }",
        ".eq-offset { stroke: #94a3b8; stroke-width: 1; }",
        ".endpoint-label { font: 700 10px sans-serif; fill: #0f766e; }",
        ".legend-label { font: 500 11px sans-serif; fill: #1f2937; }",
        "</style>",
    ]

    def draw_simplex(
        title: str,
        subtitle: str,
        vertices: list[tuple[float, float]],
        strategy_labels: list[str],
        region_polygons: list[list[tuple[float, float, float]]],
        region_colors: list[str],
    ) -> None:
        parts.append(
            f'<text class="title" x="{vertices[0][0]:.2f}" y="36" text-anchor="middle">{title}</text>'
        )
        parts.append(
            f'<text class="subtitle" x="{vertices[0][0]:.2f}" y="56" text-anchor="middle">{subtitle}</text>'
        )
        points = " ".join(f"{x:.3f},{y:.3f}" for x, y in vertices)

        for idx, bary_poly in enumerate(region_polygons):
            if len(bary_poly) < 3:
                continue
            cart_poly = [
                _barycentric_to_cartesian(point, vertices) for point in bary_poly
            ]
            poly_points = " ".join(f"{x:.3f},{y:.3f}" for x, y in cart_poly)
            parts.append(
                f'<polygon class="br-region" points="{poly_points}" fill="{region_colors[idx]}"/>'
            )

        parts.append(f'<polygon class="simplex" points="{points}" fill="none"/>')

        offsets = [(0, -10), (-14, 18), (14, 18)]
        anchors = ["middle", "end", "start"]
        for (x, y), label, (dx, dy), anchor in zip(
            vertices, strategy_labels, offsets, anchors
        ):
            parts.append(
                f'<text class="axis-label" x="{x + dx:.2f}" y="{y + dy:.2f}" text-anchor="{anchor}">{label}</text>'
            )

    row_region_colors = ["#fecaca", "#bbf7d0", "#bfdbfe"]
    col_region_colors = ["#fde68a", "#fbcfe8", "#c7d2fe"]

    q_region_polygons = [
        _best_response_region_polygon(
            i,
            [
                (
                    p1[k][0] - p1[i][0],
                    p1[k][1] - p1[i][1],
                    p1[k][2] - p1[i][2],
                )
                for k in range(3)
            ],
        )
        for i in range(3)
    ]

    p_region_polygons = [
        _best_response_region_polygon(
            j,
            [
                (
                    p2[0][k] - p2[0][j],
                    p2[1][k] - p2[1][j],
                    p2[2][k] - p2[2][j],
                )
                for k in range(3)
            ],
        )
        for j in range(3)
    ]

    draw_simplex(
        "Q-simplex",
        "Player II strategy space",
        q_vertices,
        col_labels,
        q_region_polygons,
        row_region_colors,
    )
    draw_simplex(
        "P-simplex",
        "Player I strategy space",
        p_vertices,
        row_labels,
        p_region_polygons,
        col_region_colors,
    )

    def draw_legend(
        x: float,
        y: float,
        title: str,
        labels: list[str],
        colors: list[str],
        prefix: str,
    ) -> None:
        parts.append(
            f'<text class="legend-label" x="{x:.2f}" y="{y:.2f}">{title}</text>'
        )
        for idx, (label, color) in enumerate(zip(labels, colors)):
            y0 = y + 8 + idx * 18
            parts.append(
                f'<rect x="{x:.2f}" y="{y0:.2f}" width="12" height="12" fill="{color}" stroke="#475569" stroke-width="0.5"/>'
            )
            parts.append(
                f'<text class="legend-label" x="{x + 18:.2f}" y="{y0 + 10:.2f}">{prefix} {label}</text>'
            )

    draw_legend(
        q_vertices[1][0] - 8,
        q_vertices[1][1] + 34,
        "Q background",
        row_labels,
        row_region_colors,
        "BR",
    )
    draw_legend(
        p_vertices[1][0] - 8,
        p_vertices[1][1] + 34,
        "P background",
        col_labels,
        col_region_colors,
        "BR",
    )

    def draw_indifference_line(
        vertices: list[tuple[float, float]],
        d: tuple[float, float, float],
        strategy_i: int,
        strategy_k: int,
        strategy_labels: list[str],
        payoff_symbol: str,
    ) -> None:
        anchors = _indifference_anchors(d)
        if len(anchors) < 2:
            return

        a = anchors[0]
        b = anchors[1]
        ax, ay = _barycentric_to_cartesian(a, vertices)
        bx, by = _barycentric_to_cartesian(b, vertices)
        parts.append(
            f'<line class="indiff" x1="{ax:.3f}" y1="{ay:.3f}" x2="{bx:.3f}" y2="{by:.3f}"/>'
        )

        mx = (ax + bx) / 2.0
        my = (ay + by) / 2.0
        parts.append(
            f'<text class="line-label" x="{mx:.2f}" y="{my - 6:.2f}" text-anchor="middle">{payoff_symbol}({strategy_labels[strategy_i]})={payoff_symbol}({strategy_labels[strategy_k]})</text>'
        )

        anchor_mid = (
            (a[0] + b[0]) / 2.0,
            (a[1] + b[1]) / 2.0,
            (a[2] + b[2]) / 2.0,
        )
        basis = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]

        vals = [_dot3(d, vertex_bary) for vertex_bary in basis]

        def region_label_point(indices: list[int]) -> tuple[float, float, float]:
            if len(indices) == 1:
                vertex = basis[indices[0]]
                return (
                    0.55 * vertex[0] + 0.45 * anchor_mid[0],
                    0.55 * vertex[1] + 0.45 * anchor_mid[1],
                    0.55 * vertex[2] + 0.45 * anchor_mid[2],
                )

            v0 = basis[indices[0]]
            v1 = basis[indices[1]]
            return (
                (v0[0] + v1[0] + anchor_mid[0]) / 3.0,
                (v0[1] + v1[1] + anchor_mid[1]) / 3.0,
                (v0[2] + v1[2] + anchor_mid[2]) / 3.0,
            )

        eps = 1e-7
        pos_indices = [idx for idx, value in enumerate(vals) if value > eps]
        neg_indices = [idx for idx, value in enumerate(vals) if value < -eps]

        if pos_indices:
            probe = region_label_point(pos_indices)
            px, py = _barycentric_to_cartesian(probe, vertices)
            parts.append(
                f'<text class="br-label" x="{px:.2f}" y="{py:.2f}" text-anchor="middle">BR {strategy_labels[strategy_i]}</text>'
            )

        if neg_indices:
            probe = region_label_point(neg_indices)
            px, py = _barycentric_to_cartesian(probe, vertices)
            parts.append(
                f'<text class="br-label" x="{px:.2f}" y="{py:.2f}" text-anchor="middle">BR {strategy_labels[strategy_k]}</text>'
            )

    strategy_pairs = [(0, 1), (0, 2), (1, 2)]

    for i, k in strategy_pairs:
        d_q = (
            p1[i][0] - p1[k][0],
            p1[i][1] - p1[k][1],
            p1[i][2] - p1[k][2],
        )
        draw_indifference_line(q_vertices, d_q, i, k, row_labels, "u")

    for j, k in strategy_pairs:
        d_p = (
            p2[0][j] - p2[0][k],
            p2[1][j] - p2[1][k],
            p2[2][j] - p2[2][k],
        )
        draw_indifference_line(p_vertices, d_p, j, k, col_labels, "v")

    palette = ["#dc2626", "#059669", "#7c3aed", "#d97706", "#0ea5e9", "#be123c"]

    q_positions = [
        _barycentric_to_cartesian(q_vec, q_vertices) for _, q_vec in parsed_equilibria
    ]
    p_positions = [
        _barycentric_to_cartesian(p_vec, p_vertices) for p_vec, _ in parsed_equilibria
    ]
    q_displayed, q_shifted = _spread_overlapping_points(q_positions)
    p_displayed, p_shifted = _spread_overlapping_points(p_positions)

    p_component_groups: dict[tuple[float, ...], list[int]] = {}
    q_component_groups: dict[tuple[float, ...], list[int]] = {}
    for idx0, (p_vec, q_vec) in enumerate(parsed_equilibria):
        p_component_groups.setdefault(_rounded_key3(q_vec), []).append(idx0)
        q_component_groups.setdefault(_rounded_key3(p_vec), []).append(idx0)

    endpoint_on_q: set[int] = set()
    endpoint_on_p: set[int] = set()

    def draw_component_segments(
        groups: dict[tuple[float, ...], list[int]],
        positions: list[tuple[float, float]],
        endpoint_set: set[int],
    ) -> None:
        for group in groups.values():
            if len(group) < 2:
                continue
            ordered = sorted(group, key=lambda i: (positions[i][0], positions[i][1]))
            segment = " ".join(
                f"{positions[i][0]:.3f},{positions[i][1]:.3f}" for i in ordered
            )
            parts.append(f'<polyline class="eq-component" points="{segment}"/>')
            endpoint_set.add(ordered[0])
            endpoint_set.add(ordered[-1])

    draw_component_segments(p_component_groups, p_positions, endpoint_on_p)
    draw_component_segments(q_component_groups, q_positions, endpoint_on_q)

    for idx, (p_vec, q_vec) in enumerate(parsed_equilibria, start=1):
        color = palette[(idx - 1) % len(palette)]

        qx, qy = q_positions[idx - 1]
        px, py = p_positions[idx - 1]
        qx_draw, qy_draw = q_displayed[idx - 1]
        px_draw, py_draw = p_displayed[idx - 1]

        if q_shifted[idx - 1]:
            parts.append(
                f'<line class="eq-offset" x1="{qx:.3f}" y1="{qy:.3f}" x2="{qx_draw:.3f}" y2="{qy_draw:.3f}"/>'
            )
        if p_shifted[idx - 1]:
            parts.append(
                f'<line class="eq-offset" x1="{px:.3f}" y1="{py:.3f}" x2="{px_draw:.3f}" y2="{py_draw:.3f}"/>'
            )

        parts.append(
            f'<circle cx="{qx_draw:.3f}" cy="{qy_draw:.3f}" r="4.5" fill="{color}"/>'
        )
        parts.append(
            f'<circle cx="{px_draw:.3f}" cy="{py_draw:.3f}" r="4.5" fill="{color}"/>'
        )

        q_label = _fraction_label(q_vec)
        p_label = _fraction_label(p_vec)
        parts.append(
            f'<text class="eq-label" x="{qx_draw + 8:.2f}" y="{qy_draw - 8:.2f}">NE{idx} q={q_label}</text>'
        )
        parts.append(
            f'<text class="eq-label" x="{px_draw + 8:.2f}" y="{py_draw - 8:.2f}">NE{idx} p={p_label}</text>'
        )

        if (idx - 1) in endpoint_on_q:
            parts.append(
                f'<text class="endpoint-label" x="{qx_draw - 8:.2f}" y="{qy_draw + 16:.2f}" text-anchor="end">end</text>'
            )
        if (idx - 1) in endpoint_on_p:
            parts.append(
                f'<text class="endpoint-label" x="{px_draw - 8:.2f}" y="{py_draw + 16:.2f}" text-anchor="end">end</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)
