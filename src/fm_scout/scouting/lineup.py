"""Meilleure affectation joueurs -> postes, chaque joueur servant une seule fois."""
from math import inf

FORBIDDEN = 1e6


def assign(scores: list[list[float | None]]) -> list[int | None]:
    """scores[poste][joueur] (None = impossible) -> joueur de chaque poste.

    Algorithme hongrois : maximise d'abord le nombre de postes pourvus, puis le score total.
    """
    rows = len(scores)
    if not rows:
        return []
    cols = max(len(scores[0]), rows)  # colonnes fictives si trop peu de joueurs
    cost = [[FORBIDDEN if j >= len(r) or r[j] is None else -r[j] for j in range(cols)] for r in scores]
    u, v = [0.0] * (rows + 1), [0.0] * (cols + 1)
    match, way = [0] * (cols + 1), [0] * (cols + 1)
    for i in range(1, rows + 1):
        match[0], j0 = i, 0
        low, used = [inf] * (cols + 1), [False] * (cols + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = match[j0], inf, 0
            for j in range(1, cols + 1):
                if not used[j]:
                    current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < low[j]:
                        low[j], way[j] = current, j0
                    if low[j] < delta:
                        delta, j1 = low[j], j
            for j in range(cols + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    low[j] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while j0:
            match[j0] = match[way[j0]]
            j0 = way[j0]
    result: list[int | None] = [None] * rows
    for j in range(1, cols + 1):
        if match[j] and cost[match[j] - 1][j - 1] < FORBIDDEN:
            result[match[j] - 1] = j - 1
    return result
