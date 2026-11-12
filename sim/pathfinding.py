"""A* on the 4-connected grid."""
import heapq


def neighbors4(c):
    x, y = c
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def neighbors8(c):
    x, y = c
    return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def astar(start, goals, passable, extra_cost=None, limit=6000):
    """Shortest path from `start` to any cell in `goals`, excluding `start`; None if unreachable.

    `extra_cost(c)` adds a penalty for entering c, which is how robots keep away from hazards
    without ever being trapped by them.
    """
    goals = set(goals)
    if start in goals:
        return []
    goals = {g for g in goals if passable(g)}
    if not goals:
        return None

    def h(c):
        return min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goals)

    tie = 0
    frontier = [(h(start), tie, start)]
    came = {start: None}
    cost = {start: 0}
    while frontier and len(came) < limit:
        _, _, cur = heapq.heappop(frontier)
        if cur in goals:
            path = []
            while cur != start:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        for n in neighbors4(cur):
            if not passable(n):
                continue
            g = cost[cur] + 1 + (extra_cost(n) if extra_cost else 0)
            if g < cost.get(n, float("inf")):
                cost[n] = g
                came[n] = cur
                tie += 1
                heapq.heappush(frontier, (g + h(n), tie, n))
    return None
