
from collections import deque, defaultdict

# TRACE network topology

NODES = ["A", "B", "C", "D", "E"]

LINKS = [
    ("A", "B"),
    ("B", "C"),
    ("B", "D"),
    ("C", "E"),
    ("D", "E"),
]


def normalize_link(a, b):
    """Treat links as undirected."""
    return tuple(sorted((a, b)))


def build_graph():
    graph = {node: [] for node in NODES}

    for a, b in LINKS:
        graph[a].append(b)
        graph[b].append(a)

    return graph


def shortest_paths(graph, source, destination):
    """Find all shortest paths between two nodes."""

    distances = {source: 0}
    queue = deque([source])

    while queue:
        node = queue.popleft()

        for neighbour in graph[node]:
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)

    if destination not in distances:
        return []

    paths = []

    def explore(node, path):
        if node == destination:
            paths.append(path.copy())
            return

        for neighbour in graph[node]:
            if (
                distances.get(neighbour) == distances[node] + 1
                and distances[neighbour] <= distances[destination]
            ):
                explore(neighbour, path + [neighbour])

    explore(source, [source])

    return paths


def rank_suspect_links(observations):
    """
    Rank links using failed, healthy and degraded path observations.

    Supported statuses:
        failed
        healthy
        degraded

    Degraded observations can include a severity value.
    """

    graph = build_graph()

    failed_scores = defaultdict(float)
    healthy_scores = defaultdict(float)
    degraded_scores = defaultdict(float)

    for observation in observations:

        source = observation["source"]
        destination = observation["destination"]
        status = observation["status"].lower()

        if source not in graph or destination not in graph:
            raise ValueError("Unknown network node")

        if source == destination:
            raise ValueError("Source and destination must differ")

        paths = shortest_paths(graph, source, destination)

        if not paths:
            continue

        weight = 1 / len(paths)

        if status == "failed":
            target_scores = failed_scores

        elif status == "healthy":
            target_scores = healthy_scores

        elif status == "degraded":
            target_scores = degraded_scores

        else:
            raise ValueError(
                "Status must be failed, healthy or degraded"
            )

        severity = float(observation.get("severity", 1.0))

        for path in paths:

            for i in range(len(path) - 1):

                link = normalize_link(path[i], path[i + 1])

                target_scores[link] += weight * (
                    severity if status == "degraded" else 1.0
                )

    results = []

    for link in LINKS:

        key = normalize_link(*link)

        failed = failed_scores[key]
        healthy = healthy_scores[key]
        degraded = degraded_scores[key]

        # Failed and degraded evidence increases suspicion.
        # Healthy evidence reduces suspicion.
        score = failed + degraded - healthy

        if failed > 0 or degraded > 0:

            results.append({
                "link": f"{key[0]}-{key[1]}",
                "failed_path_support": round(failed, 2),
                "healthy_path_support": round(healthy, 2),
                "degraded_path_support": round(degraded, 2),
                "suspicion_score": round(score, 2)
            })

    return sorted(
        results,
        key=lambda item: item["suspicion_score"],
        reverse=True
    )


if __name__ == "__main__":

    observations = [
        {
            "source": "A",
            "destination": "B",
            "status": "healthy"
        },
        {
            "source": "B",
            "destination": "C",
            "status": "degraded",
            "severity": 2.0
        },
        {
            "source": "B",
            "destination": "D",
            "status": "healthy"
        },
        {
            "source": "C",
            "destination": "E",
            "status": "healthy"
        },
        {
            "source": "D",
            "destination": "E",
            "status": "healthy"
        }
    ]

    print("\nTRACE - Topology-Aware Analysis")
    print("=" * 45)

    results = rank_suspect_links(observations)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}: Link {result['link']}")
        print("Failed support:", result["failed_path_support"])
        print("Degraded support:", result["degraded_path_support"])
        print("Healthy support:", result["healthy_path_support"])
        print("Suspicion score:", result["suspicion_score"])
