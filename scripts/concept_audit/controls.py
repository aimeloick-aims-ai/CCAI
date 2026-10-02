import numpy as np

from .config import (
    NODE_CONCEPTS,
    NUM_RANDOM_CONTROLS,
)


def make_random_directions(
    node_dim,
    graph_dim,
    seed,
):
    rng = np.random.default_rng(seed)

    node_directions = {}

    for concept in NODE_CONCEPTS:
        directions = []

        for _ in range(
            NUM_RANDOM_CONTROLS
        ):
            d = rng.normal(
                size=node_dim
            )

            d /= np.linalg.norm(d)
            directions.append(d)

        node_directions[concept] = directions

    graph_directions = []

    for _ in range(
        NUM_RANDOM_CONTROLS
    ):
        d = rng.normal(
            size=graph_dim
        )

        d /= np.linalg.norm(d)
        graph_directions.append(d)

    return {
        "node": node_directions,
        "graph": graph_directions,
    }
