"""CPU preprocessing for bidirectional rake--compress contraction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from enum import Enum, auto
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray


class ContractionSchedule(Enum):
    """Host-side policy used to construct tree-contraction rounds."""

    RAKE_COMPRESS = auto()
    RAKE_ONLY = auto()


class ContractionRound(NamedTuple):
    """One conflict-free rake--compress round.

    ``rakes`` contains ``(edge, parent, leaf)`` records. ``compressions``
    contains ``(middle, left_edge, right_edge, parent, child)`` records.
    The remaining fields describe a balanced reduction of rake messages that
    target the same parent.
    """

    rakes: jnp.ndarray
    compressions: jnp.ndarray
    rake_reduction_stages: tuple[jnp.ndarray, ...]
    rake_roots: jnp.ndarray
    rake_parents: jnp.ndarray


class TreeContractionPlan(NamedTuple):
    """A topology-only contraction plan represented entirely as a PyTree."""

    parents: jnp.ndarray
    edge_parents: jnp.ndarray
    edge_children: jnp.ndarray
    root: jnp.ndarray
    rounds: tuple[ContractionRound, ...]

    @property
    def num_nodes(self) -> int:
        return self.parents.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_children.shape[0]


class PlanStatistics(NamedTuple):
    """Shape-derived statistics that do not transfer plan data from a device."""

    num_nodes: int
    num_edges: int
    num_rounds: int
    num_rakes: int
    num_compressions: int
    max_rakes_per_round: int
    max_compressions_per_round: int


def _normalize_parents(
    parents: ArrayLike, root: int | None
) -> tuple[NDArray[np.int64], int]:
    result = np.asarray(parents)
    if result.ndim != 1:
        raise ValueError("parents must be a one-dimensional integer array")
    if result.size == 0:
        raise ValueError("a tree must contain at least one node")
    if not np.issubdtype(result.dtype, np.integer):
        raise TypeError("parents must contain integers")

    result = np.array(result, dtype=np.int64, copy=True)
    negative_roots = np.flatnonzero(result < 0)
    self_roots = np.flatnonzero(result == np.arange(result.size))

    if root is None:
        candidates = np.unique(np.concatenate((negative_roots, self_roots)))
        if candidates.size != 1:
            raise ValueError(
                "parents must identify exactly one root with a negative or "
                "self parent, or root must be provided"
            )
        root = int(candidates[0])
    elif not 0 <= root < result.size:
        raise ValueError(f"root {root} is outside [0, {result.size})")

    assert root is not None
    for node, parent in enumerate(result):
        if node == root:
            if parent >= 0 and parent != root:
                raise ValueError("the root must have a negative or self parent")
            continue
        if parent < 0 or parent >= result.size:
            raise ValueError(f"invalid parent {parent} for node {node}")
        if parent == node:
            raise ValueError(f"non-root node {node} cannot parent itself")

    result[root] = -1
    children: list[list[int]] = [[] for _ in range(result.size)]
    for node, parent in enumerate(result):
        if node != root:
            children[int(parent)].append(node)

    visited: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        visited.append(node)
        stack.extend(children[node])
    if len(visited) != result.size or len(set(visited)) != result.size:
        raise ValueError("parents must describe one acyclic tree rooted at root")
    return result, root


def _message_reduction_plan(
    rake_parents: Sequence[int],
) -> tuple[list[NDArray[np.int32]], NDArray[np.int32], NDArray[np.int32]]:
    """Build balanced, conflict-free reductions for messages by parent."""

    groups: dict[int, list[int]] = defaultdict(list)
    for message, parent in enumerate(rake_parents):
        groups[int(parent)].append(message)

    ordered_parents = sorted(groups)
    active = {parent: groups[parent] for parent in ordered_parents}
    stages: list[NDArray[np.int32]] = []
    while any(len(indices) > 1 for indices in active.values()):
        pairs: list[tuple[int, int]] = []
        next_active: dict[int, list[int]] = {}
        for parent in ordered_parents:
            indices = active[parent]
            survivors: list[int] = []
            for index in range(0, len(indices) - 1, 2):
                destination = indices[index]
                source = indices[index + 1]
                pairs.append((destination, source))
                survivors.append(destination)
            if len(indices) % 2:
                survivors.append(indices[-1])
            next_active[parent] = survivors
        if pairs:
            stages.append(np.asarray(pairs, dtype=np.int32).reshape(-1, 2))
        active = next_active

    roots = np.asarray(
        [active[parent][0] for parent in ordered_parents], dtype=np.int32
    )
    parents = np.asarray(ordered_parents, dtype=np.int32)
    return stages, roots, parents


def make_tree_contraction_plan(
    parents: ArrayLike,
    *,
    root: int | None = None,
    schedule: ContractionSchedule = ContractionSchedule.RAKE_COMPRESS,
) -> TreeContractionPlan:
    """Precompute a deterministic rake--compress plan on the CPU.

    Args:
        parents: Parent index for every node. The root must have a negative or
            self parent unless ``root`` is supplied.
        root: Optional explicit root index.
        schedule: Contraction policy. ``RAKE_COMPRESS`` gives logarithmic-depth
            parallel contraction; ``RAKE_ONLY`` removes leaves level by level.

    Returns:
        A JAX PyTree containing only integer topology arrays. Edge summaries
        are ordered by ``plan.edge_children``: one incoming edge for every
        non-root node.
    """

    if not isinstance(schedule, ContractionSchedule):
        raise TypeError(f"unsupported contraction schedule: {schedule!r}")

    parent_array, root = _normalize_parents(parents, root)
    num_nodes = parent_array.size
    edge_children = np.asarray(
        [node for node in range(num_nodes) if node != root], dtype=np.int32
    )
    edge_parents = parent_array[edge_children].astype(np.int32, copy=False)
    num_edges = edge_children.size

    incoming = np.full(num_nodes, -1, dtype=np.int64)
    active_parent = np.full(num_nodes, -1, dtype=np.int64)
    edge_parent = np.asarray(edge_parents, dtype=np.int64).copy()
    edge_child = np.asarray(edge_children, dtype=np.int64).copy()
    active_children: list[set[int]] = [set() for _ in range(num_nodes)]
    for edge in range(num_edges):
        parent = int(edge_parent[edge])
        child = int(edge_child[edge])
        incoming[child] = edge
        active_parent[child] = parent
        active_children[parent].add(edge)

    active = np.ones(num_nodes, dtype=np.bool_)
    active_count = num_nodes
    host_rounds: list[
        tuple[
            NDArray[np.int32],
            NDArray[np.int32],
            list[NDArray[np.int32]],
            NDArray[np.int32],
            NDArray[np.int32],
        ]
    ] = []

    while active_count > 1:
        rakes: list[tuple[int, int, int]] = []
        for node in range(num_nodes):
            if active[node] and node != root and not active_children[node]:
                rakes.append(
                    (
                        int(incoming[node]),
                        int(active_parent[node]),
                        node,
                    )
                )

        for edge, parent, child in rakes:
            active_children[parent].discard(edge)
            active[child] = False
            active_count -= 1

        selected: list[int] = []
        if schedule is ContractionSchedule.RAKE_COMPRESS:
            blocked = np.zeros(num_nodes, dtype=np.bool_)
            for node in range(num_nodes):
                if not active[node] or node == root or len(active_children[node]) != 1:
                    continue
                parent = int(active_parent[node])
                child_edge = next(iter(active_children[node]))
                child = int(edge_child[child_edge])
                if blocked[parent] or blocked[node] or blocked[child]:
                    continue
                selected.append(node)
                blocked[parent] = blocked[node] = blocked[child] = True

        compressions: list[tuple[int, int, int, int, int]] = []
        for node in selected:
            left_edge = int(incoming[node])
            right_edge = next(iter(active_children[node]))
            parent = int(active_parent[node])
            child = int(edge_child[right_edge])
            compressions.append((node, left_edge, right_edge, parent, child))

        for node, left_edge, _right_edge, parent, child in compressions:
            edge_child[left_edge] = child
            active_parent[child] = parent
            incoming[child] = left_edge
            active[node] = False
            active_count -= 1

        if not rakes and not compressions:
            raise RuntimeError("tree contraction planning made no progress")

        rake_array = np.asarray(rakes, dtype=np.int32).reshape(-1, 3)
        compression_array = np.asarray(compressions, dtype=np.int32).reshape(-1, 5)
        reduction_stages, reduction_roots, reduction_parents = _message_reduction_plan(
            rake_array[:, 1]
        )
        host_rounds.append(
            (
                rake_array,
                compression_array,
                reduction_stages,
                reduction_roots,
                reduction_parents,
            )
        )

    rounds = tuple(
        ContractionRound(
            rakes=jnp.asarray(rakes),
            compressions=jnp.asarray(compressions),
            rake_reduction_stages=tuple(
                jnp.asarray(stage) for stage in reduction_stages
            ),
            rake_roots=jnp.asarray(reduction_roots),
            rake_parents=jnp.asarray(reduction_parents),
        )
        for (
            rakes,
            compressions,
            reduction_stages,
            reduction_roots,
            reduction_parents,
        ) in host_rounds
    )
    return TreeContractionPlan(
        parents=jnp.asarray(parent_array, dtype=jnp.int32),
        edge_parents=jnp.asarray(edge_parents),
        edge_children=jnp.asarray(edge_children),
        root=jnp.asarray(root, dtype=jnp.int32),
        rounds=rounds,
    )


def plan_statistics(plan: TreeContractionPlan) -> PlanStatistics:
    """Return contraction counts using only statically known array shapes."""

    rake_counts = [round_.rakes.shape[0] for round_ in plan.rounds]
    compression_counts = [round_.compressions.shape[0] for round_ in plan.rounds]
    return PlanStatistics(
        num_nodes=plan.num_nodes,
        num_edges=plan.num_edges,
        num_rounds=len(plan.rounds),
        num_rakes=sum(rake_counts),
        num_compressions=sum(compression_counts),
        max_rakes_per_round=max(rake_counts, default=0),
        max_compressions_per_round=max(compression_counts, default=0),
    )
