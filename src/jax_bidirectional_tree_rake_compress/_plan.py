"""CPU preprocessing for bidirectional rake--compress contraction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray


class ContractionSchedule(Enum):
    """Host-side policy used to construct tree-contraction rounds."""

    RAKE_COMPRESS = auto()
    RAKE_ONLY = auto()


class ContractionExecutor(Enum):
    """JAX control-flow implementation used to execute a contraction."""

    AUTO = auto()
    UNROLLED = auto()
    SCAN = auto()
    ASSOCIATIVE_SCAN = auto()


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


class _DependencyLevel(NamedTuple):
    """Mutually independent primitive operations at one dependency level.

    ``rakes`` contains ``(edge, parent, leaf, branch)`` records,
    ``branch_reductions`` contains ``(destination, source)`` branch slots,
    ``branch_absorptions`` contains ``(parent, branch)`` records, and
    ``compressions`` uses the same schema as :class:`ContractionRound`.
    """

    rakes: jnp.ndarray
    branch_reductions: jnp.ndarray
    branch_absorptions: jnp.ndarray
    compressions: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class TreeContractionPlan:
    """A topology-only contraction plan represented entirely as a PyTree."""

    parents: jnp.ndarray
    edge_parents: jnp.ndarray
    edge_children: jnp.ndarray
    root: jnp.ndarray
    rounds: tuple[ContractionRound | _DependencyLevel, ...]
    chain_nodes: jnp.ndarray
    chain_edges: jnp.ndarray
    executor: ContractionExecutor

    @property
    def num_nodes(self) -> int:
        return self.parents.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_children.shape[0]

    def tree_flatten(self):
        children = (
            self.parents,
            self.edge_parents,
            self.edge_children,
            self.root,
            self.rounds,
            self.chain_nodes,
            self.chain_edges,
        )
        return children, self.executor

    @classmethod
    def tree_unflatten(cls, executor, children):
        return cls(*children, executor=executor)


class PlanStatistics(NamedTuple):
    """Shape-derived statistics that do not transfer plan data from a device."""

    num_nodes: int
    num_edges: int
    num_rounds: int
    num_rakes: int
    num_compressions: int
    max_rakes_per_round: int
    max_compressions_per_round: int
    num_operation_levels: int


HostRound = tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    list[NDArray[np.int32]],
    NDArray[np.int32],
    NDArray[np.int32],
]


class _HostDependencyLevel(NamedTuple):
    rakes: NDArray[np.int32]
    branch_reductions: NDArray[np.int32]
    branch_absorptions: NDArray[np.int32]
    compressions: NDArray[np.int32]


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


def _make_chain_order(
    parents: NDArray[np.int64],
    root: int,
    edge_children: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Order a rooted chain from its root to its leaf."""
    child_by_parent = np.full(parents.size, -1, dtype=np.int32)
    for child, parent in enumerate(parents):
        if child == root:
            continue
        if child_by_parent[parent] >= 0:
            raise ValueError("scan executors require a chain topology")
        child_by_parent[parent] = child

    chain_nodes = [root]
    while child_by_parent[chain_nodes[-1]] >= 0:
        chain_nodes.append(int(child_by_parent[chain_nodes[-1]]))
    if len(chain_nodes) != parents.size:
        raise ValueError("scan executors require a chain topology")

    incoming_edge = np.full(parents.size, -1, dtype=np.int32)
    incoming_edge[edge_children] = np.arange(edge_children.size, dtype=np.int32)
    chain_nodes_array = np.asarray(chain_nodes, dtype=np.int32)
    return chain_nodes_array, incoming_edge[chain_nodes_array[1:]]


def _dependency_level_plan(
    num_nodes: int,
    num_edges: int,
    host_rounds: Sequence[HostRound],
) -> tuple[_HostDependencyLevel, ...]:
    """Schedule rake--compress primitives at their earliest dependency level."""

    node_producers = np.full(num_nodes, -1, dtype=np.int64)
    path_producers = np.full(num_edges, -1, dtype=np.int64)
    branch_producers: list[int] = []
    levels: list[dict[str, list[tuple[int, ...]]]] = []

    def add(level: int, operation: str, record: tuple[int, ...]) -> None:
        while len(levels) <= level:
            levels.append(
                {
                    "rakes": [],
                    "branch_reductions": [],
                    "branch_absorptions": [],
                    "compressions": [],
                }
            )
        levels[level][operation].append(record)

    for (
        rakes,
        compressions,
        reduction_stages,
        reduction_roots,
        reduction_parents,
    ) in host_rounds:
        round_branches = np.arange(
            len(branch_producers),
            len(branch_producers) + rakes.shape[0],
            dtype=np.int64,
        )
        for (edge, parent, leaf), branch in zip(rakes, round_branches, strict=True):
            level = 1 + max(path_producers[edge], node_producers[leaf])
            branch_producers.append(int(level))
            add(
                int(level),
                "rakes",
                (int(edge), int(parent), int(leaf), int(branch)),
            )

        for stage in reduction_stages:
            for destination, source in stage:
                destination = int(round_branches[destination])
                source = int(round_branches[source])
                level = 1 + max(branch_producers[destination], branch_producers[source])
                branch_producers[destination] = int(level)
                add(
                    int(level),
                    "branch_reductions",
                    (destination, source),
                )

        for root, parent in zip(reduction_roots, reduction_parents, strict=True):
            branch = int(round_branches[root])
            parent = int(parent)
            level = 1 + max(branch_producers[branch], node_producers[parent])
            node_producers[parent] = int(level)
            add(int(level), "branch_absorptions", (parent, branch))

        for middle, left_edge, right_edge, parent, child in compressions:
            level = 1 + max(
                node_producers[middle],
                path_producers[left_edge],
                path_producers[right_edge],
            )
            path_producers[left_edge] = int(level)
            add(
                int(level),
                "compressions",
                (
                    int(middle),
                    int(left_edge),
                    int(right_edge),
                    int(parent),
                    int(child),
                ),
            )

    return tuple(
        _HostDependencyLevel(
            rakes=np.asarray(level["rakes"], dtype=np.int32).reshape(-1, 4),
            branch_reductions=np.asarray(
                level["branch_reductions"], dtype=np.int32
            ).reshape(-1, 2),
            branch_absorptions=np.asarray(
                level["branch_absorptions"], dtype=np.int32
            ).reshape(-1, 2),
            compressions=np.asarray(level["compressions"], dtype=np.int32).reshape(
                -1, 5
            ),
        )
        for level in levels
    )


def _materialize_dependency_levels(
    levels: Sequence[_HostDependencyLevel],
) -> tuple[_DependencyLevel, ...]:
    return tuple(
        _DependencyLevel(*(jnp.asarray(field) for field in level)) for level in levels
    )


def _materialize_rounds(
    host_rounds: Sequence[HostRound],
) -> tuple[ContractionRound, ...]:
    return tuple(
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


def _synchronous_operation_levels(host_rounds: Sequence[HostRound]) -> int:
    """Return the primitive span of synchronous round execution."""
    return sum(
        (1 if rakes.shape[0] else 0)
        + len(reduction_stages)
        + (1 if rakes.shape[0] else 0)
        + (1 if compressions.shape[0] else 0)
        for rakes, compressions, reduction_stages, _, _ in host_rounds
    )


def make_tree_contraction_plan(
    parents: ArrayLike,
    *,
    root: int | None = None,
    schedule: ContractionSchedule = ContractionSchedule.RAKE_COMPRESS,
    executor: ContractionExecutor = ContractionExecutor.UNROLLED,
) -> TreeContractionPlan:
    """Precompute a deterministic rake--compress plan on the CPU.

    Args:
        parents: Parent index for every node. The root must have a negative or
            self parent unless ``root`` is supplied.
        root: Optional explicit root index.
        schedule: Contraction policy. ``RAKE_COMPRESS`` gives logarithmic-depth
            parallel contraction and ``RAKE_ONLY`` removes leaves level by
            level.
        executor: JAX control-flow implementation. ``SCAN`` executes a
            rake-only chain through :func:`jax.lax.scan`, while
            ``ASSOCIATIVE_SCAN`` executes a rake--compress chain through
            :func:`jax.lax.associative_scan`. ``AUTO`` selects the corresponding
            scan on chains and ``UNROLLED`` otherwise. ``UNROLLED`` supports
            every topology and schedule.

    Returns:
        A JAX PyTree containing only integer topology arrays. Edge summaries
        are ordered by ``plan.edge_children``: one incoming edge for every
        non-root node.
    """

    if not isinstance(schedule, ContractionSchedule):
        raise TypeError(f"unsupported contraction schedule: {schedule!r}")
    if not isinstance(executor, ContractionExecutor):
        raise TypeError(f"unsupported contraction executor: {executor!r}")

    parent_array, root = _normalize_parents(parents, root)
    num_nodes = parent_array.size
    edge_children = np.asarray(
        [node for node in range(num_nodes) if node != root], dtype=np.int32
    )
    edge_parents = parent_array[edge_children].astype(np.int32, copy=False)
    num_edges = edge_children.size
    is_chain = bool(np.all(np.bincount(edge_parents, minlength=num_nodes) <= 1))
    if executor is ContractionExecutor.AUTO:
        if is_chain:
            executor = (
                ContractionExecutor.SCAN
                if schedule is ContractionSchedule.RAKE_ONLY
                else ContractionExecutor.ASSOCIATIVE_SCAN
            )
        else:
            executor = ContractionExecutor.UNROLLED
    common = dict(
        parents=jnp.asarray(parent_array, dtype=jnp.int32),
        edge_parents=jnp.asarray(edge_parents),
        edge_children=jnp.asarray(edge_children),
        root=jnp.asarray(root, dtype=jnp.int32),
        executor=executor,
    )

    if executor is not ContractionExecutor.UNROLLED:
        expected_schedule = (
            ContractionSchedule.RAKE_ONLY
            if executor is ContractionExecutor.SCAN
            else ContractionSchedule.RAKE_COMPRESS
        )
        if schedule is not expected_schedule:
            raise ValueError(
                f"{executor.name} requires schedule={expected_schedule.name}"
            )
        chain_nodes, chain_edges = _make_chain_order(parent_array, root, edge_children)
        return TreeContractionPlan(
            **common,
            rounds=(),
            chain_nodes=jnp.asarray(chain_nodes),
            chain_edges=jnp.asarray(chain_edges),
        )

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
    host_rounds: list[HostRound] = []

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

    rounds: tuple[ContractionRound | _DependencyLevel, ...]
    max_reduction_stages = max(
        (len(reduction_stages) for _, _, reduction_stages, _, _ in host_rounds),
        default=0,
    )
    may_have_global_reduction_barriers = (
        len(host_rounds) > 1 and max_reduction_stages > 1
    )
    if (
        schedule is ContractionSchedule.RAKE_COMPRESS
        and may_have_global_reduction_barriers
    ):
        dependency_levels = _dependency_level_plan(num_nodes, num_edges, host_rounds)
        synchronous_levels = _synchronous_operation_levels(host_rounds)
        # Dependency-level execution needs a persistent branch workspace and
        # finer-grained kernels. Use it only when it removes a non-constant
        # fraction of the synchronous span. Otherwise the synchronous span is
        # already within a factor of two of the dependency depth, preserving
        # the O(log N) bound without paying that overhead on ordinary trees.
        if 2 * len(dependency_levels) < synchronous_levels:
            rounds = _materialize_dependency_levels(dependency_levels)
        else:
            rounds = _materialize_rounds(host_rounds)
    else:
        rounds = _materialize_rounds(host_rounds)
    return TreeContractionPlan(
        **common,
        rounds=rounds,
        chain_nodes=jnp.empty(0, dtype=jnp.int32),
        chain_edges=jnp.empty(0, dtype=jnp.int32),
    )


def plan_statistics(plan: TreeContractionPlan) -> PlanStatistics:
    """Return contraction counts using only statically known array shapes."""

    if plan.executor is ContractionExecutor.SCAN:
        return PlanStatistics(
            num_nodes=plan.num_nodes,
            num_edges=plan.num_edges,
            num_rounds=plan.num_edges,
            num_rakes=plan.num_edges,
            num_compressions=0,
            max_rakes_per_round=min(plan.num_edges, 1),
            max_compressions_per_round=0,
            num_operation_levels=2 * plan.num_edges,
        )
    if plan.executor is ContractionExecutor.ASSOCIATIVE_SCAN:
        depth = 0 if plan.num_edges == 0 else plan.num_edges.bit_length() + 1
        return PlanStatistics(
            num_nodes=plan.num_nodes,
            num_edges=plan.num_edges,
            num_rounds=depth,
            num_rakes=min(plan.num_edges, 1),
            num_compressions=max(plan.num_edges - 1, 0),
            max_rakes_per_round=min(plan.num_edges, 1),
            max_compressions_per_round=plan.num_edges // 2,
            num_operation_levels=depth,
        )

    rake_counts = [round_.rakes.shape[0] for round_ in plan.rounds]
    compression_counts = [round_.compressions.shape[0] for round_ in plan.rounds]
    dependency_levels = bool(plan.rounds) and isinstance(
        plan.rounds[0], _DependencyLevel
    )
    if dependency_levels:
        num_operation_levels = len(plan.rounds)
    else:
        num_operation_levels = sum(
            (1 if round_.rakes.shape[0] else 0)
            + len(round_.rake_reduction_stages)
            + (1 if round_.rakes.shape[0] else 0)
            + (1 if round_.compressions.shape[0] else 0)
            for round_ in plan.rounds
            if isinstance(round_, ContractionRound)
        )
    return PlanStatistics(
        num_nodes=plan.num_nodes,
        num_edges=plan.num_edges,
        num_rounds=len(plan.rounds),
        num_rakes=sum(rake_counts),
        num_compressions=sum(compression_counts),
        max_rakes_per_round=max(rake_counts, default=0),
        max_compressions_per_round=max(compression_counts, default=0),
        num_operation_levels=num_operation_levels,
    )
