from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from jax_bidirectional_tree_rake_compress import (
    ContractionExecutor,
    ContractionSchedule,
    make_tree_contraction_plan,
    plan_statistics,
)


def test_plan_removes_every_non_root_node_once() -> None:
    parents = np.array([4, 4, 0, 1, -1, 1, 5, 2, 2])
    plan = make_tree_contraction_plan(parents)

    removed: list[int] = []
    for round_ in plan.rounds:
        removed.extend(np.asarray(round_.rakes[:, 2]).tolist())
        removed.extend(np.asarray(round_.compressions[:, 0]).tolist())

    assert sorted(removed) == [node for node in range(9) if node != 4]
    assert np.asarray(plan.edge_children).tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
    assert np.asarray(plan.edge_parents).tolist() == [4, 4, 0, 1, 1, 5, 2, 2]


def test_star_has_balanced_sibling_reduction() -> None:
    num_nodes = 1025
    plan = make_tree_contraction_plan([-1, *([0] * (num_nodes - 1))])
    stats = plan_statistics(plan)

    assert stats.num_rounds == 1
    assert stats.num_rakes == num_nodes - 1
    assert stats.num_compressions == 0
    assert stats.num_operation_levels == 2 + math.ceil(math.log2(num_nodes - 1))
    assert len(plan.rounds[0].rake_reduction_stages) == math.ceil(
        math.log2(num_nodes - 1)
    )
    assert plan.rounds[0].rake_roots.shape == (1,)
    assert plan.rounds[0].rake_parents.shape == (1,)


def test_chain_has_logarithmic_number_of_rounds() -> None:
    num_nodes = 8192
    plan = make_tree_contraction_plan([-1, *range(num_nodes - 1)])
    stats = plan_statistics(plan)

    assert stats.num_rakes + stats.num_compressions == num_nodes - 1
    assert stats.num_compressions > 0
    assert stats.num_rounds <= 3 * math.ceil(math.log2(num_nodes))


@pytest.mark.parametrize(
    ("executor", "schedule"),
    (
        (ContractionExecutor.SCAN, ContractionSchedule.RAKE_ONLY),
        (ContractionExecutor.ASSOCIATIVE_SCAN, ContractionSchedule.RAKE_COMPRESS),
    ),
)
def test_scan_executors_order_a_permuted_chain(executor, schedule) -> None:
    plan = make_tree_contraction_plan(
        [2, 3, 1, -1], schedule=schedule, executor=executor
    )

    assert np.asarray(plan.chain_nodes).tolist() == [3, 1, 2, 0]
    assert np.asarray(plan.chain_edges).tolist() == [1, 2, 0]
    assert plan.rounds == ()


@pytest.mark.parametrize(
    ("parents", "schedule", "expected"),
    (
        ([-1, 0, 1], ContractionSchedule.RAKE_ONLY, ContractionExecutor.SCAN),
        (
            [-1, 0, 1],
            ContractionSchedule.RAKE_COMPRESS,
            ContractionExecutor.ASSOCIATIVE_SCAN,
        ),
        (
            [-1, 0, 0],
            ContractionSchedule.RAKE_ONLY,
            ContractionExecutor.UNROLLED,
        ),
        (
            [-1, 0, 0],
            ContractionSchedule.RAKE_COMPRESS,
            ContractionExecutor.UNROLLED,
        ),
    ),
)
def test_auto_executor_selects_from_schedule_and_topology(
    parents, schedule, expected
) -> None:
    plan = make_tree_contraction_plan(
        parents, schedule=schedule, executor=ContractionExecutor.AUTO
    )

    assert plan.executor is expected


@pytest.mark.parametrize(
    "executor", (ContractionExecutor.SCAN, ContractionExecutor.ASSOCIATIVE_SCAN)
)
@pytest.mark.parametrize("parents", ([-1, 0, 0], [1, -1, 1]))
def test_scan_executors_reject_non_chain_topologies(executor, parents) -> None:
    schedule = (
        ContractionSchedule.RAKE_ONLY
        if executor is ContractionExecutor.SCAN
        else ContractionSchedule.RAKE_COMPRESS
    )
    with pytest.raises(ValueError, match="require a chain topology"):
        make_tree_contraction_plan(parents, schedule=schedule, executor=executor)


@pytest.mark.parametrize(
    ("executor", "schedule", "expected"),
    (
        (
            ContractionExecutor.SCAN,
            ContractionSchedule.RAKE_COMPRESS,
            "RAKE_ONLY",
        ),
        (
            ContractionExecutor.ASSOCIATIVE_SCAN,
            ContractionSchedule.RAKE_ONLY,
            "RAKE_COMPRESS",
        ),
    ),
)
def test_scan_executors_require_their_corresponding_schedule(
    executor, schedule, expected
) -> None:
    with pytest.raises(ValueError, match=expected):
        make_tree_contraction_plan([-1, 0], schedule=schedule, executor=executor)


def delayed_star(groups: int, width: int) -> list[int]:
    """Make groups of root paths that finish after successively longer delays."""
    parents = [-1]
    for group in range(groups):
        for _ in range(width):
            parent = 0
            for _ in range(2**group):
                parents.append(parent)
                parent = len(parents) - 1
    return parents


def test_rake_compress_removes_global_sibling_reduction_barriers() -> None:
    parents = delayed_star(groups=6, width=16)
    plan = make_tree_contraction_plan(parents)
    stats = plan_statistics(plan)

    assert stats.num_operation_levels == 20
    assert stats.num_operation_levels <= 2 * math.ceil(math.log2(len(parents)))

    node_producers = np.full(plan.num_nodes, -1)
    path_producers = np.full(plan.num_edges, -1)
    branch_producers = np.full(stats.num_rakes, -1)
    removed = [
        int(node)
        for level in plan.rounds
        for node in np.concatenate(
            (
                np.asarray(level.rakes[:, 2]),
                np.asarray(level.compressions[:, 0]),
            )
        )
    ]
    assert sorted(removed) == list(range(1, len(parents)))

    for level_index, level in enumerate(plan.rounds):
        rakes = np.asarray(level.rakes)
        reductions = np.asarray(level.branch_reductions)
        absorptions = np.asarray(level.branch_absorptions)
        compressions = np.asarray(level.compressions)

        assert np.all(path_producers[rakes[:, 0]] < level_index)
        assert np.all(node_producers[rakes[:, 2]] < level_index)
        assert np.all(branch_producers[reductions] < level_index)
        assert np.all(node_producers[absorptions[:, 0]] < level_index)
        assert np.all(branch_producers[absorptions[:, 1]] < level_index)
        assert np.all(node_producers[compressions[:, 0]] < level_index)
        assert np.all(path_producers[compressions[:, 1:3]] < level_index)

        rake_writes = set(np.asarray(level.rakes[:, 3]).tolist())
        reduction_writes = set(np.asarray(level.branch_reductions[:, 0]).tolist())
        assert rake_writes.isdisjoint(reduction_writes)
        assert np.unique(np.asarray(level.branch_absorptions[:, 0])).size == len(
            level.branch_absorptions
        )
        assert np.unique(np.asarray(level.compressions[:, 1])).size == len(
            level.compressions
        )

        branch_producers[rakes[:, 3]] = level_index
        branch_producers[reductions[:, 0]] = level_index
        node_producers[absorptions[:, 0]] = level_index
        path_producers[compressions[:, 1]] = level_index


@pytest.mark.parametrize(
    ("parents", "expected_rounds"),
    [
        ([-1, 0, 1, 2, 3, 4, 5, 6], 7),
        ([-1, 0, 0, 0, 0, 0, 0, 0], 1),
        ([-1, 0, 0, 1, 1, 2, 2, 3, 3, 6], 3),
        ([-1, 0, 0, 1, 1, 3, 2, 6, 6, 8, 3], 4),
    ],
    ids=("chain", "star", "balanced", "irregular"),
)
def test_rake_only_schedule_removes_one_level_per_round(
    parents: list[int], expected_rounds: int
) -> None:
    plan = make_tree_contraction_plan(parents, schedule=ContractionSchedule.RAKE_ONLY)
    stats = plan_statistics(plan)

    removed = [
        int(node) for round_ in plan.rounds for node in np.asarray(round_.rakes[:, 2])
    ]
    assert sorted(removed) == list(range(1, len(parents)))
    assert stats.num_rakes == len(parents) - 1
    assert stats.num_compressions == 0
    assert stats.num_rounds == expected_rounds


def test_invalid_schedule_is_rejected_for_single_node_tree() -> None:
    with pytest.raises(TypeError, match="unsupported contraction schedule"):
        make_tree_contraction_plan([-1], schedule="rake-only")  # type: ignore[arg-type]


def test_invalid_executor_is_rejected_for_single_node_tree() -> None:
    with pytest.raises(TypeError, match="unsupported contraction executor"):
        make_tree_contraction_plan([-1], executor="scan")  # type: ignore[arg-type]


def test_plan_is_a_jax_pytree() -> None:
    plan = make_tree_contraction_plan([-1, 0, 0, 1])
    leaves = jax.tree.leaves(plan)

    assert leaves
    assert all(hasattr(leaf, "dtype") for leaf in leaves)


@pytest.mark.parametrize(
    ("parents", "message"),
    [
        ([], "at least one"),
        ([-1, -1], "exactly one root"),
        ([-1, 2], "invalid parent"),
        ([-1, 1], "exactly one root"),
        ([-1, 2, 1], "one acyclic tree"),
    ],
)
def test_invalid_topologies(parents: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_tree_contraction_plan(parents)


def test_explicit_nonzero_root_and_self_parent() -> None:
    plan = make_tree_contraction_plan([2, 0, 2, 2], root=2)

    assert int(plan.root) == 2
    assert np.asarray(plan.parents).tolist() == [2, 0, -1, 2]
