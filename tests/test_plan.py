from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from jax_bidirectional_tree_rake_compress import (
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
