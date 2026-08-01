from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_bidirectional_tree_rake_compress import (
    ContractionExecutor,
    ContractionSchedule,
    make_tree_contraction_plan,
    tree_contract,
    tree_expand,
)


@dataclass(frozen=True)
class AffineBroadcastAlgebra:
    def rake(self, path, leaf):
        return jnp.zeros_like(leaf), path

    def combine_branches(self, left, right):
        return left + right

    def absorb_branch(self, node, message):
        return node + message

    def compress(self, left, middle, right):
        del middle
        left_matrix, left_offset = left
        right_matrix, right_offset = right
        composed = (
            right_matrix @ left_matrix,
            right_matrix @ left_offset + right_offset,
        )
        return composed, left

    def expand_compress(self, residual, parent_output, child_output):
        del child_output
        matrix, offset = residual
        return matrix @ parent_output + offset

    def expand_rake(self, residual, parent_output):
        matrix, offset = residual
        return matrix @ parent_output + offset


def sequential_affine_broadcast(plan, matrices, offsets, root_value):
    result = np.zeros((plan.num_nodes, root_value.size), dtype=np.float32)
    result[int(plan.root)] = root_value
    children = [[] for _ in range(plan.num_nodes)]
    for edge, (parent, child) in enumerate(
        zip(plan.edge_parents, plan.edge_children, strict=True)
    ):
        children[int(parent)].append((edge, int(child)))
    stack = [int(plan.root)]
    while stack:
        parent = stack.pop()
        for edge, child in children[parent]:
            result[child] = matrices[edge] @ result[parent] + offsets[edge]
            stack.append(child)
    return result


@pytest.mark.parametrize(
    "parents",
    [
        [-1],
        [-1, 0, 1, 2, 3, 4, 5, 6, 7],
        [-1, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 1, 1, 3, 2, 6, 6, 8, 3],
    ],
)
@pytest.mark.parametrize("schedule", list(ContractionSchedule))
def test_affine_broadcast_recovery(parents, schedule) -> None:
    dimension = 3
    plan = make_tree_contraction_plan(parents, schedule=schedule)
    key = jax.random.key(len(parents))
    matrix_key, offset_key, root_key = jax.random.split(key, 3)
    matrices = 0.15 * jax.random.normal(
        matrix_key, (plan.num_edges, dimension, dimension)
    )
    matrices += jnp.eye(dimension)
    offsets = jax.random.normal(offset_key, (plan.num_edges, dimension))
    root_value = jax.random.normal(root_key, (dimension,))
    nodes = jnp.zeros(plan.num_nodes, dtype=jnp.float32)
    algebra = AffineBroadcastAlgebra()

    def run(node_values, paths, root):
        _, tape = tree_contract(plan, node_values, paths, algebra)
        return tree_expand(plan, tape, root, algebra)

    actual = jax.jit(run)(nodes, (matrices, offsets), root_value)
    expected = sequential_affine_broadcast(
        plan,
        np.asarray(matrices),
        np.asarray(offsets),
        np.asarray(root_value),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("schedule", list(ContractionSchedule))
def test_affine_broadcast_is_differentiable(schedule) -> None:
    plan = make_tree_contraction_plan([-1, 0, 1, 1, 3, 0], schedule=schedule)
    dimension = 2
    matrices = jnp.tile(jnp.eye(dimension)[None, :, :], (plan.num_edges, 1, 1))
    offsets = jnp.ones((plan.num_edges, dimension))
    nodes = jnp.zeros(plan.num_nodes)
    algebra = AffineBroadcastAlgebra()

    def objective(root):
        _, tape = tree_contract(plan, nodes, (matrices, offsets), algebra)
        outputs = tree_expand(plan, tape, root, algebra)
        return outputs.sum()

    gradient = jax.jit(jax.grad(objective))(jnp.array([2.0, 3.0]))
    np.testing.assert_allclose(
        gradient, jnp.full((dimension,), plan.num_nodes, dtype=jnp.float32)
    )


@pytest.mark.parametrize(
    ("executor", "schedule"),
    (
        (ContractionExecutor.SCAN, ContractionSchedule.RAKE_ONLY),
        (ContractionExecutor.ASSOCIATIVE_SCAN, ContractionSchedule.RAKE_COMPRESS),
    ),
)
def test_chain_executors_recover_affine_broadcast(executor, schedule) -> None:
    plan = make_tree_contraction_plan(
        [2, 3, 1, -1], schedule=schedule, executor=executor
    )
    dimension = 3
    key = jax.random.key(17)
    matrix_key, offset_key, root_key = jax.random.split(key, 3)
    matrices = jnp.eye(dimension)[None, :, :] + 0.15 * jax.random.normal(
        matrix_key, (plan.num_edges, dimension, dimension)
    )
    offsets = jax.random.normal(offset_key, (plan.num_edges, dimension))
    root_value = jax.random.normal(root_key, (dimension,))
    nodes = jnp.zeros(plan.num_nodes, dtype=jnp.float32)
    algebra = AffineBroadcastAlgebra()

    def run(node_values, paths, root):
        _, tape = tree_contract(plan, node_values, paths, algebra)
        return tree_expand(plan, tape, root, algebra)

    actual = jax.jit(run)(nodes, (matrices, offsets), root_value)
    expected = sequential_affine_broadcast(
        plan,
        np.asarray(matrices),
        np.asarray(offsets),
        np.asarray(root_value),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
