from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_bidirectional_tree_rake_compress import (
    make_tree_contraction_plan,
    tree_contract,
    tree_contract_and_expand,
    tree_expand,
    tree_reduce,
)


@dataclass(frozen=True)
class SubtreeSumAlgebra:
    def rake(self, path, leaf):
        return path + leaf, leaf

    def combine_branches(self, left, right):
        return left + right

    def absorb_branch(self, node, message):
        return node + message

    def compress(self, left, middle, right):
        return left + middle + right, (middle, right)

    def expand_compress(self, residual, parent_output, child_output):
        del parent_output
        middle, right = residual
        return middle + right + child_output

    def expand_rake(self, residual, parent_output):
        del parent_output
        return residual


def sequential_subtree_sums(parents, values, root):
    children = [[] for _ in parents]
    for child, parent in enumerate(parents):
        if child != root:
            children[parent].append(child)

    result = np.zeros_like(values)

    def visit(node):
        result[node] = values[node]
        for child in children[node]:
            result[node] += visit(child)
        return result[node]

    visit(root)
    return result


TOPOLOGIES = [
    [-1],
    [-1, 0],
    [-1, 0, 1, 2, 3, 4, 5, 6],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 1, 1, 3, 2, 6, 6, 8, 3],
    [4, 4, 0, 1, -1, 1, 5, 2, 2],
]


@pytest.mark.parametrize("parents", TOPOLOGIES)
def test_contract_and_recover_subtree_sums(parents) -> None:
    plan = make_tree_contraction_plan(parents)
    values = jnp.arange(1, len(parents) + 1, dtype=jnp.float32)
    paths = jnp.zeros(plan.num_edges, dtype=jnp.float32)
    algebra = SubtreeSumAlgebra()

    compiled = jax.jit(
        lambda node_values, edge_values: tree_contract(
            plan, node_values, edge_values, algebra
        )
    )
    root, tape = compiled(values, paths)
    recovered = jax.jit(
        lambda contraction_tape, root_value: tree_expand(
            plan, contraction_tape, root_value, algebra
        )
    )(tape, root)

    expected = sequential_subtree_sums(
        np.asarray(parents), np.asarray(values), int(plan.root)
    )
    np.testing.assert_allclose(root, expected[int(plan.root)])
    np.testing.assert_allclose(recovered, expected)


def test_plan_can_be_a_dynamic_jitted_argument() -> None:
    plan = make_tree_contraction_plan([-1, 0, 0, 1, 1, 2, 5])
    values = jnp.arange(plan.num_nodes, dtype=jnp.float32)
    paths = jnp.zeros(plan.num_edges, dtype=jnp.float32)

    compiled = jax.jit(
        lambda dynamic_plan, nodes, edges: tree_reduce(
            dynamic_plan, nodes, edges, SubtreeSumAlgebra()
        )
    )

    np.testing.assert_allclose(compiled(plan, values, paths), values.sum())


def test_vmap_and_grad() -> None:
    plan = make_tree_contraction_plan([-1, 0, 0, 1, 1, 2, 5])
    paths = jnp.zeros(plan.num_edges, dtype=jnp.float32)
    batch = jnp.arange(3 * plan.num_nodes, dtype=jnp.float32).reshape(3, plan.num_nodes)

    def reduce_one(values):
        return tree_reduce(plan, values, paths, SubtreeSumAlgebra())

    roots = jax.jit(jax.vmap(reduce_one))(batch)
    gradient = jax.jit(jax.grad(reduce_one))(batch[0])

    np.testing.assert_allclose(roots, batch.sum(axis=1))
    np.testing.assert_allclose(gradient, jnp.ones(plan.num_nodes))


def test_contract_and_expand_convenience_function() -> None:
    plan = make_tree_contraction_plan([-1, 0, 1, 1, 0])
    values = jnp.arange(1, 6, dtype=jnp.float32)
    paths = jnp.zeros(plan.num_edges, dtype=jnp.float32)

    summary, root_output, outputs = tree_contract_and_expand(
        plan,
        values,
        paths,
        SubtreeSumAlgebra(),
        finish_root=lambda root: 2 * root,
    )

    assert float(summary) == 15.0
    assert float(root_output) == 30.0
    # Expansion semantics belong to the algebra. This algebra recovers subtree
    # sums, so only the externally supplied root output is doubled.
    assert float(outputs[int(plan.root)]) == 30.0


def test_leading_axis_validation() -> None:
    plan = make_tree_contraction_plan([-1, 0, 0])

    with pytest.raises(ValueError, match="expected 3"):
        tree_reduce(
            plan,
            jnp.ones(2),
            jnp.zeros(2),
            SubtreeSumAlgebra(),
        )


def test_random_permuted_trees() -> None:
    rng = np.random.default_rng(42)
    algebra = SubtreeSumAlgebra()

    for num_nodes in (3, 7, 16, 31, 64):
        for _ in range(4):
            ordered_parents = np.array(
                [-1, *(rng.integers(0, node) for node in range(1, num_nodes))]
            )
            permutation = rng.permutation(num_nodes)
            inverse = np.empty(num_nodes, dtype=np.int64)
            inverse[permutation] = np.arange(num_nodes)
            parents = np.full(num_nodes, -1, dtype=np.int64)
            for old_child in range(1, num_nodes):
                parents[inverse[old_child]] = inverse[ordered_parents[old_child]]

            plan = make_tree_contraction_plan(parents)
            values = rng.normal(size=num_nodes).astype(np.float32)
            paths = jnp.zeros(plan.num_edges, dtype=jnp.float32)
            root, tape = tree_contract(plan, jnp.asarray(values), paths, algebra)
            recovered = tree_expand(plan, tape, root, algebra)
            expected = sequential_subtree_sums(parents, values, int(plan.root))

            np.testing.assert_allclose(root, expected[int(plan.root)], atol=1e-5)
            np.testing.assert_allclose(recovered, expected, atol=1e-5)
