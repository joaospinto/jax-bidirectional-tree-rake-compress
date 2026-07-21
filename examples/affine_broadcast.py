"""Broadcast a root vector through affine maps on an irregular tree."""

import jax
import jax.numpy as jnp

import jax_bidirectional_tree_rake_compress as jtrc


class AffineBroadcast:
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
        return (
            right_matrix @ left_matrix,
            right_matrix @ left_offset + right_offset,
        ), left

    def expand_compress(self, residual, parent_output, child_output):
        del child_output
        matrix, offset = residual
        return matrix @ parent_output + offset

    def expand_rake(self, residual, parent_output):
        matrix, offset = residual
        return matrix @ parent_output + offset


parents = [-1, 0, 0, 1, 1, 3, 2]
plan = jtrc.make_tree_contraction_plan(parents)
dimension = 2
matrices = jnp.tile(jnp.eye(dimension)[None, :, :], (plan.num_edges, 1, 1))
offsets = jnp.arange(plan.num_edges * dimension, dtype=jnp.float32).reshape(
    plan.num_edges, dimension
)
dummy_nodes = jnp.zeros(plan.num_nodes, dtype=jnp.float32)
root = jnp.array([1.0, -1.0])
algebra = AffineBroadcast()


@jax.jit
def run(node_values, path_values, root_value):
    _, tape = jtrc.tree_contract(plan, node_values, path_values, algebra)
    return jtrc.tree_expand(plan, tape, root_value, algebra)


print(run(dummy_nodes, (matrices, offsets), root))
