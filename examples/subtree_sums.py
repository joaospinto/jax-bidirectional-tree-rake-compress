"""Recover every subtree sum using bidirectional contraction."""

import jax
import jax.numpy as jnp

import jax_bidirectional_tree_rake_compress as jtrc


class SubtreeSum:
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


parents = [-1, 0, 0, 1, 1, 3, 2]
plan = jtrc.make_tree_contraction_plan(parents)
nodes = jnp.arange(1, 8, dtype=jnp.float32)
paths = jnp.zeros(plan.num_edges, dtype=jnp.float32)
algebra = SubtreeSum()


@jax.jit
def run(node_values, path_values):
    root, tape = jtrc.tree_contract(plan, node_values, path_values, algebra)
    return root, jtrc.tree_expand(plan, tape, root, algebra)


root, subtree_sums = run(nodes, paths)
print("root:", root)
print("subtree sums:", subtree_sums)
