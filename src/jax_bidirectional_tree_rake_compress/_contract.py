"""Pure-JAX execution of a precomputed bidirectional contraction plan."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

import jax
import jax.numpy as jnp

from ._algebra import TreeContractionAlgebra
from ._plan import TreeContractionPlan

PyTree = Any
RootOutput = TypeVar("RootOutput")


class RoundTape(NamedTuple):
    """Numerical residuals produced by one contraction round."""

    rake: PyTree
    compress: PyTree


class ContractionTape(NamedTuple):
    """Residuals consumed by :func:`tree_expand` in reverse round order."""

    rounds: tuple[RoundTape, ...]


def _take(tree: PyTree, indices: jax.Array) -> PyTree:
    return jax.tree.map(lambda value: value[indices], tree)


def _set(tree: PyTree, indices: jax.Array, values: PyTree) -> PyTree:
    return jax.tree.map(
        lambda destination, source: destination.at[indices].set(source),
        tree,
        values,
    )


def _validate_leading_axis(name: str, tree: PyTree, size: int) -> None:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise ValueError(f"{name} must contain at least one array leaf")
    for leaf in leaves:
        if not hasattr(leaf, "shape") or not leaf.shape:
            raise ValueError(f"every {name} leaf must have a leading axis")
        if leaf.shape[0] != size:
            raise ValueError(
                f"{name} has leading size {leaf.shape[0]}, expected {size}"
            )


def tree_contract(
    plan: TreeContractionPlan,
    node_summaries: PyTree,
    path_summaries: PyTree,
    algebra: TreeContractionAlgebra[Any, Any, Any, Any, Any, Any],
) -> tuple[PyTree, ContractionTape]:
    """Contract a rooted tree to its root using a user-defined algebra.

    This function is deliberately not decorated with :func:`jax.jit`; callers
    can close over ``plan`` and ``algebra`` or pass the plan as a PyTree to a
    larger compiled computation.
    """

    _validate_leading_axis("node_summaries", node_summaries, plan.num_nodes)
    _validate_leading_axis("path_summaries", path_summaries, plan.num_edges)

    nodes = node_summaries
    paths = path_summaries
    tape: list[RoundTape] = []

    for round_ in plan.rounds:
        if round_.rakes.shape[0]:
            rake_edges = round_.rakes[:, 0]
            rake_leaves = round_.rakes[:, 2]
            messages, rake_residual = jax.vmap(algebra.rake)(
                _take(paths, rake_edges),
                _take(nodes, rake_leaves),
            )

            for pairs in round_.rake_reduction_stages:
                destinations = pairs[:, 0]
                sources = pairs[:, 1]
                reduced = jax.vmap(algebra.combine_branches)(
                    _take(messages, destinations),
                    _take(messages, sources),
                )
                messages = _set(messages, destinations, reduced)

            parent_messages = _take(messages, round_.rake_roots)
            updated_parents = jax.vmap(algebra.absorb_branch)(
                _take(nodes, round_.rake_parents), parent_messages
            )
            nodes = _set(nodes, round_.rake_parents, updated_parents)
        else:
            rake_residual = ()

        if round_.compressions.shape[0]:
            middle = round_.compressions[:, 0]
            left_edges = round_.compressions[:, 1]
            right_edges = round_.compressions[:, 2]
            compressed, compress_residual = jax.vmap(algebra.compress)(
                _take(paths, left_edges),
                _take(nodes, middle),
                _take(paths, right_edges),
            )
            paths = _set(paths, left_edges, compressed)
        else:
            compress_residual = ()

        tape.append(RoundTape(rake_residual, compress_residual))

    return _take(nodes, plan.root), ContractionTape(tuple(tape))


def tree_reduce(
    plan: TreeContractionPlan,
    node_summaries: PyTree,
    path_summaries: PyTree,
    algebra: TreeContractionAlgebra[Any, Any, Any, Any, Any, Any],
) -> PyTree:
    """Contract a tree and return only the root summary."""

    root, _ = tree_contract(plan, node_summaries, path_summaries, algebra)
    return root


def tree_expand(
    plan: TreeContractionPlan,
    tape: ContractionTape,
    root_output: PyTree,
    algebra: TreeContractionAlgebra[Any, Any, Any, Any, Any, Any],
) -> PyTree:
    """Reverse a contraction and recover one output per original node."""

    if len(tape.rounds) != len(plan.rounds):
        raise ValueError("tape and plan have different numbers of contraction rounds")

    outputs = jax.tree.map(
        lambda value: (
            jnp.zeros((plan.num_nodes, *value.shape), dtype=value.dtype)
            .at[plan.root]
            .set(value)
        ),
        root_output,
    )

    for round_, round_tape in zip(
        reversed(plan.rounds), reversed(tape.rounds), strict=True
    ):
        # Forward order is rake then compress, so expansion reverses compress
        # before rake. A raked leaf's parent may have been compressed in the
        # same forward round.
        if round_.compressions.shape[0]:
            middle = round_.compressions[:, 0]
            parents = round_.compressions[:, 3]
            children = round_.compressions[:, 4]
            middle_outputs = jax.vmap(algebra.expand_compress)(
                round_tape.compress,
                _take(outputs, parents),
                _take(outputs, children),
            )
            outputs = _set(outputs, middle, middle_outputs)

        if round_.rakes.shape[0]:
            parents = round_.rakes[:, 1]
            leaves = round_.rakes[:, 2]
            leaf_outputs = jax.vmap(algebra.expand_rake)(
                round_tape.rake,
                _take(outputs, parents),
            )
            outputs = _set(outputs, leaves, leaf_outputs)

    return outputs


def tree_contract_and_expand(
    plan: TreeContractionPlan,
    node_summaries: PyTree,
    path_summaries: PyTree,
    algebra: TreeContractionAlgebra[Any, Any, Any, Any, Any, Any],
    finish_root: Callable[[PyTree], RootOutput],
) -> tuple[PyTree, RootOutput, PyTree]:
    """Convenience composition of contraction, root work, and expansion."""

    root_summary, tape = tree_contract(plan, node_summaries, path_summaries, algebra)
    root_output = finish_root(root_summary)
    outputs = tree_expand(plan, tape, root_output, algebra)
    return root_summary, root_output, outputs
