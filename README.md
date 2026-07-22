# JAX Bidirectional Tree Rake–Compress

Generic, topology-independent rake–compress tree contraction and reverse
recovery for JAX.

The package separates a one-time CPU planning step from repeated numerical
execution:

```text
parents ──CPU──> TreeContractionPlan
                         │
node/path arrays ──JAX───┼──> root summary + residual tape
                         │
root output ─────────────┴──> output at every original node
```

The numerical meaning of a contraction is supplied by a small user-defined
algebra. The same topology plan can therefore evaluate subtree reductions,
conditional value functions, probabilistic messages, affine maps, and other
fixed-shape tree computations.

## Why bidirectional?

An ordinary tree reduction returns only a root value. Many applications also
need quantities at the nodes eliminated during the reduction. For example,
parallel optimal-control factorization must recover every intermediate value
function, and its solve must propagate a root solution back to all states.

`tree_contract` therefore returns both a root summary and a residual tape.
`tree_expand` visits the same plan backward and uses application-defined
recovery operations to reconstruct one output per original node.

## Installation

Install from PyPI:

```sh
python -m pip install jax-bidirectional-tree-rake-compress
```

From a source checkout:

```sh
python -m pip install -e .
```

For development:

```sh
python -m pip install -e '.[dev]'
pytest
```

The package uses whichever JAX accelerator installation is available. See the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for
platform-specific GPU or TPU wheels.

## Quick start: subtree sums

Each node starts with a local scalar. An edge path starts at zero. Raking and
compressing accumulate completed subtrees, while expansion reconstructs the
subtree sum at every node.

```python
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
plan = jtrc.make_tree_contraction_plan(parents)  # CPU, once

node_values = jnp.arange(1, 8, dtype=jnp.float32)
edge_values = jnp.zeros(plan.num_edges, dtype=jnp.float32)
algebra = SubtreeSum()


@jax.jit
def run(nodes, edges):
    root, tape = jtrc.tree_contract(plan, nodes, edges, algebra)
    return root, jtrc.tree_expand(plan, tape, root, algebra)


root, subtree_sums = run(node_values, edge_values)
```

The result is

```text
root = 28
subtree_sums = [28, 17, 10, 10, 5, 6, 7]
```

All numerical payloads may be PyTrees of arrays. Every leaf needs a leading
node or edge axis, and every value in a role must have the same static shape.

## Public API

- `make_tree_contraction_plan(parents, root=None, schedule=..., executor=...)` performs
  validated CPU preprocessing and returns a JAX PyTree of integer schedule
  arrays. Rake--compress remains the default and rake-only traversal is an
  opt-in policy. Executors change only the JAX control-flow representation.
- `tree_contract(plan, nodes, paths, algebra)` returns the root summary and a
  numerical recovery tape.
- `tree_reduce(...)` returns only the root summary.
- `tree_expand(plan, tape, root_output, algebra)` reverses the schedule.
- `tree_contract_and_expand(...)` composes contraction, root work, and
  expansion.
- `plan_statistics(plan)` reports topology counts without copying schedule
  values back from a device.

The plan orders input path summaries by `plan.edge_children`: one incoming
edge for each non-root node. `plan.edge_parents` gives the corresponding
parent indices.

## JAX transformations

The executor is written in pure JAX. It can be used inside `jax.jit`, mapped
over batches with `jax.vmap`, and differentiated with `jax.grad`, `jax.jvp`,
or `jax.vjp`. The topology indices are discrete and are not differentiated.

Plans can either be closed over by a compiled function or passed as ordinary
PyTree arguments. Closing over a plan specializes compilation to that
topology; passing it as an argument permits reuse by plans with the same round
and array shapes.

## Algorithmic notes

Every non-root node is removed exactly once by either:

- **rake:** close an active leaf component and send a branch message to its
  parent;
- **compress:** eliminate a unary middle node and join its two oriented path
  summaries.

Sibling rakes never race. Messages targeting the same parent are combined by
a CPU-precomputed balanced reduction before the parent is updated. This makes
the generic implementation independent of atomic addition and supports any
compatible associative branch operation.

For rake--compress, planning compares synchronous rounds with the earliest
legal static levels of the same rake, branch-combination, branch-absorption,
and compression operations. Dependency levels are used internally only when
they remove more than half of the synchronous primitive span. This lets work
in unrelated subtrees proceed without waiting for the largest sibling
reduction while retaining the lower-overhead round representation on ordinary
trees.

Chains also support loop-based executors without introducing another
contraction policy. `ContractionExecutor.SCAN` executes `RAKE_ONLY` through
`jax.lax.scan`; `ContractionExecutor.ASSOCIATIVE_SCAN` executes
`RAKE_COMPRESS` through `jax.lax.associative_scan`. Each executor rejects the
other schedule and any branching topology. `AUTO` selects those mappings on a
chain and `UNROLLED` otherwise. The default `UNROLLED` executor continues to
support both schedules on every rooted tree.

For bounded-degree trees, the plan has linear work and logarithmic contraction
depth. High-degree branch reductions are balanced as well; actual device
kernel count and fusion depend on JAX and XLA.

See [the design document](docs/design.md) for the precise invariants and
[the optimal-control mapping](docs/optimal_control.md) for the intended LQR
application.

## Current limitations

- Numerical roles have uniform static shapes. Variable-dimensional tree
  problems need padding, dimension bucketing, or separate compiled plans.
- The planner is deterministic but does not optimize a hardware cost model.
- A pure-JAX call can lower to multiple device kernels; this API does not
  promise one dispatch.
- The current planner is for static rooted trees. Dynamic topology updates are
  out of scope.

## Development

```sh
ruff check .
ruff format --check .
pytest
python -m build
python -m twine check dist/*
```

Benchmarks are described in [benchmarks/README.md](benchmarks/README.md).

## License

MIT
