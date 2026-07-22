# Design and algebraic contract

This document describes the generic executor independently of any particular
numerical application.

## 1. Static topology and dynamic payloads

`make_tree_contraction_plan` accepts one parent index per original node. It
validates that the indices describe one directed rooted tree, creates one edge
slot per non-root node, and repeatedly plans:

1. all currently active leaf rakes;
2. a conflict-free set of unary-node compressions;
3. balanced reductions for sibling rake messages.

The result contains only integer JAX arrays. Planning uses NumPy and ordinary
Python data structures; it is never called by `tree_contract` or
`tree_expand`.

A round stores records with these schemas:

```text
rake:       edge, parent, leaf
compress:   middle, left edge, right edge, parent, child
```

The particular schedule affects floating-point parenthesization and runtime,
but a lawful algebra gives the same exact-arithmetic result for every legal
schedule.

The rake--compress planner also expands the rounds into four kinds of primitive
operation and records the latest producer of each node, path, and branch slot.
Each primitive is placed one level after its latest input. When this cuts the
synchronous primitive span by more than half, the plan uses those levels
internally. This removes global sibling-reduction barriers: an unrelated
subtree can advance while a high-degree parent is still combining messages.
Operations within a level read the same input state and have disjoint writes.

## 2. Active-tree invariant

At every forward stage:

- each active node owns one `Node` summary;
- each active edge owns one oriented `Path` summary between its active
  endpoints;
- active summaries partition all original contributions already represented;
- an inactive node has been removed exactly once by rake or compress.

The executor stores numerical values in immutable JAX arrays and realizes
logical mutation with indexed updates. Removed slots may remain physically
present but are never read as active summaries again.

Internally dependency-leveled execution retains one branch slot per rake. This
linear workspace allows a message to survive across levels until its
deterministic reduction and absorption operations become ready.

## 3. Forward algebra

The user supplies four operations.

### 3.1 Rake

```python
message, residual = algebra.rake(path, leaf)
```

`leaf` is complete relative to the active tree. `path` connects its active
parent to that leaf. The result is a branch message expressed at the parent
boundary and an arbitrary fixed-shape residual for later expansion.

### 3.2 Branch reduction

```python
combined = algebra.combine_branches(left, right)
```

Messages targeting the same parent are reduced in a balanced order. The
operation must be associative. To make results independent of sibling order
and contraction schedule it should also be commutative.

No simultaneous writes target the parent. After reduction, exactly one update
is applied:

```python
node = algebra.absorb_branch(node, combined)
```

`absorb_branch` must be compatible with message combination. Informally,

```text
absorb(absorb(node, a), b) = absorb(node, combine(a, b)).
```

### 3.3 Compress

```python
path, residual = algebra.compress(left, middle, right)
```

The two paths share the active unary node `middle`. The operation eliminates
that boundary and returns a path from the old parent boundary to the old child
boundary. Concurrent compressions use disjoint
`(parent, middle, child)` triples.

Path composition is generally oriented and need not be commutative.

## 4. The contraction tape

`tree_contract` retains the rake and compression residual PyTrees for each
round. Residuals are chosen by the algebra, not the executor. Examples include:

- a completed leaf value;
- an interval adjacent to a removed middle node;
- a local affine transition map;
- factorizations needed to reconstruct an eliminated variable.

Returning no numerical residual is allowed by using an empty tuple or `None`.
JAX dead-code elimination can remove unused residual calculations when a
caller uses only `tree_reduce`.

The tape has linear aggregate cardinality because every non-root node is
removed once.

## 5. Reverse expansion

Expansion begins with an application-defined output at the root. It visits
rounds backward and invokes:

```python
middle_output = algebra.expand_compress(
    residual, parent_output, child_output
)

leaf_output = algebra.expand_rake(residual, parent_output)
```

Within a reversed synchronous round, compressions are undone before rakes
because the forward order was rake then compress. This ordering matters: the
parent of a raked leaf may itself have been compressed later in that forward
round. In an internally dependency-leveled plan, true dependencies put those operations in
different levels; independent rake and compression recoveries within one
reversed level read the same boundary-output state and write distinct nodes.

Both surviving boundary outputs are offered to `expand_compress`. An algebra
may ignore either one.

The same traversal supports two superficially different recoveries:

- upward recovery, where a compressed node is reconstructed from its child
  boundary value;
- downward broadcast, where a removed node is reconstructed from its parent
  boundary solution.

## 6. PyTrees and shapes

`Node`, `Path`, messages, residuals, and outputs may each be arbitrary PyTrees
of JAX arrays. The executor vectorizes each local callback with `jax.vmap`.

For a given role:

- PyTree structure is fixed;
- leaf rank, dtype, and non-leading dimensions are fixed;
- the executor adds or consumes the leading operation axis.

This model deliberately follows JAX's static-shape compilation discipline.
Ragged numerical dimensions must be padded or bucketed by shape.

## 7. JIT structure

`TreeContractionPlan` is itself a PyTree. Its tuple of rounds is structural,
so an ordinary Python loop over rounds is unrolled while tracing. Work within
each round remains batched: compiled program size grows with contraction depth,
not with the number of nodes.

Plans can be used in two ways:

```python
# Specialize to one topology.
compiled = jax.jit(lambda nodes, paths: tree_reduce(plan, nodes, paths, alg))

# Pass index values dynamically. Plans with identical leaf shapes can reuse
# the executable.
compiled = jax.jit(lambda p, nodes, paths: tree_reduce(p, nodes, paths, alg))
```

## 8. Differentiation

The implementation contains only JAX operations and therefore receives JVP
and VJP behavior by composition. Expansion is application-level recovery; it
is not an autodiff backward pass.

A future optimized implementation could define a custom derivative that uses
the same reverse topology while propagating cotangents. That is an orthogonal
layer and should not change the public contraction algebra.
