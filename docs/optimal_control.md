# Mapping to tree-structured optimal control

The package was designed to express the three tree passes needed by parallel
Riccati factorization and solve without making optimal control part of the
generic API.

## Quadratic factorization

Use a node summary for a completed unary quadratic and a path summary for a
two-boundary conditional quadratic:

```text
Node = (P, p)                    # or P in a quadratic-only factor pass
Path = (P_left, p_left, C, A, c)
```

- `rake` terminalizes the path with the completed child value and produces a
  unary contribution at the parent.
- `combine_branches` and `absorb_branch` add unary quadratics.
- `compress` eliminates the shared state between adjacent conditional values.
- the compression residual retains the right interval needed to recover the
  removed middle-node value after the child value is known.

Expansion then reconstructs every intermediate `P_i` or `(P_i, p_i)`.

## Affine right-hand-side solve

With the quadratic factorization fixed, a path carries an affine message

```text
p_parent contribution = Z @ p_child + z.
```

Rake applies this message to a completed child. Compress is ordinary affine
composition. Reverse expansion recovers the affine value coefficient at
compressed middle nodes.

## State down-sweep

Each original edge carries a local transition

```text
x_child = T @ x_parent + b.
```

For this pass, raking has no parent contribution. Compress composes adjacent
transition maps and saves the original left map. Once the root state is
solved, expansion applies saved maps to recover compressed nodes and raked
leaves in dependency order.

This is demonstrated abstractly by `examples/affine_broadcast.py` and tested
on chains, stars, and irregular trees.

## Expected integration structure

```python
plan = make_tree_contraction_plan(parents)  # once

# Repeated when the quadratic data changes.
root_P, factor_tape = tree_contract(plan, nodes_P, paths_P, quadratic_alg)
all_P = tree_expand(plan, factor_tape, root_P, quadratic_alg)

# Repeated for right-hand sides with the same factorization.
root_p, solve_tape = tree_contract(plan, nodes_p, paths_affine, solve_alg)
all_p = tree_expand(plan, solve_tape, root_p, solve_alg)

# Root solve followed by a parallel down-sweep.
_, transition_tape = tree_contract(
    plan, dummy_nodes, local_transitions, transition_alg
)
all_x = tree_expand(plan, transition_tape, x_root, transition_alg)
```

The topology arrays are reusable across every pass. Only numerical summaries
and residual tapes reside in factor/solve calls.

For a chain, use `schedule=RAKE_ONLY` with `executor=SCAN` for a sequential
solve, or `schedule=RAKE_COMPRESS` with `executor=ASSOCIATIVE_SCAN` for a
parallel solve. These executor choices only change JAX control flow; branching
OCPs continue to use the unrolled executor.
