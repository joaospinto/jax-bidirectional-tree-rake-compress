"""Microbenchmark generic affine contraction and reverse expansion."""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import jax_bidirectional_tree_rake_compress as jtrc


@dataclass(frozen=True)
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


def chain(nodes: int) -> list[int]:
    return [-1, *range(nodes - 1)]


def balanced(nodes: int) -> list[int]:
    return [-1, *((node - 1) // 2 for node in range(1, nodes))]


def star(nodes: int) -> list[int]:
    return [-1, *([0] * (nodes - 1))]


def comb(nodes: int) -> list[int]:
    """A long spine with side leaves, truncated to exactly ``nodes``."""

    parents = [-1]
    spine = 0
    while len(parents) < nodes:
        next_spine = len(parents)
        parents.append(spine)
        if len(parents) < nodes:
            parents.append(spine)
        spine = next_spine
    return parents


def delayed_star(nodes: int) -> list[int]:
    """Root paths in groups whose messages become ready at different rounds."""
    if nodes == 1:
        return [-1]
    groups = max(1, int(math.log2(nodes)) // 2)
    nodes_per_width = 2**groups - 1
    width = max(1, (nodes - 1) // nodes_per_width)
    parents = [-1]
    for group in range(groups):
        for _ in range(width):
            parent = 0
            for _ in range(2**group):
                parents.append(parent)
                parent = len(parents) - 1
    parents.extend([0] * (nodes - len(parents)))
    return parents


TOPOLOGIES: dict[str, Callable[[int], list[int]]] = {
    "chain": chain,
    "balanced": balanced,
    "comb": comb,
    "delayed-star": delayed_star,
    "star": star,
}

MODES = {
    "rake-compress": (
        jtrc.ContractionSchedule.RAKE_COMPRESS,
        jtrc.ContractionExecutor.UNROLLED,
    ),
    "rake-only": (
        jtrc.ContractionSchedule.RAKE_ONLY,
        jtrc.ContractionExecutor.UNROLLED,
    ),
    "scan": (
        jtrc.ContractionSchedule.RAKE_ONLY,
        jtrc.ContractionExecutor.SCAN,
    ),
    "associative-scan": (
        jtrc.ContractionSchedule.RAKE_COMPRESS,
        jtrc.ContractionExecutor.ASSOCIATIVE_SCAN,
    ),
}


def block(result) -> None:
    jax.tree.leaves(result)[0].block_until_ready()


def measure(function, arguments, repeats: int) -> tuple[float, float]:
    start = time.perf_counter()
    first = function(*arguments)
    block(first)
    first_ms = 1e3 * (time.perf_counter() - start)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*arguments)
        block(result)
        samples.append(1e3 * (time.perf_counter() - start))
    return first_ms, statistics.median(samples)


def benchmark_case(
    topology: str,
    mode: str,
    nodes: int,
    dimension: int,
    repeats: int,
) -> tuple[float, int, int, float, float]:
    start = time.perf_counter()
    schedule, executor = MODES[mode]
    plan = jtrc.make_tree_contraction_plan(
        TOPOLOGIES[topology](nodes), schedule=schedule, executor=executor
    )
    setup_ms = 1e3 * (time.perf_counter() - start)
    stats = jtrc.plan_statistics(plan)
    key = jax.random.key(nodes + dimension)
    matrix_key, offset_key, root_key = jax.random.split(key, 3)
    matrices = jnp.eye(dimension)[None, :, :] + 0.01 * jax.random.normal(
        matrix_key, (plan.num_edges, dimension, dimension)
    )
    offsets = jax.random.normal(offset_key, (plan.num_edges, dimension))
    root = jax.random.normal(root_key, (dimension,))
    dummy_nodes = jnp.zeros(plan.num_nodes, dtype=jnp.float32)
    algebra = AffineBroadcast()

    @jax.jit
    def run(node_values, path_values, root_value):
        _, tape = jtrc.tree_contract(plan, node_values, path_values, algebra)
        return jtrc.tree_expand(plan, tape, root_value, algebra)

    first_ms, execution_ms = measure(
        run, (dummy_nodes, (matrices, offsets), root), repeats
    )
    return (
        setup_ms,
        stats.num_rounds,
        stats.num_operation_levels,
        first_ms,
        execution_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--topologies", nargs="+", choices=TOPOLOGIES, default=list(TOPOLOGIES)
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=["rake-compress"],
    )
    args = parser.parse_args()

    print("device:", jax.devices()[0])
    print("dimension:", args.dimension)
    print()
    header = (
        f"{'mode':<18} {'topology':<12} {'nodes':>7} "
        f"{'rounds':>7} {'levels':>7} "
        f"{'setup ms':>11} {'compile+first ms':>18} {'execute ms':>12}"
    )
    print(header)
    print("-" * len(header))
    for nodes in args.nodes:
        for topology in args.topologies:
            for mode in args.modes:
                if mode in ("scan", "associative-scan") and topology != "chain":
                    continue
                setup_ms, rounds, levels, first_ms, execution_ms = benchmark_case(
                    topology, mode, nodes, args.dimension, args.repeats
                )
                print(
                    f"{mode:<18} {topology:<12} {nodes:>7} "
                    f"{rounds:>7} {levels:>7} {setup_ms:>11.3f} "
                    f"{first_ms:>18.3f} {execution_ms:>12.3f}"
                )


if __name__ == "__main__":
    main()
