"""Typing protocols for bidirectional tree-contraction algebras."""

from __future__ import annotations

from typing import Protocol, TypeVar

Node = TypeVar("Node")
Path = TypeVar("Path")
Branch = TypeVar("Branch")
RakeResidual = TypeVar("RakeResidual")
CompressResidual = TypeVar("CompressResidual")
Output = TypeVar("Output")


class TreeContractionAlgebra(
    Protocol[Node, Path, Branch, RakeResidual, CompressResidual, Output]
):
    """Local operations required by :func:`tree_contract` and expansion.

    Every value may be any PyTree of arrays. All values of a given role must
    have the same structure and leaf shapes. Leading batch axes are added by
    the contraction executor.
    """

    def rake(self, path: Path, leaf: Node) -> tuple[Branch, RakeResidual]:
        """Turn a completed leaf component into a parent branch message."""
        ...

    def combine_branches(self, left: Branch, right: Branch) -> Branch:
        """Associatively combine two messages targeting the same parent."""
        ...

    def absorb_branch(self, node: Node, message: Branch) -> Node:
        """Absorb one reduced branch message into an active node."""
        ...

    def compress(
        self, left: Path, middle: Node, right: Path
    ) -> tuple[Path, CompressResidual]:
        """Eliminate a unary middle node between two oriented paths."""
        ...

    def expand_compress(
        self,
        residual: CompressResidual,
        parent_output: Output,
        child_output: Output,
    ) -> Output:
        """Recover a middle-node output while undoing one compression."""
        ...

    def expand_rake(self, residual: RakeResidual, parent_output: Output) -> Output:
        """Recover a leaf-component output while undoing one rake."""
        ...
