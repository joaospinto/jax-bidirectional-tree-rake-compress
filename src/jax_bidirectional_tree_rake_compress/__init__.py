"""Generic bidirectional rake--compress contraction for JAX."""

from importlib.metadata import version as _distribution_version

from ._algebra import TreeContractionAlgebra
from ._contract import (
    ContractionTape,
    RoundTape,
    tree_contract,
    tree_contract_and_expand,
    tree_expand,
    tree_reduce,
)
from ._plan import (
    ContractionRound,
    ContractionSchedule,
    PlanStatistics,
    TreeContractionPlan,
    make_tree_contraction_plan,
    plan_statistics,
)

__version__ = _distribution_version("jax-bidirectional-tree-rake-compress")

__all__ = [
    "ContractionRound",
    "ContractionSchedule",
    "ContractionTape",
    "PlanStatistics",
    "RoundTape",
    "TreeContractionAlgebra",
    "TreeContractionPlan",
    "__version__",
    "make_tree_contraction_plan",
    "plan_statistics",
    "tree_contract",
    "tree_contract_and_expand",
    "tree_expand",
    "tree_reduce",
]
