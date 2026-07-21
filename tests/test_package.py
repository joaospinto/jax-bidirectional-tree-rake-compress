from __future__ import annotations

from importlib.metadata import version

import jax_bidirectional_tree_rake_compress as jtrc


def test_runtime_and_distribution_versions_match() -> None:
    assert jtrc.__version__ == version("jax-bidirectional-tree-rake-compress")
