# Contributing

Create an isolated environment and install the development dependencies:

```sh
python -m pip install -e '.[dev]'
```

Before submitting a change, run:

```sh
ruff check .
ruff format --check .
pytest
python -m build
python -m twine check dist/*
```

Changes to the scheduler should add property tests covering chains, stars,
balanced trees, and irregular trees. Changes to the algebra API should include
at least one upward-recovery and one downward-broadcast example.

## Releasing

1. Update the version in `pyproject.toml` and add the corresponding dated entry
   to `CHANGELOG.md`.
2. Run the complete verification commands above and commit the release.
3. Create a GitHub Release whose tag is `v<version>` (for example, `v0.1.0`).

The distribution workflow builds the source and wheel archives, validates
them with Twine, installs the wheel on Linux, macOS, and Windows, verifies that
the release tag matches the wheel metadata, and publishes to PyPI using Trusted
Publishing. The GitHub `pypi` environment and PyPI Trusted Publisher must be
configured for this repository before the first release.
