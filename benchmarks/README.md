# Benchmarks

`benchmark.py` measures three separately reported costs:

- CPU topology setup;
- JAX compilation plus the first execution;
- median steady-state contraction plus reverse expansion.

The payload is a dense affine map with configurable dimension. Available
topologies are chain, balanced binary, comb, star, and a delayed star designed
to expose global sibling-reduction barriers. Rake--compress planning selects
dependency levels internally only when they substantially reduce span.

```sh
python benchmarks/benchmark.py \
  --nodes 256 512 1024 2048 4096 8192 \
  --dimension 8 \
  --repeats 50
```

Use `--modes rake-compress rake-only scan associative-scan` to compare
execution modes. Scan modes are chain-only and imply their corresponding
schedule: ordinary scan is rake-only and associative scan is rake--compress.
The `rounds` column reports logical contraction rounds and `levels` reports
primitive dependency depth; they differ for synchronous plans because one
round can contain several sibling-reduction stages.

Run the benchmark in a fresh process when comparing devices or JAX versions.
JAX execution is asynchronous, so the benchmark blocks on an output leaf for
every timed invocation.

These timings measure the generic executor, not an LQR implementation. An LQR
benchmark should additionally separate factorization, affine solve, root
solve, and state recovery.
