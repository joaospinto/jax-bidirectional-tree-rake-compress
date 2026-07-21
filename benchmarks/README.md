# Benchmarks

`benchmark.py` measures three separately reported costs:

- CPU topology setup;
- JAX compilation plus the first execution;
- median steady-state contraction plus reverse expansion.

The payload is a dense affine map with configurable dimension. Available
topologies are chain, balanced binary, comb, and star.

```sh
python benchmarks/benchmark.py \
  --nodes 256 512 1024 2048 4096 8192 \
  --dimension 8 \
  --repeats 50
```

Run the benchmark in a fresh process when comparing devices or JAX versions.
JAX execution is asynchronous, so the benchmark blocks on an output leaf for
every timed invocation.

These timings measure the generic executor, not an LQR implementation. An LQR
benchmark should additionally separate factorization, affine solve, root
solve, and state recovery.
