# MeshCFD performance baseline

These benchmarks define Phase 0 of the optimization work. They measure geometry construction, one finite-volume residual evaluation, and a complete solver run separately.

Run the smoke benchmark from the repository root:

```bash
PYTHONPATH=src python -m benchmarks.run_benchmarks --quick --iterations 10 --repeats 2
```

Run the full baseline:

```bash
PYTHONPATH=src python -m benchmarks.run_benchmarks \
  --iterations 50 --repeats 5 --output benchmark_results.json
```

Create a `cProfile` file for the first selected case:

```bash
PYTHONPATH=src python -m benchmarks.run_benchmarks \
  --quick --iterations 50 --profile meshcfd.prof
python -m pstats meshcfd.prof
```

## Comparison protocol

1. Use the same machine, Python environment, power mode, and process affinity.
2. Close the GUI and avoid VTU export during measurements.
3. Run at least five repetitions and compare medians.
4. Retain JSON results from the unmodified baseline.
5. Accept a change only when numerical regression tests pass and the representative case improves without a material regression in secondary cases.

The primary speedup is computed as:

```text
speedup = baseline median solve time / candidate median solve time
```

The Phase 0 target is measurement, not acceleration. Later phases should target at least 4.0x on the agreed representative case.
