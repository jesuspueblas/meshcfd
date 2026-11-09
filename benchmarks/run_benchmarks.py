"""Run repeatable MeshCFD solver benchmarks and write machine-readable results."""

from __future__ import annotations

import argparse
import cProfile
import csv
import json
import pstats
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from meshcfd.core.solver import _build_geometry, _residual, solve_case

from benchmarks.common import (
    BenchmarkSpec,
    conservative_states,
    make_case,
    spec_as_dict,
    system_metadata,
)

DEFAULT_SPECS = (
    BenchmarkSpec("euler_first_order_2500", 50, 50, "First order", "Explicit RK2"),
    BenchmarkSpec("euler_muscl_10000", 100, 100, "MUSCL linear", "Explicit RK2"),
    BenchmarkSpec("euler_muscl_rk4_10000", 100, 100, "MUSCL linear", "Explicit RK4"),
    BenchmarkSpec(
        "laminar_muscl_10000", 100, 100, "MUSCL linear", "Explicit RK2", "Constant"
    ),
    BenchmarkSpec(
        "sst_muscl_10000",
        100,
        100,
        "MUSCL linear",
        "Explicit RK2",
        "Constant",
        "Menter k-omega SST",
    ),
)

QUICK_SPECS = (
    BenchmarkSpec("quick_euler_muscl_1600", 40, 40, "MUSCL linear", "Explicit RK2"),
)


def _measure(operation: Callable[[], object], repeats: int) -> tuple[list[float], object]:
    samples: list[float] = []
    result: object = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - started)
    return samples, result


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
        "mean_seconds": statistics.fmean(samples),
    }


def benchmark_spec(spec: BenchmarkSpec, iterations: int, repeats: int) -> dict[str, object]:
    geometry_samples: list[float] = []
    residual_samples: list[float] = []
    solve_samples: list[float] = []
    final_residual_norm = float("nan")

    for _ in range(repeats):
        case, states = make_case(spec, iterations)
        conservative = conservative_states(states, case.numerics.gamma)

        started = time.perf_counter()
        geometry = _build_geometry(case.mesh, case)
        geometry_samples.append(time.perf_counter() - started)

        # Warm up imported code paths and allocator state before timing the kernel.
        _residual(conservative, geometry, case.numerics)
        started = time.perf_counter()
        derivative = _residual(conservative, geometry, case.numerics)
        residual_samples.append(time.perf_counter() - started)
        final_residual_norm = float(np.linalg.norm(derivative))

        solve_case_obj, solve_states = make_case(spec, iterations)
        started = time.perf_counter()
        result = solve_case(solve_case_obj, solve_states)
        solve_samples.append(time.perf_counter() - started)
        steps = result.steps

    median_solve = statistics.median(solve_samples)
    return {
        "spec": spec_as_dict(spec),
        "iterations": iterations,
        "repeats": repeats,
        "geometry": _summary(geometry_samples),
        "residual": _summary(residual_samples),
        "solve": _summary(solve_samples),
        "steps": steps,
        "cell_steps_per_second": spec.cells * steps / median_solve,
        "final_residual_norm": final_residual_norm,
    }


def write_csv(path: Path, results: list[dict[str, object]]) -> None:
    fields = (
        "name",
        "cells",
        "iterations",
        "repeats",
        "geometry_median_seconds",
        "residual_median_seconds",
        "solve_median_seconds",
        "cell_steps_per_second",
        "final_residual_norm",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            spec = result["spec"]
            writer.writerow(
                {
                    "name": spec["name"],
                    "cells": spec["cells"],
                    "iterations": result["iterations"],
                    "repeats": result["repeats"],
                    "geometry_median_seconds": result["geometry"]["median_seconds"],
                    "residual_median_seconds": result["residual"]["median_seconds"],
                    "solve_median_seconds": result["solve"]["median_seconds"],
                    "cell_steps_per_second": result["cell_steps_per_second"],
                    "final_residual_norm": result["final_residual_norm"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run one small smoke benchmark.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--profile", type=Path, help="Write cProfile data for the first case.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations < 1 or args.repeats < 1:
        raise SystemExit("--iterations and --repeats must be positive")
    specs = QUICK_SPECS if args.quick else DEFAULT_SPECS

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        first = benchmark_spec(specs[0], args.iterations, 1)
        profiler.disable()
        profiler.dump_stats(args.profile)
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(30)
        results = [first]
    else:
        results = [benchmark_spec(spec, args.iterations, args.repeats) for spec in specs]

    document = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "results": results,
    }
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    write_csv(csv_path, results)

    for result in results:
        spec = result["spec"]
        print(
            f"{spec['name']}: geometry={result['geometry']['median_seconds']:.6f}s, "
            f"residual={result['residual']['median_seconds']:.6f}s, "
            f"solve={result['solve']['median_seconds']:.6f}s, "
            f"throughput={result['cell_steps_per_second']:.0f} cell-steps/s"
        )
    print(f"JSON: {args.output}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
