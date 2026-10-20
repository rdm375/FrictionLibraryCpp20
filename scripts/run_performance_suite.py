#!/usr/bin/env python3
"""Build, run, and summarize the friction library's CPU/CUDA benchmarks."""

from __future__ import annotations

import argparse
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README_RESULTS_BEGIN = "<!-- BEGIN GENERATED PERFORMANCE RESULTS -->"
README_RESULTS_END = "<!-- END GENERATED PERFORMANCE RESULTS -->"


def run(command: list[str], *, cwd: Path = ROOT) -> str:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode:
        print(result.stdout, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def run_live(command: list[str], *, cwd: Path = ROOT) -> None:
    """Run a build command with output attached to the terminal for progress."""
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd)
    if result.returncode:
        raise SystemExit(result.returncode)


def run_streaming(command: list[str], *, cwd: Path = ROOT) -> str:
    print("+", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=cwd, text=True, bufsize=1,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout is not None
    output = []
    for line in process.stdout:
        output.append(line)
        print(line, end="", flush=True)
    status = process.wait()
    if status:
        raise SystemExit(status)
    return "".join(output)


def save_runs(executable: Path, stem: str, count: int, raw: Path) -> list[str]:
    outputs = []
    for index in range(1, count + 1):
        print(f"Running {stem}: {index}/{count}", flush=True)
        started = time.monotonic()
        output = run_streaming([str(executable)])
        (raw / f"{stem}-{index:02d}.txt").write_text(output)
        outputs.append(output)
        print(f"Finished {stem}: {time.monotonic() - started:.1f} s", flush=True)
    return outputs


def throughput_rows(outputs: list[str]) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    pattern = re.compile(r"^\s*(.*?)\s*([0-9]+\.[0-9]+) M (solves|eval)/s", re.M)
    for output in outputs:
        for name, value, _unit in pattern.findall(output):
            data.setdefault(name.strip(), []).append(float(value))
    return data


def median_range(values: list[float]) -> str:
    return f"{statistics.median(values):.2f} ({min(values):.2f}–{max(values):.2f})"


def fastest_first(names: list[str] | tuple[str, ...],
                  throughputs: dict[str, list[float]]) -> list[str]:
    """Order table rows by descending median throughput."""
    return sorted(names, key=lambda name: statistics.median(throughputs[name]),
                  reverse=True)


def cpu_surface_rows(outputs: list[str]) -> dict[str, list[float]]:
    labels = {
        "Direct Colebrook + gradients": "Direct Colebrook + gradients",
        "Adaptive bilinear + gradients": "Adaptive bilinear + gradients",
        "Tensor table + gradients": "Adaptive bicubic + gradients",
    }
    result = {label: [] for label in labels.values()}
    for output in outputs:
        for line in output.splitlines():
            for prefix, label in labels.items():
                if line.startswith(prefix):
                    match = re.search(r"([0-9]+\.[0-9]+) M eval/s", line)
                    if match:
                        result[label].append(float(match.group(1)))
    return result


def cuda_direct_rows(outputs: list[str]) -> dict[str, dict[str, list[float]]]:
    names = {
        "Colebrook primal": "Clamond Colebrook, primal",
        "Colebrook + gradients": "Clamond Colebrook + derivatives",
        "Swamee-Jain primal": "Swamee–Jain, primal",
        "Swamee-Jain + gradients": "Swamee–Jain + derivatives",
        "Colebrook fixed point": "Colebrook fixed point, primal",
        "Colebrook Newton": "Colebrook Newton, primal",
    }
    rows = {name: {"CPU": [], "GPU kernel": [], "GPU + copies": []}
            for name in names.values()}
    for output in outputs:
        for line in output.splitlines():
            if "n=1048576" not in line:
                continue
            label = next((name for prefix, name in names.items()
                          if line.lstrip().startswith(prefix)), None)
            if label is None:
                continue
            values = {}
            for key, column in (("CPU", "CPU"), ("GPU-kernel", "GPU kernel"),
                                ("GPU+copies", "GPU + copies")):
                match = re.search(re.escape(key) + r"=\s*([0-9.]+)\s+us", line)
                if match:
                    values[column] = 1048576.0 / float(match.group(1))
            for column, value in values.items():
                rows[label][column].append(value)
    return rows


def cuda_tensor_rows(outputs: list[str]) -> dict[str, list[float]]:
    labels = {
        "Direct Colebrook": "Direct Colebrook",
        "Adaptive bilinear": "Adaptive bilinear",
        "Adaptive bicubic": "Adaptive bicubic",
    }
    rows = {label: {"GPU kernel": [], "GPU + copies": []} for label in labels.values()}
    for output in outputs:
        for line in output.splitlines():
            label = next((name for prefix, name in labels.items()
                          if line.startswith(prefix)), None)
            if label is None:
                continue
            kernel = re.search(r"kernel=\s*([0-9.]+) M/s", line)
            e2e = re.search(r"E2E=\s*([0-9.]+) M/s", line)
            if kernel:
                rows[label]["GPU kernel"].append(float(kernel.group(1)))
            if e2e:
                rows[label]["GPU + copies"].append(float(e2e.group(1)))
    return rows


def cuda_tensor_accuracy(outputs: list[str]) -> dict[str, tuple[str, str]]:
    result = {"Direct Colebrook": ("reference", "0"),
              "Adaptive bilinear": ("not reported", "not reported"),
              "Adaptive bicubic": ("not reported", "not reported")}
    for output in outputs:
        for line in output.splitlines():
            label = next((name for name in ("Adaptive bilinear", "Adaptive bicubic")
                          if line.startswith(name)), None)
            if label is None:
                continue
            errors = re.search(r"scaled max errors=([^ ]+)", line)
            failures = re.search(r"failures=([^ ]+)", line)
            if errors and failures:
                result[label] = (errors.group(1), failures.group(1))
    return result


def get_first_matching(outputs: list[str], pattern: str) -> str:
    regex = re.compile(pattern)
    for output in outputs:
        for line in output.splitlines():
            if regex.search(line):
                return line.strip()
    return "not reported"


def fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def sort_markdown_tables(markdown: str) -> str:
    """Sort generated benchmark tables by their primary throughput column."""
    lines = markdown.splitlines()
    result: list[str] = []
    i = 0
    in_fence = False
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if (not in_fence and line.startswith("|") and i + 1 < len(lines)
                and re.fullmatch(r"\|?[\s:|\\-]+\|?", lines[i + 1])):
            table = [line, lines[i + 1]]
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                table.append(lines[j])
                j += 1
            headers = [cell.strip() for cell in table[0].strip("|").split("|")]
            metric_column = None
            for column, header in enumerate(headers):
                if header == "GPU + copies M/s":
                    metric_column = column
                    break
            if metric_column is None:
                for column, header in enumerate(headers):
                    if header.startswith("CPU M/s") or header.startswith("CPU M eval/s"):
                        metric_column = column
                        break
            if metric_column is not None:
                body = table[2:]
                def rate(row: str) -> float:
                    cells = [cell.strip() for cell in row.strip("|").split("|")]
                    if metric_column >= len(cells):
                        return float("-inf")
                    match = re.search(r"[0-9]+(?:\.[0-9]+)?", cells[metric_column])
                    return float(match.group()) if match else float("-inf")
                result.extend(table[:2])
                result.extend(sorted(body, key=rate, reverse=True))
            else:
                result.extend(table)
            i = j
            continue
        result.append(line)
        i += 1
    return "\n".join(result)


def insert_results_into_readme(results_path: Path, readme_path: Path) -> None:
    """Replace the README's generated-results block with the latest report."""
    readme = readme_path.read_text()
    if readme.count(README_RESULTS_BEGIN) != 1 or readme.count(README_RESULTS_END) != 1:
        raise SystemExit(
            f"Expected exactly one generated-results marker pair in {readme_path}"
        )

    results = sort_markdown_tables(results_path.read_text()).strip()
    report_lines = results.splitlines()
    if report_lines and report_lines[0].startswith("# "):
        # The README already has the project title; keep the report's content
        # and demote its section headings by one level when embedding it.
        report_lines = report_lines[1:]
    embedded_lines = []
    for line in report_lines:
        if line.startswith("## "):
            line = "### " + line[3:]
        elif line.startswith("### "):
            line = "#### " + line[4:]
        embedded_lines.append(line)
    embedded = "\n".join(embedded_lines).strip()

    start = readme.index(README_RESULTS_BEGIN) + len(README_RESULTS_BEGIN)
    end = readme.index(README_RESULTS_END)
    updated = readme[:start] + "\n\n" + embedded + "\n\n" + readme[end:]
    readme_path.write_text(updated)


def require_rows(rows: dict, expected: list[str], description: str) -> None:
    missing = [key for key in expected if key not in rows or not rows[key]]
    if missing:
        raise SystemExit(f"Could not parse {description}: {', '.join(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true",
                        help="also build/run CUDA direct and adaptive tensor benchmarks")
    parser.add_argument("--cuda-arch", default="52",
                        help="CMAKE_CUDA_ARCHITECTURES (default: 52, Quadro M3000M)")
    parser.add_argument("--runs", type=int, default=7,
                        help="separate process runs per benchmark (default: 7)")
    parser.add_argument("--gpu-runs", type=int, default=3,
                        help="separate CUDA benchmark runs (default: 3)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel build jobs (default: 1 to limit memory use)")
    parser.add_argument("--build-dir", default="build-performance",
                        help="CMake build directory (default: build-performance)")
    parser.add_argument("--insert-results", type=Path,
                        help="insert an existing performance-results.md into README and exit")
    args = parser.parse_args()
    if args.insert_results is not None:
        results_path = args.insert_results
        if not results_path.is_absolute():
            results_path = ROOT / results_path
        if not results_path.is_file():
            parser.error(f"results file does not exist: {results_path}")
        insert_results_into_readme(results_path, ROOT / "README.md")
        print(f"Updated {ROOT / 'README.md'} from {results_path}")
        return 0
    if args.runs < 1 or args.gpu_runs < 1:
        parser.error("--runs and --gpu-runs must be at least 1")

    build = (ROOT / args.build_dir).resolve()
    raw = build / "performance-raw"
    raw.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs
    configure = ["cmake", "-S", str(ROOT), "-B", str(build),
                 "-DCMAKE_BUILD_TYPE=Release", "-DFRICTION_BUILD_BENCHMARKS=ON",
                 "-DFRICTION_BUILD_CUDA_BENCHMARK=" + ("ON" if args.cuda else "OFF"),
                 "-DBUILD_TESTING=ON"]
    if args.cuda:
        configure.append(f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}")
    run(configure)
    cpu_targets = [
        "test_friction", "test_analytic_derivatives", "test_batch",
        "test_colebrook_methods", "test_adaptive_tensor_surrogate",
        "test_adaptive_bilinear_tensor_surrogate", "colebrook_solver_benchmark",
        "friction_surrogate_benchmark",
    ]
    run_live(["cmake", "--build", str(build), "--target", *cpu_targets,
              "--parallel", str(jobs)])
    if args.cuda:
        run_live(["cmake", "--build", str(build), "--target",
                  "friction_cuda_benchmark", "friction_cuda_adaptive_bilinear_benchmark",
                  "--parallel", str(jobs)])
    tests = run(["ctest", "--test-dir", str(build), "--output-on-failure"])
    (raw / "ctest.txt").write_text(tests)

    cpu_solver = save_runs(build / "colebrook_solver_benchmark", "cpu-solvers",
                           args.runs, raw)
    cpu_tensor = save_runs(build / "friction_surrogate_benchmark", "cpu-tensor",
                           args.runs, raw)
    solver_data = throughput_rows(cpu_solver)
    require_rows(solver_data, ["Clamond (existing)", "Fixed point", "Newton",
                               "Swamee-Jain primal", "Colebrook gradients (production)",
                               "Swamee-Jain gradients (production)"], "CPU solver benchmark")
    tensor_data = cpu_surface_rows(cpu_tensor)
    require_rows(tensor_data, ["Direct Colebrook + gradients",
                               "Adaptive bilinear + gradients",
                               "Adaptive bicubic + gradients"], "CPU tensor benchmark")

    cpu_model = "unknown CPU"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    cache = (build / "CMakeCache.txt").read_text(errors="replace")
    compiler_match = re.search(r"^CMAKE_CXX_COMPILER:FILEPATH=(.+)$", cache, re.M)
    compiler = compiler_match.group(1) if compiler_match else "c++"
    compiler_info = subprocess.run([compiler, "--version"], cwd=ROOT, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   check=True).stdout.splitlines()[0]
    lines = ["# FrictionLibraryCpp20 performance results", "",
             f"Generated: {platform.platform()}", f"CPU: {cpu_model}",
             f"C++ compiler: {compiler_info}",
             f"CPU benchmark repetitions: {args.runs} process runs; CPU rows show median (min–max) throughput.",
             "CPU compiler flags: CMake Release plus `-O3 -march=native` for benchmark targets.",
             "Tests:", "", "```text", tests.rstrip(), "```", "",
             "## CPU direct methods", "",
             "Primal rows are friction-factor solves/evaluations; derivative rows compute the primal plus both first derivatives.", ""]
    cpu_method_specs = [
        ("Clamond Colebrook", "Clamond (existing)", "Primal; 2 updates"),
        ("Newton Colebrook", "Newton", "Primal; converged solve"),
        ("Fixed-point Colebrook", "Fixed point", "Primal; converged solve"),
        ("Swamee–Jain", "Swamee-Jain primal", "Primal explicit correlation"),
        ("Clamond Colebrook + derivatives", "Colebrook gradients (production)",
         "Primal + `df/dRe` + `df/d(k_s/D)`"),
        ("Swamee–Jain + derivatives", "Swamee-Jain gradients (production)",
         "Primal + `df/dRe` + `df/d(k_s/D)`"),
    ]
    ordered_solver_keys = fastest_first([spec[1] for spec in cpu_method_specs], solver_data)
    specs_by_key = {spec[1]: spec for spec in cpu_method_specs}
    cpu_rows = [[label, median_range(solver_data[key]), outputs]
                for label, key, outputs in
                (specs_by_key[key] for key in ordered_solver_keys)]
    lines.append(fmt_table(["Method", "CPU M/s (median; range)", "Outputs"], cpu_rows))
    lines += ["", "All direct CPU methods use the same deterministic 8,192-point turbulent input set and benchmark executable. It performs 250 passes per method. Solver results also report iteration counts and checksums in the raw logs.",
              "", "## CPU adaptive 2D interpolants", "",
              "All three methods evaluate the same 1,048,576 deterministic `(Re, k_s/D)` inputs and return the factor and both derivatives. Each executable run reports the median of five passes; the table below reports the median (range) across process runs.", ""]
    cpu_tensor_order = fastest_first(list(tensor_data), tensor_data)
    cpu_tensor_rows = [[name, median_range(tensor_data[name])]
                       for name in cpu_tensor_order]
    lines.append(fmt_table(["Method", "CPU M eval/s (median; range)"], cpu_tensor_rows))
    lines += ["", get_first_matching(cpu_tensor, r"^Adaptive bilinear tensor:"),
              get_first_matching(cpu_tensor, r"^Adaptive tensor table:"), ""]

    if args.cuda:
        direct_outputs = save_runs(build / "friction_cuda_benchmark", "gpu-direct",
                                   args.gpu_runs, raw)
        tensor_outputs = save_runs(build / "friction_cuda_adaptive_bilinear_benchmark",
                                   "gpu-adaptive-2d", args.gpu_runs, raw)
        direct_data = cuda_direct_rows(direct_outputs)
        expected = ["Clamond Colebrook, primal", "Clamond Colebrook + derivatives",
                    "Swamee–Jain, primal", "Swamee–Jain + derivatives",
                    "Colebrook fixed point, primal", "Colebrook Newton, primal"]
        require_rows(direct_data, expected, "CUDA direct benchmark at batch size 1,048,576")
        tensor_gpu_data = cuda_tensor_rows(tensor_outputs)
        require_rows(tensor_gpu_data, ["Direct Colebrook", "Adaptive bilinear", "Adaptive bicubic"],
                     "CUDA adaptive tensor benchmark")
        gpu_name = get_first_matching(tensor_outputs, r"^GPU:")
        tensor_accuracy = cuda_tensor_accuracy(tensor_outputs)
        lines += ["## GPU direct methods", "", gpu_name,
                  f"Each row is the 1,048,576-element case from the batch-size sweep, summarized across {args.gpu_runs} process runs. CPU and GPU timings are from the same benchmark executable. Kernel timing keeps inputs resident; `GPU + copies` transfers both inputs and output for each pass.", ""]
        direct_e2e_rates = {name: direct_data[name]["GPU + copies"] for name in expected}
        direct_gpu_order = fastest_first(expected, direct_e2e_rates)
        direct_gpu_rows = []
        for name in direct_gpu_order:
            row = direct_data[name]
            direct_gpu_rows.append([name,
                median_range(row["CPU"]) if row["CPU"] else "n/a",
                median_range(row["GPU kernel"]) if row["GPU kernel"] else "n/a",
                median_range(row["GPU + copies"]) if row["GPU + copies"] else "n/a"])
        lines.append(fmt_table(["Method", "CPU M/s", "GPU kernel M/s", "GPU + copies M/s"], direct_gpu_rows))
        lines += ["", "Newton and fixed-point GPU kernels are direct ports of the library's converged residual iterations. Their GPU primal results are checked elementwise against the production CPU implementations. Derivatives are not currently benchmarked for these iterative variants.",
                  "", "## GPU adaptive 2D interpolants", "", gpu_name,
                  f"The adaptive CUDA benchmark uses a shared input batch and compares the direct Colebrook reference with bilinear and bicubic interpolation over {args.gpu_runs} process runs. Rates are medians (min–max); accuracy and setup details are recorded in the raw logs.", ""]
        tensor_names = ("Direct Colebrook", "Adaptive bilinear", "Adaptive bicubic")
        tensor_e2e_rates = {name: tensor_gpu_data[name]["GPU + copies"]
                            for name in tensor_names}
        tensor_gpu_order = fastest_first(tensor_names, tensor_e2e_rates)
        tensor_gpu_rows = [[name,
            median_range(tensor_gpu_data[name]["GPU kernel"]),
            median_range(tensor_gpu_data[name]["GPU + copies"]),
            tensor_accuracy[name][0], tensor_accuracy[name][1]]
            for name in tensor_gpu_order]
        lines.append(fmt_table(["Method", "GPU kernel M/s", "GPU + copies M/s",
                                "Max scaled errors (value/Re/roughness)", "Failed checks"],
                               tensor_gpu_rows))
        lines += ["", get_first_matching(tensor_outputs, r"^Scaled error limits"),
                  get_first_matching(tensor_outputs, r"^Bilinear grid="),
                  get_first_matching(tensor_outputs, r"^Initial upload="), "",
                  "Raw CUDA logs include error maxima, tolerance failures, checksums, and every batch size:", ""]
    else:
        lines += ["## GPU results", "", "CUDA was not requested. Re-run with `--cuda` on a host with the CUDA toolkit and an NVIDIA GPU to build and execute both GPU suites. The CUDA direct suite includes Clamond, Swamee–Jain, Newton, and fixed-point primal evaluation; the adaptive suite compares direct Colebrook with bilinear and bicubic tensors.", ""]
    lines += [f"Raw logs: `{raw.relative_to(ROOT)}/`", ""]

    report = build / "performance-results.md"
    report.write_text("\n".join(lines))
    print(f"Wrote {report}")
    insert_results_into_readme(report, ROOT / "README.md")
    print(f"Updated {ROOT / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
