#!/usr/bin/env python3
"""Plot the Colebrook contours and the friction library's 2D sampler knots.

The adaptive knot axes are obtained from the C++ library itself (both bilinear
and bicubic builders). A matched-size regular grid is included for comparison.
Requires a C++20 compiler, NumPy, and Matplotlib.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


CPP_SOURCE = r"""
#include "friction/adaptive_bilinear_tensor_surrogate.hpp"
#include "friction/adaptive_tensor_surrogate.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>

template<class Table>
void dump(char kind, Table& table) {
    const auto report = table.build();
    std::cout << kind << ' ' << report.reynolds_axis_points << ' '
              << report.roughness_axis_points << ' ' << report.refinement_rounds << '\n';
    std::cout << "U";
    for (double x : table.u_nodes()) std::cout << ' ' << x;
    std::cout << '\n' << "V";
    for (double x : table.v_nodes()) std::cout << ' ' << x;
    std::cout << '\n';
}

int main(int argc, char** argv) {
    if (argc != 6) return 2;
    friction::surrogate::AdaptiveTensorConfig c;
    c.reynolds_min = std::strtod(argv[1], nullptr);
    c.reynolds_max = std::strtod(argv[2], nullptr);
    c.relative_roughness_max = std::strtod(argv[3], nullptr);
    c.roughness_scale = std::strtod(argv[4], nullptr);
    c.relative_tolerance = std::strtod(argv[5], nullptr);
    std::cout << std::setprecision(17);
    friction::surrogate::AdaptiveBilinearColebrookTensor bilinear(c);
    dump('B', bilinear);
    friction::surrogate::AdaptiveColebrookTensor bicubic(c);
    dump('C', bicubic);
}
"""


def get_sampler_axes(repo: Path, args: argparse.Namespace) -> dict[str, dict]:
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("No C++ compiler found. Set CXX to a C++20 compiler.")

    with tempfile.TemporaryDirectory(prefix="friction-sampler-plot-") as temp:
        temp_path = Path(temp)
        source = temp_path / "dump_sampler_axes.cpp"
        executable = temp_path / "dump_sampler_axes"
        source.write_text(CPP_SOURCE)
        subprocess.run(
            [compiler, "-std=c++20", "-O3", f"-I{repo / 'include'}", str(source), "-o", str(executable)],
            check=True,
        )
        result = subprocess.run(
            [str(executable), str(args.re_min), str(args.re_max), str(args.rr_max),
             str(args.roughness_scale), str(args.tolerance)],
            check=True, capture_output=True, text=True,
        )

    lines = iter(result.stdout.splitlines())
    output: dict[str, dict] = {}
    for _ in range(2):
        kind, nu, nv, rounds = next(lines).split()
        u = np.fromstring(next(lines)[1:], sep=" ")
        v = np.fromstring(next(lines)[1:], sep=" ")
        output[kind] = {"u": u, "v": v, "nu": int(nu), "nv": int(nv), "rounds": int(rounds)}
    return output


def colebrook_factor(re: np.ndarray, rr: np.ndarray) -> np.ndarray:
    """Vectorized Newton solve of 1/sqrt(f) Colebrook equation."""
    # Haaland is a good starting point; Newton converges rapidly over this domain.
    y = -1.8 * np.log10(np.power(rr / 3.7, 1.11) + 6.9 / re)
    for _ in range(12):
        z = rr / 3.7 + 2.51 * y / re
        residual = y + 2.0 * np.log10(z)
        derivative = 1.0 + (2.0 / np.log(10.0)) * (2.51 / re) / z
        step = residual / derivative
        y -= step
        if np.max(np.abs(step)) < 2.0e-14:
            break
    return 1.0 / np.square(y)


def physical_ticks(args: argparse.Namespace):
    re_values = np.array([2e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    re_values = re_values[(re_values >= args.re_min) & (re_values <= args.re_max)]
    u_positions = np.log10(re_values / args.re_min) / np.log10(args.re_max / args.re_min)
    rr_values = np.array([0.0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2])
    rr_values = rr_values[rr_values <= args.rr_max]
    v_span = np.log1p(args.rr_max / args.roughness_scale)
    v_positions = np.log1p(rr_values / args.roughness_scale) / v_span
    return u_positions, [f"{x:.0e}" for x in re_values], v_positions, [f"{x:g}" for x in rr_values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("colebrook-domain-samplers.png"))
    parser.add_argument("--re-min", type=float, default=2_000.0)
    parser.add_argument("--re-max", type=float, default=1.0e8)
    parser.add_argument("--rr-max", type=float, default=0.05)
    parser.add_argument("--roughness-scale", type=float, default=1.0e-5)
    parser.add_argument("--tolerance", type=float, default=2.0e-4)
    parser.add_argument("--contour-resolution", type=int, default=300)
    args = parser.parse_args()
    if args.re_min < 2_000 or args.re_max <= args.re_min or args.rr_max <= 0 or args.roughness_scale <= 0:
        parser.error("domain must satisfy Re_min >= 2000, Re_max > Re_min, rr_max > 0, scale > 0")
    if args.contour_resolution < 50:
        parser.error("--contour-resolution must be at least 50")

    repo = Path(__file__).resolve().parents[1]
    samplers = get_sampler_axes(repo, args)
    bicubic = samplers["C"]
    bilinear = samplers["B"]

    # The regular comparison grid has the same axis counts/node budget as the
    # adaptive grids, sampled uniformly in the same normalized coordinates.
    uniform_u = np.linspace(0.0, 1.0, bicubic["nu"])
    uniform_v = np.linspace(0.0, 1.0, bicubic["nv"])

    q = np.linspace(0.0, 1.0, args.contour_resolution)
    u_mesh, v_mesh = np.meshgrid(q, q, indexing="xy")
    re_mesh = args.re_min * np.power(args.re_max / args.re_min, u_mesh)
    v_span = np.log1p(args.rr_max / args.roughness_scale)
    rr_mesh = args.roughness_scale * np.expm1(v_mesh * v_span)
    friction = colebrook_factor(re_mesh, rr_mesh)
    levels = np.geomspace(max(0.006, float(friction.min())), float(friction.max()), 11)
    u_ticks, u_labels, v_ticks, v_labels = physical_ticks(args)

    panels = [
        ("Uniform tensor grid", uniform_u, uniform_v,
         f"{len(uniform_u) * len(uniform_v):,} knots (matched budget)"),
        ("Adaptive bilinear tensor", bilinear["u"], bilinear["v"],
         f"{bilinear['nu'] * bilinear['nv']:,} knots; {bilinear['rounds']} rounds"),
        ("Adaptive bicubic tensor", bicubic["u"], bicubic["v"],
         f"{bicubic['nu'] * bicubic['nv']:,} knots; {bicubic['rounds']} rounds"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharex=True, sharey=True, constrained_layout=True)
    for ax, (title, u_nodes, v_nodes, subtitle) in zip(axes, panels):
        contour = ax.contour(u_mesh, v_mesh, friction, levels=levels, cmap="viridis", linewidths=0.9)
        ax.clabel(contour, inline=True, fontsize=7, fmt="%.3g")
        uu, vv = np.meshgrid(u_nodes, v_nodes, indexing="ij")
        ax.scatter(uu.ravel(), vv.ravel(), s=1.0, c="#d62728", alpha=0.28, linewidths=0,
                   rasterized=True, label="sample knots")
        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xticks(u_ticks, u_labels)
        ax.set_yticks(v_ticks, v_labels)
        ax.grid(alpha=0.18)
        ax.set_xlabel("Reynolds number, Re (log scale)")
    axes[0].set_ylabel("Relative roughness, ε/D")
    axes[0].legend(loc="upper left", framealpha=0.9, markerscale=4)
    fig.suptitle("Colebrook friction-factor contours and tensor sampler knots", fontsize=14)
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output.resolve()}")
    print(f"Adaptive bilinear: {bilinear['nu']} x {bilinear['nv']} knots, {bilinear['rounds']} refinement rounds")
    print(f"Adaptive bicubic:  {bicubic['nu']} x {bicubic['nv']} knots, {bicubic['rounds']} refinement rounds")


if __name__ == "__main__":
    main()
