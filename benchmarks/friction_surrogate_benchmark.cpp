// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#include "friction/adaptive_tensor_surrogate.hpp"
#include "friction/adaptive_bilinear_tensor_surrogate.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
struct Point { double re,rr; };
constexpr std::size_t count=1U<<20;

template<class F>
double time_pass(const char* label,const std::vector<Point>& points,F&& f) {
    constexpr int repetitions=5;
    std::array<double,repetitions> seconds{};
    double checksum=0.0;
    for (int repeat=0;repeat<repetitions;++repeat) {
        double pass_checksum=0.0;
        const auto start=Clock::now();
        for (const Point& p:points) {
            const auto r=f(p.re,p.rr);
            pass_checksum=pass_checksum+r.value+1.0e-3*r.d_reynolds+1.0e-6*r.d_relative_roughness;
        }
        seconds[repeat]=std::chrono::duration<double>(Clock::now()-start).count();
        checksum=pass_checksum;
    }
    std::sort(seconds.begin(),seconds.end());
    const double median_seconds=seconds[repetitions/2];
    std::cout<<std::left<<std::setw(28)<<label<<std::right<<std::fixed<<std::setprecision(2)
             <<(static_cast<double>(points.size())/median_seconds/1.0e6)<<" M eval/s (median of "
             <<repetitions<<" passes)"
             <<" checksum="<<std::setprecision(10)<<checksum<<'\n';
    return median_seconds;
}
}

int main() {
    friction::surrogate::AdaptiveTensorConfig config;
    friction::surrogate::AdaptiveBilinearColebrookTensor bilinear_table(config);
    const auto bilinear_build_start=Clock::now();
    const auto bilinear_report=bilinear_table.build();
    const double bilinear_build_seconds=std::chrono::duration<double>(
        Clock::now()-bilinear_build_start).count();

    friction::surrogate::AdaptiveColebrookTensor table(config);
    const auto build_start=Clock::now();
    const auto report=table.build();
    const double build_seconds=std::chrono::duration<double>(Clock::now()-build_start).count();
    std::cout<<"Adaptive tensor table: "<<report.grid_points<<" nodes ("
             <<report.reynolds_axis_points<<" x "<<report.roughness_axis_points<<"), "
             <<std::fixed<<std::setprecision(2)
             <<(static_cast<double>(report.estimated_storage_bytes)/(1024.0*1024.0))<<" MiB, "
             <<report.refinement_rounds<<" refinement rounds, build="
             <<std::fixed<<std::setprecision(3)<<build_seconds<<" s, final indicator="
             <<report.largest_unresolved_indicator<<'\n';
    std::cout<<"Adaptive bilinear tensor: "<<bilinear_report.grid_points<<" nodes ("
             <<bilinear_report.reynolds_axis_points<<" x "
             <<bilinear_report.roughness_axis_points<<"), "
             <<std::fixed<<std::setprecision(2)
             <<(static_cast<double>(bilinear_report.estimated_storage_bytes)/(1024.0*1024.0))
             <<" MiB, "<<bilinear_report.refinement_rounds<<" refinement rounds, build="
             <<std::fixed<<std::setprecision(3)<<bilinear_build_seconds<<" s\n";

    std::vector<Point> points;
    points.reserve(count);
    std::mt19937_64 rng(20261019);
    std::uniform_real_distribution<double> unit(0.0,1.0);
    const double rr_span=std::log1p(config.relative_roughness_max/config.roughness_scale);
    for (std::size_t i=0;i<count;++i) {
        const double u=unit(rng),v=unit(rng);
        points.push_back({config.reynolds_min*std::pow(
                              config.reynolds_max/config.reynolds_min,u),
                          config.roughness_scale*std::expm1(v*rr_span)});
    }
    const double direct_seconds=time_pass("Direct Colebrook + gradients",points,
        [](double re,double rr) {
            return friction::analytic::darcy_friction_factor(
                re,rr,friction::Correlation::colebrook);
        });
    const double bilinear_seconds=time_pass("Adaptive bilinear + gradients",points,
        [&bilinear_table](double re,double rr) {
            return bilinear_table.evaluate(re,rr).derivatives;
        });
    const double table_seconds=time_pass("Tensor table + gradients",points,
        [&table](double re,double rr) { return table.evaluate(re,rr).derivatives; });
    std::cout<<"Throughput ratios (bilinear/direct, bicubic/direct): "
             <<std::fixed<<std::setprecision(3)<<direct_seconds/bilinear_seconds<<"x, "
             <<direct_seconds/table_seconds<<"x\n";

    // In a pipe solve, roughness is normally fixed across the flow evaluations
    // in one objective/forward pass. Reuse the same Reynolds samples so the
    // comparison isolates the input pattern rather than changing the sample.
    const double fixed_rr=config.roughness_scale*std::expm1(0.5*rr_span);
    std::vector<Point> fixed_roughness_points=points;
    for (Point& p:fixed_roughness_points) p.rr=fixed_rr;
    std::cout<<"\nFixed relative roughness case: rr="<<std::scientific
             <<std::setprecision(6)<<fixed_rr<<'\n';
    const double fixed_direct_seconds=time_pass("Direct Colebrook + gradients",fixed_roughness_points,
        [](double re,double rr) {
            return friction::analytic::darcy_friction_factor(
                re,rr,friction::Correlation::colebrook);
        });
    const double fixed_table_seconds=time_pass("Tensor table + gradients",fixed_roughness_points,
        [&table](double re,double rr) { return table.evaluate(re,rr).derivatives; });
    std::cout<<"Throughput ratio (table/direct): "<<std::fixed<<std::setprecision(3)
             <<fixed_direct_seconds/fixed_table_seconds<<"x\n";

    const auto prepare_start=Clock::now();
    const auto prepared=table.prepare_roughness(fixed_rr);
    const double prepare_seconds=std::chrono::duration<double>(Clock::now()-prepare_start).count();
    std::cout<<"Prepared roughness setup: "<<std::fixed<<std::setprecision(3)
             <<prepare_seconds*1.0e3<<" ms\n";
    const double prepared_seconds=time_pass("Prepared slice + gradients",fixed_roughness_points,
        [&prepared](double re,double) { return prepared.evaluate(re).derivatives; });
    std::cout<<"Throughput ratio (prepared/direct): "<<std::fixed<<std::setprecision(3)
             <<fixed_direct_seconds/prepared_seconds<<"x\n";
}
