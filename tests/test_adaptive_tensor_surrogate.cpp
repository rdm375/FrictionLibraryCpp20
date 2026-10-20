#include "friction/adaptive_tensor_surrogate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {
void require(bool condition,const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
}

int main() {
    friction::surrogate::AdaptiveTensorConfig config;
    config.relative_tolerance=2.0e-4;
    friction::surrogate::AdaptiveColebrookTensor table(config);
    const auto report=table.build();
    require(report.grid_points>9,"adaptive build refines the initial tensor grid");
    require(report.largest_unresolved_indicator<=config.relative_tolerance,
            "adaptive build meets its configured cell indicator");

    double max_value_error=0.0;
    double max_log_re_slope_error=0.0;
    double max_log_roughness_slope_error=0.0;
    constexpr int re_points=37,rr_points=29;
    for (int i=0;i<re_points;++i) {
        const double u=(static_cast<double>(i)+0.371)/re_points;
        const double re=config.reynolds_min*std::pow(
            config.reynolds_max/config.reynolds_min,u);
        for (int j=0;j<rr_points;++j) {
            const double v=(static_cast<double>(j)+0.613)/rr_points;
            const double rr=config.roughness_scale*std::expm1(
                v*std::log1p(config.relative_roughness_max/config.roughness_scale));
            const auto approximate=table.evaluate(re,rr);
            const auto exact=friction::analytic::darcy_friction_factor(
                re,rr,friction::Correlation::colebrook);
            require(approximate.used_surrogate,"in-domain point uses the table");
            max_value_error=std::max(max_value_error,
                std::abs(approximate.derivatives.value-exact.value)/exact.value);
            max_log_re_slope_error=std::max(max_log_re_slope_error,
                re*std::abs(approximate.derivatives.d_reynolds-exact.d_reynolds)/exact.value);
            max_log_roughness_slope_error=std::max(max_log_roughness_slope_error,
                (config.roughness_scale+rr)*std::abs(
                    approximate.derivatives.d_relative_roughness-
                    exact.d_relative_roughness)/exact.value);
        }
    }
    // Include the domain edges explicitly: smooth-wall roughness, the maximum
    // calibrated roughness, and both Reynolds limits.
    for (int i=0;i<=40;++i) {
        const double u=static_cast<double>(i)/40.0;
        const double re=config.reynolds_min*std::pow(
            config.reynolds_max/config.reynolds_min,u);
        for (double rr : {0.0,config.relative_roughness_max}) {
            const auto approximate=table.evaluate(re,rr).derivatives;
            const auto exact=friction::analytic::darcy_friction_factor(
                re,rr,friction::Correlation::colebrook);
            max_value_error=std::max(max_value_error,
                std::abs(approximate.value-exact.value)/exact.value);
            max_log_re_slope_error=std::max(max_log_re_slope_error,
                re*std::abs(approximate.d_reynolds-exact.d_reynolds)/exact.value);
            max_log_roughness_slope_error=std::max(max_log_roughness_slope_error,
                (config.roughness_scale+rr)*std::abs(
                    approximate.d_relative_roughness-exact.d_relative_roughness)/exact.value);
        }
    }
    require(max_value_error<2.0e-4,"held-out friction-factor relative error");
    require(max_log_re_slope_error<2.0e-2,"held-out Reynolds sensitivity error");
    require(max_log_roughness_slope_error<2.0e-3,"held-out roughness sensitivity error");

    const auto laminar=table.evaluate(1500.0,0.01);
    require(!laminar.used_surrogate,"laminar input falls back to exact wrapper");
    require(std::abs(laminar.derivatives.value-64.0/1500.0)<1.0e-15,
            "laminar fallback returns 64/Re");
    const auto outside=table.evaluate(1.0e6,0.06);
    require(!outside.used_surrogate,"roughness outside the table falls back to direct");
    const auto direct=friction::analytic::darcy_friction_factor(
        1.0e6,0.06,friction::Correlation::colebrook);
    require(std::abs(outside.derivatives.value-direct.value)<1.0e-15,
            "out-of-domain fallback matches direct Colebrook");

    const double fixed_rr=config.roughness_scale*std::expm1(
        0.5*std::log1p(config.relative_roughness_max/config.roughness_scale));
    const auto prepared=table.prepare_roughness(fixed_rr);
    const auto device_view=prepared.device_view();
    require(device_view.row_count==report.reynolds_axis_points-1 &&
            device_view.reynolds_nodes!=nullptr && device_view.rows!=nullptr,
            "prepared slice exposes a consistent flat view for accelerator kernels");
    for (int i=0;i<=100;++i) {
        const double u=static_cast<double>(i)/100.0;
        const double re=config.reynolds_min*std::pow(
            config.reynolds_max/config.reynolds_min,u);
        const auto ordinary=table.evaluate(re,fixed_rr);
        const auto cached=prepared.evaluate(re);
        require(cached.used_surrogate,"prepared in-domain point uses the table");
        require(std::abs(cached.derivatives.value-ordinary.derivatives.value)<2.0e-14,
                "prepared value agrees with ordinary table lookup");
        require(std::abs(cached.derivatives.d_reynolds-
                         ordinary.derivatives.d_reynolds)<2.0e-14,
                "prepared Reynolds derivative agrees with ordinary table lookup");
        require(std::abs(cached.derivatives.d_relative_roughness-
                         ordinary.derivatives.d_relative_roughness)<2.0e-14,
                "prepared roughness derivative agrees with ordinary table lookup");
    }
    const auto prepared_laminar=prepared.evaluate(1500.0);
    require(!prepared_laminar.used_surrogate &&
            std::abs(prepared_laminar.derivatives.value-64.0/1500.0)<1.0e-15,
            "prepared evaluator retains the exact out-of-domain fallback");

    std::cout << "Adaptive tensor surrogate checks passed; grid=" << report.grid_points
              << ", max value error=" << max_value_error
              << ", max scaled d/dlog(Re) error=" << max_log_re_slope_error
              << ", max scaled roughness-slope error=" << max_log_roughness_slope_error
              << '\n';
}
