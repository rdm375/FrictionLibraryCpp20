// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#pragma once

#include "friction/adaptive_tensor_surrogate.hpp"

namespace friction::surrogate {

// Adaptive tensor-product, piecewise-bilinear Colebrook approximation.
// The field being interpolated is g=1/sqrt(f) in normalized log coordinates;
// f and its physical derivatives are recovered analytically at evaluation.
class AdaptiveBilinearColebrookTensor {
public:
    struct CellPolynomial {
        // g(x,y) = c[0] + c[1] x + c[2] y + c[3] x*y,
        // with x,y normalized to this cell.
        std::array<double, 4> coefficients{};
    };

    struct Evaluation {
        analytic::FirstDerivatives derivatives;
        bool used_surrogate{};
    };

    explicit AdaptiveBilinearColebrookTensor(AdaptiveTensorConfig config = {})
        : config_(config) {
        validate_config();
    }

    AdaptiveTensorBuildReport build() {
        u_nodes_ = {0.0, 0.5, 1.0};
        v_nodes_ = {0.0, 0.5, 1.0};
        report_ = {};
        sample_nodes();
        build_cells();

        for (;;) {
            std::vector<bool> split_u(u_nodes_.size() - 1, false);
            std::vector<bool> split_v(v_nodes_.size() - 1, false);
            double worst = 0.0;
            for (std::size_t i = 0; i + 1 < u_nodes_.size(); ++i) {
                for (std::size_t j = 0; j + 1 < v_nodes_.size(); ++j) {
                    ++report_.cells_checked;
                    const auto check = check_cell(i, j);
                    worst = std::max(worst, check.indicator);
                    if (check.indicator <= config_.relative_tolerance) continue;
                    if (check.u_indicator >= 0.35 * check.indicator) split_u[i] = true;
                    if (check.v_indicator >= 0.35 * check.indicator) split_v[j] = true;
                    if (!split_u[i] && !split_v[j]) {
                        if (check.u_indicator >= check.v_indicator) split_u[i] = true;
                        else split_v[j] = true;
                    }
                }
            }
            report_.largest_unresolved_indicator = worst;
            if (std::none_of(split_u.begin(), split_u.end(), [](bool x) { return x; }) &&
                std::none_of(split_v.begin(), split_v.end(), [](bool x) { return x; })) break;

            std::vector<double> next_u, next_v;
            next_u.reserve(u_nodes_.size() * 2);
            next_v.reserve(v_nodes_.size() * 2);
            for (std::size_t i = 0; i + 1 < u_nodes_.size(); ++i) {
                next_u.push_back(u_nodes_[i]);
                if (split_u[i]) next_u.push_back(0.5 * (u_nodes_[i] + u_nodes_[i + 1]));
            }
            next_u.push_back(u_nodes_.back());
            for (std::size_t j = 0; j + 1 < v_nodes_.size(); ++j) {
                next_v.push_back(v_nodes_[j]);
                if (split_v[j]) next_v.push_back(0.5 * (v_nodes_[j] + v_nodes_[j + 1]));
            }
            next_v.push_back(v_nodes_.back());
            if (next_u.size() > config_.max_axis_points ||
                next_v.size() > config_.max_axis_points ||
                next_u.size() * next_v.size() > config_.max_grid_points) break;

            u_nodes_.swap(next_u);
            v_nodes_.swap(next_v);
            ++report_.refinement_rounds;
            sample_nodes();
            build_cells();
        }

        report_.grid_points = nodes_.size();
        report_.reynolds_axis_points = u_nodes_.size();
        report_.roughness_axis_points = v_nodes_.size();
        report_.estimated_storage_bytes = nodes_.capacity() * sizeof(Node) +
            cells_.capacity() * sizeof(CellPolynomial) +
            (u_nodes_.capacity() + v_nodes_.capacity()) * sizeof(double);
        return report_;
    }

    [[nodiscard]] bool is_built() const noexcept { return !nodes_.empty(); }
    [[nodiscard]] bool contains(double re, double rr) const noexcept {
        return std::isfinite(re) && std::isfinite(rr) &&
            re >= config_.reynolds_min && re <= config_.reynolds_max &&
            rr >= 0.0 && rr <= config_.relative_roughness_max;
    }
    [[nodiscard]] const AdaptiveTensorConfig& config() const noexcept { return config_; }
    [[nodiscard]] const AdaptiveTensorBuildReport& report() const noexcept { return report_; }
    [[nodiscard]] const std::vector<double>& u_nodes() const noexcept { return u_nodes_; }
    [[nodiscard]] const std::vector<double>& v_nodes() const noexcept { return v_nodes_; }
    [[nodiscard]] const std::vector<CellPolynomial>& cell_polynomials() const noexcept {
        return cells_;
    }

    [[nodiscard]] Evaluation evaluate(double re, double rr) const {
        if (!is_built()) throw std::logic_error("adaptive bilinear tensor has not been built");
        if (!contains(re, rr)) return {direct(re, rr), false};
        const double u = to_u(re), v = to_v(rr);
        const std::size_t i = interval(u_nodes_, u), j = interval(v_nodes_, v);
        const auto s = interpolate(i, j, u, v);
        if (!(s.value > 0.0) || !std::isfinite(s.value) ||
            !std::isfinite(s.du) || !std::isfinite(s.dv)) return {direct(re, rr), false};
        const double f = 1.0 / (s.value * s.value);
        const double factor = -2.0 / (s.value * s.value * s.value);
        analytic::FirstDerivatives d{
            f,
            factor * s.du / (re * ln10 * log_re_span()),
            factor * s.dv / ((config_.roughness_scale + rr) * log_rr_span())};
        if (!(f > 0.0) || !std::isfinite(f) || !std::isfinite(d.d_reynolds) ||
            !std::isfinite(d.d_relative_roughness)) return {direct(re, rr), false};
        return {d, true};
    }

private:
    static constexpr double ln10 = 2.3025850929940456840179914546843642;
    struct Node { double value{}, du{}, dv{}; };
    struct Surface { double value{}, du{}, dv{}; };
    struct CellCheck { double indicator{}, u_indicator{}, v_indicator{}; };

    AdaptiveTensorConfig config_;
    AdaptiveTensorBuildReport report_{};
    std::vector<double> u_nodes_, v_nodes_;
    std::vector<Node> nodes_;
    std::vector<CellPolynomial> cells_;

    void validate_config() const {
        if (!std::isfinite(config_.reynolds_min) || !std::isfinite(config_.reynolds_max) ||
            config_.reynolds_min < 2000.0 || config_.reynolds_max <= config_.reynolds_min)
            throw std::invalid_argument("surrogate requires 2000 <= Re_min < Re_max");
        if (!std::isfinite(config_.relative_roughness_max) ||
            config_.relative_roughness_max <= 0.0 || config_.relative_roughness_max >= 3.7)
            throw std::invalid_argument("relative roughness maximum must be in (0, 3.7)");
        if (!std::isfinite(config_.roughness_scale) || config_.roughness_scale <= 0.0 ||
            !std::isfinite(config_.relative_tolerance) || config_.relative_tolerance <= 0.0)
            throw std::invalid_argument("roughness scale and tolerance must be positive");
        if (config_.max_axis_points < 3 || config_.max_grid_points < 9)
            throw std::invalid_argument("grid limits are too small");
    }
    [[nodiscard]] double log_re_span() const {
        return std::log10(config_.reynolds_max / config_.reynolds_min);
    }
    [[nodiscard]] double log_rr_span() const {
        return std::log1p(config_.relative_roughness_max / config_.roughness_scale);
    }
    [[nodiscard]] double to_u(double re) const {
        return std::log10(re / config_.reynolds_min) / log_re_span();
    }
    [[nodiscard]] double to_v(double rr) const {
        return std::log1p(rr / config_.roughness_scale) / log_rr_span();
    }
    [[nodiscard]] double from_u(double u) const {
        return config_.reynolds_min * std::pow(10.0, u * log_re_span());
    }
    [[nodiscard]] double from_v(double v) const {
        return config_.roughness_scale * std::expm1(v * log_rr_span());
    }
    [[nodiscard]] std::size_t index(std::size_t i, std::size_t j) const {
        return i * v_nodes_.size() + j;
    }
    [[nodiscard]] static std::size_t interval(const std::vector<double>& axis, double q) {
        if (q >= axis.back()) return axis.size() - 2;
        return static_cast<std::size_t>(std::upper_bound(axis.begin(), axis.end(), q) -
                                        axis.begin() - 1);
    }
    [[nodiscard]] Node exact_node(double u, double v) const {
        const double re = from_u(u), rr = from_v(v);
        const auto d = analytic::darcy_friction_factor(re, rr, Correlation::colebrook);
        const double g = 1.0 / std::sqrt(d.value);
        const double dg_df = -0.5 * g * g * g;
        return {g,
            dg_df * d.d_reynolds * (re * ln10 * log_re_span()),
            dg_df * d.d_relative_roughness * ((config_.roughness_scale + rr) * log_rr_span())};
    }
    void sample_nodes() {
        nodes_.resize(u_nodes_.size() * v_nodes_.size());
        for (std::size_t i = 0; i < u_nodes_.size(); ++i)
            for (std::size_t j = 0; j < v_nodes_.size(); ++j)
                nodes_[index(i, j)] = exact_node(u_nodes_[i], v_nodes_[j]);
    }
    void build_cells() {
        const std::size_t nv = v_nodes_.size() - 1;
        cells_.assign((u_nodes_.size() - 1) * nv, CellPolynomial{});
        for (std::size_t i = 0; i + 1 < u_nodes_.size(); ++i) {
            for (std::size_t j = 0; j + 1 < v_nodes_.size(); ++j) {
                const double g00 = nodes_[index(i, j)].value;
                const double g10 = nodes_[index(i + 1, j)].value;
                const double g01 = nodes_[index(i, j + 1)].value;
                const double g11 = nodes_[index(i + 1, j + 1)].value;
                auto& c = cells_[i * nv + j].coefficients;
                c = {g00, g10 - g00, g01 - g00, g11 - g10 - g01 + g00};
            }
        }
    }
    [[nodiscard]] Surface interpolate(std::size_t i, std::size_t j,
                                      double u, double v) const {
        const double hu = u_nodes_[i + 1] - u_nodes_[i];
        const double hv = v_nodes_[j + 1] - v_nodes_[j];
        const double x = (u - u_nodes_[i]) / hu;
        const double y = (v - v_nodes_[j]) / hv;
        const auto& c = cells_[i * (v_nodes_.size() - 1) + j].coefficients;
        return {c[0] + c[1] * x + c[2] * y + c[3] * x * y,
                (c[1] + c[3] * y) / hu,
                (c[2] + c[3] * x) / hv};
    }
    [[nodiscard]] CellCheck check_cell(std::size_t i, std::size_t j) const {
        const double u0 = u_nodes_[i], u1 = u_nodes_[i + 1];
        const double v0 = v_nodes_[j], v1 = v_nodes_[j + 1];
        const double uc = 0.5 * (u0 + u1), vc = 0.5 * (v0 + v1);
        const double hu = u1 - u0, hv = v1 - v0;
        const Node exact = exact_node(uc, vc);
        const Surface pred = interpolate(i, j, uc, vc);
        const double scale = std::max(1.0e-12, std::abs(exact.value));
        const double direct = std::max({std::abs(exact.value - pred.value) / scale,
            hu * std::abs(exact.du - pred.du) / scale,
            hv * std::abs(exact.dv - pred.dv) / scale});

        double taylor[4]{};
        double du_min = std::numeric_limits<double>::infinity(), du_max = -du_min;
        double dv_min = du_min, dv_max = -du_min;
        for (unsigned a = 0; a < 2; ++a) for (unsigned b = 0; b < 2; ++b) {
            const Node& n = nodes_[index(i + a, j + b)];
            const double u = a ? u1 : u0, v = b ? v1 : v0;
            taylor[a + 2 * b] = n.value + n.du * (uc - u) + n.dv * (vc - v);
            du_min = std::min(du_min, n.du); du_max = std::max(du_max, n.du);
            dv_min = std::min(dv_min, n.dv); dv_max = std::max(dv_max, n.dv);
        }
        const double u_dis = std::max(std::abs(taylor[0] - taylor[1]),
                                      std::abs(taylor[2] - taylor[3])) +
                             0.25 * hu * (du_max - du_min);
        const double v_dis = std::max(std::abs(taylor[0] - taylor[2]),
                                      std::abs(taylor[1] - taylor[3])) +
                             0.25 * hv * (dv_max - dv_min);
        const double iu = u_dis / scale, iv = v_dis / scale;
        return {std::max({direct, iu, iv}), iu, iv};
    }
    [[nodiscard]] analytic::FirstDerivatives direct(double re, double rr) const {
        return analytic::darcy_friction_factor(re, rr, Correlation::colebrook);
    }
};

} // namespace friction::surrogate
