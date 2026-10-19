#include <friction/friction.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

long double reference_bisection(long double re, long double roughness) {
    // Independently solve for y=1/sqrt(f):
    // y + 2 log10(k/D / 3.7 + 2.51 y / Re) = 0.
    constexpr long double ln10 = 2.3025850929940456840179914546843642L;
    auto residual = [=](long double y) {
        return y + (2.0L / ln10) * std::log(roughness / 3.7L + 2.51L * y / re);
    };
    long double lo = 0.0L;
    long double hi = 1000.0L;
    for (int i = 0; i < 400; ++i) {
        const long double mid = (lo + hi) / 2.0L;
        if (residual(mid) > 0.0L) hi = mid;
        else lo = mid;
    }
    const long double y = (lo + hi) / 2.0L;
    return 1.0L / (y * y);
}

void test_reference_grid() {
    const std::vector<double> reynolds{2000.0, 2300.0, 3000.0, 1.0e4, 1.0e5, 7.0e5, 1.0e8, 1.0e13, 1.0e100};
    const std::vector<double> roughness{0.0, 1.0e-6, 1.0e-3, 1.0e-2, 1.0e-1};
    for (double re : reynolds) {
        for (double k : roughness) {
            const double actual = friction::colebrook_turbulent(re, k).darcy_friction_factor;
            const long double expected = reference_bisection(re, k);
            const long double relative_error = std::abs((static_cast<long double>(actual) - expected) / expected);
            if (!(relative_error < 3.0e-14L)) { std::cerr << "case Re=" << re << " k=" << k << " actual=" << actual << " expected=" << static_cast<double>(expected) << " rel=" << static_cast<double>(relative_error) << "\n"; require(false, "Clamond result agrees with independent bisection"); }
            require(actual > 0.0 && std::isfinite(actual), "friction factor is finite and positive");
            require(friction::colebrook_turbulent(re, k).iterations == 2, "paper's two updates are reported");
        }
    }
}

void test_residual_and_trends() {
    for (const auto& [re, k] : {std::pair{4000.0, 0.0}, std::pair{1.0e6, 1.0e-3}, std::pair{1.0e10, 1.0e-2}}) {
        const double f = friction::darcy_friction_factor(re, k);
        const double y = 1.0 / std::sqrt(f);
        const double residual = y + 2.0 * std::log10(k / 3.7 + 2.51 * y / re);
        require(std::abs(residual) < 2.0e-14, "Colebrook equation residual is near double precision");
    }
    require(friction::darcy_friction_factor(1.0e5, 1.0e-3) >
                friction::darcy_friction_factor(1.0e6, 1.0e-3),
            "friction decreases as Reynolds number increases at fixed roughness");
    require(friction::darcy_friction_factor(1.0e6, 1.0e-2) >
                friction::darcy_friction_factor(1.0e6, 1.0e-4),
            "friction increases as roughness increases at fixed Reynolds number");
}

void test_all_regime_wrapper() {
    constexpr double transition_start = 2000.0;
    constexpr double transition_end = 2300.0;
    constexpr double roughness = 1.0e-3;

    require(friction::darcy_friction_factor(1500.0, roughness) == 64.0 / 1500.0,
            "laminar branch returns 64/Re below 2000");
    require(friction::darcy_friction_factor(transition_start, roughness) ==
                64.0 / transition_start,
            "blend starts at the laminar value");
    require(friction::darcy_friction_factor(transition_end, roughness) ==
                friction::colebrook_turbulent(transition_end, roughness).darcy_friction_factor,
            "blend ends at the Colebrook value");

    const double midpoint = 2150.0;
    const double t = (midpoint - transition_start) / (transition_end - transition_start);
    const double weight = t * t * (3.0 - 2.0 * t);
    const double expected = (1.0 - weight) * (64.0 / midpoint) +
        weight * friction::colebrook_turbulent(midpoint, roughness).darcy_friction_factor;
    require(friction::darcy_friction_factor(midpoint, roughness) == expected,
            "transition interval uses the documented cubic smoothstep blend");

    constexpr double h = 1.0e-4;
    const double at_start = friction::darcy_friction_factor(transition_start, roughness);
    const double next_to_start = friction::darcy_friction_factor(transition_start + h, roughness);
    require(std::abs(next_to_start - at_start) < 1.0e-8,
            "friction factor is continuous at Re=2000");
    const double at_end = friction::darcy_friction_factor(transition_end, roughness);
    const double before_end = friction::darcy_friction_factor(transition_end - h, roughness);
    require(std::abs(at_end - before_end) < 1.0e-8,
            "Colebrook closure is continuous at Re=2300");

    constexpr double sj_end = 5000.0;
    const double sj_at_end = friction::darcy_friction_factor(
        sj_end, roughness, friction::Correlation::swamee_jain);
    require(sj_at_end == friction::swamee_jain_turbulent(sj_end, roughness),
            "Swamee-Jain blend ends at its supported turbulent value");
    require(friction::darcy_friction_factor(1500.0, roughness,
                friction::Correlation::swamee_jain) == 64.0 / 1500.0,
            "Swamee-Jain all-regime wrapper uses laminar law below Re=2000");
    const double sj_before_end = friction::darcy_friction_factor(
        sj_end - h, roughness, friction::Correlation::swamee_jain);
    require(std::abs(sj_at_end - sj_before_end) < 1.0e-8,
            "Swamee-Jain closure is continuous at Re=5000");
}

void test_swamee_jain() {
    const double re = 1.0e6;
    const double roughness = 1.0e-3;
    const double log_argument = roughness / 3.7 + 5.74 / std::pow(re, 0.9);
    const double expected = 0.25 / std::pow(std::log10(log_argument), 2.0);
    const double actual = friction::swamee_jain_turbulent(re, roughness);
    require(std::abs(actual - expected) <= 5.0e-15 * std::abs(expected),
            "Swamee-Jain matches its explicit equation to roundoff");

    for (double k : {1.0e-6, 1.0e-4, 1.0e-2, 5.0e-2}) {
        for (double r : {5000.0, 1.0e4, 1.0e6, 1.0e8}) {
            const double sj = friction::swamee_jain_turbulent(r, k);
            const double pow_reference = 0.25 / std::pow(
                std::log10(k / 3.7 + 5.74 / std::pow(r, 0.9)), 2.0);
            require(std::abs(sj - pow_reference) <= 5.0e-15 * std::abs(pow_reference),
                    "Swamee-Jain agrees with the pow reference to roundoff");
            const double cw = friction::colebrook_turbulent(r, k).darcy_friction_factor;
            const double relative_error = std::abs((sj - cw) / cw);
            require(relative_error < 0.03, "Swamee-Jain tracks Colebrook within a few percent over its working range");
        }
    }
    bool threw = false;
    try { (void)friction::swamee_jain_turbulent(4999.0, roughness); }
    catch (const std::domain_error&) { threw = true; }
    require(threw, "direct Swamee-Jain API rejects Reynolds numbers below 5000");
}

void test_invalid_inputs() {
    for (const auto& [re, k] : {std::pair{0.0, 0.0}, std::pair{1999.0, -1.0e-6},
                               std::pair{std::numeric_limits<double>::infinity(), 0.0},
                               std::pair{1.0e5, -1.0e-6},
                               std::pair{1.0e5, std::numeric_limits<double>::quiet_NaN()}}) {
        bool threw = false;
        try { (void)friction::darcy_friction_factor(re, k); }
        catch (const std::exception&) { threw = true; }
        require(threw, "invalid physical input is rejected");
    }
}

} // namespace

int main() {
    test_reference_grid();
    test_residual_and_trends();
    test_all_regime_wrapper();
    test_swamee_jain();
    test_invalid_inputs();
    std::cout << "All friction tests passed.\n";
}
