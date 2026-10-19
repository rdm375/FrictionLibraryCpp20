#include <friction/friction.hpp>

#include <iomanip>
#include <iostream>
#include <utility>

int main() {
    std::cout << std::setprecision(12);
    for (const auto& [re, roughness] : {
             std::pair{1500.0, 1.0e-3},
             std::pair{2150.0, 1.0e-3},
             std::pair{7.0e5, 1.0e-2},
             std::pair{1.0e8, 1.0e-4}}) {
        const double colebrook = friction::darcy_friction_factor(
            re, roughness, friction::Correlation::colebrook);
        const double swamee_jain = friction::darcy_friction_factor(
            re, roughness, friction::Correlation::swamee_jain);
        std::cout << "Re=" << re << "  k/D=" << roughness
                  << "  f_D(Colebrook)=" << colebrook
                  << "  f_D(Swamee-Jain)=" << swamee_jain << '\n';
    }
}
