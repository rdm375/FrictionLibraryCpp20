#pragma once

namespace friction::ad {

struct FirstDerivatives {
    double value{};
    double d_reynolds{};
    double d_relative_roughness{};
};

// These functions are defined in friction_enzyme_ad and require the optional
// ClangEnzyme-compiled library target.
[[nodiscard]] FirstDerivatives colebrook_turbulent(double reynolds_number,
                                                    double relative_roughness);
[[nodiscard]] FirstDerivatives swamee_jain_turbulent(double reynolds_number,
                                                      double relative_roughness);
[[nodiscard]] FirstDerivatives darcy_friction_factor(
    double reynolds_number,
    double relative_roughness,
    bool use_swamee_jain = false);

} // namespace friction::ad
