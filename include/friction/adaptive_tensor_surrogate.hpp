#pragma once

#include "friction/analytic_derivatives.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace friction::surrogate {

// Experimental two-input Colebrook surrogate. It stores an adaptively refined
// tensor grid in normalized log(Re) and log1p(relative_roughness / scale)
// coordinates. A bicubic Hermite patch returns the friction factor and its two
// first derivatives. Refinement combines Taylor-model disagreement with direct
// midpoint checks. The latter makes construction more expensive but provides
// a practical error control; neither indicator is a formal error certificate.
struct AdaptiveTensorConfig {
    double reynolds_min{2000.0};
    double reynolds_max{1.0e8};
    double relative_roughness_max{0.05};
    double roughness_scale{1.0e-5};
    double relative_tolerance{2.0e-4};
    unsigned max_axis_points{1025};
    std::size_t max_grid_points{200000};
};

struct AdaptiveTensorBuildReport {
    std::size_t grid_points{};
    std::size_t reynolds_axis_points{};
    std::size_t roughness_axis_points{};
    std::size_t refinement_rounds{};
    std::size_t cells_checked{};
    std::size_t estimated_storage_bytes{};
    double largest_unresolved_indicator{};
};

class AdaptiveColebrookTensor {
    struct Surface { double value{},du{},dv{}; };

public:
    // Public so accelerator callers can upload the prebuilt cell coefficients.
    struct CellPolynomial { std::array<double,16> coefficients{}; };

    struct Evaluation {
        analytic::FirstDerivatives derivatives;
        bool used_surrogate{};
    };

    // Reuse this context when many evaluations share one pipe roughness. It
    // caches the transformed roughness coordinate and reduces each bicubic
    // patch to a one-dimensional cubic in the Reynolds coordinate.
    class PreparedRoughness {
    public:
        struct Row {
            double value[4]{};
            double d_roughness[4]{};
        };
        struct DeviceView {
            const double* reynolds_nodes{};
            const double* inverse_reynolds_widths{};
            const std::size_t* interval_bins{};
            const Row* rows{};
            std::size_t row_count{};
            std::size_t interval_bin_count{};
            double log_re_min{};
            double inverse_log_re_span{};
            double inverse_re_derivative_scale{};
            double roughness_derivative_scale{};
            double reynolds_min{};
            double reynolds_max{};
        };

        [[nodiscard]] Evaluation evaluate(double reynolds) const;
        [[nodiscard]] double relative_roughness() const noexcept { return rr_; }
        [[nodiscard]] DeviceView device_view() const {
            if (!owner_) throw std::logic_error("prepared roughness has no source table");
            return {owner_->u_nodes_.data(), owner_->inv_hu_.data(),
                    owner_->u_interval_bins_.data(), rows_.data(), rows_.size(),
                    interval_bin_count, owner_->log_re_min_, owner_->inv_log_re_span_,
                    owner_->inv_re_derivative_scale_, roughness_derivative_scale_,
                    owner_->config_.reynolds_min, owner_->config_.reynolds_max};
        }

    private:
        friend class AdaptiveColebrookTensor;
        PreparedRoughness() = default;
        const AdaptiveColebrookTensor* owner_{};
        double rr_{};
        double roughness_derivative_scale_{};
        std::size_t generation_{};
        std::vector<Row> rows_;
    };

    explicit AdaptiveColebrookTensor(AdaptiveTensorConfig config = {})
        : config_(config),
          log_re_min_(std::log10(config.reynolds_min)),
          log_re_span_(std::log10(config.reynolds_max/config.reynolds_min)),
          inv_log_re_span_(1.0/log_re_span_),
          log_rr_span_(std::log1p(config.relative_roughness_max/config.roughness_scale)),
          inv_log_rr_span_(1.0/log_rr_span_),
          inv_re_derivative_scale_(1.0/(ln10*log_re_span_)),
          inv_roughness_scale_(1.0/config.roughness_scale) { validate_config(); }

    // Read-only normalized knot axes and bicubic cell coefficients.
    [[nodiscard]] const std::vector<double>& u_nodes() const noexcept { return u_nodes_; }
    [[nodiscard]] const std::vector<double>& v_nodes() const noexcept { return v_nodes_; }
    [[nodiscard]] const std::vector<CellPolynomial>& cell_polynomials() const noexcept {
        return polynomials_;
    }

    AdaptiveTensorBuildReport build() {
        ++generation_;
        u_nodes_ = {0.0, 0.5, 1.0};
        v_nodes_ = {0.0, 0.5, 1.0};
        report_ = {};
        sample_grid();

        for (;;) {
            std::vector<bool> split_u(u_nodes_.size()-1, false);
            std::vector<bool> split_v(v_nodes_.size()-1, false);
            double worst = 0.0;
            for (std::size_t i=0; i+1<u_nodes_.size(); ++i) {
                for (std::size_t j=0; j+1<v_nodes_.size(); ++j) {
                    ++report_.cells_checked;
                    const CellCheck check = check_cell(i,j);
                    worst = std::max(worst, check.indicator);
                    if (check.indicator <= config_.relative_tolerance) continue;
                    if (check.u_indicator >= 0.35*check.indicator) split_u[i]=true;
                    if (check.v_indicator >= 0.35*check.indicator) split_v[j]=true;
                    if (!split_u[i] && !split_v[j]) {
                        if (check.u_indicator >= check.v_indicator) split_u[i]=true;
                        else split_v[j]=true;
                    }
                }
            }
            report_.largest_unresolved_indicator = worst;
            if (std::none_of(split_u.begin(),split_u.end(),[](bool x){return x;}) &&
                std::none_of(split_v.begin(),split_v.end(),[](bool x){return x;})) break;

            std::vector<double> next_u, next_v;
            next_u.reserve(u_nodes_.size()*2);
            next_v.reserve(v_nodes_.size()*2);
            for (std::size_t i=0; i+1<u_nodes_.size(); ++i) {
                next_u.push_back(u_nodes_[i]);
                if (split_u[i]) next_u.push_back(0.5*(u_nodes_[i]+u_nodes_[i+1]));
            }
            next_u.push_back(u_nodes_.back());
            for (std::size_t j=0; j+1<v_nodes_.size(); ++j) {
                next_v.push_back(v_nodes_[j]);
                if (split_v[j]) next_v.push_back(0.5*(v_nodes_[j]+v_nodes_[j+1]));
            }
            next_v.push_back(v_nodes_.back());
            if (next_u.size()>config_.max_axis_points ||
                next_v.size()>config_.max_axis_points ||
                next_u.size()*next_v.size()>config_.max_grid_points) break;
            u_nodes_.swap(next_u);
            v_nodes_.swap(next_v);
            ++report_.refinement_rounds;
            sample_grid();
        }
        report_.grid_points = grid_.size();
        report_.reynolds_axis_points = u_nodes_.size();
        report_.roughness_axis_points = v_nodes_.size();
        report_.estimated_storage_bytes = grid_.capacity()*sizeof(Node) +
            polynomials_.capacity()*sizeof(CellPolynomial) +
            (u_nodes_.capacity()+v_nodes_.capacity())*sizeof(double);
        build_interval_bins();
        report_.estimated_storage_bytes +=
            (u_interval_bins_.capacity()+v_interval_bins_.capacity())*sizeof(std::size_t) +
            (inv_hu_.capacity()+inv_hv_.capacity())*sizeof(double);
        return report_;
    }

    [[nodiscard]] bool is_built() const noexcept { return !grid_.empty(); }
    [[nodiscard]] bool contains(double re, double rr) const noexcept {
        return std::isfinite(re) && std::isfinite(rr) &&
            re>=config_.reynolds_min && re<=config_.reynolds_max &&
            rr>=0.0 && rr<=config_.relative_roughness_max;
    }
    [[nodiscard]] const AdaptiveTensorConfig& config() const noexcept { return config_; }
    [[nodiscard]] const AdaptiveTensorBuildReport& report() const noexcept { return report_; }

    // Out-of-domain calls use the direct analytic correlation, so laminar flow
    // and optimizer excursions retain the library's existing behavior.
    [[nodiscard]] Evaluation evaluate(double re, double rr) const {
        if (!is_built()) throw std::logic_error("adaptive Colebrook table has not been built");
        if (!contains(re,rr)) return {direct(re,rr),false};
        const double u=to_u(re), v=to_v(rr);
        const std::size_t i=fast_interval(u_nodes_,u_interval_bins_,u);
        const std::size_t j=fast_interval(v_nodes_,v_interval_bins_,v);
        return finish_evaluation(re,rr,interpolate(i,j,u,v),
                                 inv_log_rr_span_/(config_.roughness_scale+rr));
    }

    [[nodiscard]] PreparedRoughness prepare_roughness(double rr) const {
        if (!is_built()) throw std::logic_error("adaptive Colebrook table has not been built");
        if (!std::isfinite(rr) || rr<0.0 || rr>config_.relative_roughness_max)
            throw std::invalid_argument("prepared roughness must be inside the table domain");
        PreparedRoughness prepared;
        prepared.owner_=this;
        prepared.rr_=rr;
        prepared.generation_=generation_;
        prepared.roughness_derivative_scale_=inv_log_rr_span_/(config_.roughness_scale+rr);
        const double v=to_v(rr);
        const std::size_t j=fast_interval(v_nodes_,v_interval_bins_,v);
        const double y=(v-v_nodes_[j])*inv_hv_[j];
        prepared.rows_.resize(u_nodes_.size()-1);
        const std::size_t nv=v_nodes_.size()-1;
        for (std::size_t i=0;i<prepared.rows_.size();++i) {
            const auto& c=polynomials_[i*nv+j].coefficients;
            auto& row=prepared.rows_[i];
            for (unsigned a=0;a<4;++a) {
                const double c0=c[a*4],c1=c[a*4+1],c2=c[a*4+2],c3=c[a*4+3];
                row.value[a]=((c3*y+c2)*y+c1)*y+c0;
                row.d_roughness[a]=((3.0*c3*y+2.0*c2)*y+c1)*inv_hv_[j];
            }
        }
        return prepared;
    }

private:
    [[nodiscard]] Evaluation evaluate_prepared(const PreparedRoughness& prepared,
                                               double re) const {
        if (prepared.owner_!=this)
            throw std::invalid_argument("prepared roughness belongs to another table");
        if (prepared.generation_!=generation_)
            throw std::logic_error("prepared roughness was invalidated by rebuilding its table");
        if (!contains(re,prepared.rr_)) return {direct(re,prepared.rr_),false};
        const double u=to_u(re);
        const std::size_t i=fast_interval(u_nodes_,u_interval_bins_,u);
        const double x=(u-u_nodes_[i])*inv_hu_[i];
        const auto& row=prepared.rows_[i];
        const double value=((row.value[3]*x+row.value[2])*x+row.value[1])*x+row.value[0];
        const double du=((3.0*row.value[3]*x+2.0*row.value[2])*x+row.value[1])*inv_hu_[i];
        const double dv=((row.d_roughness[3]*x+row.d_roughness[2])*x+
                         row.d_roughness[1])*x+row.d_roughness[0];
        return finish_evaluation(re,prepared.rr_,{value,du,dv},
                                 prepared.roughness_derivative_scale_);
    }

    [[nodiscard]] Evaluation finish_evaluation(double re,double rr,Surface g,
                                                double roughness_derivative_scale) const {
        if (!(g.value>0.0) || !std::isfinite(g.value) ||
            !std::isfinite(g.du) || !std::isfinite(g.dv)) return {direct(re,rr),false};
        const double inv_g=1.0/g.value;
        const double f=inv_g*inv_g;
        const double df_du=-2.0*g.du*f*inv_g;
        const double df_dv=-2.0*g.dv*f*inv_g;
        analytic::FirstDerivatives result{
            f,df_du*(inv_re_derivative_scale_/re),
            df_dv*roughness_derivative_scale};
        if (!(f>0.0) || !std::isfinite(f) || !std::isfinite(result.d_reynolds) ||
            !std::isfinite(result.d_relative_roughness)) return {direct(re,rr),false};
        return {result,true};
    }

    static constexpr double ln10=2.3025850929940456840179914546843642;
    struct Node { double value{},du{},dv{},duv{}; }; // y=1/sqrt(f), in transformed coordinates
    struct CellCheck { double indicator{},u_indicator{},v_indicator{}; };

    AdaptiveTensorConfig config_;
    std::size_t generation_{};
    // Cache invariant transforms once. Recomputing log spans in evaluate() was
    // doing two unnecessary libm calls for every table lookup.
    double log_re_min_{};
    double log_re_span_{};
    double inv_log_re_span_{};
    double log_rr_span_{};
    double inv_log_rr_span_{};
    double inv_re_derivative_scale_{};
    double inv_roughness_scale_{};
    AdaptiveTensorBuildReport report_{};
    std::vector<double> u_nodes_,v_nodes_;
    std::vector<double> inv_hu_,inv_hv_;
    std::vector<Node> grid_;
    std::vector<CellPolynomial> polynomials_;
    static constexpr std::size_t interval_bin_count=2048;
    std::vector<std::size_t> u_interval_bins_,v_interval_bins_;

    void validate_config() const {
        if (!std::isfinite(config_.reynolds_min) || !std::isfinite(config_.reynolds_max) ||
            config_.reynolds_min<2000.0 || config_.reynolds_max<=config_.reynolds_min)
            throw std::invalid_argument("surrogate requires 2000 <= Re_min < Re_max");
        if (!std::isfinite(config_.relative_roughness_max) ||
            config_.relative_roughness_max<=0.0 || config_.relative_roughness_max>=3.7)
            throw std::invalid_argument("relative roughness maximum must be in (0, 3.7)");
        if (!std::isfinite(config_.roughness_scale) || config_.roughness_scale<=0.0 ||
            !std::isfinite(config_.relative_tolerance) || config_.relative_tolerance<=0.0)
            throw std::invalid_argument("roughness scale and tolerance must be positive");
        if (config_.max_axis_points<3 || config_.max_grid_points<9)
            throw std::invalid_argument("grid limits are too small");
    }
    [[nodiscard]] double log_re_span() const { return log_re_span_; }
    [[nodiscard]] double to_u(double re) const { return (std::log10(re)-log_re_min_)*inv_log_re_span_; }
    [[nodiscard]] double to_v(double rr) const { return std::log1p(rr*inv_roughness_scale_)*inv_log_rr_span_; }
    [[nodiscard]] double from_u(double u) const { return config_.reynolds_min*std::pow(10.0,u*log_re_span()); }
    [[nodiscard]] double from_v(double v) const { return config_.roughness_scale*std::expm1(v*log_rr_span_); }
    [[nodiscard]] std::size_t index(std::size_t i,std::size_t j) const { return i*v_nodes_.size()+j; }
    [[nodiscard]] static std::size_t interval(const std::vector<double>& x,double q) {
        if (q>=x.back()) return x.size()-2;
        const auto it=std::upper_bound(x.begin(),x.end(),q);
        return static_cast<std::size_t>(it-x.begin()-1);
    }
    void build_interval_bins() {
        auto fill=[](const std::vector<double>& nodes,std::vector<std::size_t>& bins) {
            bins.resize(interval_bin_count);
            for (std::size_t b=0;b<interval_bin_count;++b)
                bins[b]=interval(nodes,static_cast<double>(b)/interval_bin_count);
        };
        fill(u_nodes_,u_interval_bins_);
        fill(v_nodes_,v_interval_bins_);
    }
    [[nodiscard]] static std::size_t fast_interval(const std::vector<double>& nodes,
                                                   const std::vector<std::size_t>& bins,
                                                   double q) {
        if (q>=1.0) return nodes.size()-2;
        const std::size_t b=std::min(static_cast<std::size_t>(q*interval_bin_count),
                                     interval_bin_count-1);
        std::size_t i=bins[b];
        while (i+1<nodes.size()-1 && nodes[i+1]<=q) ++i;
        while (i>0 && nodes[i]>q) --i;
        return i;
    }

    [[nodiscard]] Node exact_node(double u,double v) const {
        const double re=from_u(u), rr=from_v(v);
        const auto d=analytic::darcy_friction_factor(re,rr,Correlation::colebrook);
        const double y=1.0/std::sqrt(d.value);
        const double dg_df=-0.5*y*y*y;
        return {y,
                dg_df*d.d_reynolds*(re*ln10*log_re_span()),
                dg_df*d.d_relative_roughness*((config_.roughness_scale+rr)*log_rr_span_),
                0.0};
    }

    static double derivative_1d(const std::vector<double>& x,
                                const std::vector<double>& y,std::size_t i) {
        if (x.size()<3) return (y[1]-y[0])/(x[1]-x[0]);
        if (i==0) {
            const double h1=x[1]-x[0],h2=x[2]-x[1];
            return -(2*h1+h2)/(h1*(h1+h2))*y[0] + (h1+h2)/(h1*h2)*y[1] -
                h1/(h2*(h1+h2))*y[2];
        }
        if (i+1==x.size()) {
            const std::size_t n=x.size();
            const double h1=x[n-2]-x[n-3],h2=x[n-1]-x[n-2];
            return h2/(h1*(h1+h2))*y[n-3] - (h1+h2)/(h1*h2)*y[n-2] +
                (h1+2*h2)/(h2*(h1+h2))*y[n-1];
        }
        const double h1=x[i]-x[i-1],h2=x[i+1]-x[i];
        return -h2/(h1*(h1+h2))*y[i-1] + (h2-h1)/(h1*h2)*y[i] +
            h1/(h2*(h1+h2))*y[i+1];
    }

    void sample_grid() {
        grid_.resize(u_nodes_.size()*v_nodes_.size());
        for (std::size_t i=0;i<u_nodes_.size();++i)
            for (std::size_t j=0;j<v_nodes_.size();++j)
                grid_[index(i,j)]=exact_node(u_nodes_[i],v_nodes_[j]);
        // Cross partials are estimated from the exact first derivatives. They
        // complete the bicubic Hermite data while keeping the oracle first-order.
        for (std::size_t i=0;i<u_nodes_.size();++i) {
            std::vector<double> gu(v_nodes_.size()),xv=v_nodes_;
            for (std::size_t j=0;j<v_nodes_.size();++j) gu[j]=grid_[index(i,j)].du;
            for (std::size_t j=0;j<v_nodes_.size();++j)
                grid_[index(i,j)].duv=derivative_1d(xv,gu,j);
        }
        for (std::size_t j=0;j<v_nodes_.size();++j) {
            std::vector<double> gv(u_nodes_.size()),xu=u_nodes_;
            for (std::size_t i=0;i<u_nodes_.size();++i) gv[i]=grid_[index(i,j)].dv;
            for (std::size_t i=0;i<u_nodes_.size();++i) {
                const double from_v=grid_[index(i,j)].duv;
                const double from_u=derivative_1d(xu,gv,i);
                grid_[index(i,j)].duv=0.5*(from_v+from_u);
            }
        }
        build_polynomials();
    }

    void build_polynomials() {
        static constexpr double m[4][4]={{1,0,0,0},{0,0,1,0},
                                          {-3,3,-2,-1},{2,-2,1,1}};
        const std::size_t nu=u_nodes_.size()-1,nv=v_nodes_.size()-1;
        inv_hu_.resize(nu);
        inv_hv_.resize(nv);
        for (std::size_t i=0;i<nu;++i) inv_hu_[i]=1.0/(u_nodes_[i+1]-u_nodes_[i]);
        for (std::size_t j=0;j<nv;++j) inv_hv_[j]=1.0/(v_nodes_[j+1]-v_nodes_[j]);
        polynomials_.assign(nu*nv,CellPolynomial{});
        for (std::size_t i=0;i<nu;++i) for (std::size_t j=0;j<nv;++j) {
            const double hu=u_nodes_[i+1]-u_nodes_[i],hv=v_nodes_[j+1]-v_nodes_[j];
            double q[4][4]{};
            for (unsigned a=0;a<2;++a) for (unsigned b=0;b<2;++b) {
                const Node& n=grid_[index(i+a,j+b)];
                q[a][b]=n.value;
                q[a][b+2]=hv*n.dv;
                q[a+2][b]=hu*n.du;
                q[a+2][b+2]=hu*hv*n.duv;
            }
            auto& c=polynomials_[i*nv+j].coefficients;
            for (unsigned a=0;a<4;++a) for (unsigned b=0;b<4;++b)
                for (unsigned p=0;p<4;++p) for (unsigned r=0;r<4;++r)
                    c[a*4+b]+=m[a][p]*q[p][r]*m[b][r];
        }
    }

    [[nodiscard]] Surface interpolate(std::size_t i,std::size_t j,double u,double v) const {
        const double inv_hu=inv_hu_[i],inv_hv=inv_hv_[j];
        const double x=(u-u_nodes_[i])*inv_hu,y=(v-v_nodes_[j])*inv_hv;
        const auto& c=polynomials_[i*(v_nodes_.size()-1)+j].coefficients;
        double row[4],drow_dv[4];
        for (unsigned a=0;a<4;++a) {
            const double c0=c[a*4],c1=c[a*4+1],c2=c[a*4+2],c3=c[a*4+3];
            row[a]=((c3*y+c2)*y+c1)*y+c0;
            drow_dv[a]=((3.0*c3*y+2.0*c2)*y+c1)*inv_hv;
        }
        const double value=((row[3]*x+row[2])*x+row[1])*x+row[0];
        const double du=((3.0*row[3]*x+2.0*row[2])*x+row[1])*inv_hu;
        const double dv=((drow_dv[3]*x+drow_dv[2])*x+drow_dv[1])*x+drow_dv[0];
        return {value,du,dv};
    }

    [[nodiscard]] CellCheck check_cell(std::size_t i,std::size_t j) const {
        const double u0=u_nodes_[i],u1=u_nodes_[i+1],v0=v_nodes_[j],v1=v_nodes_[j+1];
        const double uc=0.5*(u0+u1),vc=0.5*(v0+v1);
        const double hu=u1-u0,hv=v1-v0;
        const Node exact=exact_node(uc,vc);
        const Surface pred=interpolate(i,j,uc,vc);
        const double scale=std::max(1.0e-12,std::abs(exact.value));
        const double direct_defect=std::max({
            std::abs(exact.value-pred.value)/scale,
            hu*std::abs(exact.du-pred.du)/scale,
            hv*std::abs(exact.dv-pred.dv)/scale});
        std::array<double,4> taylor{};
        double gu_min=std::numeric_limits<double>::infinity(),gu_max=-gu_min;
        double gv_min=gu_min,gv_max=-gu_min;
        for (unsigned a=0;a<2;++a) for (unsigned b=0;b<2;++b) {
            const Node& n=grid_[index(i+a,j+b)];
            const double x=a?u1:u0,y=b?v1:v0;
            taylor[a+2U*b]=n.value+n.du*(uc-x)+n.dv*(vc-y);
            gu_min=std::min(gu_min,n.du);gu_max=std::max(gu_max,n.du);
            gv_min=std::min(gv_min,n.dv);gv_max=std::max(gv_max,n.dv);
        }
        const double u_dis=std::max(std::abs(taylor[0]-taylor[1]),std::abs(taylor[2]-taylor[3]))+
            0.25*hu*(gu_max-gu_min);
        const double v_dis=std::max(std::abs(taylor[0]-taylor[2]),std::abs(taylor[1]-taylor[3]))+
            0.25*hv*(gv_max-gv_min);
        const double u_indicator=u_dis/scale;
        const double v_indicator=v_dis/scale;
        return {std::max({direct_defect,u_indicator,v_indicator}),u_indicator,v_indicator};
    }

    [[nodiscard]] analytic::FirstDerivatives direct(double re,double rr) const {
        return analytic::darcy_friction_factor(re,rr,Correlation::colebrook);
    }
};

inline AdaptiveColebrookTensor::Evaluation
AdaptiveColebrookTensor::PreparedRoughness::evaluate(double reynolds) const {
    if (!owner_) throw std::logic_error("prepared roughness has no source table");
    return owner_->evaluate_prepared(*this,reynolds);
}

} // namespace friction::surrogate
