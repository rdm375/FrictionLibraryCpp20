#include "friction/adaptive_tensor_surrogate.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Config = friction::surrogate::AdaptiveTensorConfig;
using Table = friction::surrogate::AdaptiveColebrookTensor;
constexpr std::size_t count = 1U << 20;
constexpr int repetitions = 12;
constexpr int block_size = 256;
constexpr double ln10 = 2.3025850929940456840179914546843642;

struct Result { double value, d_re, d_rr; };
struct Cell { double c[16]; };
struct Node { double value, du, dv, duv; };
struct Point { double re, rr; };
struct UniformData {
    std::vector<double> u, v;
    std::vector<Node> nodes;
    std::vector<Cell> cells;
};
struct Error {
    double max_value{}, max_scaled_dre{}, max_scaled_drr{};
    std::size_t fail_value{}, fail_dre{}, fail_drr{};
};

void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::cerr << where << ": " << cudaGetErrorString(e) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

__device__ int interval(const double* x, int n, double q) {
    if (q >= x[n - 1]) return n - 2;
    int lo = 0, hi = n - 1;
    while (lo + 1 < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (x[mid] <= q) lo = mid; else hi = mid;
    }
    return lo;
}

__device__ Result eval_cell(double re, double rr, double u, double v,
                            int i, int j, const double* u_axis,
                            const double* v_axis, int nv, const Cell* cells,
                            double u_span, double v_span, double rr_scale) {
    const double hu = u_axis[i + 1] - u_axis[i];
    const double hv = v_axis[j + 1] - v_axis[j];
    const double x = (u - u_axis[i]) / hu;
    const double y = (v - v_axis[j]) / hv;
    const Cell& cell = cells[static_cast<std::size_t>(i) * (nv - 1) + j];
    double row[4], row_dv[4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const double c0 = cell.c[a * 4], c1 = cell.c[a * 4 + 1];
        const double c2 = cell.c[a * 4 + 2], c3 = cell.c[a * 4 + 3];
        row[a] = ((c3 * y + c2) * y + c1) * y + c0;
        row_dv[a] = ((3.0 * c3 * y + 2.0 * c2) * y + c1) / hv;
    }
    const double z = ((row[3] * x + row[2]) * x + row[1]) * x + row[0];
    const double zu = ((3.0 * row[3] * x + 2.0 * row[2]) * x + row[1]) / hu;
    const double zv = ((row_dv[3] * x + row_dv[2]) * x + row_dv[1]) * x + row_dv[0];
    const double f = 1.0 / (z * z);
    const double scale = -2.0 / (z * z * z);
    return {f, scale * zu / (re * ln10 * u_span),
            scale * zv / ((rr_scale + rr) * v_span)};
}

__device__ Result eval_adaptive(double re, double rr, const double* u_axis,
                                const double* v_axis, int nu, int nv,
                                const Cell* cells, double u_span,
                                double v_span, double rr_scale) {
    const double u = log10(re / 2000.0) / u_span;
    const double v = log1p(rr / rr_scale) / v_span;
    const int i = interval(u_axis, nu, u);
    const int j = interval(v_axis, nv, v);
    return eval_cell(re, rr, u, v, i, j, u_axis, v_axis, nv, cells,
                     u_span, v_span, rr_scale);
}

__device__ Result eval_uniform(double re, double rr, const double* u_axis,
                               const double* v_axis, int nu, int nv,
                               const Cell* cells, double u_span,
                               double v_span, double rr_scale) {
    const double u = log10(re / 2000.0) / u_span;
    const double v = log1p(rr / rr_scale) / v_span;
    int i = static_cast<int>(u * (nu - 1));
    int j = static_cast<int>(v * (nv - 1));
    i = i < 0 ? 0 : (i >= nu - 1 ? nu - 2 : i);
    j = j < 0 ? 0 : (j >= nv - 1 ? nv - 2 : j);
    return eval_cell(re, rr, u, v, i, j, u_axis, v_axis, nv, cells,
                     u_span, v_span, rr_scale);
}

__device__ Result colebrook(double re, double rr) {
    const double x1 = rr * re * (ln10 / 18.574);
    const double x2 = log(re) - 0.7793974884556818028;
    double z = x2 - 0.2;
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const double q = x1 + z;
        const double e = (z + log(q) - x2) / (1.0 + q);
        z -= ((1.0 + q + 0.5 * e) / (1.0 + q + e * (1.0 + e / 3.0))) * e * q;
    }
    const double y = (2.0 / ln10) * z;
    const double f = 1.0 / (y * y), root = 1.0 / sqrt(f);
    const double a = rr / 3.7 + 2.51 * root / re;
    const double ry = 1.0 + (2.0 / ln10) * (2.51 / re) / a;
    const double yre = ((2.0 / ln10) * 2.51 * root / (re * re * a)) / ry;
    const double yrr = (-(2.0 / ln10) / (3.7 * a)) / ry;
    const double dfdy = -2.0 / (root * root * root);
    return {f, dfdy * yre, dfdy * yrr};
}

__global__ void direct_kernel(const Point* points, Result* out, std::size_t n) {
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < n) out[k] = colebrook(points[k].re, points[k].rr);
}
__global__ void tensor_kernel(const Point* points, Result* out, std::size_t n,
                              const double* u, const double* v, int nu, int nv,
                              const Cell* cells, double u_span, double v_span,
                              double rr_scale, bool adaptive) {
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const Point p = points[k];
    out[k] = adaptive
        ? eval_adaptive(p.re, p.rr, u, v, nu, nv, cells, u_span, v_span, rr_scale)
        : eval_uniform(p.re, p.rr, u, v, nu, nv, cells, u_span, v_span, rr_scale);
}

Node exact_node(const Config& cfg, double u, double v) {
    const double re_span = std::log10(cfg.reynolds_max / cfg.reynolds_min);
    const double rr_span = std::log1p(cfg.relative_roughness_max / cfg.roughness_scale);
    const double re = cfg.reynolds_min * std::pow(10.0, u * re_span);
    const double rr = cfg.roughness_scale * std::expm1(v * rr_span);
    const auto d = friction::analytic::darcy_friction_factor(
        re, rr, friction::Correlation::colebrook);
    const double y = 1.0 / std::sqrt(d.value);
    const double dg_df = -0.5 * y * y * y;
    return {y, dg_df * d.d_reynolds * (re * ln10 * re_span),
            dg_df * d.d_relative_roughness * ((cfg.roughness_scale + rr) * rr_span), 0.0};
}

double derivative_1d(const std::vector<double>& x, const std::vector<double>& y, std::size_t i) {
    if (x.size() < 3) return (y[1] - y[0]) / (x[1] - x[0]);
    if (i == 0) {
        const double h1=x[1]-x[0], h2=x[2]-x[1];
        return -(2*h1+h2)/(h1*(h1+h2))*y[0] + (h1+h2)/(h1*h2)*y[1] - h1/(h2*(h1+h2))*y[2];
    }
    if (i + 1 == x.size()) {
        const std::size_t n=x.size(); const double h1=x[n-2]-x[n-3], h2=x[n-1]-x[n-2];
        return h2/(h1*(h1+h2))*y[n-3] - (h1+h2)/(h1*h2)*y[n-2] + (h1+2*h2)/(h2*(h1+h2))*y[n-1];
    }
    const double h1=x[i]-x[i-1], h2=x[i+1]-x[i];
    return -h2/(h1*(h1+h2))*y[i-1] + (h2-h1)/(h1*h2)*y[i] + h1/(h2*(h1+h2))*y[i+1];
}

std::vector<Cell> make_cells(const Config& cfg, const std::vector<double>& u,
                             const std::vector<double>& v, std::vector<Node>& nodes) {
    const std::size_t nu=u.size(), nv=v.size();
    nodes.resize(nu*nv);
    auto at=[&](std::size_t i,std::size_t j)->Node& { return nodes[i*nv+j]; };
    for (std::size_t i=0;i<nu;++i) for (std::size_t j=0;j<nv;++j)
        at(i,j)=exact_node(cfg,u[i],v[j]);
    for (std::size_t i=0;i<nu;++i) {
        std::vector<double> row(nv);
        for (std::size_t j=0;j<nv;++j) row[j]=at(i,j).du;
        for (std::size_t j=0;j<nv;++j) at(i,j).duv=derivative_1d(v,row,j);
    }
    for (std::size_t j=0;j<nv;++j) {
        std::vector<double> col(nu);
        for (std::size_t i=0;i<nu;++i) col[i]=at(i,j).dv;
        for (std::size_t i=0;i<nu;++i)
            at(i,j).duv=0.5*(at(i,j).duv+derivative_1d(u,col,i));
    }
    static constexpr double m[4][4]={{1,0,0,0},{0,0,1,0},{-3,3,-2,-1},{2,-2,1,1}};
    std::vector<Cell> cells((nu-1)*(nv-1), Cell{});
    for (std::size_t i=0;i+1<nu;++i) for (std::size_t j=0;j+1<nv;++j) {
        const double hu=u[i+1]-u[i], hv=v[j+1]-v[j];
        double q[4][4]{};
        for (unsigned a=0;a<2;++a) for (unsigned b=0;b<2;++b) {
            const Node& n=at(i+a,j+b);
            q[a][b]=n.value; q[a][b+2]=hv*n.dv;
            q[a+2][b]=hu*n.du; q[a+2][b+2]=hu*hv*n.duv;
        }
        Cell& cell=cells[i*(nv-1)+j];
        for (unsigned a=0;a<4;++a) for (unsigned b=0;b<4;++b)
            for (unsigned p=0;p<4;++p) for (unsigned r=0;r<4;++r)
                cell.c[a*4+b]+=m[a][p]*q[p][r]*m[b][r];
    }
    return cells;
}

UniformData build_uniform(const Config& cfg, std::size_t nu, std::size_t nv) {
    UniformData out;
    out.u.resize(nu); out.v.resize(nv);
    for (std::size_t i=0;i<nu;++i) out.u[i]=static_cast<double>(i)/(nu-1);
    for (std::size_t j=0;j<nv;++j) out.v[j]=static_cast<double>(j)/(nv-1);
    out.cells=make_cells(cfg,out.u,out.v,out.nodes);
    return out;
}

Error compare(const Result* exact,const Result* approx,const Point* points) {
    constexpr double tol[3]={2.0e-4,2.0e-2,2.0e-3};
    Error e{};
    for (std::size_t k=0;k<count;++k) {
        if (!std::isfinite(approx[k].value) || !std::isfinite(approx[k].d_re) ||
            !std::isfinite(approx[k].d_rr)) {
            e.max_value=e.max_scaled_dre=e.max_scaled_drr=
                std::numeric_limits<double>::infinity();
            ++e.fail_value; ++e.fail_dre; ++e.fail_drr;
            continue;
        }
        const double scale=std::max(std::abs(exact[k].value),1.0e-300);
        const double value=std::abs(approx[k].value-exact[k].value)/scale;
        const double dre=points[k].re*std::abs(approx[k].d_re-exact[k].d_re)/scale;
        const double drr=(1.0e-5+points[k].rr)*std::abs(approx[k].d_rr-exact[k].d_rr)/scale;
        e.max_value=std::max(e.max_value,value);
        e.max_scaled_dre=std::max(e.max_scaled_dre,dre);
        e.max_scaled_drr=std::max(e.max_scaled_drr,drr);
        e.fail_value+=value>tol[0]; e.fail_dre+=dre>tol[1]; e.fail_drr+=drr>tol[2];
    }
    return e;
}

double checksum(const std::vector<Result>& values) {
    double s=0.0; for (const auto& x:values) s+=x.value+1e-3*x.d_re+1e-6*x.d_rr; return s;
}

void run(const char* label, bool adaptive, Point* d_points, const Point* h_points,
         const Result* exact, Result* d_out, std::vector<Result>& h_out,
         const double* d_u, const double* d_v, int nu, int nv, const Cell* d_cells,
         double u_span, double v_span, double rr_scale) {
    const dim3 block(block_size), grid(static_cast<unsigned>((count+block_size-1)/block_size));
    auto launch=[&] { tensor_kernel<<<grid,block>>>(d_points,d_out,count,d_u,d_v,nu,nv,d_cells,
                                                    u_span,v_span,rr_scale,adaptive); };
    launch(); check(cudaGetLastError(),"warmup interpolant launch"); check(cudaDeviceSynchronize(),"warmup sync");
    cudaEvent_t start{},stop{}; check(cudaEventCreate(&start),"event create"); check(cudaEventCreate(&stop),"event create");
    check(cudaEventRecord(start),"event start");
    for(int k=0;k<repetitions;++k) launch();
    check(cudaGetLastError(),"timed launch"); check(cudaEventRecord(stop),"event stop"); check(cudaEventSynchronize(stop),"event wait");
    float elapsed_ms=0; check(cudaEventElapsedTime(&elapsed_ms,start,stop),"event elapsed");
    check(cudaMemcpy(h_out.data(),d_out,count*sizeof(Result),cudaMemcpyDeviceToHost),"copy result");
    const Error e=compare(exact,h_out.data(),h_points);
    const double kernel_s=elapsed_ms*1e-3;
    const double kernel_rate=static_cast<double>(count)*repetitions/kernel_s/1e6;
    const auto e2e_start=Clock::now();
    for(int k=0;k<repetitions;++k) {
        check(cudaMemcpy(d_points,h_points,count*sizeof(Point),cudaMemcpyHostToDevice),"copy points in");
        launch(); check(cudaGetLastError(),"E2E launch");
        check(cudaMemcpy(h_out.data(),d_out,count*sizeof(Result),cudaMemcpyDeviceToHost),"copy points out");
    }
    const double e2e_s=std::chrono::duration<double>(Clock::now()-e2e_start).count();
    std::cout<<std::left<<std::setw(25)<<label<<std::right<<" kernel="<<std::fixed<<std::setprecision(1)<<kernel_rate
             <<" M/s E2E="<<(static_cast<double>(count)*repetitions/e2e_s/1e6)<<" M/s"
             <<" scaled max errors="<<std::scientific<<e.max_value<<'/'<<e.max_scaled_dre<<'/'<<e.max_scaled_drr
             <<" fails="<<e.fail_value<<'/'<<e.fail_dre<<'/'<<e.fail_drr
             <<" checksum="<<std::fixed<<std::setprecision(9)<<checksum(h_out)<<'\n';
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

} // namespace

int main() {
    Config cfg;
    Table adaptive_table(cfg);
    auto t0=Clock::now();
    const auto report=adaptive_table.build();
    const double adaptive_build_ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    const std::size_t nu=adaptive_table.u_nodes().size(), nv=adaptive_table.v_nodes().size();
    t0=Clock::now();
    UniformData uniform=build_uniform(cfg,nu,nv);
    const double uniform_build_ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();

    std::vector<Cell> adaptive_cells((nu-1)*(nv-1));
    for(std::size_t i=0;i<adaptive_cells.size();++i)
        for(int k=0;k<16;++k) adaptive_cells[i].c[k]=adaptive_table.cell_polynomials()[i].coefficients[k];
    const double u_span=std::log10(cfg.reynolds_max/cfg.reynolds_min);
    const double v_span=std::log1p(cfg.relative_roughness_max/cfg.roughness_scale);
    std::cout<<"2D bicubic Hermite tensor interpolation; same basis and node budget for both grids.\n"
             <<"Coordinates: u=normalized log10(Re), v=normalized log1p(rr/scale); derivatives included.\n"
             <<"Scaled validation limits (value, Re slope, roughness slope)="
             <<"2e-4, 2e-2, 2e-3.\n"
             <<"Grid adaptive="<<nu<<'x'<<nv<<" ("<<nu*nv<<" nodes), uniform="<<nu<<'x'<<nv
             <<" ("<<nu*nv<<" nodes); build ms adaptive/uniform="<<std::fixed<<std::setprecision(2)
             <<adaptive_build_ms<<'/'<<uniform_build_ms<<"; GPU coefficient MiB="
             <<(static_cast<double>(adaptive_cells.size()*sizeof(Cell))/(1024.0*1024.0))<<" each.\n";

    std::vector<Point> points(count); std::vector<Result> exact(count),output(count);
    std::mt19937_64 rng(20261020); std::uniform_real_distribution<double> unit(0.0,1.0);
    for(std::size_t k=0;k<count;++k) {
        const double u=unit(rng),v=unit(rng);
        points[k]={cfg.reynolds_min*std::pow(cfg.reynolds_max/cfg.reynolds_min,u),
                   cfg.roughness_scale*std::expm1(v*v_span)};
        const auto d=friction::analytic::darcy_friction_factor(points[k].re,points[k].rr,
                                                                friction::Correlation::colebrook);
        exact[k]={d.value,d.d_reynolds,d.d_relative_roughness};
    }
    int device=0; check(cudaGetDevice(&device),"get device"); cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop,device),"get properties");
    std::cout<<"GPU: "<<prop.name<<" (compute "<<prop.major<<'.'<<prop.minor<<")\n"
             <<"Batch="<<count<<", repetitions="<<repetitions<<"; kernel keeps data resident; E2E copies points and results.\n";
    Point *d_points=nullptr; Result*d_out=nullptr; double *d_ua=nullptr,*d_va=nullptr,*d_uu=nullptr,*d_vu=nullptr;
    Cell *d_ca=nullptr,*d_cu=nullptr;
    check(cudaMalloc((void**)&d_points,count*sizeof(Point)),"alloc points");
    check(cudaMalloc((void**)&d_out,count*sizeof(Result)),"alloc output");
    check(cudaMalloc((void**)&d_ua,nu*sizeof(double)),"alloc adaptive u"); check(cudaMalloc((void**)&d_va,nv*sizeof(double)),"alloc adaptive v");
    check(cudaMalloc((void**)&d_uu,nu*sizeof(double)),"alloc uniform u"); check(cudaMalloc((void**)&d_vu,nv*sizeof(double)),"alloc uniform v");
    check(cudaMalloc((void**)&d_ca,adaptive_cells.size()*sizeof(Cell)),"alloc adaptive cells"); check(cudaMalloc((void**)&d_cu,uniform.cells.size()*sizeof(Cell)),"alloc uniform cells");
    auto upload_start=Clock::now();
    check(cudaMemcpy(d_ua,adaptive_table.u_nodes().data(),nu*sizeof(double),cudaMemcpyHostToDevice),"upload adaptive u");
    check(cudaMemcpy(d_va,adaptive_table.v_nodes().data(),nv*sizeof(double),cudaMemcpyHostToDevice),"upload adaptive v");
    check(cudaMemcpy(d_uu,uniform.u.data(),nu*sizeof(double),cudaMemcpyHostToDevice),"upload uniform u");
    check(cudaMemcpy(d_vu,uniform.v.data(),nv*sizeof(double),cudaMemcpyHostToDevice),"upload uniform v");
    check(cudaMemcpy(d_ca,adaptive_cells.data(),adaptive_cells.size()*sizeof(Cell),cudaMemcpyHostToDevice),"upload adaptive cells");
    check(cudaMemcpy(d_cu,uniform.cells.data(),uniform.cells.size()*sizeof(Cell),cudaMemcpyHostToDevice),"upload uniform cells");
    check(cudaMemcpy(d_points,points.data(),count*sizeof(Point),cudaMemcpyHostToDevice),"upload points");
    const double upload_ms=std::chrono::duration<double,std::milli>(Clock::now()-upload_start).count();
    std::cout<<"GPU initial upload="<<std::fixed<<std::setprecision(3)<<upload_ms<<" ms (outside kernel timings).\n";
    // Direct baseline, then same-size regular and adaptive tensors.
    const dim3 block(block_size),grid(static_cast<unsigned>((count+block_size-1)/block_size));
    direct_kernel<<<grid,block>>>(d_points,d_out,count); check(cudaGetLastError(),"direct warmup"); check(cudaDeviceSynchronize(),"direct sync");
    cudaEvent_t ds{},de{}; check(cudaEventCreate(&ds),"direct event");check(cudaEventCreate(&de),"direct event");
    check(cudaEventRecord(ds),"direct start");for(int k=0;k<repetitions;++k)direct_kernel<<<grid,block>>>(d_points,d_out,count);
    check(cudaEventRecord(de),"direct stop");check(cudaEventSynchronize(de),"direct wait");float direct_ms=0;check(cudaEventElapsedTime(&direct_ms,ds,de),"direct time");
    std::cout<<"Direct Colebrook kernel="<<std::fixed<<std::setprecision(1)
             <<(static_cast<double>(count)*repetitions/(direct_ms*1e-3)/1e6)<<" M/s\n";
    run("Uniform regular grid",false,d_points,points.data(),exact.data(),d_out,output,d_uu,d_vu,
        static_cast<int>(nu),static_cast<int>(nv),d_cu,u_span,v_span,cfg.roughness_scale);
    run("Adaptive sampled grid",true,d_points,points.data(),exact.data(),d_out,output,d_ua,d_va,
        static_cast<int>(nu),static_cast<int>(nv),d_ca,u_span,v_span,cfg.roughness_scale);
    cudaEventDestroy(ds);cudaEventDestroy(de);
    cudaFree(d_points);cudaFree(d_out);cudaFree(d_ua);cudaFree(d_va);cudaFree(d_uu);cudaFree(d_vu);cudaFree(d_ca);cudaFree(d_cu);
    return EXIT_SUCCESS;
}
