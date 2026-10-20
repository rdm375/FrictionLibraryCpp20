// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Richard Myers

#include "friction/analytic_derivatives.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {
constexpr double re_min = 2300.0, re_max = 1.0e8;
constexpr double roughness = 6.971775e-4;
constexpr double u_min = 7.740664401917241, u_max = 18.420680743952367;
constexpr double u_span = u_max - u_min;
constexpr std::size_t uniform_n = 8192, bucket_n = 256, batch_n = 1U << 20;
constexpr int reps = 20, block_n = 256, max_depth = 24;
constexpr double rtol[3] = {2e-7, 2e-3, 2e-6};
constexpr double atol[3] = {2e-12, 2e-12, 2e-10};

struct Result { double f, dre, drr; };
struct Segment { double u0, invdu, f0, df, rr0, drr; };
struct UniformSegment { double f0, df, rr0, drr; };
struct Sample { double u, re; Result r; };
struct Error { double abs[3]{}, rel[3]{}; std::size_t fail[3]{}; };
struct BuildStats { std::size_t probes{}, unresolved{}; int max_depth{}; };

__device__ __forceinline__ Result colebrook(double re, double rr) {
    constexpr double ln10 = 2.3025850929940456840179914546843642;
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

__device__ __forceinline__ Result line_eval(double re, double u, const Segment& s) {
    const double t = fmin(1.0, fmax(0.0, (u - s.u0) * s.invdu));
    return {s.f0 + s.df * t, s.df * s.invdu / re, s.rr0 + s.drr * t};
}

__device__ __forceinline__ Result eval_uniform(double re, const UniformSegment* s) {
    const double u = log(re);
    double q = fmax(0.0, (u - u_min) * (static_cast<double>(uniform_n) / u_span));
    std::size_t i = static_cast<std::size_t>(q);
    if (i >= uniform_n) i = uniform_n - 1;
    const double t = fmin(1.0, fmax(0.0, q - static_cast<double>(i)));
    return {s[i].f0 + s[i].df * t,
            s[i].df * static_cast<double>(uniform_n) / (u_span * re),
            s[i].rr0 + s[i].drr * t};
}

__device__ __forceinline__ Result eval_global(double re, const Segment* s, std::size_t n) {
    const double u = log(re);
    std::size_t lo = 0, hi = n - 1;
    while (lo < hi) {
        const std::size_t m = lo + (hi - lo + 1) / 2;
        if (s[m].u0 <= u) lo = m; else hi = m - 1;
    }
    return line_eval(re, u, s[lo]);
}

__device__ __forceinline__ Result eval_bucket(double re, const Segment* s,
                                              const std::size_t* offsets) {
    const double u = log(re);
    std::size_t b = static_cast<std::size_t>(fmax(0.0,
        (u - u_min) * (static_cast<double>(bucket_n) / u_span)));
    if (b >= bucket_n) b = bucket_n - 1;
    std::size_t lo = offsets[b], hi = offsets[b + 1] - 1;
    while (lo < hi) {
        const std::size_t m = lo + (hi - lo + 1) / 2;
        if (s[m].u0 <= u) lo = m; else hi = m - 1;
    }
    return line_eval(re, u, s[lo]);
}

__global__ void direct_kernel(const double* re, Result* out, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = colebrook(re[i], roughness);
}
__global__ void uniform_kernel(const double* re, Result* out, std::size_t n,
                               const UniformSegment* s) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = eval_uniform(re[i], s);
}
__global__ void global_kernel(const double* re, Result* out, std::size_t n,
                              const Segment* s, std::size_t ns) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = eval_global(re[i], s, ns);
}
__global__ void bucket_kernel(const double* re, Result* out, std::size_t n,
                              const Segment* s, const std::size_t* offsets) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = eval_bucket(re[i], s, offsets);
}

Sample sample(double u) {
    const double re = std::exp(u);
    auto x = friction::analytic::darcy_friction_factor(re, roughness,
                                                        friction::Correlation::colebrook);
    return {u, re, {x.value, x.d_reynolds, x.d_relative_roughness}};
}
bool within(double e, int k, double ref) { return e <= atol[k] + rtol[k] * std::abs(ref); }
bool accept(const Sample& a, const Sample& b, const Sample& m) {
    const double h = b.u - a.u, slope = (b.r.f - a.r.f) / h;
    const double chord = 0.5 * (a.r.f + b.r.f);
    const double tl = a.r.f + 0.5 * h * a.re * a.r.dre;
    const double tr = b.r.f - 0.5 * h * b.re * b.r.dre;
    if (!within(std::abs(tl - chord), 0, m.r.f) ||
        !within(std::abs(tr - chord), 0, m.r.f) ||
        !within(std::abs(m.r.f - chord), 0, m.r.f)) return false;
    if (!within(std::abs(slope / a.re - a.r.dre), 1, a.r.dre) ||
        !within(std::abs(slope / b.re - b.r.dre), 1, b.r.dre) ||
        !within(std::abs(slope / m.re - m.r.dre), 1, m.r.dre)) return false;
    return within(std::abs(m.r.drr - 0.5 * (a.r.drr + b.r.drr)), 2, m.r.drr);
}
void refine(const Sample& a, const Sample& b, int depth, std::vector<Sample>& knots,
            BuildStats& stats) {
    stats.max_depth = std::max(stats.max_depth, depth);
    const Sample m = sample(0.5 * (a.u + b.u));
    ++stats.probes;
    if (accept(a, b, m)) { knots.push_back(b); return; }
    if (depth >= max_depth) { ++stats.unresolved; knots.push_back(b); return; }
    refine(a, m, depth + 1, knots, stats);
    refine(m, b, depth + 1, knots, stats);
}
Segment make_segment(const Sample& a, const Sample& b) {
    const double inv = 1.0 / (b.u - a.u);
    return {a.u, inv, a.r.f, b.r.f - a.r.f, a.r.drr, b.r.drr - a.r.drr};
}
std::vector<UniformSegment> build_uniform() {
    std::vector<Sample> p(uniform_n + 1);
    for (std::size_t i = 0; i <= uniform_n; ++i)
        p[i] = sample(u_min + u_span * (static_cast<double>(i) / uniform_n));
    std::vector<UniformSegment> s(uniform_n);
    for (std::size_t i = 0; i < uniform_n; ++i)
        s[i] = {p[i].r.f, p[i+1].r.f-p[i].r.f,
                p[i].r.drr, p[i+1].r.drr-p[i].r.drr};
    return s;
}
std::vector<Segment> build_global(BuildStats& stats) {
    Sample a = sample(u_min), b = sample(u_max);
    std::vector<Sample> p{a}; p.reserve(4096);
    refine(a, b, 0, p, stats);
    std::vector<Segment> s; s.reserve(p.size()-1);
    for (std::size_t i=0; i+1<p.size(); ++i) s.push_back(make_segment(p[i],p[i+1]));
    return s;
}
std::vector<Segment> build_bucketed(std::vector<std::size_t>& offsets,
                                    BuildStats& stats, std::size_t& max_local) {
    std::vector<Segment> s; s.reserve(4096); offsets.assign(bucket_n+1,0);
    Sample a = sample(u_min);
    for (std::size_t b=0; b<bucket_n; ++b) {
        const double u1 = u_min + u_span * (static_cast<double>(b+1)/bucket_n);
        const Sample z = (b+1==bucket_n) ? sample(u_max) : sample(u1);
        std::vector<Sample> p{a}; p.reserve(32);
        refine(a,z,0,p,stats);
        offsets[b]=s.size();
        for(std::size_t i=0;i+1<p.size();++i) s.push_back(make_segment(p[i],p[i+1]));
        max_local=std::max(max_local,p.size()-1);
        offsets[b+1]=s.size(); a=z;
    }
    return s;
}
void check(cudaError_t e,const char* what){if(e!=cudaSuccess){std::cerr<<what<<": "<<cudaGetErrorString(e)<<'\n';std::exit(1);}}
Result host(double re){auto x=friction::analytic::darcy_friction_factor(re,roughness,friction::Correlation::colebrook);return{x.value,x.d_reynolds,x.d_relative_roughness};}
Error compare(const std::vector<Result>& a,const std::vector<Result>& b){Error e{};for(std::size_t i=0;i<a.size();++i){double x[]={a[i].f,a[i].dre,a[i].drr},y[]={b[i].f,b[i].dre,b[i].drr};for(int k=0;k<3;++k){double d=std::abs(x[k]-y[k]);e.abs[k]=std::max(e.abs[k],d);e.rel[k]=std::max(e.rel[k],d/std::max(std::abs(x[k]),1e-300));if(!std::isfinite(y[k])||!within(d,k,x[k]))++e.fail[k];}}return e;}
double sum(const std::vector<Result>& x){double s=0;for(auto q:x)s+=q.f+1e-3*q.dre+1e-6*q.drr;return s;}
enum class Method{direct,uniform,global,bucket};
}

int main(){
    int dev=0;check(cudaGetDevice(&dev),"get device");cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,dev),"device props");
    std::cout<<"GPU: "<<prop.name<<" (compute "<<prop.major<<'.'<<prop.minor<<")\n";
    auto t0=std::chrono::steady_clock::now();auto uniform=build_uniform();double uniform_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    BuildStats gs{};t0=std::chrono::steady_clock::now();auto global=build_global(gs);double global_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    BuildStats bs{};std::size_t max_local=0;std::vector<std::size_t> offsets;t0=std::chrono::steady_clock::now();auto bucket=build_bucketed(offsets,bs,max_local);double bucket_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    std::cout<<"Build ms uniform/global-adaptive/bucketed="<<std::fixed<<std::setprecision(3)<<uniform_ms<<'/'<<global_ms<<'/'<<bucket_ms<<"\n"
             <<"Segments uniform/global/bucketed="<<uniform.size()<<'/'<<global.size()<<'/'<<bucket.size()<<" max local="<<max_local<<" buckets="<<bucket_n<<"\n"
             <<"Samples (midpoint probes) global/bucketed="<<gs.probes<<'/'<<bs.probes<<" unresolved="<<gs.unresolved<<'/'<<bs.unresolved<<"\n";
    std::vector<double> rr(batch_n),sorted;std::vector<Result> ref(batch_n),out(batch_n);std::mt19937_64 rng(20261020);std::uniform_real_distribution<double> un(0,1);
    for(std::size_t i=0;i<batch_n;++i){rr[i]=re_min*std::pow(re_max/re_min,un(rng));ref[i]=host(rr[i]);}sorted=rr;std::sort(sorted.begin(),sorted.end());std::vector<Result> ref_sorted(batch_n);for(std::size_t i=0;i<batch_n;++i)ref_sorted[i]=host(sorted[i]);
    double *d_r=nullptr,*d_rs=nullptr;Result*d_out=nullptr;UniformSegment*d_u=nullptr;Segment*d_g=nullptr,*d_b=nullptr;std::size_t*d_offsets=nullptr;
    check(cudaMalloc((void**)&d_r,batch_n*sizeof(double)),"alloc Re");check(cudaMalloc((void**)&d_rs,batch_n*sizeof(double)),"alloc sorted Re");check(cudaMalloc((void**)&d_out,batch_n*sizeof(Result)),"alloc output");
    check(cudaMalloc((void**)&d_u,uniform.size()*sizeof(UniformSegment)),"alloc uniform");check(cudaMalloc((void**)&d_g,global.size()*sizeof(Segment)),"alloc global");check(cudaMalloc((void**)&d_b,bucket.size()*sizeof(Segment)),"alloc bucket");check(cudaMalloc((void**)&d_offsets,offsets.size()*sizeof(std::size_t)),"alloc offsets");
    t0=std::chrono::steady_clock::now();check(cudaMemcpy(d_u,uniform.data(),uniform.size()*sizeof(UniformSegment),cudaMemcpyHostToDevice),"upload uniform");check(cudaMemcpy(d_g,global.data(),global.size()*sizeof(Segment),cudaMemcpyHostToDevice),"upload global");check(cudaMemcpy(d_b,bucket.data(),bucket.size()*sizeof(Segment),cudaMemcpyHostToDevice),"upload bucket");check(cudaMemcpy(d_offsets,offsets.data(),offsets.size()*sizeof(std::size_t),cudaMemcpyHostToDevice),"upload offsets");double upload_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    check(cudaMemcpy(d_r,rr.data(),batch_n*sizeof(double),cudaMemcpyHostToDevice),"upload random input");check(cudaMemcpy(d_rs,sorted.data(),batch_n*sizeof(double),cudaMemcpyHostToDevice),"upload sorted input");
    dim3 block(block_n),grid((batch_n+block_n-1)/block_n);
    auto launch=[&](Method m,const double* x){switch(m){case Method::direct:direct_kernel<<<grid,block>>>(x,d_out,batch_n);break;case Method::uniform:uniform_kernel<<<grid,block>>>(x,d_out,batch_n,d_u);break;case Method::global:global_kernel<<<grid,block>>>(x,d_out,batch_n,d_g,global.size());break;case Method::bucket:bucket_kernel<<<grid,block>>>(x,d_out,batch_n,d_b,d_offsets);break;}};
    std::cout<<"Coefficient upload="<<upload_ms<<" ms; resident kernel excludes transfers; E2E includes Re and output copies. Sorting excluded.\n";
    bool failed=gs.unresolved||bs.unresolved;
    auto run=[&](const char* order,const std::vector<double>& input,const std::vector<Result>& expected,double* device_re){
        std::cout<<"\nInput order: "<<order<<'\n';
        for(Method m:{Method::direct,Method::uniform,Method::global,Method::bucket}){
            cudaEvent_t a{},z{};check(cudaEventCreate(&a),"event start");check(cudaEventCreate(&z),"event stop");launch(m,device_re);check(cudaGetLastError(),"warmup launch");check(cudaDeviceSynchronize(),"warmup sync");check(cudaEventRecord(a),"record start");for(int k=0;k<reps;++k)launch(m,device_re);check(cudaGetLastError(),"timed launch");check(cudaEventRecord(z),"record stop");check(cudaEventSynchronize(z),"wait stop");float ms=0;check(cudaEventElapsedTime(&ms,a,z),"elapsed");check(cudaMemcpy(out.data(),d_out,batch_n*sizeof(Result),cudaMemcpyDeviceToHost),"download");double kernel_s=ms*1e-3;double checksum=sum(out);Error e=compare(expected,out);
            auto start=std::chrono::steady_clock::now();for(int k=0;k<reps;++k){check(cudaMemcpy(device_re,input.data(),batch_n*sizeof(double),cudaMemcpyHostToDevice),"copy input");launch(m,device_re);check(cudaGetLastError(),"E2E launch");check(cudaMemcpy(out.data(),d_out,batch_n*sizeof(Result),cudaMemcpyDeviceToHost),"copy output");}double e2e=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            const char* name=m==Method::direct?"Direct Colebrook":m==Method::uniform?"Uniform linear slice":m==Method::global?"Global adaptive": "Bucketed adaptive";
            double evals=static_cast<double>(batch_n)*reps;
            std::cout<<std::left<<std::setw(22)<<name<<std::right<<" kernel="<<std::fixed<<std::setprecision(2)<<evals/kernel_s/1e6<<" M/s E2E="<<evals/e2e/1e6<<" M/s"
                     <<" max_rel="<<std::scientific<<e.rel[0]<<'/'<<e.rel[1]<<'/'<<e.rel[2]<<" max_abs="<<e.abs[0]<<'/'<<e.abs[1]<<'/'<<e.abs[2]
                     <<" failures="<<e.fail[0]<<'/'<<e.fail[1]<<'/'<<e.fail[2]<<" checksum="<<std::fixed<<std::setprecision(10)<<checksum<<'\n';
            if(e.fail[0]||e.fail[1]||e.fail[2])failed=true;cudaEventDestroy(a);cudaEventDestroy(z);
        }
    };
    run("random log-uniform Re",rr,ref,d_r);run("sorted by Re",sorted,ref_sorted,d_rs);
    cudaFree(d_r);cudaFree(d_rs);cudaFree(d_out);cudaFree(d_u);cudaFree(d_g);cudaFree(d_b);cudaFree(d_offsets);
    return failed?EXIT_FAILURE:EXIT_SUCCESS;
}
