module Normalize

using LinearAlgebra
using Optim, NLSolversBase
using Random, Statistics, StatsBase, NMF
using SpecialFunctions: loggamma, erfinv, erf
using GSL, ProgressMeter

include("scrna.jl")
include("util.jl")
include("infer.jl")
using .scRNA: Count
using .Inference: virtualembryo

incbeta(a,b,x) = GSL.sf_beta_inc(a,b,x)
incgamma(a,x)  = GSL.sf_gamma_inc_P(a,x)

export normalize

const ∞ = Inf

"""
    logmean(x;ϵ=1)

Compute the geometric mean: ``\exp\\left(\\langle \\log\\left(x + \\epsilon \\right) \\rangle\\right) - \\epsilon``
"""
const logmean = function (x; dims=:, ϵ=1)
    μ = mean(log.(x .+ ϵ); dims=dims)
    exp.(μ) .- ϵ
end

"""
    logmean(x;ϵ=1)

Compute the geometric variance: ``exp\\left(\\langle \\log\\left(x + \\epsilon \\right)^2 \\rangle_c\\right) - \\epsilon``
"""
const logvar(x;ϵ=1)  = exp.(var(log.(x.+ϵ)))-ϵ


_modelkey(x) = x isa Symbol ? x :
               x isa AbstractString ? Symbol(x) :
               error("Model key must be Symbol or String")

struct GLMTable{T} <: AbstractArray{T,2}
    priors
    residual::Matrix{T}
    quantile::Matrix{T}
    cdf::Matrix{T}
    Θ₁::Vector{T}; Θ₂::Vector{T}; Θ₃::Vector{T}
    δΘ₁::Vector{T}; δΘ₂::Vector{T}; δΘ₃::Vector{T}
    likelihood::Vector{T}
    model::Vector{Any}
end

function GLMTable(; priors, fits, model=nothing)
    _mat_rows(vs) = Matrix(reduce(hcat, vs)')
    T = try eltype(fits[1].residual) catch; Float64 end

    residual = _mat_rows(map(f -> f.residual,   fits))::Matrix{T}
    quantile = _mat_rows(map(f -> f.quantile,   fits))::Matrix{T}
    cdf      = _mat_rows(map(f -> f.cumulative, fits))::Matrix{T}
    Θ₁  = map(f -> T(f.parameters[1]),  fits)
    Θ₂  = map(f -> T(f.parameters[2]),  fits)
    Θ₃  = map(f -> T(f.parameters[3]),  fits)
    δΘ₁ = map(f -> T(f.uncertainty[1]), fits)
    δΘ₂ = map(f -> T(f.uncertainty[2]), fits)
    δΘ₃ = map(f -> T(f.uncertainty[3]), fits)
    likelihood = map(f -> f.likelihood, fits)

    G = length(Θ₁)
    model = isnothing(model) ? fill("unspecified", G) : fill("$(model)", G)

    return GLMTable{T}(priors, residual, quantile, cdf,
                       Θ₁, Θ₂, Θ₃, δΘ₁, δΘ₂, δΘ₃, likelihood,
                       model)
end

# ---------- Array interface (extended) ----------
Base.size(R::GLMTable) = size(R.cdf)
Base.IndexStyle(::Type{GLMTable}) = IndexCartesian()
Base.getindex(R::GLMTable, I::Vararg{Int,2}) = @inbounds R.cdf[I...]

# Row-subset, keep everything aligned
Base.getindex(R::GLMTable, I::AbstractArray{<:Integer}, J) =
    GLMTable{eltype(R.cdf)}(
        R.priors,
        R.residual[I, J],
        R.quantile[I, J],
        R.cdf[I, J],
        R.Θ₁[I], R.Θ₂[I], R.Θ₃[I],
        R.δΘ₁[I], R.δΘ₂[I], R.δΘ₃[I],
        R.likelihood[I],
        R.model[I],
        R.gene[I],
    )

# Column-subset
Base.getindex(R::GLMTable, I, J::AbstractArray{<:Integer}) =
    GLMTable{eltype(R.cdf)}(
        R.priors,
        R.residual[I, J],
        R.quantile[I, J],
        R.cdf[I, J],
        R.Θ₁[I], R.Θ₂[I], R.Θ₃[I],
        R.δΘ₁[I], R.δΘ₂[I], R.δΘ₃[I],
        R.likelihood[I],
        R.model[collect(axes(R.cdf,1))],  # models are row-aligned
        R.gene[collect(axes(R.cdf,1))],
    )

# Row+col subset
Base.getindex(R::GLMTable, I::AbstractArray{<:Integer}, J::AbstractArray{<:Integer}) =
    GLMTable{eltype(R.cdf)}(
        R.priors,
        R.residual[I, J],
        R.quantile[I, J],
        R.cdf[I, J],
        R.Θ₁[I], R.Θ₂[I], R.Θ₃[I],
        R.δΘ₁[I], R.δΘ₂[I], R.δΘ₃[I],
        R.likelihood[I],
        R.model[I],
        R.gene[I],
    )

nrows(R::GLMTable) = size(R.cdf,1)
ncols(R::GLMTable) = size(R.cdf,2)

"""
    merge_models(nb::GLMTable, cmp::GLMTable, which;
                 gene=nothing, priors=nothing)

Combine Model 1 and Model 2 GLMTables into one, using `which[g]` to pick the model for gene g.
`which` accepts Bool/Int (1=Model 1,2=Model 2)

Works whether `model1` and `model2` each have all genes (full) or only their own subsets.
"""
function merge_models(model1::GLMTable, model2::GLMTable, which, priors=nothing)
    @assert ncols(model1) == ncols(model2) "Model 1/Model 2 must have same # of cells"
    G = length(which)
    C = ncols(model1)

    # Determine element type
    T = promote_type(eltype(model1.cdf), eltype(model2.cdf))

    # Preallocate
    residual  = Matrix{T}(undef, G, C)
    quantile  = Matrix{T}(undef, G, C)
    cdf       = Matrix{T}(undef, G, C)
    Θ₁  = Vector{T}(undef, G); Θ₂  = Vector{T}(undef, G); Θ₃  = Vector{T}(undef, G)
    δΘ₁ = Vector{T}(undef, G); δΘ₂ = Vector{T}(undef, G); δΘ₃ = Vector{T}(undef, G)
    likelihood = Vector{T}(undef, G)
    model = Vector{Any}(undef, G)

    full_model1  = nrows(model1)  == G
    full_model2 = nrows(model2) == G
    i_model1 = 0; i_model2 = 0

    @inbounds @views for g in 1:G
        if which[g]
            row = full_model1 ? g : (i_model1 += 1)
            residual[g,:]   .= model1.residual[row,:]
            quantile[g,:]   .= model1.quantile[row,:]
            cdf[g,:]        .= model1.cdf[row,:]
            Θ₁[g] = model1.Θ₁[row]; Θ₂[g] = model1.Θ₂[row]; Θ₃[g] = model1.Θ₃[row]
            δΘ₁[g] = model1.δΘ₁[row]; δΘ₂[g] = model1.δΘ₂[row]; δΘ₃[g] = model1.δΘ₃[row]
            likelihood[g] = model1.likelihood[row]
            model[g] = model1.model[row]
        else
            row = full_model2 ? g : (i_model2 += 1)
            residual[g,:]   .= model2.residual[row,:]
            quantile[g,:]   .= model2.quantile[row,:]
            cdf[g,:]        .= model2.cdf[row,:]
            Θ₁[g] = model2.Θ₁[row]; Θ₂[g] = model2.Θ₂[row]; Θ₃[g] = model2.Θ₃[row]
            δΘ₁[g] = model2.δΘ₁[row]; δΘ₂[g] = model2.δΘ₂[row]; δΘ₃[g] = model2.δΘ₃[row]
            likelihood[g] = model2.likelihood[row]
            model[g] = model2.model[row]
        end
    end

    merged_priors = isnothing(priors) ? (; GLM_Model1 = model1.priors, GLM_Model2 = model2.priors) : priors

    return GLMTable{T}(merged_priors, residual, quantile, cdf,
                       Θ₁, Θ₂, Θ₃, δΘ₁, δΘ₂, δΘ₃, likelihood,
                       model)
end


"""
    negativebinomial(count, depth)

Compute the log likelihood of a negative binomial generalized linear model (GLM) with log link function for `count` of a single gene.
The sequencing depth for each sequenced cell is assumed to be the only confounding variables.
A prior on Θ₂ is assumed to be Gaussian with mean `Θ̄₂` and variance `δΘ₂⁻²`.
"""
function negativebinomial(count, depth; Θ̄₂=1, δΘ₂⁻²=0, Θ₃_priors = nothing)
    if isnothing(Θ₃_priors)
        loglikelihood = function(Θ)
            Θ₁,Θ₂,Θ₃ = Θ
            # threaded reduction over data
            nt  = Base.Threads.nthreads()
            acc = fill(zero(Θ₁), nt)
            Base.Threads.@threads for i in eachindex(count)
                n = count[i]; d = depth[i]
                acc[Base.Threads.threadid()] += (
                    loggamma(n+Θ₃)
                    - loggamma(n+1)
                    - loggamma(Θ₃)
                    + n*(Θ₁+Θ₂*d)
                    + Θ₃*log(Θ₃)
                    - (n+Θ₃)*log(exp(Θ₁+Θ₂*d)+Θ₃)
                )
            end
            s = zero(Θ₁)
            @inbounds for j in 1:nt
                s += acc[j]
            end
            return -s + 0.5*δΘ₂⁻²*(Θ₂-Θ̄₂)^2
        end
    else
        loglikelihood = function(Θ)
            Θ₁,Θ₂,Θ₃ = Θ
            μ,σ,ν = Θ₃_priors
            nt  = Base.Threads.nthreads()
            acc = fill(zero(Θ₁), nt)
            Base.Threads.@threads for i in eachindex(count)
                n = count[i]; d = depth[i]
                acc[Base.Threads.threadid()] += (
                    loggamma(n+Θ₃)
                    - loggamma(n+1)
                    - loggamma(Θ₃)
                    + n*(Θ₁+Θ₂*d)
                    + Θ₃*log(Θ₃)
                    - (n+Θ₃)*log(exp(Θ₁+Θ₂*d)+Θ₃)
                )
            end
            s = zero(Θ₁)
            @inbounds for j in 1:nt
                s += acc[j]
            end
            return -s + 0.5*δΘ₂⁻²*(Θ₂-Θ̄₂)^2 + (abs(log(Θ₃)-μ)/σ)^ν
        end
    end

    μ  = logmean(count)
    Θ₀ = [log(μ), 1, logvar(count)/μ - 1]
    if Θ₀[end] < 0 || isinf(Θ₀[end]) || isnan(Θ₀[end])
        Θ₀[end] = 1
    end

    return (
        Θ₀         = Θ₀,
        likelihood = TwiceDifferentiable(loglikelihood, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-∞,-∞,0],[+∞,+∞,+∞]),
        residual   = function(Θ)
            Θ₁,Θ₂,Θ₃ = Θ
            μ = @. exp(Θ₁ + Θ₂*depth)
            σ = @. √(μ + μ^2/Θ₃)
            z = @. (count - μ) / σ
            z[ z .< -5 ] .= -5
            z[ z .> +5 ] .= +5
            return z
        end,
        quantile = function(Θ)
            Θ₁, Θ₂, Θ₃ = Θ
            μ = @. exp(Θ₁ .+ Θ₂ .* depth)
            ρ = similar(μ)
            @inbounds for i in eachindex(μ)
                y = Int(count[i])
                pi = μ[i] / (μ[i] + Θ₃)
                F_lo = (y == 0) ? 0.0 : incbeta(y, Θ₃, pi)

                log_py = loggamma(y + Θ₃) - loggamma(Θ₃) - loggamma(y+1) + y*log(μ[i]) + Θ₃*log(Θ₃) - (y + Θ₃)*log(μ[i] + Θ₃)
                p_y = exp(log_py)

                # Dunn–Smyth randomization
                Fi = F_lo + rand()*p_y
                Fi = clamp(Fi, 1e-15, 1-1e-15)
                ρ[i] = erfinv(2*Fi - 1) * √2
            end
            clamp!(ρ, -10, +10)
            return ρ
        end,
        cumulative = function(Θ)
            Θ₁,Θ₂,Θ₃ = Θ
            μ = @. exp(Θ₁ + Θ₂*depth)
            p = @. μ / (μ + Θ₃)
            return @. incbeta(count+1, Θ₃, p)
        end
    )
end


"""
    generalizedpoisson(count, depth; Θ̄₂=1, δΘ₂⁻²=0, Θ₃_priors=nothing)

Compute the negative log likelihood of a Generalized Poisson (Consul–Jain) GLM with log link
for `count` of a single gene. Sequencing depth is the only covariate.

Parameters (kept analogous to your NB):
- Θ₁: intercept
- Θ₂: depth coefficient (Gaussian prior N(Θ̄₂, δΘ₂⁻²⁻¹))
- Θ₃ ≡ θ ∈ (-1,1): dispersion (θ>0 overdispersion, θ<0 underdispersion)

With μᵢ = exp(Θ₁ + Θ₂ dᵢ):
    E[Yᵢ|dᵢ] = μᵢ/(1-θ),   Var[Yᵢ|dᵢ] = μᵢ/(1-θ)^3

If `Θ₃_priors = (μ,σ,ν)` is provided, a penalty (abs(atanh(Θ₃)−μ)/σ)^ν is added (domain-safe analogue of your NB shape prior).
"""


function generalizedpoisson(count, depth; Θ̄₂=1, δΘ₂⁻²=0, Θ₃_priors=nothing)
    # ---- GP log-pmf and stable CDF (kept local, no API change) ----
    gp_logpmf = function(y, μ, θ)
        y == 0 ? -μ : (log(μ) + (y-1)*log(μ + θ*y) - loggamma(y+1) - (μ + θ*y))
    end
    gp_cdf = function(y::Int, μ::Float64, θ::Float64)
        y < 0 && return 0.0
        # P(0)
        logPk = -μ
        S = exp(logPk)
        @inbounds for k in 0:(y-1)
            a = μ + θ*(k+1); b = (k==0 ? 1.0 : μ + θ*k)
            (a <= 0 || b <= 0) && return NaN
            logPk += k*log(a) - (k-1)*log(b) - log(k+1) - θ
            S    += exp(logPk)
            S > 1-1e-15 && return 1-1e-15
        end
        return clamp(S, 1e-15, 1-1e-15)
    end

    if isnothing(Θ₃_priors)
        loglikelihood = function(Θ)
            Θ₁, Θ₂, θ = Θ
            abs(θ) ≥ 1-1e-6 && return Inf
            s = 0.0
            @inbounds for (n,d) in zip(count, depth)
                μ = exp(Θ₁ + Θ₂*d)
                (μ <= 0 || μ + θ*n <= 0) && return Inf
                s += gp_logpmf(n, μ, θ)
            end
            return -s + 0.5*δΘ₂⁻²*(Θ₂-Θ̄₂)^2
        end
    else
        μp, σp, νp = Θ₃_priors
        loglikelihood = function(Θ)
            Θ₁, Θ₂, θ = Θ
            abs(θ) ≥ 1-1e-6 && return Inf
            s = 0.0
            @inbounds for (n,d) in zip(count, depth)
                μ = exp(Θ₁ + Θ₂*d)
                (μ <= 0 || μ + θ*n <= 0) && return Inf
                s += gp_logpmf(n, μ, θ)
            end
            return -s + 0.5*δΘ₂⁻²*(Θ₂-Θ̄₂)^2 + (abs(atanh(θ) - μp)/σp)^νp
        end
    end

    # ---- initialization (moment match, mirrors your NB style) ----
    μ̂ = mean(count); v̂ = var(count)
    θ0 = clamp(v̂ > 0 ? 1 - sqrt(clamp(μ̂/v̂, 1e-12, 1e12)) : 0.0, -0.9, 0.9)
    μ0 = max(μ̂*(1-θ0), 1e-6)
    Θ₀ = [log(μ0), 1.0, θ0]
    if any(!isfinite, Θ₀); Θ₀ .= (Θ₀ .== Θ₀) ? Θ₀ : [log(max(μ̂,1e-6)), 1.0, 0.0]; end

    return (
        Θ₀         = Θ₀,
        likelihood = TwiceDifferentiable(loglikelihood, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-∞,-∞,-1+1e-6],[+∞,+∞,+1-1e-6]),
        residual   = function(Θ)
            Θ₁, Θ₂, θ = Θ
            μ = @. exp(Θ₁ + Θ₂*depth)
            m = @. μ/(1-θ)
            σ = @. √(μ/(1-θ)^3)
            z = @. (count - m) / σ
            z[z .< -5] .= -5;  z[z .> +5] .= +5
            z
        end,
        quantile   = function(Θ)
            Θ₁, Θ₂, θ = Θ
            μ = @. exp(Θ₁ + Θ₂*depth)
            ρ = similar(μ)
            @inbounds for i in eachindex(μ)
                yi   = Int(count[i])
                # same cdf and pmf as above:
                Fi_lo = gp_cdf(yi-1, μ[i], θ)                 # F(y-1)
                pi    = exp( yi==0 ? -μ[i] :
                            log(μ[i]) + (yi-1)*log(μ[i] + θ*yi) - loggamma(yi+1) - (μ[i] + θ*yi) )
                Fi    = clamp(Fi_lo + rand()*pi, 1e-15, 1-1e-15)
                ρ[i]  = erfinv(2Fi - 1) * √2
            end
            ρ
        end,
        cumulative = function(Θ)
            Θ₁, Θ₂, θ = Θ
            μ = @. exp(Θ₁ + Θ₂*depth)
            [gp_cdf(count[i], μ[i], θ) for i in eachindex(μ)]
        end
    )
end


"""
    CMPoisson(count, depth; Θ̄₂=1, δΘ₂⁻²=0, Θ₃_priors=nothing)

Conway–Maxwell–Poisson (CMP) GLM with log link for a single gene’s `count`, using
sequencing `depth` as the sole covariate. Θ = (Θ₁, Θ₂, Θ₃) where Θ₃ = ν > 0 is the
CMP dispersion: ν = 1 reduces to Poisson; ν < 1 allows overdispersion; ν > 1 allows
underdispersion. The intensity is λᵢ = exp(Θ₁ + Θ₂ * depthᵢ).

A Gaussian prior on Θ₂ has mean Θ̄₂ and variance δΘ₂⁻². If `Θ₃_priors = (μ, σ, κ)`,
a penalty (abs(log(Θ₃) - μ)/σ)^κ is added (log–scale prior on ν).

Dunn–Smyth randomized quantile residuals are provided.
"""


function CMPoisson(count, depth; Θ̄₂=1, δΘ₂⁻²=0, Θ₃_priors=nothing)
    @assert length(count) == length(depth)
    c = collect(count)
    d = collect(depth)
    N = length(c)

    # --- Z(λ,ν) = Σ λ^k/(k!)^ν with term-based convergence (AD-friendly) ---
    @inline function _logZ(λ::T, ν::T; Kmax::Int=50_000, rtol::T = oftype(λ, eps(Float64)*100)) where {T}
        if !(λ > zero(T)) || !(ν > zero(T))
            return zero(T)
        end
        logλ  = log(λ)
        logS  = zero(T)
        k     = 0
        @inbounds while k < Kmax
            k += 1
            ak = T(k)*logλ - ν*loggamma(T(k) + one(T))
            if ak > logS
                logS = ak + log1p(exp(logS - ak))
            else
                logS = logS + log1p(exp(ak - logS))
            end
            if (ak - logS) < log(rtol)
                break
            end
        end
        return logS
    end


    # --- single pass: return raw totals S (for Z) and partial S_y up to y ---
    @inline function _sumZ_and_partial(y::Int, λ::T, ν::T; Kmax::Int=1_000, tol::T=oftype(λ, eps(Float64)*100)) where {T}
        s    = one(T)      # total
        term = one(T)
        k    = 0
        s_y  = (y >= 0) ? one(T) : zero(T)
        @inbounds while k < Kmax
            k += 1
            term *= λ * exp(-ν * log(float(k)))
            s    += term
            if k == y
                s_y = s
            end
            if abs(term) ≤ max(tol, tol*abs(s))
                break
            end
        end
        return s, s_y
    end

    # --- cache loggamma(n+1) if counts are nonnegative integers ---
    int_counts = eltype(c) <: Integer && all(x -> x ≥ 0, c)
    lg_cache = if int_counts
        maxy = maximum(c)
        v = Vector{Float64}(undef, maxy + 1) # stores loggamma(k) for k=1..maxy+1
        @inbounds @simd for k in 1:length(v)
            v[k] = loggamma(k)
        end
        v
    else
        Float64[]
    end
    @inline get_lg(n::Int) = int_counts ? @inbounds(lg_cache[n+1]) : loggamma(n+1)

    # --- log-likelihood (threaded when not differentiating) ---
    loglikelihood =
        if isnothing(Θ₃_priors)
            function (Θ::AbstractVector{T}) where {T}
                Θ₁, Θ₂, Θ₃ = Θ
                s_local = zeros(T, Threads.nthreads())
                if eltype(Θ) <: Real
                    Threads.@threads for i in 1:N
                        n   = c[i]
                        di  = d[i]
                        logλ = Θ₁ + Θ₂*di
                        λ     = exp(logλ)
                        logZ  = _logZ(λ, Θ₃)
                        s_local[Threads.threadid()] += n*logλ - Θ₃*get_lg(Int(n)) - logZ
                    end
                    s = zero(T)
                    @inbounds @simd for t in 1:length(s_local); s += s_local[t]; end
                else
                    s = zero(T)
                    @inbounds @simd for i in 1:N
                        n   = c[i]
                        di  = d[i]
                        logλ = Θ₁ + Θ₂*di
                        λ     = exp(logλ)
                        logZ  = _logZ(λ, Θ₃)
                        s    += n*logλ - Θ₃*get_lg(Int(n)) - logZ
                    end
                end
                reg = (T(0.5)*T(δΘ₂⁻²))*(Θ₂ - T(Θ̄₂))^2
                return -(s) + reg
            end
        else
            μ, σ, νp = Θ₃_priors
            function (Θ::AbstractVector{T}) where {T}
                Θ₁, Θ₂, Θ₃ = Θ
                s_local = zeros(T, Threads.nthreads())
                if eltype(Θ) <: Real
                    Threads.@threads for i in 1:N
                        n   = c[i]
                        di  = d[i]
                        logλ = Θ₁ + Θ₂*di
                        λ     = exp(logλ)
                        logZ  = _logZ(λ, Θ₃)
                        s_local[Threads.threadid()] += n*logλ - Θ₃*get_lg(Int(n)) - logZ
                    end
                    s = zero(T)
                    @inbounds @simd for t in 1:length(s_local); s += s_local[t]; end
                else
                    s = zero(T)
                    @inbounds @simd for i in 1:N
                        n   = c[i]
                        di  = d[i]
                        logλ = Θ₁ + Θ₂*di
                        λ     = exp(logλ)
                        logZ  = _logZ(λ, Θ₃)
                        s    += n*logλ - Θ₃*get_lg(Int(n)) - logZ
                    end
                end
                reg  = (T(0.5)*T(δΘ₂⁻²))*(Θ₂ - T(Θ̄₂))^2
                pen3 = (abs(log(Θ₃) - T(μ))/T(σ))^T(νp)
                return -(s) + reg + pen3
            end
        end

    # --- initializer (unchanged) ---
    μ̄ = logmean(c)
    Θ₀ = [log(μ̄), 1.0, 1.0]
    if Θ₀[end] ≤ 0 || isinf(Θ₀[end]) || isnan(Θ₀[end])
        Θ₀[end] = 1.0
    end

    # --- residuals (guarded threading; AD-safe) ---
    residual = function (Θ::AbstractVector{T}) where {T}
        Θ₁, Θ₂, Θ₃ = Θ
        ν    = Θ₃
        invν = inv(ν)
        half = T(0.5)
        out  = Vector{T}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                logλ = Θ₁ + Θ₂*d[i]
                λν   = exp(logλ * invν)
                μ    = λν + (inv(T(2)*ν) - half)
                σ2   = max(invν * λν, oftype(logλ, eps(Float64)))
                z    = (c[i] - μ) / sqrt(σ2)
                @inbounds out[i] = ifelse(z < -T(5), -T(5), ifelse(z > T(5), T(5), z))
            end
        else
            @inbounds @simd for i in 1:N
                logλ = Θ₁ + Θ₂*d[i]
                λν   = exp(logλ * invν)
                μ    = λν + (inv(T(2)*ν) - half)
                σ2   = max(invν * λν, oftype(logλ, eps(Float64)))
                z    = (c[i] - μ) / sqrt(σ2)
                out[i] = ifelse(z < -T(5), -T(5), ifelse(z > T(5), T(5), z))
            end
        end
        return out
    end

    # --- randomized normal scores (quantile) with single-pass sums (threaded) ---
    quantile = function (Θ)
        Θ₁, Θ₂, Θ₃ = Θ
        ν = Θ₃
        ρ = Vector{Float64}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                y    = Int(c[i])
                logλ = Θ₁ + Θ₂*d[i]
                λ    = exp(logλ)
                S, S_yminus = _sumZ_and_partial(max(y-1, 0), λ, ν)
                invZ = inv(S)
                py   = exp(y*logλ - ν*get_lg(y)) * invZ
                F_lo = (y == 0 ? 0.0 : S_yminus * invZ)
                Fi   = clamp(F_lo + rand()*py, 1e-15, 1-1e-15)
                @inbounds ρ[i] = sqrt(2) * erfinv(2*Fi - 1)
            end
        else
            @inbounds for i in 1:N
                y    = Int(c[i])
                logλ = Θ₁ + Θ₂*d[i]
                λ    = exp(logλ)
                S, S_yminus = _sumZ_and_partial(max(y-1, 0), λ, ν)
                invZ = inv(S)
                py   = exp(y*logλ - ν*get_lg(y)) * invZ
                F_lo = (y == 0 ? 0.0 : S_yminus * invZ)
                Fi   = clamp(F_lo + rand()*py, 1e-15, 1-1e-15)
                ρ[i] = sqrt(2) * erfinv(2*Fi - 1)
            end
        end
        clamp!(ρ, -10.0, 10.0)
        return ρ
    end

    # --- CDF at observed counts with single-pass sums (threaded) ---
    cumulative = function (Θ)
        Θ₁, Θ₂, Θ₃ = Θ
        ν = Θ₃
        cdf = Vector{Float64}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                y  = Int(c[i])
                λ  = exp(Θ₁ + Θ₂*d[i])
                S, S_y = _sumZ_and_partial(y, λ, ν)
                @inbounds cdf[i] = S_y * inv(S)
            end
        else
            @inbounds @simd for i in 1:N
                y  = Int(c[i])
                λ  = exp(Θ₁ + Θ₂*d[i])
                S, S_y = _sumZ_and_partial(y, λ, ν)
                cdf[i] = S_y * inv(S)
            end
        end
        return cdf
    end

    return (
        Θ₀         = Θ₀,
        likelihood = TwiceDifferentiable(loglikelihood, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-∞,-∞,0.0],[+∞,+∞,+∞]),
        residual   = residual,
        quantile   = quantile,
        cumulative = cumulative
    )
end

function CMPoisson(count, depth; Θ̄₂=1, δΘ₂⁻²=0, Θ₃_priors=nothing)
    @assert length(count) == length(depth)
    c = collect(count); d = collect(depth); N = length(c)

    # maps for dispersion
    @inline _ν_from_ρ(ρ::T) where {T} = exp(-ρ)   # ν = e^{-ρ} > 0
    @inline _κ_from_ρ(ρ::T) where {T} = exp(ρ)    # κ = 1/ν

    # --- log Z(λ,ν) ---
    @inline function _logZ(λ::T, ν::T; Kmax::Int=50_000, rtol::T=oftype(λ, eps(Float64)*100)) where {T}
        if !(λ > zero(T)) || !(ν > zero(T)); return zero(T) end
        logλ = log(λ); logS = zero(T); k = 0
        @inbounds while k < Kmax
            k += 1
            ak = T(k)*logλ - ν*loggamma(T(k) + one(T))
            logS = (ak > logS) ? ak + log1p(exp(logS - ak)) : logS + log1p(exp(ak - logS))
            if (ak - logS) < log(rtol); break; end
        end
        return logS
    end

    # --- single pass sums for CDF/quantile ---
    @inline function _sumZ_and_partial(y::Int, λ::T, ν::T; Kmax::Int=1_000, tol::T=oftype(λ, eps(Float64)*100)) where {T}
        s = one(T); term = one(T); k = 0; s_y = (y >= 0) ? one(T) : zero(T)
        @inbounds while k < Kmax
            k += 1
            term *= λ * exp(-ν * log(float(k)))
            s    += term
            if k == y; s_y = s; end
            if abs(term) ≤ max(tol, tol*abs(s)); break; end
        end
        return s, s_y
    end

    # --- cache loggamma(n+1) for integer counts ---
    int_counts = eltype(c) <: Integer && all(x -> x ≥ 0, c)
    lg_cache = if int_counts
        maxy = maximum(c); v = Vector{Float64}(undef, maxy + 1)
        @inbounds @simd for k in 1:length(v); v[k] = loggamma(k); end; v
    else
        Float64[]
    end
    @inline get_lg(n::Int) = int_counts ? @inbounds(lg_cache[n+1]) : loggamma(n+1)

    # --- log-likelihood ---
    loglikelihood =
        if isnothing(Θ₃_priors)
            function (Θ::AbstractVector{T}) where {T}
                Θ₁, Θ₂, ρ = Θ
                ν = _ν_from_ρ(ρ)
                s_local = zeros(T, Threads.nthreads())
                if eltype(Θ) <: Real
                    Threads.@threads for i in 1:N
                        n = c[i]; di = d[i]
                        logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                        logZ = _logZ(λ, ν)
                        s_local[Threads.threadid()] += n*logλ - ν*get_lg(Int(n)) - logZ
                    end
                    s = zero(T); @inbounds @simd for t in 1:length(s_local); s += s_local[t]; end
                else
                    s = zero(T)
                    @inbounds @simd for i in 1:N
                        n = c[i]; di = d[i]
                        logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                        logZ = _logZ(λ, ν)
                        s += n*logλ - ν*get_lg(Int(n)) - logZ
                    end
                end
                reg = (T(0.5)*T(δΘ₂⁻²))*(Θ₂ - T(Θ̄₂))^2
                return -(s) + reg
            end
        else
            μ, σ, νp = Θ₃_priors   # prior on log ν stays the same
            function (Θ::AbstractVector{T}) where {T}
                Θ₁, Θ₂, ρ = Θ
                ν = _ν_from_ρ(ρ)
                s_local = zeros(T, Threads.nthreads())
                if eltype(Θ) <: Real
                    Threads.@threads for i in 1:N
                        n = c[i]; di = d[i]
                        logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                        logZ = _logZ(λ, ν)
                        s_local[Threads.threadid()] += n*logλ - ν*get_lg(Int(n)) - logZ
                    end
                    s = zero(T); @inbounds @simd for t in 1:length(s_local); s += s_local[t]; end
                else
                    s = zero(T)
                    @inbounds @simd for i in 1:N
                        n = c[i]; di = d[i]
                        logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                        logZ = _logZ(λ, ν)
                        s += n*logλ - ν*get_lg(Int(n)) - logZ
                    end
                end
                reg  = (T(0.5)*T(δΘ₂⁻²))*(Θ₂ - T(Θ̄₂))^2
                pen3 = (abs(log(ν) - T(μ))/T(σ))^T(νp)   # = (abs(-ρ - μ)/σ)^νp
                return -(s) + reg + pen3
            end
        end

    # --- initializer (ρ₀=0 ⇒ ν₀=1) ---
    μ̄ = logmean(c)
    Θ₀ = [log(μ̄), 1.0, 0.0]

    # --- residuals (uses κ = e^{ρ}) ---
    residual = function (Θ::AbstractVector{T}) where {T}
        Θ₁, Θ₂, ρ = Θ
        ν    = _ν_from_ρ(ρ)
        invν = _κ_from_ρ(ρ)   # = exp(ρ)
        half = T(0.5)
        out  = Vector{T}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                logλ = Θ₁ + Θ₂*d[i]
                λν   = exp(logλ * invν)
                μ    = λν + (inv(T(2)*ν) - half)
                σ2   = max(invν * λν, oftype(logλ, eps(Float64)))
                z    = (c[i] - μ) / sqrt(σ2)
                @inbounds out[i] = ifelse(z < -T(5), -T(5), ifelse(z > T(5), T(5), z))
            end
        else
            @inbounds @simd for i in 1:N
                logλ = Θ₁ + Θ₂*d[i]
                λν   = exp(logλ * invν)
                μ    = λν + (inv(T(2)*ν) - half)
                σ2   = max(invν * λν, oftype(logλ, eps(Float64)))
                z    = (c[i] - μ) / sqrt(σ2)
                out[i] = ifelse(z < -T(5), -T(5), ifelse(z > T(5), T(5), z))
            end
        end
        return out
    end

    # --- randomized normal scores ---
    quantile = function (Θ)
        Θ₁, Θ₂, ρ = Θ
        ν = _ν_from_ρ(ρ)
        ρv = Vector{Float64}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                y    = Int(c[i])
                logλ = Θ₁ + Θ₂*d[i]; λ = exp(logλ)
                S, S_yminus = _sumZ_and_partial(max(y-1, 0), λ, ν)
                invZ = inv(S)
                py   = exp(y*logλ - ν*get_lg(y)) * invZ
                F_lo = (y == 0 ? 0.0 : S_yminus * invZ)
                Fi   = clamp(F_lo + rand()*py, 1e-15, 1-1e-15)
                @inbounds ρv[i] = sqrt(2) * erfinv(2*Fi - 1)
            end
        else
            @inbounds for i in 1:N
                y    = Int(c[i])
                logλ = Θ₁ + Θ₂*d[i]; λ = exp(logλ)
                S, S_yminus = _sumZ_and_partial(max(y-1, 0), λ, ν)
                invZ = inv(S)
                py   = exp(y*logλ - ν*get_lg(y)) * invZ
                F_lo = (y == 0 ? 0.0 : S_yminus * invZ)
                Fi   = clamp(F_lo + rand()*py, 1e-15, 1-1e-15)
                ρv[i] = sqrt(2) * erfinv(2*Fi - 1)
            end
        end
        clamp!(ρv, -10.0, 10.0)
        return ρv
    end

    # --- CDF at observed counts ---
    cumulative = function (Θ)
        Θ₁, Θ₂, ρ = Θ
        ν = _ν_from_ρ(ρ)
        cdf = Vector{Float64}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                y = Int(c[i]); λ = exp(Θ₁ + Θ₂*d[i])
                S, S_y = _sumZ_and_partial(y, λ, ν)
                @inbounds cdf[i] = S_y * inv(S)
            end
        else
            @inbounds @simd for i in 1:N
                y = Int(c[i]); λ = exp(Θ₁ + Θ₂*d[i])
                S, S_y = _sumZ_and_partial(y, λ, ν)
                cdf[i] = S_y * inv(S)
            end
        end
        return cdf
    end

    return (
        Θ₀         = Θ₀,
        likelihood = TwiceDifferentiable(loglikelihood, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-∞,-∞,-∞],[+∞,+∞,+∞]),  # ρ is unconstrained
        residual   = residual,
        quantile   = quantile,
        cumulative = cumulative
    )
end

function cmp_variance_exact(λ::Real, ν::Real; tol=1e-12, kmax=200_000)
    λ <= 0 && return 0.0
    λT = float(λ); νT = float(ν)
    m = floor(Int, clamp(exp(log(λT)/νT) - (νT - 1)/(2νT), 0.0, float(kmax-1)))
    Z=S1=S2=1.0; S1*=m; S2*=m*m
    p=1.0; k=m
    while k < kmax
        k+=1; p*=λT/exp(νT*log(k))
        p < tol*Z && break
        Z+=p; S1+=p*k; S2+=p*k*k
    end
    p=1.0; k=m
    while k > 0
        p*=exp(νT*log(k))/λT; k-=1
        p < tol*Z && break
        Z+=p; S1+=p*k; S2+=p*k*k
    end
    μ = S1/Z
    v = S2/Z - μ*μ
    v > 0 ? v : 0.0
end

"""
    gamma_regression(count, depth)

Compute the log likelihood of a gamma distributed generalized linear model (GLM) with log link function for the `count` of a single gene.
The sequencing depth for each sequenced cell is assumed to be the only confounding variables.
"""
function gamma_regression(count, depth)
    loglikelihood = function(Θ)
        Θ₁,Θ₂,Θ₃ = Θ
        return -sum(
            let
                α = Θ₃*exp(Θ₁+Θ₂*d)
                α*log(Θ₃) + (α-1)*log(x) - (Θ₃*x) - loggamma(α)
            end for (x,d) ∈ zip(count,depth)
        )
    end

    μ  = logmean(count; ϵ = 1e-10)
    Θ₀ = [log(μ), 1.25, μ/logvar(count; ϵ=1e-10)]
    if Θ₀[end] < 0 || isinf(Θ₀[end]) || isnan(Θ₀[end])
        Θ₀[end] = 1
    end

    return (
        Θ₀ = Θ₀,
        likelihood = TwiceDifferentiable(loglikelihood, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-∞,-∞,0],[+∞,+∞,+∞]),
        residual   = function(Θ)
            Θ₁, Θ₂, Θ₃ = Θ
            μ = @. exp(Θ₁ + Θ₂*depth)
            σ = @. sqrt(μ / Θ₃)
            z = @. (count - μ) / σ

            z[ z .< -5 ] .= -5
            z[ z .> +5 ] .= +5

            return z
        end,
        quantile = function(Θ)
            Θ₁, Θ₂, Θ₃ = Θ
            α  = @. Θ₃ * exp(Θ₁ + Θ₂*depth)
            Φ  = @. incgamma(α, count*Θ₃)
            ρ  = @. erfinv(clamp(2Φ - 1, -1, +1)) * √2
            clamp!(ρ, -10, +10)
            return ρ
        end,
        cumulative = function(Θ)
            Θ₁, Θ₂, Θ₃ = Θ
            α  = @. Θ₃ * exp(Θ₁ + Θ₂*depth)
            return @. incgamma(α, count*Θ₃)
        end
    )
end

"""
    gamma_pooled(count, overdispersion)

Compute the log likelihood of a gamma distributed generalized linear model (GLM) with log link function for the estimated mean `count` of a single gene.
The overdispersion factor `Θ₃` is assumed to be constant across all cells.
"""

function gamma_pooled(count)
    # Precompute data-only terms (fast + identical results)
    countF   = float.(count)
    N        = length(countF)
    sum_logc = sum(@. log(countF + 0.001))
    sum_c    = sum(countF)

    loglikelihood = function(Θ)
        logm, Θ₃ = Θ
        m  = exp(logm)
        α  = Θ₃ * m
        # -sum( α*log(Θ₃) + (α-1)log(count+0.001) - Θ₃*count - loggamma(α) )
        return -( N*(α*log(Θ₃) - loggamma(α)) + (α - 1)*sum_logc - Θ₃*sum_c )
    end

    logm₀  = log(mean(countF))
    θ₃₀   = var(countF) / (exp(logm₀) - 1)
    init_params = [logm₀; (isfinite(θ₃₀) && θ₃₀ > 0) ? θ₃₀ : 1.0]

    residual = function(Θ)
        logm, Θ₃ = Θ
        μ = exp(logm)
        σ = sqrt(μ / Θ₃)
        z = similar(countF)
        @inbounds Base.Threads.@threads for idx in eachindex(countF)
            t = (countF[idx] - μ) / σ
            z[idx] = t < -5 ? -5 : (t > 5 ? 5 : t)
        end
        z
    end

    quantile = function(Θ)
        logm, Θ₃ = Θ
        α = Θ₃ * exp(logm)
        ρ = similar(countF)
        @inbounds Base.Threads.@threads for idx in eachindex(countF)
            Φ = incgamma(α, countF[idx]*Θ₃)
            u = 2*Φ - 1
            u = u < -1 ? -1 : (u > 1 ? 1 : u)
            r = erfinv(u) * √2
            ρ[idx] = r < -10 ? -10 : (r > 10 ? 10 : r)
        end
        ρ
    end

    cumulative = function(Θ)
        logm, Θ₃ = Θ
        α = Θ₃ * exp(logm)
        F = similar(countF)
        @inbounds Base.Threads.@threads for idx in eachindex(countF)
            F[idx] = incgamma(α, countF[idx]*Θ₃)
        end
        F
    end

    return (
        init_params = init_params,
        likelihood  = TwiceDifferentiable(loglikelihood, init_params; autodiff=:forward),
        constraint  = TwiceDifferentiableConstraints([-Inf,0],[+Inf,+Inf]),
        residual    = residual,
        quantile    = quantile,
        cumulative  = cumulative,
    )
end

"""
    generalized_normal(params)

Compute the log likelihood of a generalized normal distribution with parameters `params`.
In practice, used to compute the empirical prior for overdispersion factor `Θ₃` in the negative binomial.
"""
function generalized_normal(params)
    objective = function(Θ)
        Θ₁, Θ₂, Θ₃ = Θ
        return sum(
            loggamma(1/Θ₃) + log(Θ₂) .+ (abs.(params .- Θ₁)./ Θ₂).^Θ₃ .- log(Θ₃)
        )
    end

    Θ₁ = mean(params)
    Θ₂ = std(params)
    Θ₀ = [abs(Θ₁),abs(Θ₂),2] # XXX Might need to do something smarter here

    return (
        Θ₀         = Θ₀,
        likelihood = TwiceDifferentiable(objective, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([0,0,0],[+∞,+∞,+∞]),
        cumulative = (Θ) -> let
            Θ₁, Θ₂, Θ₃ = Θ
            return @. .5*(1+sign(params-Θ₁)*incgamma((1/Θ₃), abs(params-Θ₁)/(Θ₂^Θ₃)))
        end
   )
end

"""
    FitType = NamedTuple{
    (
     :likelihood,
     :parameters,
     :uncertainty,
     :residual
    ),
    Tuple{Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}

Stores the result of MLE fit of _one_ gene.
"""
const FitType = NamedTuple{
    (
     :likelihood,
     :parameters,
     :uncertainty,
     :residual
    ),
    Tuple{Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}
}

"""
    fit(stochastic, count, depth)

Fit a generative model `stochastic` to gene expression `count` data, assuming confounding sequencing `depth`.
`stochastic` can be either [`negativebinomial`](@ref) or [`gamma`](@ref).
"""
function fit(stochastic, count, depth; priors = nothing)
    if isnothing(priors)
        model = stochastic(count, depth)
    else
        model = stochastic(count, depth; Θ₃_priors = priors)
    end

    param = optimize(model.likelihood, model.constraint, model.Θ₀, IPNewton())
    Θ̂ = Optim.minimizer(param)

    # println(eigvals(hessian!(model.likelihood, Optim.minimizer(param))))

    return (
        likelihood  = Optim.minimum(param),
        parameters  = Θ̂,
        uncertainty = diag(pinv(hessian!(model.likelihood, Optim.minimizer(param)))),
        cumulative  = model.cumulative(Θ̂),
        residual    = model.residual(Θ̂),
        quantile    = model.quantile(Θ̂)
    )
end

"""
    fit_generalized_normal(params)

Fit a generative model `generalized_normal` to `params`.
In practice, used to compute the empirical prior for overdispersion factor `Θ₃` in the negative binomial.
"""

function fit_generalized_normal(params)
    model = generalized_normal(params)
    param = optimize(model.likelihood, model.constraint, model.Θ₀, IPNewton())

    return (
        likelihood  = Optim.minimum(param),
        parameters  = Optim.minimizer(param),
        uncertainty = diag(inv(hessian!(model.likelihood, Optim.minimizer(param)))),
        cumulative  = model.cumulative(Optim.minimizer(param)),
    )
end


function fit_gamma_pooled(counts)
    prog_lock = ReentrantLock()
    progress = Progress(size(counts,1); desc="--> fitting:", output=stderr, color = :blue)

    fits = Vector{NamedTuple}(undef, size(counts, 1))

    for (i, gene) in enumerate(eachrow(counts))
        model = gamma_pooled(gene)
        param = optimize(model.likelihood, model.constraint, model.init_params, IPNewton())

        fits[i] = (
            likelihood  = Optim.minimum(param),
            parameters  = Optim.minimizer(param),
            uncertainty = diag(inv(hessian!(model.likelihood, Optim.minimizer(param)))),
            pearson_residual    = model.residual(Optim.minimizer(param)),
            quantile_residual    = model.quantile(Optim.minimizer(param)),
            cumulative  = model.cumulative(Optim.minimizer(param))
        )

        lock(prog_lock) do
            next!(progress)
        end
    end
        return (
                likelihood  = map((f)->f.likelihood,  fits),
                residual    = Matrix(reduce(hcat, map((f)->f.pearson_residual, fits))'),
                quantile    = Matrix(reduce(hcat, map((f)->f.quantile_residual, fits))'),
                cdf         = Matrix(reduce(hcat, map((f)->f.cumulative, fits))'),

                logm    = map((f)->f.parameters[1],  fits),
                Θ₃      = map((f)->f.parameters[2],  fits),

                δlogm   = map((f)->f.uncertainty[1], fits),
                δΘ₃   = map((f)->f.uncertainty[2], fits),
                
            )
end

"""
    bootstrap(count, depth; stochastic=negativebinomial, samples=50)

Empirically verify the MLE fit of `count`, using a GLM model generated by `stochastic` with confounding `depth` variables by bootstrap.
One third of cells are removed and the parameters are re-estimated with the remaining cells.
This process is repeated `samples` times.
The resultant distribution of estimation is returned.
"""
function bootstrap(count, depth; stochastic=negativebinomial, priors = nothing, samples=80)
    N = length(depth)

    Θ₁ = Array{Float64}(undef,samples)
    Θ₂ = Array{Float64}(undef,samples)
    Θ₃ = Array{Float64}(undef,samples)
    δΘ₁ = Array{Float64}(undef,samples)
    δΘ₂ = Array{Float64}(undef,samples)
    δΘ₃ = Array{Float64}(undef,samples)

    for n in 1:samples
        ι = randperm(N)[1:2*N÷3] # Sample without replacement
        # ι = rand(1:N, N) # Sample N with replacement
        f = fit(stochastic, count[ι], depth[ι]; priors=priors)

        Θ₁[n], Θ₂[n], Θ₃[n] = f.parameters
        δΘ₁[n], δΘ₂[n], δΘ₃[n] = f.uncertainty
    end

    return Θ₁, Θ₂, Θ₃, δΘ₁, δΘ₂, δΘ₃
end

"""
    glm(data; stochastic=negativebinomial, ϵ=1)

Fit a generalized linear model (GLM) to the matrix `data`.
Genes are assumed to be on rows, cells over columns.
The underlying generative model is passed by `stochastic`.
"""
function glm(data; stochastic=negativebinomial, priors = nothing, ϵ=1, run=(x)->true, barcolor = :red)
    # compute depth of each cell
    average(x) = logmean(x; ϵ=ϵ)
    depth = map(eachcol(data)) do col
        col |> vec |> average
    end

    selected = [i for (i, row) in enumerate(eachrow(data)) if run(row)]
    ι = zeros(Int, size(data,1))
    ι[selected] = 1:length(selected)

    progress = Progress(sum(ι .!= 0); desc="--> fitting:", output=stderr, color = barcolor)

    fits = Array{NamedTuple}(undef, length(selected))
    for (i,gene) in collect(enumerate(eachrow(data)))
        ι[i] == 0 && continue
        fits[ι[i]] = fit(stochastic,collect(vec(gene)),depth; priors=priors)
        if i % 100 == 0
            next!(progress; step = 100)
        end
    end

    return GLMTable(; priors, fits, model = stochastic)

end

using Statistics

"""
    conditional_variance_mixture(data, model; tol=1e-12, kmax=20_000)

Compute Σ[g,c] = Var(Y_{gc} | model_g, μ_{gc}) for a mixed GLM:
- NB:            Var = μ + μ^2/θ_g
- CMP (ρ):       Var via cmp_variance_exact(μ, ν_g) with ν_g = exp(-ρ_g)
- Gen. Poisson:  Var = μ / (1-α_g)^2  (when 1-α_g > 0, else Inf)

Inputs
- `data`  :: AbstractMatrix (G×C) of μ/λ per gene×cell
- `model` must expose:
    * Θ₃ :: Vector or scalar (θ for NB; ρ for CMP; α for GenPois)
    * model.model :: Vector{<:AbstractString} (e.g. "negativebinomial", "CMPoisson_log", "generalizedpoisson")

Options
- `tol`, `kmax` passed to `cmp_variance_exact`.

Returns
- Σ :: Matrix{Float64} (G×C)
"""
function conditional_variance_mixture(data, model; tol=1e-12, kmax=20_000)
    G, C = size(data)
    Σ = similar(data, Float64)

    labels = lowercase.(String.(model.model))
    Θ3 = model.Θ₃

    # helpers to fetch per-gene scalar from Θ₃ whether it's a vector or scalar
    @inline getθ(g) = Float64(Θ3 isa AbstractVector ? Θ3[g] : Θ3)   # NB θ
    @inline getρ(g) = Float64(Θ3 isa AbstractVector ? Θ3[g] : Θ3)   # CMP ρ
    @inline getα(g) = Float64(Θ3 isa AbstractVector ? Θ3[g] : Θ3)   # GenPois α

    idx_nb  = findall(l -> occursin("negativebinomial", l), labels)
    idx_cmp = findall(l -> occursin("cmp", l), labels)              # matches "cmpoisson_log", etc.
    idx_gp  = findall(l -> occursin("generalizedpoisson", l), labels)

    # --- NB genes --------------------------------------------------------------
    @inbounds for g in idx_nb
        θg = getθ(g)
        @views μg = data[g, :]
        @views Σg = Σ[g, :]
        @inbounds for c in 1:C
            μ = Float64(μg[c])
            Σg[c] = μ + (μ*μ)/θg
        end
    end

    # --- CMP genes (ρ = log κ = -log ν) ---------------------------------------
    @inbounds for g in idx_cmp
        ρg = getρ(g)
        νg = exp(-ρg)
        @views μg = data[g, :]
        @views Σg = Σ[g, :]
        @inbounds for c in 1:C
            λ = Float64(μg[c])  # here μ_mat ≡ λ for CMP parameterization in your code
            Σg[c] = cmp_variance_exact(λ, νg; tol=tol, kmax=kmax)
        end
    end

    # --- Generalized Poisson genes --------------------------------------------
    @inbounds for g in idx_gp
        αg = getα(g)
        d  = 1.0 - αg
        @views μg = data[g, :]
        @views Σg = Σ[g, :]
        if d > 0 && isfinite(d)
            invd2 = 1.0 / (d*d)
            @inbounds for c in 1:C
                μ = Float64(μg[c])
                Σg[c] = μ * invd2
            end
        else
            @inbounds for c in 1:C
                Σg[c] = Inf
            end
        end
    end

    if !all(1:G .∈ sort(idx_cmp ∪ idx_nb ∪ idx_gp))
        @warn "Some genes were not assigned a model in `model.model`"
    end

    return Σ
end


"""
    normalize(data; δ=5)

WIP
"""

function normalize(data, model; δ = 2)

    Σ = conditional_variance_mixture(data, model)

    X̃₁, Σ̃, u², v² = let
        u², v², _ = Utility.sinkhorn(Σ)

        (
        Diagonal(.√u²) * data * Diagonal(.√v²),
        Diagonal(u²) * Σ * Diagonal(v²),
        u², v²
        )
    end

    F = svd(X̃₁)
    σ = F.S
    R = count(σ .> (√size(X̃₁,1) + √size(X̃₁,2)) + δ)

    X̃, ρ = let
        nmf = nnmf(X̃₁, R; alg=:cd, init = :nndsvdar)
        M̂ = nmf.W * nmf.H
        (M̂, cor(M̂[:], X̃₁[:]))
    end

    gamma_fits = Normalize.fit_gamma_pooled(X̃)
    X̂ = gamma_fits.residual

    return (
        testing_data    = (u², v², X̃₁, Σ̃),
        counts          = scRNA.Count(X̂, data.gene, data.cell),
        rank            = R,
        corr            = ρ,
        gamma_models    = gamma_fits,
        GLM_models      = model,
    )
end

end
