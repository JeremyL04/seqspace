module Normalize

using LinearAlgebra
using Optim, NLSolversBase
using Random, Statistics, StatsBase, NMF
using SpecialFunctions: loggamma, erfinv, erf, besselk
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

############# Flexible GLMTable with δ-uncertainties ###########################

struct GLMTable{T} <: AbstractArray{T,2}
    priors
    residual::Matrix{T}
    quantile::Matrix{T}
    cdf::Matrix{T}
    params::Dict{Symbol,Vector{T}}        # :Θ₁,:Θ₂,:ρ,:p, ...
    uncertainties::Dict{Symbol,Vector{T}} # :δΘ₁,:δΘ₂,:δρ,:δp, ...
    likelihood::Vector{T}
    model::Vector{Any}
end

# basics
Base.size(R::GLMTable) = size(R.cdf)
Base.IndexStyle(::Type{GLMTable}) = IndexCartesian()
Base.getindex(R::GLMTable, I::Vararg{Int,2}) = @inbounds R.cdf[I...]
nrows(R::GLMTable) = size(R.cdf,1)
ncols(R::GLMTable) = size(R.cdf,2)

# convenience access: R.Θ₁, R.ρ, R.p, R.δΘ₁, R.δρ, ...
const _builtin = Set([:priors,:residual,:quantile,:cdf,:params,:uncertainties,:likelihood,:model])
function Base.getproperty(R::GLMTable, s::Symbol)
    if s in _builtin; return getfield(R,s) end
    haskey(R.params, s)        && return R.params[s]
    haskey(R.uncertainties, s) && return R.uncertainties[s]
    getfield(R, s)  # fall through (throws if missing)
end

# subsetting (rows carry row-wise meta/params; cols are data only)
function Base.getindex(R::GLMTable, I::AbstractVector{<:Integer}, J)
    ps  = Dict(k => @view(R.params[k][I])[:]        for k in keys(R.params))
    ups = Dict(k => @view(R.uncertainties[k][I])[:] for k in keys(R.uncertainties))
    GLMTable{eltype(R.cdf)}(R.priors, R.residual[I,J], R.quantile[I,J], R.cdf[I,J],
                            ps, ups, R.likelihood[I], R.model[I])
end
function Base.getindex(R::GLMTable, I, J::AbstractVector{<:Integer})
    GLMTable{eltype(R.cdf)}(R.priors, R.residual[I,J], R.quantile[I,J], R.cdf[I,J],
                            R.params, R.uncertainties, R.likelihood, R.model)
end
function Base.getindex(R::GLMTable, I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer})
    ps  = Dict(k => @view(R.params[k][I])[:]        for k in keys(R.params))
    ups = Dict(k => @view(R.uncertainties[k][I])[:] for k in keys(R.uncertainties))
    GLMTable{eltype(R.cdf)}(R.priors, R.residual[I,J], R.quantile[I,J], R.cdf[I,J],
                            ps, ups, R.likelihood[I], R.model[I])
end

# name helpers (Θ₁…Θ₉, then Θ_10, Θ_11, …)
const _sub = Dict(1=>'₁',2=>'₂',3=>'₃',4=>'₄',5=>'₅',6=>'₆',7=>'₇',8=>'₈',9=>'₉')
_theta_name(i) = i ≤ 9 ? Symbol("Θ", string(_sub[i])) : Symbol("Θ_", string(i))
_theta_names(k) = [ _theta_name(i) for i in 1:k ]
_delta_name(sym::Symbol) = Symbol("δ", String(sym))   # e.g. :Θ₁ -> :δΘ₁, :ρ->:δρ, :p->:δp

"""
    GLMTable(; priors, fits, model=nothing, paramnames=nothing)

- `fits[i]` expose: `parameters`, `uncertainty`, `residual`, `quantile`, `cumulative`, `likelihood`
- `paramnames`:
    * `Vector{Symbol}`: same names for all fits
    * `Vector{Vector{Symbol}}`: names per-fit
    * omitted: defaults to `[:Θ₁,:Θ₂,...]` per-fit length
"""
function GLMTable(; priors, fits, model=nothing, paramnames=nothing)
    _mat_rows(vs) = Matrix(reduce(hcat, vs)')
    T = try eltype(fits[1].residual) catch; Float64 end

    residual   = _mat_rows(map(f -> f.residual,   fits))::Matrix{T}
    quantile   = _mat_rows(map(f -> f.quantile,   fits))::Matrix{T}
    cdf        = _mat_rows(map(f -> f.cumulative, fits))::Matrix{T}
    likelihood = map(f -> f.likelihood, fits)

    G = length(fits)
    labels = model isa AbstractVector ? model :
             isnothing(model) ? fill("unspecified", G) : fill(model, G)

    # param names per-fit
    names_per_fit = Vector{Vector{Symbol}}(undef, G)
    for g in 1:G
        if paramnames isa Vector{Symbol}
            names_per_fit[g] = paramnames
        elseif paramnames isa Vector{<:Vector{Symbol}}
            names_per_fit[g] = paramnames[g]
        elseif hasproperty(fits[g], :paramnames)
            names_per_fit[g] = Symbol.(getproperty(fits[g], :paramnames))
        else
            k = length(getproperty(fits[g], :parameters))
            names_per_fit[g] = _theta_names(k)
        end
    end

    # union keys
    keyset = Set{Symbol}()
    for v in names_per_fit, s in v; push!(keyset, s); end
    ukeyset = Set{Symbol}()
    for v in names_per_fit, s in v; push!(ukeyset, _delta_name(s)); end
    pkeys, ukeys = collect(keyset), collect(ukeyset)

    nanT = T <: AbstractFloat ? T(NaN) : zero(T)
    params        = Dict(k => fill(nanT, G) for k in pkeys)
    uncertainties = Dict(k => fill(nanT, G) for k in ukeys)

    # populate
    for g in 1:G
        pars = getproperty(fits[g], :parameters)
        uncs = getproperty(fits[g], :uncertainty)
        names = names_per_fit[g]
        @inbounds for j in eachindex(pars, names)
            k  = names[j]
            dk = _delta_name(k)
            params[k][g]        = T(pars[j])
            uncertainties[dk][g]= T(uncs[j])
        end
    end

    GLMTable{T}(priors, residual, quantile, cdf, params, uncertainties, likelihood, labels)
end

# merge two GLMTables with possibly different param sets
function merge_models(A::GLMTable, B::GLMTable, which; priors=nothing)
    @assert ncols(A) == ncols(B) "A and B must have the same number of columns (cells)"
    C = ncols(A)

    # detect mode
    is_bool_sel = eltype(which) <: Bool

    # --- dimensions & checks ---
    G = is_bool_sel ? length(which) : nrows(A)
    if is_bool_sel
        @assert (nrows(A) == G) || (nrows(B) == G) ||
                (nrows(A) + nrows(B) == G) "In bool mode, rows of A/B must align or concatenate to G"
    else
        @assert nrows(A) == G "In index mode, A must provide all G rows to start from"
        idxs = collect(which)::Vector{Int}
        @assert all(1 .<= idxs .<= G) "Replacement indices must lie within 1:G"
        @assert nrows(B) >= length(idxs) "B must have at least as many rows as there are replacements"
    end

    # element type
    T = promote_type(eltype(A.cdf), eltype(B.cdf))

    # allocate outputs
    residual   = Matrix{T}(undef, G, C)
    quantile   = Matrix{T}(undef, G, C)
    cdf        = Matrix{T}(undef, G, C)
    likelihood = Vector{T}(undef, G)
    model      = Vector{Any}(undef, G)

    # params & uncertainties unions
    allP = collect(union(keys(A.params),        keys(B.params)))
    allU = collect(union(keys(A.uncertainties), keys(B.uncertainties)))
    nanT = T <: AbstractFloat ? T(NaN) : zero(T)
    params        = Dict(k => fill(nanT, G) for k in allP)
    uncertainties = Dict(k => fill(nanT, G) for k in allU)

    # small helpers to copy one row from a src table into dest row g
    @inline function _copy_row!(dest_row::Int, src_tbl::GLMTable, src_row::Int)
        @inbounds begin
            residual[dest_row, :] .= src_tbl.residual[src_row, :]
            quantile[dest_row, :] .= src_tbl.quantile[src_row, :]
            cdf[dest_row,      :] .= src_tbl.cdf[src_row,      :]
            likelihood[dest_row]   = src_tbl.likelihood[src_row]
            model[dest_row]        = src_tbl.model[src_row]
            for k in keys(src_tbl.params)
                params[k][dest_row] = src_tbl.params[k][src_row]
            end
            for k in keys(src_tbl.uncertainties)
                uncertainties[k][dest_row] = src_tbl.uncertainties[k][src_row]
            end
        end
    end

    if is_bool_sel
        # --- BOOL MODE: pick from A or B row-by-row (supports subset tables) ---
        fullA = nrows(A) == G
        fullB = nrows(B) == G
        ia = 0; ib = 0
        @inbounds for g in 1:G
            if which[g]
                row = fullA ? g : (ia += 1)
                _copy_row!(g, A, row)
            else
                row = fullB ? g : (ib += 1)
                _copy_row!(g, B, row)
            end
        end
    else
        # --- INDEX MODE: start from A, then replace specified rows with next rows of B ---
        # 1) copy all rows from A as baseline
        @inbounds for g in 1:G
            _copy_row!(g, A, g)
        end
        # 2) sequentially overwrite the requested rows with consecutive rows from B
        @inbounds for (t, g) in enumerate(idxs)
            _copy_row!(g, B, t)
        end
    end

    mpriors = isnothing(priors) ? (; A=A.priors, B=B.priors) : priors
    return GLMTable{T}(mpriors, residual, quantile, cdf, params, uncertainties, likelihood, model)
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


function poisson_inverse_gaussian(count, depth; Θ̄₂=1, δΘ₂⁻²=0, Θ₃_priors=nothing,
    debug::Bool=false, warn_limit::Int=6, ν_fast::Float64=50.0)

    # bounded warnings
    warnref = Ref(0)
    _maybe_warn(msg; kwargs...) = begin
        if debug && warnref[] < warn_limit
            @warn msg kwargs...
            warnref[] += 1
        end
    end

    # --- numerically safe log K_ν(x) (unchanged except tiny guard) ---
    @inline function _logK(ν::Float64, x::Float64)
        if !(x > 0) || !isfinite(x)
            _maybe_warn("Bessel arg x not positive/finite", x=x, ν=ν)
            return -Inf
        end
        if x > 50.0
            return 0.5*log(pi/(2x)) - x
        else
            try
                return log(besselk(abs(ν), x)) # K_ν = K_|ν|
            catch err
                _maybe_warn("besselk failed; using asymptotic", ν=ν, x=x, err=err)
                return 0.5*log(pi/(2x)) - x
            end
        end
    end

    # --- tiny helper: NB log pmf with mean μ, size r (>0) ---
    @inline function _logpmf_nb(n::Int, μ::Float64, r::Float64)
        (μ > 0 && r > 0) || return -Inf
        return loggamma(n + r) - loggamma(r) - loggamma(n + 1) +
               r*log(r) + n*log(μ) - (n + r)*log(μ + r)
    end

    # --- PIG log pmf (φ = exp(Θ₃)); FAST-PATH large-ν to NB surrogate ---
    @inline function _logpmf_pig(y::Int, μ::Float64, Θ₃::Float64)
        μ > 0 || ( _maybe_warn("μ <= 0 in pmf", μ=μ); return -Inf )
        Θ₃ > -10 || ( _maybe_warn("Θ₃ too small in pmf", Θ₃=Θ₃); return -Inf ) # φ floor for stability

        # QUICK EXIT for large ν := |y-1/2|
        νa = abs(float(y) - 0.5)
        if νa > ν_fast
            φ      = exp(Θ₃)
            r_eff  = max(1e-12, 1.0 / (φ * μ))  # match Var tail: μ + φ μ^3 ≈ μ + μ^2/r_eff
            _maybe_warn("using NB surrogate for large ν", y=y, ν=νa, r_eff=r_eff)
            return _logpmf_nb(y, μ, r_eff)
        end

        # Otherwise do the full PIG with Bessel
        φ    = exp(Θ₃)                  # φ > 0 by construction
        invφ = exp(-Θ₃)
        A = 1.0 + 0.5*invφ/(μ*μ)        # > 1
        q = (2.0*invφ) * A
        if !(q > 0) || !isfinite(q)
            _maybe_warn("PIG radicand q invalid (complex sqrt risk)", μ=μ, φ=φ, Θ₃=Θ₃, q=q, A=A)
            return -Inf
        end
        x = sqrt(q)
        if !isfinite(x)
            _maybe_warn("sqrt(q) not finite", q=q)
            return -Inf
        end

        # log p(y) = 0.5*log(2/(πφ)) + 1/(φμ) - log(y!) - 0.5*(y-0.5)log(2φA) + logK_{y-1/2}(x)
        t1 = 0.5*log(2/(π*φ))
        t2 = invφ/μ
        t3 = -loggamma(y + 1)
        t4 = -0.5*(y - 0.5)*log(2*φ*A)
        t5 = _logK(y - 0.5, x)
        return t1 + t2 + t3 + t4 + t5
    end

    # --- CDF via partial sums (PIT-friendly) ---
    @inline function _cdf_pig(y::Int, μ::Float64, Θ₃::Float64)
        y < 0 && return 0.0
        s = 0.0
        @inbounds for k in 0:y
            lk = _logpmf_pig(k, μ, Θ₃)
            s += (lk == -Inf) ? 0.0 : exp(lk)
        end
        return clamp(s, 0.0, 1.0)
    end

    # --- negative log-likelihood (threaded) ---
    loglikelihood = let Θ₃_priors=Θ₃_priors, δΘ₂⁻²=δΘ₂⁻², Θ̄₂=Θ̄₂
        function (Θ)
            Θ₁, Θ₂, Θ₃ = Θ
            nt  = Base.Threads.nthreads()
            acc = fill(zero(Θ₁), nt)
            Base.Threads.@threads for i in eachindex(count)
                n  = Int(count[i])
                μi = exp(Θ₁ + Θ₂*depth[i])
                acc[Base.Threads.threadid()] += _logpmf_pig(n, μi, Θ₃)
            end
            s = zero(Θ₁); @inbounds for j in 1:nt; s += acc[j]; end
            pen = 0.5*δΘ₂⁻²*(Θ₂-Θ̄₂)^2
            if isnothing(Θ₃_priors)
                return -s + pen
            else
                μp, σp, νp = Θ₃_priors
                return -s + pen + (abs(Θ₃ - μp)/σp)^νp   # prior on log φ
            end
        end
    end

    # --- init (Θ₃ is log φ) ---
    μ̂ = logmean(count)
    m, v = mean(count), var(count)
    φ0 = (v > m && m > 0) ? max((v - m)/max(m^3, eps()), 1e-6) : 1.0
    Θ₀ = [log(μ̂), 1.0, log(φ0)]

    return (
        Θ₀         = Θ₀,
        likelihood = TwiceDifferentiable(loglikelihood, Θ₀; autodiff=:finite),
        constraint = TwiceDifferentiableConstraints([-∞,-∞,-∞],[+∞,+∞,+∞]),
        residual   = function(Θ)
            Θ₁, Θ₂, Θ₃ = Θ
            μ = @. exp(Θ₁ + Θ₂*depth)
            φ = exp(Θ₃)
            σ = @. sqrt(μ + φ*μ^3)
            z = @. (count - μ) / σ
            clamp!(z, -5, 5)
            return z
        end,
        quantile = function(Θ)
            Θ₁, Θ₂, Θ₃ = Θ
            μ = @. exp(Θ₁ + Θ₂*depth)
            ρ = similar(μ)
            @inbounds for i in eachindex(μ)
                y   = Int(count[i])
                Flo = (y == 0) ? 0.0 : _cdf_pig(y-1, μ[i], Θ₃)
                py  = exp(_logpmf_pig(y, μ[i], Θ₃))
                Fi  = clamp(Flo + rand()*py, 1e-15, 1-1e-15)
                ρ[i] = √2 * erfinv(2*Fi - 1)
            end
            clamp!(ρ, -10, 10)
            return ρ
        end,
        cumulative = function(Θ)
            Θ₁, Θ₂, Θ₃ = Θ
            μ = @. exp(Θ₁ + Θ₂*depth)
            F = similar(μ)
            @inbounds for i in eachindex(μ)
                F[i] = _cdf_pig(Int(count[i]), μ[i], Θ₃)
            end
            return F
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

function Hurdle_CMPoisson(count, depth; Θ̄₂=1, δΘ₂⁻²=0, Θ₃_priors=nothing, Θ̄₄=:auto, δΘ₄⁻²=1e-3)  # gentle prior on Θ₄ (ψ)

    @assert length(count) == length(depth)
    c = collect(count); d = collect(depth); N = length(c)

    # maps
    @inline _ν_from_ρ(ρ::T) where {T} = exp(-ρ)           # ν = e^{-ρ} > 0
    @inline _κ_from_ρ(ρ::T) where {T} = exp(ρ)            # κ = 1/ν
    @inline _σ(x::T) where {T} = inv(one(T) + exp(-x))    # logistic
    @inline _lse(a::T,b::T) where {T} = (m=max(a,b); m + log1p(exp(min(a,b)-m)))

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

    # --- single pass sums for CDF/partial (CMP part) ---
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

    # loggamma cache
    int_counts = eltype(c) <: Integer && all(x -> x ≥ 0, c)
    lg_cache = if int_counts
        maxy = maximum(c); v = Vector{Float64}(undef, maxy + 1)
        @inbounds @simd for k in 1:length(v); v[k] = loggamma(k); end; v
    else
        Float64[]
    end
    @inline get_lg(n::Int) = int_counts ? @inbounds(lg_cache[n+1]) : loggamma(n+1)

    # --- weak prior center for ψ (Θ₄) -----------------------------------------
    zfrac_emp = mean(c .== 0)
    p0_emp    = clamp(zfrac_emp, 1e-6, 1 - 1e-6)
    ψ0_emp    = log(p0_emp/(1 - p0_emp))
    ψ̄_prior  = (Θ̄₄ === :auto) ? ψ0_emp : Float64(Θ̄₄)

    # --- log-likelihood (Hurdle–CMP) ------------------------------------------
    loglikelihood =
        if isnothing(Θ₃_priors)
            function (Θ::AbstractVector{T}) where {T}
                Θ₁, Θ₂, ρ, ψ = Θ
                ν = _ν_from_ρ(ρ); p = _σ(ψ)
                s_local = zeros(T, Threads.nthreads())
                if eltype(Θ) <: Real
                    Threads.@threads for i in 1:N
                        n = c[i]; di = d[i]
                        logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                        logZ = _logZ(λ, ν)
                        log1m_p0 = log1p(-exp(-logZ))      # log(1 - 1/Z)
                        if n == 0
                            s_local[Threads.threadid()] += log(p)
                        else
                            s_local[Threads.threadid()] += log1p(-p) + n*logλ - ν*get_lg(Int(n)) - logZ - log1m_p0
                        end
                    end
                    s = zero(T); @inbounds @simd for t in 1:length(s_local); s += s_local[t]; end
                else
                    s = zero(T)
                    @inbounds @simd for i in 1:N
                        n = c[i]; di = d[i]
                        logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                        logZ = _logZ(λ, ν)
                        log1m_p0 = log1p(-exp(-logZ))
                        if n == 0
                            s += log(p)
                        else
                            s += log1p(-p) + n*logλ - ν*get_lg(Int(n)) - logZ - log1m_p0
                        end
                    end
                end
                reg2 = (T(0.5)*T(δΘ₂⁻²))*(Θ₂ - T(Θ̄₂))^2
                reg4 = (T(0.5)*T(δΘ₄⁻²))*(ψ  - T(ψ̄_prior))^2   # gentle prior on Θ₄
                return -(s) + reg2 + reg4
            end
        else
            μ, σ, νp = Θ₃_priors   # prior on log ν (= -ρ) as before
            function (Θ::AbstractVector{T}) where {T}
                Θ₁, Θ₂, ρ, ψ = Θ
                ν = _ν_from_ρ(ρ); p = _σ(ψ)
                s_local = zeros(T, Threads.nthreads())
                if eltype(Θ) <: Real
                    Threads.@threads for i in 1:N
                        n = c[i]; di = d[i]
                        logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                        logZ = _logZ(λ, ν)
                        log1m_p0 = log1p(-exp(-logZ))
                        if n == 0
                            s_local[Threads.threadid()] += log(p)
                        else
                            s_local[Threads.threadid()] += log1p(-p) + n*logλ - ν*get_lg(Int(n)) - logZ - log1m_p0
                        end
                    end
                    s = zero(T); @inbounds @simd for t in 1:length(s_local); s += s_local[t]; end
                else
                    s = zero(T)
                    @inbounds @simd for i in 1:N
                        n = c[i]; di = d[i]
                        logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                        logZ = _logZ(λ, ν)
                        log1m_p0 = log1p(-exp(-logZ))
                        if n == 0
                            s += log(p)
                        else
                            s += log1p(-p) + n*logλ - ν*get_lg(Int(n)) - logZ - log1m_p0
                        end
                    end
                end
                reg2 = (T(0.5)*T(δΘ₂⁻²))*(Θ₂ - T(Θ̄₂))^2
                pen3 = (abs(log(ν) - T(μ))/T(σ))^T(νp)          # your ν prior
                reg4 = (T(0.5)*T(δΘ₄⁻²))*(ψ  - T(ψ̄_prior))^2   # gentle prior on Θ₄
                return -(s) + reg2 + pen3 + reg4
            end
        end

    # --- initializer (ρ₀=0 ⇒ ν₀=1; ψ from empirical zeros) --------------------
    μ̄ = logmean(c)
    ψ0 = ψ0_emp
    Θ₀ = [log(μ̄), 1.0, 0.0, ψ0]

    # --- residuals (CMP asymptotics for μ,σ²; exact p0 via logZ) --------------
    residual = function (Θ::AbstractVector{T}) where {T}
        Θ₁, Θ₂, ρ, ψ = Θ
        ν    = _ν_from_ρ(ρ)
        invν = _κ_from_ρ(ρ)
        p    = _σ(ψ)
        half = T(0.5)
        out  = Vector{T}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                logλ = Θ₁ + Θ₂*d[i]; λ = exp(logλ)
                λν   = exp(logλ * invν)                 # CMP mean approx
                μcmp = λν + (inv(T(2)*ν) - half)
                σ2cmp= max(invν * λν, oftype(logλ, eps(Float64)))
                p0cmp= exp(-_logZ(λ, ν))                # exact p0 = 1/Z
                one_m_p0 = max(1 - p0cmp, oftype(logλ, 1e-12))
                μplus = μcmp / one_m_p0
                σ2plus = ((one_m_p0)*σ2cmp - p0cmp*μcmp*μcmp) / (one_m_p0*one_m_p0)
                μh   = (one(T) - p)*μplus
                σ2h  = (one(T) - p)*(σ2plus + p*μplus*μplus)
                z    = (c[i] - μh) / sqrt(max(σ2h, oftype(logλ, eps(Float64))))
                @inbounds out[i] = clamp(z, -T(5), T(5))
            end
        else
            @inbounds @simd for i in 1:N
                logλ = Θ₁ + Θ₂*d[i]; λ = exp(logλ)
                λν   = exp(logλ * invν)
                μcmp = λν + (inv(T(2)*ν) - half)
                σ2cmp= max(invν * λν, oftype(logλ, eps(Float64)))
                p0cmp= exp(-_logZ(λ, ν))
                one_m_p0 = max(1 - p0cmp, oftype(logλ, 1e-12))
                μplus = μcmp / one_m_p0
                σ2plus = ((one_m_p0)*σ2cmp - p0cmp*μcmp*μcmp) / (one_m_p0*one_m_p0)
                μh   = (one(T) - p)*μplus
                σ2h  = (one(T) - p)*(σ2plus + p*μplus*μplus)
                z    = (c[i] - μh) / sqrt(max(σ2h, oftype(logλ, eps(Float64))))
                out[i] = clamp(z, -T(5), T(5))
            end
        end
        return out
    end

    # --- randomized normal scores (Hurdle–CMP) --------------------------------
    quantile = function (Θ)
        Θ₁, Θ₂, ρ, ψ = Θ
        ν = _ν_from_ρ(ρ); p = _σ(ψ)
        qv = Vector{Float64}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                y    = Int(c[i]); logλ = Θ₁ + Θ₂*d[i]; λ = exp(logλ)
                S, S_yminus = _sumZ_and_partial(max(y-1, 0), λ, ν)
                invZ = inv(S); p0cmp = invZ
                if y == 0
                    Fi = clamp(rand()*p, 1e-15, 1-1e-15)
                else
                    pycmp = exp(y*logλ - ν*get_lg(y)) * invZ
                    Flo   = p + (1 - p) * ((S_yminus*invZ - p0cmp) / max(1 - p0cmp, 1e-12))
                    mass  = (1 - p) * (pycmp / max(1 - p0cmp, 1e-12))
                    Fi    = clamp(Flo + rand()*mass, 1e-15, 1-1e-15)
                end
                @inbounds qv[i] = sqrt(2) * erfinv(2*Fi - 1)
            end
        else
            @inbounds for i in 1:N
                y    = Int(c[i]); logλ = Θ₁ + Θ₂*d[i]; λ = exp(logλ)
                S, S_yminus = _sumZ_and_partial(max(y-1, 0), λ, ν)
                invZ = inv(S); p0cmp = invZ
                if y == 0
                    Fi = clamp(rand()*p, 1e-15, 1-1e-15)
                else
                    pycmp = exp(y*logλ - ν*get_lg(y)) * invZ
                    Flo   = p + (1 - p) * ((S_yminus*invZ - p0cmp) / max(1 - p0cmp, 1e-12))
                    mass  = (1 - p) * (pycmp / max(1 - p0cmp, 1e-12))
                    Fi    = clamp(Flo + rand()*mass, 1e-15, 1-1e-15)
                end
                qv[i] = sqrt(2) * erfinv(2*Fi - 1)
            end
        end
        clamp!(qv, -10.0, 10.0)
        return qv
    end

    # --- CDF at observed counts (Hurdle–CMP) ----------------------------------
    cumulative = function (Θ)
        Θ₁, Θ₂, ρ, ψ = Θ
        ν = _ν_from_ρ(ρ); p = _σ(ψ)
        cdf = Vector{Float64}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                y = Int(c[i]); λ = exp(Θ₁ + Θ₂*d[i])
                S, S_y = _sumZ_and_partial(y, λ, ν)
                invZ = inv(S); p0cmp = invZ
                if y == 0
                    @inbounds cdf[i] = p
                else
                    @inbounds cdf[i] = p + (1 - p) * ((S_y*invZ - p0cmp) / max(1 - p0cmp, 1e-12))
                end
            end
        else
            @inbounds @simd for i in 1:N
                y = Int(c[i]); λ = exp(Θ₁ + Θ₂*d[i])
                S, S_y = _sumZ_and_partial(y, λ, ν)
                invZ = inv(S); p0cmp = invZ
                cdf[i] = (y == 0) ? p :
                         p + (1 - p) * ((S_y*invZ - p0cmp) / max(1 - p0cmp, 1e-12))
            end
        end
        return cdf
    end

    return (
        Θ₀         = Θ₀,
        likelihood = TwiceDifferentiable(loglikelihood, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-∞,-∞,-∞,-∞],[+∞,+∞,+∞,+∞]),
        residual   = residual,
        quantile   = quantile,
        cumulative = cumulative
    )
end


function ZI_CMPoisson(count, depth; Θ̄₂=1, δΘ₂⁻²=0, Θ₃_priors=nothing) # ZI–CMP (log param for ν, logit param for p)
    @assert length(count) == length(depth)
    c = collect(count); d = collect(depth); N = length(c)

    # maps for dispersion and zero-inflation
    @inline _ν_from_ρ(ρ::T) where {T} = exp(-ρ)           # ν = e^{-ρ} > 0
    @inline _κ_from_ρ(ρ::T) where {T} = exp(ρ)            # κ = 1/ν
    @inline _σ(x::T) where {T} = inv(one(T) + exp(-x))    # logistic
    @inline _lse(a::T,b::T) where {T} = (m = max(a,b); m + log1p(exp(min(a,b)-m)))  # logsumexp

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

    # --- single pass sums for CDF/quantile (CMP part) ---
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

    # --- log-likelihood (ZI–CMP) ---
    loglikelihood =
        if isnothing(Θ₃_priors)
            function (Θ::AbstractVector{T}) where {T}
                Θ₁, Θ₂, ρ, ψ = Θ
                ν = _ν_from_ρ(ρ)
                s_local = zeros(T, Threads.nthreads())
                if eltype(Θ) <: Real
                    Threads.@threads for i in 1:N
                        n   = c[i]; di = d[i]
                        logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                        logZ = _logZ(λ, ν)
                        p = _σ(ψ)
                        if n == 0
                            s_local[Threads.threadid()] += _lse(log(p), log1p(-p) - logZ)
                        else
                            s_local[Threads.threadid()] += log1p(-p) + n*logλ - ν*get_lg(Int(n)) - logZ
                        end
                    end
                    s = zero(T); @inbounds @simd for t in 1:length(s_local); s += s_local[t]; end
                else
                    s = zero(T)
                    @inbounds @simd for i in 1:N
                        n   = c[i]; di = d[i]
                        logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                        logZ = _logZ(λ, ν)
                        p = _σ(ψ)
                        if n == 0
                            s += _lse(log(p), log1p(-p) - logZ)
                        else
                            s += log1p(-p) + n*logλ - ν*get_lg(Int(n)) - logZ
                        end
                    end
                end
                reg = (T(0.5)*T(δΘ₂⁻²))*(Θ₂ - T(Θ̄₂))^2
                return -(s) + reg
            end
        else
            μ, σ, νp = Θ₃_priors   # prior on log ν stays the same
            function (Θ::AbstractVector{T}) where {T}
                Θ₁, Θ₂, ρ, ψ = Θ
                ν = _ν_from_ρ(ρ)
                s_local = zeros(T, Threads.nthreads())
                if eltype(Θ) <: Real
                    Threads.@threads for i in 1:N
                        n   = c[i]; di = d[i]
                        logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                        logZ = _logZ(λ, ν)
                        p = _σ(ψ)
                        if n == 0
                            s_local[Threads.threadid()] += _lse(log(p), log1p(-p) - logZ)
                        else
                            s_local[Threads.threadid()] += log1p(-p) + n*logλ - ν*get_lg(Int(n)) - logZ
                        end
                    end
                    s = zero(T); @inbounds @simd for t in 1:length(s_local); s += s_local[t]; end
                else
                    s = zero(T)
                    @inbounds @simd for i in 1:N
                        n   = c[i]; di = d[i]
                        logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                        logZ = _logZ(λ, ν)
                        p = _σ(ψ)
                        if n == 0
                            s += _lse(log(p), log1p(-p) - logZ)
                        else
                            s += log1p(-p) + n*logλ - ν*get_lg(Int(n)) - logZ
                        end
                    end
                end
                reg  = (T(0.5)*T(δΘ₂⁻²))*(Θ₂ - T(Θ̄₂))^2
                pen3 = (abs(log(ν) - T(μ))/T(σ))^T(νp)   # = (abs(-ρ - μ)/σ)^νp
                return -(s) + reg + pen3
            end
        end

    # --- initializer (ρ₀=0 ⇒ ν₀=1; ψ from zero fraction) ---
    μ̄ = logmean(c)
    zfrac = mean(c .== 0)
    p0 = clamp(0.5*zfrac, 1e-6, 1 - 1e-6)   # mild zero-inflation start
    ψ0 = log(p0/(1-p0))
    Θ₀ = [log(μ̄), 1.0, 0.0, ψ0]

    # --- residuals: ZI mean/var using CMP asymptotics used before -------------
    residual = function (Θ::AbstractVector{T}) where {T}
        Θ₁, Θ₂, ρ, ψ = Θ
        ν    = _ν_from_ρ(ρ)
        invν = _κ_from_ρ(ρ)   # = exp(ρ)
        p    = _σ(ψ)
        half = T(0.5)
        out  = Vector{T}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                logλ = Θ₁ + Θ₂*d[i]
                λν   = exp(logλ * invν)                  # CMP mean approx core
                μcmp = λν + (inv(T(2)*ν) - half)
                σ2cmp= max(invν * λν, oftype(logλ, eps(Float64)))
                μzi  = (one(T) - p)*μcmp
                σ2zi = (one(T) - p)*(σ2cmp + p*μcmp*μcmp)
                z    = (c[i] - μzi) / sqrt(max(σ2zi, oftype(logλ, eps(Float64))))
                @inbounds out[i] = ifelse(z < -T(5), -T(5), ifelse(z > T(5), T(5), z))
            end
        else
            @inbounds @simd for i in 1:N
                logλ = Θ₁ + Θ₂*d[i]
                λν   = exp(logλ * invν)
                μcmp = λν + (inv(T(2)*ν) - half)
                σ2cmp= max(invν * λν, oftype(logλ, eps(Float64)))
                μzi  = (one(T) - p)*μcmp
                σ2zi = (one(T) - p)*(σ2cmp + p*μcmp*μcmp)
                z    = (c[i] - μzi) / sqrt(max(σ2zi, oftype(logλ, eps(Float64))))
                out[i] = ifelse(z < -T(5), -T(5), ifelse(z > T(5), T(5), z))
            end
        end
        return out
    end

    # --- randomized normal scores (ZI-CMP) ------------------------------------
    quantile = function (Θ)
        Θ₁, Θ₂, ρ, ψ = Θ
        ν = _ν_from_ρ(ρ)
        p = _σ(ψ)
        qv = Vector{Float64}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                y    = Int(c[i])
                logλ = Θ₁ + Θ₂*d[i]; λ = exp(logλ)
                S, S_yminus = _sumZ_and_partial(max(y-1, 0), λ, ν)
                invZ = inv(S)
                if y == 0
                    mass0 = p + (1 - p) * invZ
                    Fi = clamp(rand()*mass0, 1e-15, 1-1e-15)
                else
                    pycmp = exp(y*logλ - ν*get_lg(y)) * invZ
                    Flo   = p + (1 - p) * (S_yminus * invZ)
                    Fi    = clamp(Flo + rand()*((1 - p)*pycmp), 1e-15, 1-1e-15)
                end
                @inbounds qv[i] = sqrt(2) * erfinv(2*Fi - 1)
            end
        else
            @inbounds for i in 1:N
                y    = Int(c[i])
                logλ = Θ₁ + Θ₂*d[i]; λ = exp(logλ)
                S, S_yminus = _sumZ_and_partial(max(y-1, 0), λ, ν)
                invZ = inv(S)
                if y == 0
                    mass0 = p + (1 - p) * invZ
                    Fi = clamp(rand()*mass0, 1e-15, 1-1e-15)
                else
                    pycmp = exp(y*logλ - ν*get_lg(y)) * invZ
                    Flo   = p + (1 - p) * (S_yminus * invZ)
                    Fi    = clamp(Flo + rand()*((1 - p)*pycmp), 1e-15, 1-1e-15)
                end
                qv[i] = sqrt(2) * erfinv(2*Fi - 1)
            end
        end
        clamp!(qv, -10.0, 10.0)
        return qv
    end

    # --- CDF at observed counts (ZI-CMP) --------------------------------------
    cumulative = function (Θ)
        Θ₁, Θ₂, ρ, ψ = Θ
        ν = _ν_from_ρ(ρ)
        p = _σ(ψ)
        cdf = Vector{Float64}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                y = Int(c[i]); λ = exp(Θ₁ + Θ₂*d[i])
                S, S_y = _sumZ_and_partial(y, λ, ν)
                @inbounds cdf[i] = p + (1 - p) * (S_y * inv(S))
            end
        else
            @inbounds @simd for i in 1:N
                y = Int(c[i]); λ = exp(Θ₁ + Θ₂*d[i])
                S, S_y = _sumZ_and_partial(y, λ, ν)
                cdf[i] = p + (1 - p) * (S_y * inv(S))
            end
        end
        return cdf
    end

    return (
        Θ₀         = Θ₀,
        likelihood = TwiceDifferentiable(loglikelihood, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-∞,-∞,-∞,-∞],[+∞,+∞,+∞,+∞]),  # ρ, ψ unconstrained
        residual   = residual,
        quantile   = quantile,
        cumulative = cumulative
    )
end

function CMPoisson(count, depth; Θ̄₂=1, δΘ₂⁻²=0, Θ₃_priors=nothing) # This is log parameterization
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

function CMPoisson_fixed_rho(count, depth, ρ; Θ̄₂=1, δΘ₂⁻²=0, Θ₃_priors=nothing)
    @assert length(count) == length(depth)
    c = collect(count); d = collect(depth); N = length(c)

    # maps
    @inline _ν_from_ρ(ρ::T) where {T} = exp(-ρ)
    @inline _κ_from_ρ(ρ::T) where {T} = exp(ρ)
    
    @inline function _logZ(λ::T, ν::T; Kmax::Int=50_000, rtol::T=oftype(λ, eps(Float64)*100)) where {T}
        if !(λ > zero(T)) || !(ν > zero(T)); return zero(T) end

        # anchor near the mode and snap to discrete bracket
        logλ = log(λ)
        m̂ = exp(logλ/ν) - (ν - one(T))/(T(2)*ν)
        m  = floor(Int, clamp(m̂, zero(T), T(Kmax - 1)))
        @inbounds while (m + 1) < Kmax && λ > exp(ν * log(T(m + 1))); m += 1; end
        @inbounds while m > 0            && λ < exp(ν * log(T(m)));     m -= 1; end

        # base log-weight at m
        a_m = T(m)*logλ - ν*loggamma(T(m) + one(T))
        S   = one(T)

        # ---------- upward tail: k = m+1, m+2, ... ----------
        k  = m
        t  = one(T)
        # ratio r_{m+1} = λ/(m+1)^ν
        rk = λ * exp(-ν * log(T(k + 1)))
        @inbounds while k < Kmax
            k  += 1
            t  *= rk
            S  += t
            if t ≤ rtol*S; break; end
            # remainder bound using monotone ratio: tail ≤ t * rk_next / (1 - rk_next)
            # Compute rk_next without extra logs via telescoping: rk_next = rk * (k/(k+1))^ν
            rk_next = rk * exp(-ν * log1p(one(T)/T(k)))
            if rk_next < one(T)  # it is, but guard keeps AD safe
                rem = t * rk_next / (one(T) - rk_next)
                if rem ≤ rtol*S; break; end
            end
            rk = rk_next
        end

        # ---------- downward tail: k = m-1, m-2, ... ----------
        k = m
        t = one(T)
        if k > 0
            invλ = inv(λ)
            # s_k = k^ν / λ  (ratio for stepping downward)
            sk = exp(ν * log(T(k))) * invλ
            @inbounds while k > 0
                kT = T(k)
                t  *= sk
                S  += t
                if t ≤ rtol*S; break; end
                # next ratio s_{k-1} = s_k * ((k-1)/k)^ν via log1p
                sk_next = (k == 1) ? zero(T) : sk * exp(ν * log1p(-one(T)/kT))
                if sk_next < one(T)
                    rem = t * sk_next / (one(T) - sk_next)
                    if rem ≤ rtol*S; break; end
                end
                k  -= 1
                sk = sk_next
            end
        end

        return a_m + log(S)
    end



    # cache
    int_counts = eltype(c) <: Integer && all(x -> x ≥ 0, c)
    lg_cache = if int_counts
        maxy = maximum(c); v = Vector{Float64}(undef, maxy + 1)
        @inbounds @simd for k in 1:length(v); v[k] = loggamma(k); end; v
    else
        Float64[]
    end
    @inline get_lg(n::Int) = int_counts ? @inbounds(lg_cache[n+1]) : loggamma(n+1)

    νfix = _ν_from_ρ(ρ)

    # --- 2D log-likelihood in (Θ₁, Θ₂) with fixed ρ ---
    loglikelihood = function (Θ::AbstractVector{T}) where {T}
        Θ₁, Θ₂ = Θ
        s_local = zeros(T, Threads.nthreads())
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                n = c[i]; di = d[i]
                logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                logZ = _logZ(λ, T(νfix))
                s_local[Threads.threadid()] += n*logλ - T(νfix)*get_lg(Int(n)) - logZ
            end
            s = zero(T); @inbounds @simd for t in 1:length(s_local); s += s_local[t]; end
        else
            s = zero(T)
            @inbounds @simd for i in 1:N
                n = c[i]; di = d[i]
                logλ = Θ₁ + Θ₂*di; λ = exp(logλ)
                logZ = _logZ(λ, T(νfix))
                s += n*logλ - T(νfix)*get_lg(Int(n)) - logZ
            end
        end
        reg = (T(0.5)*T(δΘ₂⁻²))*(Θ₂ - T(Θ̄₂))^2
        return -(s) + reg
    end

    # initializer
    μ̄ = logmean(c)
    Θ₀ = [log(μ̄), 1.0]

    # residuals/quantile/cdf identical to your CMP, just plug fixed ρ
    residual = function (Θ::AbstractVector{T}) where {T}
        Θ₁, Θ₂ = Θ
        ν    = T(νfix)
        invν = _κ_from_ρ(T(ρ))
        half = T(0.5)
        out  = Vector{T}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                logλ = Θ₁ + Θ₂*d[i]
                λν   = exp(logλ * invν)
                μ    = λν + (inv(T(2)*ν) - half)
                σ2   = max(invν * λν, oftype(logλ, eps(Float64)))
                z    = (c[i] - μ) / sqrt(σ2)
                @inbounds out[i] = clamp(z, -T(5), T(5))
            end
        else
            @inbounds @simd for i in 1:N
                logλ = Θ₁ + Θ₂*d[i]
                λν   = exp(logλ * invν)
                μ    = λν + (inv(T(2)*ν) - half)
                σ2   = max(invν * λν, oftype(logλ, eps(Float64)))
                out[i] = clamp((c[i]-μ)/sqrt(σ2), -T(5), T(5))
            end
        end
        out
    end

    # single-pass sum for cdf (same as yours)
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

    quantile = function (Θ)
        Θ₁, Θ₂ = Θ
        ν = νfix
        ρv = Vector{Float64}(undef, N)
        if eltype(Θ) <: Real
            Threads.@threads for i in 1:N
                y = Int(c[i]); λ = exp(Θ₁ + Θ₂*d[i])
                S, S_yminus = _sumZ_and_partial(max(y-1,0), λ, ν)
                invZ = inv(S)
                py   = exp(y*(Θ₁+Θ₂*d[i]) - ν*get_lg(y)) * invZ
                Flo  = (y==0 ? 0.0 : S_yminus*invZ)
                Fi   = clamp(Flo + rand()*py, 1e-15, 1-1e-15)
                @inbounds ρv[i] = sqrt(2) * erfinv(2*Fi - 1)
            end
        else
            @inbounds for i in 1:N
                y = Int(c[i]); λ = exp(Θ₁ + Θ₂*d[i])
                S, S_yminus = _sumZ_and_partial(max(y-1,0), λ, ν)
                invZ = inv(S)
                py   = exp(y*(Θ₁+Θ₂*d[i]) - ν*get_lg(y)) * invZ
                Flo  = (y==0 ? 0.0 : S_yminus*invZ)
                Fi   = clamp(Flo + rand()*py, 1e-15, 1-1e-15)
                ρv[i] = sqrt(2) * erfinv(2*Fi - 1)
            end
        end
        clamp!(ρv, -10.0, 10.0); ρv
    end

    cumulative = function (Θ)
        Θ₁, Θ₂ = Θ
        ν = νfix
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
        cdf
    end

    return (
        Θ₀         = Θ₀,
        ρ          = ρ,                          # <- fixed here
        likelihood = TwiceDifferentiable(loglikelihood, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-∞,-∞],[+∞,+∞]),
        residual   = residual,
        quantile   = quantile,
        cumulative = cumulative
    )
end

CMP_fixed(ρ) = (y, depth; priors=nothing) -> CMPoisson_fixed_rho(y, depth, ρ)

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
    println("Fitting $(length(selected)) genes out of $(size(data,1))")
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

    labels = lowercase.(replace.(string.(model.model), r".*\." => ""))
    idx_nb    = findall(==("negativebinomial"),   labels)
    idx_cmp   = findall(==("cmpoisson"),          labels)
    idx_gp    = findall(==("generalizedpoisson"), labels)
    idx_zicmp = findall(==("zi_cmpoisson"),       labels)
    idx_hcmp  = findall(l -> occursin("hurdle", l) && occursin("cmp", l), labels)

    # params
    pmap = model.params
    getv(sym, default) = haskey(pmap, sym) ? pmap[sym] : default
    θv = getv(:θ,  getv(:Θ₃, 0.0))
    ρv = getv(:ρ,  getv(:Θ₃, 0.0))
    αv = getv(:α,  getv(:Θ₃, 0.0))
    # p (probability of zero) — ensure Θ₄ is treated as logit(p)
    pv = haskey(pmap,:p)  ? pmap[:p] :
         haskey(pmap,:ψ)  ? @.(1 / (1 + exp(-pmap[:ψ]))) :
         haskey(pmap,:Θ₄) ? @.(1 / (1 + exp(-pmap[:Θ₄]))) : 0.0

    _par(v,g) = v isa AbstractVector ? Float64(v[g]) : Float64(v)

    # CMP moments (μ, var, p0) with corrected p0
    @inline function _cmp_moments(λ::Float64, ν::Float64; tol::Float64=tol, kmax::Int=kmax)
        if !(λ>0) || !(ν>0); return 0.0, 0.0, 1.0 end
        m̂ = exp(log(λ)/ν) - (ν - 1)/(2ν)
        m  = floor(Int, clamp(m̂, 0.0, float(kmax-1)))
        Zr = 1.0; S1 = m*1.0; S2 = (m*m)*1.0  # relative sums (t_m = 1)

        p = 1.0; k = m
        while k < kmax
            k += 1
            p *= λ / exp(ν*log(k))
            if p < tol*Zr; break; end
            Zr += p; S1 += p*k; S2 += p*k*k
        end
        p = 1.0; k = m
        while k > 0
            p *= exp(ν*log(k)) / λ
            k -= 1
            if p < tol*Zr; break; end
            Zr += p; S1 += p*k; S2 += p*k*k
        end

        μ = S1/Zr
        v = max(S2/Zr - μ*μ, 0.0)

        am    = m*log(λ) - ν*loggamma(m + 1.0)  # log t_m
        logp0 = -am - log(Zr)
        p0    = clamp(exp(logp0), 0.0, 1.0)

        return μ, v, p0
    end

    # NB
    @inbounds for g in idx_nb
        θg = _par(θv,g); @views μg = data[g,:]; @views Σg = Σ[g,:]
        @inbounds for c in 1:C
            μ = Float64(μg[c]); Σg[c] = μ + (μ*μ)/θg
        end
    end

    # CMP
    @inbounds for g in idx_cmp
        νg = exp(-_par(ρv,g)); @views λg = data[g,:]; @views Σg = Σ[g,:]
        @inbounds for c in 1:C
            _, vcmp, _ = _cmp_moments(Float64(λg[c]), νg; tol=tol, kmax=kmax)
            Σg[c] = vcmp
        end
    end

    # Generalized Poisson
    @inbounds for g in idx_gp
        αg = _par(αv,g); d = 1.0 - αg; @views μg = data[g,:]; @views Σg = Σ[g,:]
        if d > 0 && isfinite(d)
            invd2 = 1.0/(d*d)
            @inbounds for c in 1:C
                Σg[c] = Float64(μg[c]) * invd2
            end
        else
            fill!(Σg, Inf)
        end
    end

    # ZI-CMP (p = extra-zero prob)
    @inbounds for g in idx_zicmp
        νg = exp(-_par(ρv,g)); p = clamp(_par(pv,g), 0.0, 1.0)
        @views λg = data[g,:]; @views Σg = Σ[g,:]
        @inbounds for c in 1:C
            μcmp, vcmp, _ = _cmp_moments(Float64(λg[c]), νg; tol=tol, kmax=kmax)
            Σg[c] = (1 - p) * (vcmp + p*μcmp*μcmp)
        end
    end

    # Hurdle-CMP  (p = P(Y=0); positive part is zero-truncated CMP)
    @inbounds for g in idx_hcmp
        νg = exp(-_par(ρv,g)); p = clamp(_par(pv,g), 0.0, 1.0)
        @views λg = data[g,:]; @views Σg = Σ[g,:]
        @inbounds for c in 1:C
            μcmp, vcmp, p0 = _cmp_moments(Float64(λg[c]), νg; tol=tol, kmax=kmax)
            one_m_p0 = max(1.0 - p0, 1e-12)
            μplus    = μcmp / one_m_p0
            σ2plus   = ((vcmp + μcmp*μcmp) / one_m_p0) - (μplus*μplus)
            σ2plus   = max(σ2plus, 0.0)
            # Var(Y) = (1-p) Var_plus + p(1-p) μ_plus^2
            Σg[c]    = (1 - p) * σ2plus + p*(1 - p) * (μplus*μplus)
        end
    end

    assigned = sort!(vcat(idx_nb, idx_cmp, idx_gp, idx_zicmp, idx_hcmp))
    if length(assigned) != G
        @warn "Some genes were not assigned a model in `model.model`"
    end

    @inbounds for i in eachindex(Σ)
        if Σ[i] < 0; @warn "Negative variance detected"; end
    end

    return Σ
end




"""
    normalize(data; δ=5)

WIP
"""

function normalize(data, model; δ = 2)

    base = let
        depth = map(eachcol(data)) do col
            col |> vec |> logmean
        end
        Θ1c = reshape(model.Θ₁, :, 1)
        Θ2c = reshape(model.Θ₂, :, 1)
        η   = Θ1c .+ Θ2c .* permutedims(depth)
        exp.(η)
    end

    Σ = conditional_variance_mixture(base, model) # Getting negative variances with very low Θ₃ values. These genes should probably be filtered out.

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
