module Normalize

using LinearAlgebra
using Optim, NLSolversBase
using Random, Statistics, StatsBase, NMF
using SpecialFunctions: loggamma, erfinv, erf
using GSL, ProgressMeter

include("scrna.jl")
include("util.jl")
using .scRNA: Count

incbeta(a,b,x) = GSL.sf_beta_inc(a,b,x)
incgamma(a,x)  = GSL.sf_gamma_inc_P(a,x)

export normalize

const ∞ = Inf

"""
    logmean(x;ϵ=1)

Compute the geometric mean: ``\exp\\left(\\langle \\log\\left(x + \\epsilon \\right) \\rangle\\right) - \\epsilon``
"""
const logmean(x;ϵ=1) = exp.(mean(log.(x.+ϵ)))-ϵ

"""
    logmean(x;ϵ=1)

Compute the geometric variance: ``exp\\left(\\langle \\log\\left(x + \\epsilon \\right)^2 \\rangle_c\\right) - \\epsilon``
"""
const logvar(x;ϵ=1)  = exp.(var(log.(x.+ϵ)))-ϵ

"""
    negativebinomial(count, depth)

Compute the log likelihood of a negative binomial generalized linear model (GLM) with log link function for `count` of a single gene.
The sequencing depth for each sequenced cell is assumed to be the only confounding variables.
A prior on Θ₂ is assumed to be Gaussian with mean `Θ̄₂` and variance `δΘ₂⁻²`.
"""
function negativebinomial(count, depth; Θ̄₂=1, δΘ₂⁻²=5, Θ₃_priors = nothing)
    if isnothing(Θ₃_priors)
        loglikelihood = function(Θ)
            Θ₁,Θ₂,Θ₃ = Θ
            return -sum(
                loggamma(n+Θ₃)
            - loggamma(n+1)
            - loggamma(Θ₃)
            + n*(Θ₁+Θ₂*d)
            + Θ₃*log(Θ₃)
            - (n+Θ₃)*log(exp(Θ₁+Θ₂*d)+Θ₃)
            for (n,d) ∈ zip(count,depth)
            ) + 0.5*δΘ₂⁻²*(Θ₂-Θ̄₂)^2
        end
    else
        loglikelihood = function(Θ)
            Θ₁,Θ₂,Θ₃ = Θ
            μ,σ,ν = Θ₃_priors
            return -sum(
                loggamma(n+Θ₃)
            - loggamma(n+1)
            - loggamma(Θ₃)
            + n*(Θ₁+Θ₂*d)
            + Θ₃*log(Θ₃)
            - (n+Θ₃)*log(exp(Θ₁+Θ₂*d)+Θ₃)
            for (n,d) ∈ zip(count,depth)
            ) + 0.5*δΘ₂⁻²*(Θ₂-Θ̄₂)^2 + (abs(log(Θ₃)-μ)/σ)^ν
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
        cumulative = function(Θ)
            Θ₁,Θ₂,Θ₃ = Θ
            μ = @. exp(Θ₁ + Θ₂*depth)
            p = @. μ / (μ + Θ₃)
        return @. incbeta(count+1, Θ₃, p) # CDF of negative binomial
    end
    )
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
    loglikelihood = function(Θ)
        logm, Θ₃ = Θ
        m = @. exp(logm)
        α = @. Θ₃ * m
        return -sum(@. α*log(Θ₃) + (α - 1)*log(count + 0.001) - (Θ₃*count) - loggamma(α))
    end

    logm₀  = log(mean(count))
    θ₃₀   = logvar(count)/(exp(logm₀) - 1)
    init_params = [logm₀; θ₃₀]
    if init_params[end] < 0 || isinf(init_params[end]) || isnan(init_params[end])
        init_params[end] = 1
    end

    return (
        init_params = init_params,
        likelihood = TwiceDifferentiable(loglikelihood, init_params; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-Inf,0],[+Inf,+Inf]),
        residual   = function(Θ)
            logm, Θ₃ = Θ
            μ = @. exp(logm)
            σ = @. sqrt(μ / Θ₃)
            z = @. (count - μ) / σ

            z[ z .< -5 ] .= -5
            z[ z .> +5 ] .= +5

            return z
        end,
        quantile = function(Θ)
            logm, Θ₃ = Θ
            α  = @. Θ₃ * exp(logm)
            Φ  = @. incgamma(α, count*Θ₃)
            ρ  = @. erfinv(clamp(2Φ - 1, -1, +1)) * √2
            clamp!(ρ, -10, +10)
            return ρ
        end,
        cumulative = function(Θ)
            logm, Θ₃ = Θ
            α  = @. Θ₃ * exp(logm)
            return @. incgamma(α, count*Θ₃)
        end
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

    return (
        likelihood  = Optim.minimum(param),
        parameters  = Θ̂,
        uncertainty = diag(inv(hessian!(model.likelihood, Optim.minimizer(param)))),
        cumulative  = model.cumulative(Θ̂),
        residual    = model.residual(Θ̂),
        quantile = try 
            model.quantile(Θ̂)
        catch
            nothing
        end
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
    progress = Progress(size(counts,1); desc="--> fitting:", output=stderr, color = :yellow)

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
                likelihood          = map((f)->f.likelihood,  fits),
                residual            = Matrix(reduce(hcat, map((f)->f.pearson_residual, fits))'),
                quantile_residual   =  Matrix(reduce(hcat, map((f)->f.quantile_residual, fits))'),

                logm    = map((f)->f.parameters[1],  fits),
                Θ₃      = map((f)->f.parameters[2],  fits),

                δlogm   = map((f)->f.uncertainty[1], fits),
                δΘ₃   = map((f)->f.uncertainty[2], fits),
                
                cdf     = map((f)->f.cumulative, fits),
            )
end

"""
    bootstrap(count, depth; stochastic=negativebinomial, samples=50)

Empirically verify the MLE fit of `count`, using a GLM model generated by `stochastic` with confounding `depth` variables by bootstrap.
One third of cells are removed and the parameters are re-estimated with the remaining cells.
This process is repeated `samples` times.
The resultant distribution of estimation is returned.
"""
function bootstrap(count, depth; stochastic=negativebinomial, priors = nothing, samples=50)
    N = length(depth)

    Θ₁ = Array{Float64}(undef,samples)
    Θ₂ = Array{Float64}(undef,samples)
    Θ₃ = Array{Float64}(undef,samples)
    δL = Array{Float64}(undef,samples)

    for n in 1:samples
        ι = randperm(N)[1:2*N÷3]
        f = fit(stochastic, count[ι], depth[ι]; priors=priors)

        δL[n] = f.likelihood
        Θ₁[n], Θ₂[n], Θ₃[n] = f.parameters
    end

    return Θ₁, Θ₂, Θ₃, δL
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

    prog_lock = ReentrantLock() # for thread-safe progress bar
    progress = Progress(sum(ι .!= 0); desc="--> fitting:", output=stderr, color = barcolor)

    fits = Array{NamedTuple}(undef, length(selected))
    Threads.@threads for (i,gene) in collect(enumerate(eachrow(data)))
        ι[i] == 0 && continue
        fits[ι[i]] = fit(stochastic,vec(gene),depth; priors=priors)
        lock(prog_lock) do
            next!(progress)
        end
    end

    return (
        priors     = priors,
        likelihood = map((f)->f.likelihood,  fits),
        residual   = Matrix(reduce(hcat, map((f)->f.residual, fits))'),
        quantile = try 
                    map((f)->f.quantile, fits) 
                catch 
                    @warn("$stochastic has no implemented quantile function")
                    nothing 
                end,
        Θ₁  = map((f)->f.parameters[1],  fits),
        Θ₂  = map((f)->f.parameters[2],  fits),
        Θ₃  = map((f)->f.parameters[3],  fits),

        δΘ₁ = map((f)->f.uncertainty[1], fits),
        δΘ₂ = map((f)->f.uncertainty[2], fits),
        δΘ₃ = map((f)->f.uncertainty[3], fits),

        cdf = map((f)->f.cumulative, fits),
    )
end

"""
    normalize(data; δ=5)


"""

function normalize(data; δ = 2)

    model = glm(data; stochastic=negativebinomial, ϵ=1, barcolor = :green)
    
    X̃₁, Σ, u², v² = let
        Σ = data .* (data .+ model.Θ₃) ./ (1 .+ model.Θ₃)
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

    gamma_fits = fit_gamma_pooled(X̃)
    X̂ = gamma_fits.residual

    return (
        counts      = scRNA.Count(X̂, data.gene, data.cell),
        rank        = R,
        corr        = ρ,
        gamma_models  = gamma_fits,
        NB_models     = model,
        iid_counts    = X̃₁,
        NNMF_corr      = ρ
    )
    
end

function TestReviseNormalize()
    println("Testing Normalize.jl Revise; 8:56")
end
end
