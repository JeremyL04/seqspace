module MLE

using ProgressMeter
using LinearAlgebra, Statistics, StatsBase
using GSL, Optim, NLSolversBase
using SpecialFunctions: erfinv, loggamma

incbeta(a,b,x) = GSL.sf_beta_inc(a,b,x)
incgamma(a,x)  = GSL.sf_gamma_inc_P(a,x)

const ∞ = Inf
function clamp(x,lo,hi)
    x < lo && return lo
    x > hi && return hi
    return x
end

# ---------------------------------------------------------------------
# generalized linear models
# Γ denotes parameters for prior distributions

# the common formulation i.e. NB2
function negative_binomial(x⃗, z⃗, Γ)
    β̄,δβ¯²,Γᵧ = Γ

    f = if isnothing(Γᵧ)
        (Θ) -> let
            α,β,γ = Θ
            L  = (loggamma(x+γ) 
                - loggamma(x+1) 
                - loggamma(γ) 
                + x*(α+β*z)
                + γ*log(γ)
                - (x+γ)*log(exp(α+β*z)+γ) for (x,z) ∈ zip(x⃗,z⃗))

            return -sum(L) + 0.5*δβ¯²*(β-β̄)^2
        end
    else
        μ,σ,ν = Γᵧ
        (Θ) -> let
            α,β,γ = Θ
            L  = (loggamma(x+γ) 
                - loggamma(x+1) 
                - loggamma(γ) 
                + x*(α+β*z)
                + γ*log(γ)
                - (x+γ)*log(exp(α+β*z)+γ) for (x,z) ∈ zip(x⃗,z⃗))

            return -sum(L) + 0.5*δβ¯²*(β-β̄)^2 + (abs(log(γ)-μ)/σ)^ν
        end
    end

    # TODO: 1st and 2nd derivatives

    μ  = mean(x⃗)
    Θ₀ = [
        log(μ),
        β̄,
        var(x⃗)/μ - 1,
    ]

    if Θ₀[end] < 0 || isinf(Θ₀[end])
        Θ₀[end] = 1
    end

    return (
        Θ₀         = Θ₀,
        loss       = TwiceDifferentiable(f, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-∞,-∞,0],[+∞,+∞,+∞]),
        residual   = (Θ) -> let
            α, β, γ = Θ
            μ = @. exp(α + β*z⃗)
            σ = @. √(μ + μ^2/γ)
            return @. (x⃗ - μ) / σ
        end,
        cumulative = (Θ) -> let
            α, β, γ = Θ
            μ = @. exp(α + β*z⃗)
            p = @. μ / (μ + γ)
            return @. 1 - incbeta(x⃗+1, γ, p)
        end
    )
end

function gamma(x⃗, z⃗, Γ) 
    β̄, δβ¯² = Γ

    function f(Θ)
        α, β, γ = Θ

        k⃗ = (exp(α+β*z)/γ for z ∈ z⃗)
        L = (-loggamma(k)
             -k*log(γ)
             +(k-1)*log(x)
             -x/γ for (k,x) ∈ zip(k⃗,x⃗))

        return -sum(L) + 0.5*δβ¯²*(β-β̄)^2
    end

    μ  = mean(x⃗)
    Θ₀ = [
        log(μ),
        β̄,
        μ^2 / (var(x⃗)-μ),
    ]

    if Θ₀[end] < 0 || isinf(Θ₀[end])
        Θ₀[end] = 1
    end

    return (
        Θ₀         = Θ₀,
        loss       = TwiceDifferentiable(f, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-∞,-∞,0],[+∞,+∞,+∞]),
        residual   = (Θ) -> let
            α, β, γ = Θ
            μ = @. exp(α + β*z⃗)
            k = @. μ / γ
            Φ = @. incgamma(k, x⃗/γ)
            ρ = @. erfinv(clamp(2*Φ-1,-1,1))
            ρ[isinf.(ρ)] = 10*sign.(ρ[isinf.(ρ)])
            return ρ
        end,
        cumulative = (Θ) -> let
            α, β, γ = Θ
            μ = @. exp(α + β*z⃗)
            k = @. μ / γ
            return @. incgamma(k, x⃗/γ)
        end
    )
end

# ---------------------------------------------------------------------
# simple distributions (no latent variables)

# univariate gamma
function gamma(x)
    f = (Θ) -> let
        k, θ = Θ
        return -sum((k-1)*log.(x) .- (x./θ) .- loggamma(k) .- k*log(θ))
    end

    μ  = mean(x)
    σ² = var(x)
    Θ₀ = [
        μ^2/σ²,
        σ²/μ,
    ]

    return (
        Θ₀         = Θ₀,
        loss       = TwiceDifferentiable(f, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([0,0],[+∞,+∞]),
        cumulative = (Θ) -> let
            k, θ = Θ
            return @. incgamma(k, x/θ)
        end
    )
end

# univariate lognormal
function log_normal(x)
    f = (Θ) -> let
        μ, σ = Θ
        return sum(log(σ) .+ (log.(x) .- μ).^2 ./ (2*σ^2))
    end
    
    μ  = mean(log.(x))
    σ  = std(log.(x))
    Θ₀ = [
        μ,
        σ
    ]
    
    return (
        Θ₀         = Θ₀,
        loss       = TwiceDifferentiable(f, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([0,0],[+∞,+∞]),
        cumulative = (Θ) -> let
            μ, σ = Θ 
            return @. .5*(1+erf((log(x) - μ)/(√2*σ)))
        end
    )
end

# univariate generalized normal (variant 1 wikipedia)
function generalized_normal(x)
    f = (Θ) -> let
        μ, σ, β = Θ
        return sum(loggamma(1/β) + log(σ) .+ (abs.(x .- μ)./ σ).^β .- log(β))
    end

    μ  = mean(x)
    σ  = std(x)
    Θ₀ = [
        μ,
        σ,
        2
    ]
    
    return (
        Θ₀         = Θ₀,
        loss       = TwiceDifferentiable(f, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([0,0,0],[+∞,+∞,+∞]),
        cumulative = (Θ) -> let
            μ, σ, β = Θ
            return @. .5*(1+sign(x-μ)*incgamma((1/β), abs(x-μ)/(σ^β)))
        end
   )
end

# ---------------------------------------------------------------------
# drivers

const FitType = NamedTuple{
    (
        :likelihood,
        :parameters,
        :uncertainty,
        :cumulative,
        :residuals,
    ),
    Tuple{
        Float64,
        Array{Float64,1},
        Array{Float64,1},
        Array{Float64,1},
        Array{Float64,1},
    }
}

# log link function is assumed
function fit_glm(model::Symbol, data; Γ=(β̄=1,δβ¯²=10,Γᵧ=nothing), run=(x)->true)
    Σ = vec(mean(data, dims=1))

    foundmodel = try
        getfield(MLE, model)
    catch
        error("model '$model' not implemented")
    end
    modelfor(x) = foundmodel(x, log.(Σ), Γ)

    function fit(row, i)::FitType
        model    = modelfor(vec(row))
        estimate = optimize(model.loss, model.constraint, model.Θ₀, IPNewton())

        Θ  = Optim.minimizer(estimate)
        E  = Optim.minimum(estimate)
        δΘ = diag(inv(hessian!(model.loss, Θ)))

        return (
            likelihood  = E,
            parameters  = Θ,
            uncertainty = δΘ,
            cumulative  = model.cumulative(Θ),
            residuals   = model.residual(Θ),
        )
    end

    ι = zeros(Int, size(data,1))
    j = 1
    for (i, row) ∈ enumerate(eachrow(data))
        if run(row)
            ι[i] = j
            j   += 1
        end
    end

    fits     = Array{FitType,1}(undef,j-1)
    progress = Progress(sum(ι .!= 0); desc="--> fitting:", output=stderr)
    for (i, row) ∈ enumerate(eachrow(data))
        ι[i] == 0 && continue

        fits[ι[i]] = fit(row,i) 
        next!(progress)
    end

    return (
        likelihood  = map((f)->f.likelihood,  fits),

        # XXX: assumes particular form for parameters
        α  = map((f)->f.parameters[1],  fits),
        β  = map((f)->f.parameters[2],  fits),
        γ  = map((f)->f.parameters[3],  fits),

        δα = map((f)->f.uncertainty[1], fits),
        δβ = map((f)->f.uncertainty[2], fits),
        δγ = map((f)->f.uncertainty[3], fits),

        cdf = map((f)->f.cumulative, fits),
    )

end

function fit(model)
    estimate = optimize(model.loss, model.constraint, model.Θ₀, IPNewton())
    return Optim.minimizer(estimate)
end

end

