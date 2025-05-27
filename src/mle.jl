module MLE

using ProgressMeter
using ForwardDiff ## ADDED FOR DEBUGGING
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

                # for (i, (xx, zz)) in enumerate(zip(x⃗, z⃗)) # FOR DEBUGGING
                #     # Unwrap the dual numbers to get plain Float64 values
                #     xx_val = ForwardDiff.value(xx)
                #     zz_val = ForwardDiff.value(zz)
                #     α_val  = ForwardDiff.value(α)
                #     β_val  = ForwardDiff.value(β)
                #     γ_val  = ForwardDiff.value(γ)
                    
                #     μ_val = exp(α_val + β_val * zz_val)
                #     val = loggamma(xx_val + γ_val) - loggamma(xx_val + 1) - loggamma(γ_val) +
                #           xx_val * (α_val + β_val * zz_val) + γ_val * log(γ_val) - (xx_val + γ_val) * log(μ_val + γ_val)
                    
                #     if isnan(val) || isinf(val)
                #         println("Error at index $i:")
                #         println("  xx = $xx_val")
                #         println("  zz = $zz_val")
                #         println("  μ  = $μ_val")
                #         println("  val = $val")
                #     end
                # end
                
            return -sum(L) #+ 0.5*δβ¯²*(β-β̄)^2
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

    μ  = logmean(x⃗)
    Θ₀ = [log(μ),2,logvar(x⃗)/μ - 1,] # XXX Understand this better 

    μ  = mean(x⃗)
    Θ₀ = [
        log(μ),
        β̄,
        var(x⃗)/μ - 1,
    ]


    if Θ₀[end] < 0 || isinf(Θ₀[end]) || isnan(Θ₀[end])
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
            z = @. (x⃗ - μ) / σ
            z[ z .< -5 ] .= -5
            z[ z .> +5 ] .= +5
            return z
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
function fit_glm(model::Symbol, data; Γ=(β̄=0,δβ¯²=0,Γᵧ=nothing), run=(x)->true)
    
    foundmodel = try
        getfield(MLE, model)
    catch
        error("model '$model' not implemented")
    end

    average(x) = logmean(x; ϵ=1)
    depth = map(eachcol(data)) do col
        col |> vec |> average
    end

    modelfor(x) = foundmodel(x, depth, Γ)

    function fit(row)::FitType
        model    = modelfor(vec(row))
        estimate = optimize(model.loss, model.constraint, model.Θ₀, IPNewton())

        E  = Optim.minimum(estimate)
        Θ  = Optim.minimizer(estimate)
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
    progress = Progress(sum(ι .!= 0); desc="--> fitting:", output=stderr, color=:blue)
    for (i, row) ∈ enumerate(eachrow(data))
        ι[i] == 0 && continue

        fits[ι[i]] = fit(row) 
        next!(progress)
    end

    return (
        likelihood  = map((f)->f.likelihood,  fits),
        residual   = Matrix(reduce(hcat, map((f)->f.residuals, fits))'),
        priors = Γ,

        # XXX: assumes particular form for parameters
        Θ₁  = map((f)->f.parameters[1],  fits),
        Θ₂  = map((f)->f.parameters[2],  fits),
        Θ₃  = map((f)->f.parameters[3],  fits),

        δΘ₁ = map((f)->f.uncertainty[1], fits),
        δΘ₂ = map((f)->f.uncertainty[2], fits),
        δΘ₃ = map((f)->f.uncertainty[3], fits),

        cdf = map((f)->f.cumulative, fits),
    )

end

function bootstrap_fit(model::Symbol, count, depth, ; Γ=(β̄=1,δβ¯²=0,Γᵧ=nothing))
    
    # foundmodel = try
    #     getfield(MLE, model)
    # catch
    #     error("model '$model' not implemented")
    # end
    # modelfor(x) = foundmodel(count, depth, Γ)

    # model = modelfor(vec(data))
    # println("model: ", model)
    model = negative_binomial(count, depth, Γ)
    param = optimize(model.loss, model.constraint, model.Θ₀, IPNewton())

    Θ  = Optim.minimizer(param)
    E  = Optim.minimum(param)
    δΘ = diag(inv(hessian!(model.loss, Θ)))

    return (
            likelihood  = E,
            parameters  = Θ,
            uncertainty = δΘ,
            cumulative  = model.cumulative(Θ), # This was giving an error
            residuals   = model.residual(Θ),
        )

end

function fit(model)
    estimate = optimize(model.loss, model.constraint, model.Θ₀, IPNewton())
    return Optim.minimizer(estimate)
end

function TestReviseMLE()
    println("TestReviseMLE 6:19")
end

end

