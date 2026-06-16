module SeqSpace

using GZip
using BSON: @save, @load
using LinearAlgebra: norm, svd, Diagonal, dot, eigvals, tr
using Statistics: quantile, std, var
import Statistics
using Flux, Zygote
using ProgressMeter
using DelaunayTriangulation

import BSON

include("io.jl")
include("rank.jl")
include("model.jl")
include("radon.jl")
include("pointcloud.jl")
include("generate.jl")
include("normalize.jl")
include("manifold.jl")
include("infer.jl")
include("scrna.jl")
include("voronoi.jl")
include("util.jl")
include("mle.jl")
include("distance.jl")


using .PointCloud, .DataIO, .SoftRank, .ML, .Voronoi, .Radon, .Normalize, .Inference

export Result, HyperParams
export linearprojection, fitmodel, extendfit
export build_transfer_model, train_transfer_model

# ------------------------------------------------------------------------
# globals

const square = Float32.([-1.0 -1.0 1.0 1.0; -1.0 1.0 1.0 -1.0])

# ------------------------------------------------------------------------
# types

"""
    mutable struct HyperParams
        dₒ :: Int
        Ws :: Array{Int,1}
        BN :: Array{Int,1}
        DO :: Array{Int,1}
        N  :: Int
        δ  :: Int
        η  :: Float64
        λ  :: Float64
        B  :: Int
        V  :: Int
        k  :: Int
        γₓ :: Float32
        γᵤ :: Float32
        g  :: Function
    end

HyperParams is a collection of parameters that specify the network architecture and training hyperparameters of the autoencoder.
`dₒ` is the desired output dimensionality of the encoding layer
`Ws` is a collection of the network layer widths. The number of entries controls the depth. The decoder is mirror-symmetric.
`BN` is the collection of layers that will be _followed_ by batch normalization.
`DO` is the collection of layers that will be _followed_ by dropout.
`N`  is the number of epochs to train against
`δ`  is the number of epochs that will be sampled for logging
`η`  is the learning rate
`λ`  is the weight decay factor
`B`  is the batch size
`V`  is the number of batches to partition for validation purposes, i.e. won't be training against
`k`  is the number of neighbors to use to estimate geodesics
`γₓ` is the prefactor of distance soft rank loss
`γᵤ` is the prefactor of uniform density loss
`g ` is the metric given to latent space
"""
mutable struct HyperParams
    dₒ :: Int          # output dimensionality
    Ws :: Array{Int,1} # (latent) layer widths
    BN :: Array{Int,1} # (latent) layers followed by batch normalization
    DO :: Array{Int,1} # (latent) layers followed by drop outs
    N  :: Int          # number of epochs to run
    δ  :: Int          # epoch subsample factor for logging
    η  :: Float64      # learning rate
    λ  :: Float64      # weight decay factor
    B  :: Int          # batch size
    V  :: Int          # number of points to partition for validation
    k  :: Int          # number of neighbors to use to estimate geodesics
    γₓ :: Float32      # prefactor of distance soft rank loss
    γᵤ :: Float32      # prefactor of uniform density loss
    g  :: Function     # metric given to latent space
    Ψ  :: Matrix{Float32}          # placeholder for future use #XXX Testing here
end

"""
    euclidean²(x)

Generate the matrix of pairwise distances between vectors `x`, assuming the Euclidean metric.
`x` is assumed to be ``d \times N`` where ``d`` denotes the dimensionality of the vector and ``N`` denotes the number.
"""
const euclidean²(x) = sum( (x[d,:]' .- x[d,:]).^2 for d in axes(x,1) )

"""
    cylinders²(x)

Generate the matrix of pairwise distances between vectors `x`, assuming the points are distributed on a cylinder.
`x` is assumed to be ``d \times N`` where ``d`` denotes the dimensionality of the vector and ``N`` denotes the number.
The first coordinate of `x` is assumed to be the polar coordinate.
"""
function cylinder²(x)
    c = cos.(π.*(x[1,:]))
    s = sin.(π.*(x[1,:]))

    return (c' .- c).^2 .+ (s' .- s).^2 .+ euclidean²(x[2:end,:])
end

HyperParams(; dₒ=2, Ws=Int[], BN=Int[], DO=Int[], N=200, δ=10, η=1e-3, λ = 0, B=64, V=1, k=12, γₓ=1, γᵤ=1e-1, g=euclidean², Ψ = zeros(Float32, 2, 2)) = HyperParams(dₒ, Ws, BN, DO, N, δ, η, λ, B, V, k, γₓ, γᵤ, g, Ψ)

"""
    struct Result
        param :: HyperParams
        loss  :: NamedTuple{(:train, :valid), Tuple{Array{Float64,1},Array{Float64,1}} }
        model
    end

Store the output of a trained autoencoder.
`param` stores the input hyperparameters used to design and train the neural network.
`loss` traces the dynamics of the optimization found during training.
`model` represents the learned pullback and pushforward functions.
"""
struct Result
    param :: HyperParams
    loss  :: NamedTuple{(:train, :valid), Tuple{Array{Float64,1},Array{Float64,1}} }
    info :: NamedTuple{(:𝕃ᵣ, :𝕃ₓ, :𝕃ᵤ, :history), Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Any}}
    model
end

# ------------------------------------------------------------------------
# utility functions

# α := rescale data by
# δ := subsample data by
function pointcloud(;α=1, δ=1)
    verts, _ = open("$root/gut/mesh_apical_stab_000153.ply") do io
        read_ply(io)
    end

    return α*vcat(
        map(v->v.x, verts)',
        map(v->v.y, verts)',
        map(v->v.z, verts)'
    )[:,1:δ:end]
end

function expression(;raw=false)
    scrna, genes, _  = if raw
        GZip.open("$root/dvex/dge_raw.txt.gz") do io
            read_matrix(io; named_cols=false, named_rows=true)
        end
    else
        GZip.open("$root/dvex/dge_normalized.txt.gz") do io
            read_matrix(io; named_cols=true, named_rows=true)
        end
    end

    return scrna, genes
end

# ------------------------------------------------------------------------
# i/o

# assumes a BSON i/o
function load(io::IO)
    database = BSON.parse(io)
    result, input = database[:result], database[:in]
end

mean(x) = length(x) > 0 ? sum(x) / length(x) : 0
mean(x::Matrix;dims=1) = sum(x;dims=dims) / size(x,dims)

function cor(x, y)
    μ = (
        x = mean(x),
        y = mean(y)
    )
    var = (
        x = mean(x.^2) .- μ.x^2,
        y = mean(y.^2) .- μ.y^2
    )
    return (mean(x .* y) .- μ.x .* μ.y) / sqrt(var.x * var.y)
end

function procrustes_align(Z, Y)
    # Align Z to Y using optimal rotation/scaling
    # Returns aligned Z
    Z_centered = Z .- Statistics.mean(Z, dims=2)
    Y_centered = Y .- Statistics.mean(Y, dims=2)

    H = Z_centered * Y_centered'
    U, _, V = svd(H)
    R = V * U'  # Optimal rotation
    
    # Scale
    s = tr(Y_centered' * R * Z_centered) / tr(Z_centered * Z_centered')
    
    return s * R * Z_centered .+ Statistics.mean(Y, dims=2)
end

function uniform_loss(z::AbstractMatrix{T}; n_queries::Int=1000, area::Real=4) where {T<:AbstractFloat}
    num_points = size(z, 2)

    σ = sqrt(T(area) / T(num_points))
    inv2σ2 = inv(T(2) * σ * σ)

    # queries in [-1, 1]^2 (same as before, but typed to T)
    q = T(2) .* rand(T, 2, n_queries) .- one(T)

    # ||q - z||^2 = ||q||^2 + ||z||^2 - 2 q'z
    q2 = sum(abs2, q; dims=1)          # 1 × n_queries
    z2 = sum(abs2, z; dims=1)          # 1 × num_points
    G  = transpose(q) * z              # n_queries × num_points (GEMM)

    d2 = q2' .+ z2 .- T(2) .* G        # n_queries × num_points

    ρ = vec(sum(exp.(-d2 .* inv2σ2), dims=2))  # n_queries
    return Statistics.var(log.(ρ .+ T(1e-8)))
end


"""
    buildloss(model, D², param)

Return a loss function used to train a neural network `model` according to input hyperparameters `param`.
`model` is a object with three fields, `pullback`, `pushforward`, and `identity`.
`pullback` and `pushforward` refers to the encoder and decoder layers respectively, while the identity is the composition.
`D²` is a matrix of pairwise distances that will be used as a quenched hyperparameter in the distance soft rank loss.
"""
function buildloss(model, D², param)
    # Define constants for the uniform density loss outside of loss function so that they are not recomputed
    # Put a flag here to determine if latent activation has custom actiavation layers
    if propertynames(model.pullback[end]) == (:linear, :activations)
        println("Detected custom activation layer in latent space...")
        Λ = 2*[model.pullback[end].activations[i](0) for i in 1:param.dₒ]
        boundary_points = Float32.(sort_points_cw(corners_and_edges(Λ)))
        latent_area = Float32.(prod(Λ))
    end

    ϕ = collect(0:0.01:π) .- 0.001
    ICDF_Splines = approximate_functions(ϕ, 0:0.01:1)
    Nₛ = 20
    I = rand(1:length(ϕ), Nₛ)


    return function(x, i::T, output::Bool) where T <: AbstractArray{Int,1}
        z = model.pullback(x)
        y = model.pushforward(z)

        # reconstruction loss
        ϵᵣ = sum(sum((x.-y).^2, dims=2)) / sum(sum(x.^2,dims=2))

        # distance softranks
        Dz² = param.g(z)
        Dx² = D²[i,i]

        if param.γₓ == 0
            ϵₓ = 0
        else
            ϵₓ = 1 - mean([
                cor(
                    (softrank(cx ./ mean(cx))),
                    (softrank(cz ./ mean(cz)))
                )
                for (cx,cz) in zip(eachcol(Dx²),eachcol(Dz²))
            ])
        end
        
        if param.γᵤ == 0
            ϵᵤ = 0
        else
         
            # XXX: Central Force Repulsion
            # ϵᵤ = let
            #     N = size(z, 2)
            #     A = 1.0
            #     d₀² = 0.25 * A / N
            #     Dz² = SeqSpace.upper_tri(Dz²) .+ 1e-8
                
            #     harmonic = (Dz² .- d₀²).^2
            #     repel = d₀² ./ Dz²  # blows up as Dz² → 0
                
            #     mean(harmonic .+ repel)
            # end

            ϵᵤ = let
                    N = size(z, 2)
                    sum(sum(z.^2; dims=1)) / N
                end

        end

        return (ϵᵣ,ϵₓ,ϵᵤ)
    end
end


# ------------------------------------------------------------------------
# main functions

"""
    linearprojection(x, d; Δ=1, Λ=nothing)

Project an empirical distance matrix `x` onto `d` top principal components.
Centers the result to have zero mean.
Returns the projection, as well as a function to transform a projected vector back to the embedding space.
Ignores top `Δ` principal components.
If `Λ` is not nothing, assumes it is a precomputed SVD decomposition.
"""
function linearprojection(x, d; Δ=1, Λ=nothing)
    Λ = isnothing(Λ) ? svd(x) : Λ

    ι = (1:d) .+ Δ
    ψ = Diagonal(Λ.S[ι])*Λ.Vt[ι,:]
    μ = Statistics.mean(ψ, dims=2)

    x₀ = (Δ > 0) ? Λ.U[:,1:Δ]*Diagonal(Λ.S[1:Δ])*Λ.Vt[1:Δ,:] : 0

    embed(x)   = (x₀ .+ (Λ.U[:,ι]*(x.+μ)))
    embed(x,i) = (x₀ .+ (Λ.U[:,ι]*(x.+μ[i])))

    return (
        embed      = embed,
        projection = (ψ .- μ),
    )
end

"""
    fitmodel(data, param; D²=nothing, dev=true, interior_activation=elu, exterior_activation=tanh_fast)

Train an autoencoder model, specified with `param` hyperparams, to fit `data`.
`data` is assumed to be sized ``d \times N`` where ``d`` and ``N`` are dimensionality and cardinality respectively.
If not nothing, `D²` is assumed to be a precomputed distance matrix of point cloud `data`.
If `dev` is true, function will print to `stdout` and values of compoents of the loss (ϵₓ,ϵᵣ,ϵᵤ) will be recorded.
Returns a `Result` type.
"""
function fitmodel(
        data,
        param;
        D²=nothing,
        dev=false,
        interior_activation=celu,
        exterior_activation=celu,
        initial_activation=celu,
    )
    D² = isnothing(D²) ? geodesics(data, param.k).^2 : D²
    if dev
        println(stderr, "done computing geodesics...")
    end

    M = model(size(data,1), param.dₒ;
          Ws         = param.Ws,
          normalizes = param.BN,
          dropouts   = param.DO,
          initial_activation = initial_activation,
          interior_activation = interior_activation,
          exterior_activation = exterior_activation,
    )


    nvalid = size(data,2) - ((size(data,2)÷param.B)-param.V)*param.B
    batch, index = validate(data, nvalid)

    loss_peices = buildloss(M, D², param)
    loss = (args...) -> dot((1, param.γₓ, param.γᵤ), loss_peices(args...))

    E    = (
        train = Float64[],
        valid = Float64[],
    )
    Info = (
        𝕃ᵣ = Float64[],
        𝕃ₓ = Float64[],
        𝕃ᵤ = Float64[],
        history = [] #TODO figure out the data struct of history and add it here 
    )
    progress = Progress(Int(round(param.N/10)); desc=">training model", output = stdout)
    log = (n) -> begin
        if (n-1) % param.δ == 0
            push!(E.train, loss(batch.train, index.train, false))
            push!(E.valid, loss(batch.valid, index.valid, false))

            if dev
                Εᵣ, Εₓ, Εᵤ = loss_peices(batch.train, index.train, false)
                push!(Info.𝕃ᵣ, Εᵣ)
                push!(Info.𝕃ₓ, Εₓ)
                push!(Info.𝕃ᵤ, Εᵤ)
            end
        end

        if (n-1) % 10 == 0
            next!(progress)
        end
        
        if dev
            push!(Info.history, M.pullback(data))
        end
        nothing
    end

    Flux.trainmode!(M)
    train!(M, batch.train, index.train, loss;
        η   = param.η,
        λ   = param.λ,
        B   = param.B,
        N   = param.N,
        log = log
    )
    Flux.testmode!(M)
    
    return Result(param, E, Info, M),
        ( # should probably be a datatype
            batch   = batch,
            index   = index,
            D²      = D²,
            # log     = log, # This makes BSON serialization fail becuase it's an anonymous function
            initial_activation  = initial_activation,
            interior_activation = interior_activation,
            exterior_activation = exterior_activation
        )
end

"""
    build_transfer_model(model, new_IO_dim)

Build a transfer model from an existing model `model` with a new input/output dimensionality `new_IO_dim`.
The transfer model will have the same architecture as `model` except for the first and last layers, which will be replaced with new `Dense` layers that have the specified input/output dimensionality.
Additionally, the transfer model will not have any dropout layers in the encoder and decoder.
Returns a tuple containing the new model with pullback, pushforward, identity, and first_and_last layers.
"""

function build_transfer_model(model, new_IO_dim)
    F, F¯¹, 𝕀 = deepcopy(model)
    ιᵢ = size(F[1].weight,1)
    ιₒ = size(F¯¹[end].weight,2)

    lᵢ = Dense(new_IO_dim, ιᵢ, F[1].σ)
    lₒ = Dense(ιₒ, new_IO_dim, F¯¹[1].σ)

    # We remove dropout layers from the encoder and decoder
    encoder = Chain(lᵢ, [layer for layer in F[2:end] if !(typeof(layer) <: Dropout)]...)
    # encoder = Chain(lᵢ, F[2:end]...)
    decoder = Chain([layer for layer in F¯¹[1:end-1] if !(typeof(layer) <: Dropout)]..., lₒ)
    # decoder = Chain(F¯¹[1:end-1]..., lₒ)
    𝕀 = Chain(encoder, decoder)
    transfer_layers = (lᵢ,lₒ)

    return (
        pullback = encoder,
        pushforward = decoder,
        identity = 𝕀
    )
end

"""
    train_transfer_model(result, data, Epochs; D²=nothing, new_hyperparams=nothing, dev=false)

Train a transfer model based on the `result` of a previous training run.
`data` is the new data to train against.
`Epochs` is the number of epochs to train for.
If `D²` is not provided, it will compute the geodesics from `data`.
If `new_hyperparams` is provided, it will use those hyperparameters instead of the ones from `result`.
If `dev` is true, it will record the pullback mapping of `data` each epoch.
Returns a `Result` type with the trained model.
"""

function train_transfer_model(result, data, Epochs; D² = nothing, new_hyperparams = nothing, schedule = nothing, dev = false)
    # Build the transfer model and initialize history
    param = isnothing(new_hyperparams) ? result.param : new_hyperparams
    model = build_transfer_model(result.model, size(data,1))

        println("Completed building transfer model")

    train_hist = copy(result.loss.train)
    valid_hist = copy(result.loss.valid)
    info_hist  = deepcopy(result.info)

    # Batch the new data:
    nvalid = size(data,2) - ((size(data,2)÷param.B)-param.V)*param.B
    batch, index = validate(data, nvalid)

    # Create loss function with new geodesics
    D² = isnothing(D²) ? geodesics(data, param.k).^2 : D²
    loss_peices = buildloss(model, D², param)
    loss = (args...) -> dot((1, param.γₓ, param.γᵤ), loss_peices(args...))

    # Initialize progress bar and logging function
    progress = Progress(Int(round(0)); desc=">training model", output = stdout)
    log = (n) -> begin
        if (n-1) % param.δ == 0
            push!(train_hist, loss(batch.train, index.train, false))
            push!(valid_hist, loss(batch.valid, index.valid, false))
            if dev
                push!(info_hist.𝕃ᵣ, loss_peices(batch.train, index.train, false)[1])
                push!(info_hist.𝕃ₓ, loss_peices(batch.train, index.train, false)[2])
                push!(info_hist.𝕃ᵤ, loss_peices(batch.train, index.train, false)[3])
            end
        end

        if (n-1) % 10 == 0
            next!(progress)
        end

        # if dev
        #     push!(info_hist.history, model.pullback(data))
        # end
        nothing
    end

    schedule = isnothing(schedule) ? [Epochs] : schedule
    println("Training schedule: ", schedule)
    
    Flux.trainmode!(model)
    for (i, epochs) ∈ enumerate(schedule)
        progress = Progress(Int(round(epochs/10)); desc=">training model", output = stdout)
        train!(model, batch.train, index.train, loss;
            layers_to_train = (model.pullback[1:i], model.pushforward[end-(i-1):end]),
            η   = param.η,
            B   = param.B,
            λ   = param.λ,
            N   = epochs,
            log = log,
        )
        println("Completed training for epoch group ", i)
    end

    Flux.testmode!(model)

    return Result(param, (train_hist, valid_hist), info_hist, model), (batch = batch, index = index)
end

function test_revise()
    return "Working 2/27"
end

end
