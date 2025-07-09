module SeqSpace

using GZip
using BSON: @save
using LinearAlgebra: norm, svd, Diagonal, dot, eigvals
using Statistics: quantile, std
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
export marshal, unmarshal
export version_info

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
`B`  is the batch size
`V`  is the number of points to partition for validation purposes, i.e. won't be training against
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
    B  :: Int          # batch size
    V  :: Int          # number of points to partition for validation
    k  :: Int          # number of neighbors to use to estimate geodesics
    γₓ :: Float32      # prefactor of distance soft rank loss
    γᵤ :: Float32      # prefactor of uniform density loss
    g  :: Function     # metric given to latent space
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

HyperParams(; dₒ=2, Ws=Int[], BN=Int[], DO=Int[], N=200, δ=10, η=1e-3, B=64, V=1, k=12, γₓ=1, γᵤ=1e-1, g=euclidean²) = HyperParams(dₒ, Ws, BN, DO, N, δ, η, B, V, k, γₓ, γᵤ, g)

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

"""
    marshal(r::Result)

Serialize a trained autoencoder to binary format suitable for disk storage.
Store parameters of model as contiguous array
"""
function marshal(TrainingOutput::Tuple{Result,Any}) 
    r = TrainingOutput[1]
    output = TrainingOutput[2]
    # trainable parameters
    ω₁ = r.model.pullback    |> cpu |> Flux.params |> collect
    ω₂ = r.model.pushforward |> cpu |> Flux.params |> collect

    # untrainables
    β₁ = [ (μ=layer.μ,σ²=layer.σ²) for layer in r.model.pullback.layers    if isa(layer,Flux.BatchNorm) ]
    β₂ = [ (μ=layer.μ,σ²=layer.σ²) for layer in r.model.pushforward.layers if isa(layer,Flux.BatchNorm) ]

    return Result(r.param,r.loss,r.info,
                    (
                        pullback=(
                            params=ω₁,
                            batchs=β₁,
                        ),
                        pushforward=(
                            params=ω₂,
                            batchs=β₂,
                        ),
                        size=size(r.model.pullback.layers[1].weight,2)
                    )
                ),
                (
                batch=output.batch,
                index=output.index,
                D²=output.D²,
                log=output.log,
                initial_activation=output.initial_activation,
                interior_activation=output.interior_activation,
                exterior_activation=output.exterior_activation
                )
end

"""
    unmarshal(r::Result)

Deserialize a trained autoencoder from binary format to semantic format.
Represents model as a collection of functors.
"""
function unmarshal(MarshaledModel::Tuple{Result,Any}) # This needs to be re-worked with new data structures
    r = MarshaledModel[1]
    output = MarshaledModel[2]
    autoencoder = model(r.model.size, r.param.dₒ;
          Ws         = r.param.Ws,
          normalizes = r.param.BN,
          dropouts   = r.param.DO,
          initial_activation = output.initial_activation,
          interior_activation = output.interior_activation,
          exterior_activation = output.exterior_activation
    )

    Flux.loadparams!(autoencoder.pullback, r.model.pullback.params)
    Flux.loadparams!(autoencoder.pushforward, r.model.pushforward.params)
    Flux.trainmode!(autoencoder.identity, false)

    i = 1
    for layer in autoencoder.pullback.layers
        if isa(layer, Flux.BatchNorm)
            layer.μ  = r.model.pullback.batchs[i].μ
            layer.σ² = r.model.pullback.batchs[i].σ²
            i += 1
        end
    end

    i = 1
    for layer in autoencoder.pushforward.layers
        if isa(layer, Flux.BatchNorm)
            layer.μ  = r.model.pushforward.batchs[i].μ
            layer.σ² = r.model.pushforward.batchs[i].σ²
            i += 1
        end
    end

    param = HyperParams(;
        dₒ = r.param.dₒ,
        Ws = r.param.Ws,
        BN = r.param.BN,
        DO = r.param.DO,
        N  = r.param.N,
        δ  = r.param.δ,
        η  = r.param.η,
        B  = r.param.B,
        V  = r.param.V,
        k  = r.param.k,
        γₓ = r.param.γₓ,
        γᵤ = r.param.γᵤ,
    )

    return Result(param, r.loss, r.info, autoencoder),(
        batch=output.batch,
        index=output.index,
        D²=output.D²,
        log=output.log,
        initial_activation=output.initial_activation,
        interior_activation=output.interior_activation,
        exterior_activation=output.exterior_activation
    )
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
        x=mean(x),
        y=mean(y)
    )
    var = (
       x=mean(x.^2) .- μ.x^2,
       y=mean(y.^2) .- μ.y^2
    )

    return (mean(x.*y) .- μ.x.*μ.y) / sqrt(var.x*var.y)
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
        Λ = 2*[model.pullback[end].activations[i](0) for i in 1:param.dₒ]
        boundary_points = Float32.(sort_points_cw(corners_and_edges(Λ)))
        latent_area = Float32.(prod(Λ))
    end

    # ϕ = collect(0:0.01:π) .- 0.001
    # ICDF_Splines = approximate_functions(ϕ, 0:0.01:1)
    # Nₛ = 9


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
            # Voronoi/Dulaney

            # # True Voronoi
            # ϵᵤ = let
            #     N = size(z,2)
            #     sum(abs.(voronoi_areas(z, boundary_points) .- (latent_area / Float32(N))))
            # end

            # # Dulaney Triangles
            # ϵᵤ = let
            #     A = areas(z, boundary_points)
            #     N_triangles = size(A,1)
            #     sum(abs.(A .- (latent_area/N_triangles)))
            # end

            # # Radon Slicing
            # ϵᵤ = let
            #     I = rand(1:length(ϕ), Nₛ)
            #     Θ, ICDFs = ϕ[I], ICDF_Splines[I]
            #     z̃ = z
            #     # z̃ = ((2 * [1/Λ[1] 0; 0 1/Λ[2]]) * z) .- 1
            #     ψₚ = [[dot(point, [cos(θ), sin(θ)]) for point ∈ eachcol(z̃)] for θ ∈ Θ]
            #     Ranks = [softrank(ψ) for ψ ∈ ψₚ]
            #     Y = [(r .- minimum(r)) ./ (1 .- minimum(r)) for r ∈ Ranks]
            #     InvCDFs = [ICDFs[i].(Y[i]) for i ∈ eachindex(Θ)]
            #     mean(mean.((F⁻¹ .- ψ).^4 for (F⁻¹, ψ) in zip(InvCDFs, ψₚ)))
            # end

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
    μ = mean(ψ, dims=2)

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
    dev=true,
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
            # print("\r ϵᵣ,ϵₓ,ϵᵤ = $(loss_peices(batch.train, index.train, false)); Epoch: $n")

            # if dev
            #     push!(Info.𝕃ᵣ, data_loss(batch.train, index.train, false)[1])
            #     push!(Info.𝕃ₓ, data_loss(batch.train, index.train, false)[2])
            #     push!(Info.𝕃ᵤ, data_loss(batch.train, index.train, false)[3])
            # end
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
        B   = param.B,
        N   = param.N,
        log = log
    )
    Flux.testmode!(M)
    
    return Result(param, E, Info, M), (
        batch=batch,
        index=index,
        D²=D²,
        log=log,
        initial_activation = initial_activation,
        interior_activation = interior_activation,
        exterior_activation = exterior_activation
    )
end

"""
    extendfit(result::Result, input, epochs)

Retrain model within `result` on `input` data for `epochs` more iterations.
Returns a new `Result`.
"""
function extendfit(result::Result, input, new_params; dev = false, data = nothing)
    loss_peices = buildloss(result.model, input.D², new_params)
    loss = (args...) -> dot((1, new_params.γₓ, new_params.γᵤ), loss_peices(args...))
    data_loss = buildloss(result.model, input.D², new_params)

    progress = Progress(Int(round(new_params.N/10)); desc=">training model", output = stdout)
    log = (n) -> begin
        if (n-1) % new_params.δ == 0
            push!(result.loss.train, loss(input.batch.train, input.index.train, false))
            push!(result.loss.valid, loss(input.batch.valid, input.index.valid, false))

            push!(result.info.𝕃ᵣ, data_loss(input.batch.train, input.index.train, false)[1])
            push!(result.info.𝕃ₓ, data_loss(input.batch.train, input.index.train, false)[2])
            push!(result.info.𝕃ᵤ, data_loss(input.batch.train, input.index.train, false)[3])
        end

        if (n-1) % 10 == 0
            next!(progress)
        end
        
        if dev
            push!(result.info.history, result.model.pullback(data))
        end
        nothing
    end

    Flux.trainmode!(result.model.identity, true)
    train!(result.model, input.batch.train, input.index.train, loss; 
        η   = result.param.η,
        B   = result.param.B,
        N   = new_params.N,
        log = log
    )
    Flux.trainmode!(result.model.identity, false)


    return Result(new_params, result.loss, result.info, result.model), input
end

function test_Revise_main(x)
    return println("Testing 1.0")
end


end
