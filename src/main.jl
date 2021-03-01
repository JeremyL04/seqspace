module SeqSpace

using GZip
using BSON: @save

include("geo.jl")
include("io.jl")
include("model.jl")

using .PointCloud, .DataIO, .ML

export Result, HyperParams

# ------------------------------------------------------------------------
# globals

const root = "$(homedir())/root/data/seqspace"

# ------------------------------------------------------------------------
# types

struct HyperParams
    dₒ :: Int          # output dimensionality
    Ws :: Array{Int,1} # network layer widths
    BN :: Array{Int,1} # (latent) layers followed by batch normalization
    DO :: Array{Int,1} # (latent) layers followed by drop outs
    N  :: Int          # number of epochs to run
    δ  :: Int          # epoch subsample factor for logging
    η  :: Float64      # learning rate
    B  :: Int          # batch size
    V  :: Int          # number of points to partition for validation
    kₙ :: Int          # number of neighbors to use to estimate geodesics
    kₗ :: Int          # average number of neighbors to impose isometry on 
    γ  :: Float64      # prefactor of neighborhood isometry loss
end

HyperParams(; dₒ=2, Ws=Int[], BN=Int[], DO=Int[], N=1000, δ=5, η=1e-3, B=64, V=128, kₙ=12, kₗ=4, γ=.01) = HyperParams(dₒ, Ws, BN, DO, N, δ, η, B, V, kₙ, kₗ, γ)

struct Result
    param :: HyperParams
    loss  :: NamedTuple{(:train, :valid), Tuple{Array{Float64,1},Array{Float64,1}} }
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

function expression()
    scrna, genes, _ = GZip.open("$root/dvex/dge_normalized.txt.gz") do io
        read_matrix(io; named_cols=true, named_rows=true)
    end

    @size scrna

    return scrna, genesk
end

mean(x)    = sum(x) / length(x)
ball(D, k) = mean(sort(view(D,:,i))[k+1] for i in 1:size(D,2))

# ------------------------------------------------------------------------
# main functions

function run(params::Array{HyperParams,1}, niter::Int)
    ptcloud = embed(pointcloud(; α=1/500, δ=10), 50; σ = .05)
    x⃗, ω, ϕ = preprocess(ptcloud)

    results = Array{Result}(undef, length(params)*niter)
    for (iₚ, p) in enumerate(params)
        for iᵢₜ in 1:niter
            M = model(size(x⃗, 1), p.dₒ; Ws = p.Ws, normalizes = p.BN, dropouts = p.DO)
            y⃗, I = validate(x⃗, p.V)

            D² = geodesics(ϕ(x⃗), p.kₙ).^2

            loss = (x, i) -> begin
                z = M.pullback(x)
                x̂ = M.pushforward(z)

                # reconstruction
                ϵᵣ = sum(sum((x.-x̂).^2, dims=2).*ω)/size(x,2)

                # neighborhood isometry
                D̂² = distance²(z)

                R = ball(D²[i,i], p.kₗ)
                d = upper_tri(D²[i,i])
                d̂ = upper_tri(D̂²)

                n = d .<= R
                n̂ = d̂ .<= R

                ϵₓ = .5*(sqrt(mean( (d[n] .- d̂[n]).^2 )) 
                       + sqrt(mean( (d̂[n̂] .- d[n̂]).^2 )))

                return ϵᵣ + p.γ*ϵₓ
            end

            E = (
                train = zeros(p.N÷p.δ),
                valid = zeros(p.N÷p.δ),
            )
            log  = (n, loss, model) -> begin
                if (n-1) % p.δ == 0 
                    @show n

                    E.train[(n-1)÷p.δ+1] = loss(y⃗.train, I.train)
                    E.valid[(n-1)÷p.δ+1] = loss(y⃗.valid, I.valid)
                end

                nothing
            end

            train!(M, y⃗.train, loss; η=p.η, B = p.B, N = p.N, log = log)

            results[niter*(iₚ-1)+iᵢₜ] = Result(p, E, M)
        end
    end

    results
end

function main(input, niter, output)
    params = Result[]
    open(input, "r") do io 
        params = eval(Meta.parse(read(io, String)))
    end
    
    result = run(params, niter)
    @save "$root/result/$output.bson" result
end

end