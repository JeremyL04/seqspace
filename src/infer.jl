module Inference

using GZip, JLD2, FileIO
using LinearAlgebra, Statistics, StatsBase

include("io.jl")
using .DataIO

include("mixtures.jl")
using .Mixtures

export inversion, virtualembryo

# ------------------------------------------------------------------------
# globals

Maybe{T} = Union{T, Missing}
rank(x) = invperm(sortperm(x))

# ------------------------------------------------------------------------
# helper functions

function cumulative(data)
    d = sort(data)
    function F(x)
        i = searchsortedfirst(d, x)
        return (i-1)/length(d)
    end

    return F
end

const columns(genes) = Dict(g=>i for (i,g) ∈ enumerate(genes))
const badgenes = Set{String}(["CG14427", "cenG1A", "CG13333", "CG31670", "CG8965", "HLHm5", "Traf1"])

function cohortdatabase(stage::Int)
    cohort   = load("$root/drosophila/bdntp/database.jld2", "cohort")
    keepgene = [g ∉ badgenes for g in cohort[stage].gene]

    left = cohort[stage].point[:,2] .≥ 0
    gene = cohort[stage].gene[keepgene]
    return (
        expression = (
            real = cohort[stage].data[left,findall(keepgene)],
            data = hcat((rank(col)/length(col) for col in eachcol(cohort[stage].data[left,findall(keepgene)]))...),
            gene = columns(gene),
        ),
        position = cohort[stage].point[left,:],
        gene = gene
    )
end

"""
    virtualembryo(;directory="")

Load the Berkeley Drosophila Transcriptional Network Project database.
`directory` should be path to folder containing two folders:
  1. bdtnp.txt.gz    : gene expression over point cloud of virtual cells
  2. geometry.txt.gz : spatial position (x,y,z) of point cloud of virtual cells.
"""
function virtualembryo(;directory="/Users/jeremy/Desktop/Research/Positional Information/Drosophila_FISH_Data")
    expression, _, genes = GZip.open("$directory/bdtnp.txt.gz") do io
        read_matrix(io; named_cols=true)
    end

    expression = collect(expression')
    size(expression,1) == 84 || error("Expected 84 BDTNP genes genes")

    positions, _, _, = GZip.open("$directory/geometry_reduced.txt.gz") do io
        read_matrix(io; named_cols=true)
    end

    badgenes = ["CG14427", "cenG1A", "CG13333", "CG31670", "CG8965", "HLHm5", "Traf1"]
    mask = .!([g in badgenes for g in genes])
    expression = expression[mask, :]
    genes      = genes[mask]

    return (
        expression = (
            real = expression,
            data = hcat(fitmixture.(eachcol(expression))...), # TODO: Figure out what this does
            gene = genes,
        ),
        position  = positions,
    )
end



function scrna()
    expression, genes, _ = GZip.open("$root/drosophila/dvex/dge_normalized.txt.gz") do io
        read_matrix(io; named_cols=true, named_rows=true)
    end

    return (
        data = expression',
        gene = columns(genes),
    )
end

function match(x,y)
    dict = Dict(name => i for (i,name) in enumerate(y))
    return [ get(dict, gene, nothing) for gene in x ]
end

# function match(x, y)
#     index = Array{String}(undef,length(x))
#     for (g,i) in x
#         index[i] = g
#     end
#     return [k ∈ keys(y) ? y[k] : nothing for k in index]
# end

# ------------------------------------------------------------------------
# main functions

"""
    cost(ref, qry; α=1, β=1, γ=0, ω=nothing)

Return the cost matrix ``J_{i\alpha}`` associated to matching cells in `qry` to cells in `ref`.
The cost matrix is computed by a heuristic distance between quantiles.
Deprecated.
"""
function cost(ref, qry; α=1, β=1, γ=0, ω=nothing)
    ϕ = match(ref.gene, qry.gene)

    Σ = zeros(size(ref.data,1), size(qry.data,1))
    for i in 1:size(ref.data,2)
        isnothing(ϕ[i]) && continue

        r  = ref.data[:,i]
        q  = qry.data[:,ϕ[i]]
        ω₀ = isnothing(ω) ? 1 : ω[i]

        f = sum(q .== 0) / length(q)
        χ = quantile(r, f)

        F₀ = cumulative(r[r.≤χ])
        F₊ = cumulative(r[r.>χ])
        F₌ = cumulative(q[q.>0])

        for j in 1:size(ref.data,1)
            for k in 1:size(qry.data,1)
                if r[j] > χ && q[k] > 0
                    Σ[j,k] += -ω₀*(2*F₊(r[j])-1)*(2*F₌(q[k])-1)
                elseif r[j] > χ && q[k] == 0
                    Σ[j,k] += +ω₀*(α*F₊(r[j])+γ)
                elseif r[j] <= χ && q[k] > 0
                    Σ[j,k] += +ω₀*(α*F₌(q[k])+γ)
                else
                    Σ[j,k] += +ω₀*β*F₀(r[j])
                end
            end
        end
    end

    return Matrix(Σ), ϕ
end

"""
    cost_simple(ref, qry)

Return the cost matrix ``J_{i\\alpha}`` associated to matching cells in `qry` to cells in `ref`.
The cost matrix is computed by hamming distance between cells via transforming quantiles to continuous spin variables.
Deprecated.
"""
function cost_simple(ref, qry)
    ϕ = match(ref.gene, qry.gene)
    Σ = zeros(size(ref.data,1), size(qry.data,1))

    σ(x) = x
    for i in 1:size(ref.data,2)
        isnothing(ϕ[i]) && continue

        r  = ref.real[:,i]
        q  = qry.data[:,ϕ[i]]

        R = 2*σ.((rank(r)./length(r))) .- 1
        Q = 2*σ.((rank(q)./length(q))) .- 1

        Σ -= (R*Q')

        #=
        for j in 1:size(ref.data,1)
            for k in 1:size(qry.data,1)
                Σ[j,k] += -*(2*σ(R(r[j]).^4)-1)*(2*σ(Q(q[k]).^4)-1)
            end
        end
        =#
    end

    return Matrix(Σ), ϕ
end

function cost_scan(ref, qry, ν, ω)
    ϕ = match(ref.gene, qry.gene)
    Σ = zeros(size(ref.data,1), size(qry.data,1))

    # XXX: try out the other option???
    # k    = 1
    # σ(x) = 1/(1+((1-x)/x)^k)
    σ(x) = x

    for i in 1:size(ref.data,2)
        isnothing(ϕ[i]) && continue

        r = ref.real[:,i]
        q = qry.data[:,ϕ[i]]

        # χ = σ.(rank(q)./(length(q)).^ν[i])

        R = (2 .* rank(r)./length(r)) .- 1
        Q = (2 .* σ.((rank(q)./length(q)).^ν[i])) .- 1
        # Q = (2 .* χ) .- 1

        Σ -= ω[i]*(R*Q')
    end

    return Matrix(Σ), ϕ
end

"""
    transform(src, dst, ν)

Transform distribution `src` to distribution `dst` by minimizing the Wasserstein metric.
This is equivalent to ``x \\to F^{-1}_{dst}\\left(F_{src}\\left(x\\right)\\right)`` where ``F`` denotes the cumulative density function.
"""
function transform(src, dst, ν)
    ref = sort(dst)
    pos = collect(1:length(dst))/length(dst)

    σ(x) = 1/(1+((1-x)/x)^ν)
    qry  = σ.(rank(src) / length(src))

    return [
        let
            i = searchsorted(pos, q)
            if first(i) == last(i)
                ref[first(i)]
            elseif last(i) == 0
                ref[1]
            else
                @assert first(i) > last(i)
                δy = ref[first(i)] - ref[last(i)]
                δx = pos[first(i)] - pos[last(i)]
                δq = q - pos[last(i)]

                ref[last(i)] + (δy/δx)*δq
            end
        end for q in qry
    ]
end

"""
    cost_transform(ref, qry; ω=nothing, ν=nothing)

Return the cost matrix ``J_{i\\alpha}`` associated to matching cells in `qry` to cells in `ref`. Assumes `ref` and `qry` are genes \\cross cells.
The cost matrix is computed by:
  1. Transforming the `qry` distribution to the `ref` distribution.
  2. Looking at the SSE across transformed genes.
Use this unless you know what you are doing.
"""
function cost_transform(ref, qry; ω=nothing, ν=nothing)
    ϕ = match(ref.gene,qry.gene)
    Σ = zeros(size(ref.data,2), size(qry.data,2))

    ω = isnothing(ω) ? ones(size(ref.real,1)) : ω
    ω = ω / sum(ω)

    ν = isnothing(ν) ? ones(size(ref.real,1)) : ν

    for i in 1:size(ref.data,1)
        isnothing(ϕ[i]) && continue

        r = ref.real[i,:]
        q = transform(qry.data[ϕ[i],:], r, ν[i])

        Σ += ω[i]*(reshape(r, length(r), 1) .- reshape(q, 1, length(q))).^2
    end

    return Matrix(Σ), ϕ
end

"""
    sinkhorn(M::Array{Float64,2};
                  a::Maybe{Array{Float64}} = missing,
                  b::Maybe{Array{Float64}} = missing,
                  maxᵢ::Integer            = 1000,
                  τ::Real                  = 1e-5,
                  verbose::Bool            = false
    )

Rescale matrix `M` to have row & column marginals `a` and `b` respectively.
Will terminate either when constraints are held to within tolerance `τ` or the number of iterations exceed `maxᵢ`.
"""
function sinkhorn(M::Array{Float64,2};
                  a::Maybe{Array{Float64}} = missing,
                  b::Maybe{Array{Float64}} = missing,
                  maxᵢ::Integer            = 1000,
                  τ::Real                  = 1e-5,
                  verbose::Bool            = false)
    c = 1 ./sum(M, dims=1)
    r = 1 ./(M*c')

    if ismissing(a)
        a = ones(size(M,1), 1) ./ size(M,1)
    end
    if ismissing(b)
        b = ones(size(M,2), 1) ./ size(M,2)
    end

    if length(a) != size(M,1)
        throw(error("invalid size for row prior"))
    end
    if length(b) != size(M,2)
        throw(error("invalid size for column prior"))
    end

    i = 0
    rdel, cdel = Inf, Inf
    while i < maxᵢ && (rdel > τ || cdel > τ)
        i += 1

        cinv = M'*r
        cdel = maximum(abs.(cinv.*c .- b))
        c    = b./cinv

        rinv = M*c
        rdel = maximum(abs.(rinv.*r .- a))
        r    = a./rinv

        if verbose
            println("Iteration $i. Row = $rdel, Col = $cdel")
        end

    end

    if verbose
        println("Terminating at iteration $i. Row = $rdel, Col = $cdel")
    end

    return M.*(r*c')
end

function inversion()
    ref, pointcloud = virtualembryo()
    qry = scrna()

    Σ, _ = cost(ref, qry; α=1.0, β=2.6, γ=0.65) # TODO: expose parameters?

    return (
        invert = (β) -> sinkhorn(exp.(-(1 .+ β*Σ))),
        cost = Σ,
        pointcloud = pointcloud,
    )
end

"""
    inversion(counts, genes; ν=nothing, ω=nothing, refdb=nothing)

Infer the original position of scRNAseq data `counts` where genes are arranged along rows.
The sampling probability over space is computed by regularized optimal transport by comparing to the Berkeley Drosophila Transcription Network Project database.
The cost matrix is determined by summing over the 1D Wasserstein metric over all genes within the BDTNP databse.
Returns the inversion as a function of inverse temperature.
"""
function inversion(counts, genes; ν=nothing, ω=nothing, refdb=nothing)
    ref, pointcloud = refdb === nothing ? virtualembryo() : refdb
    qry = (
        data = counts,
        gene = genes,
    )

    Σ, ϕ =
        if isnothing(ν) || isnothing(ω)
            cost_transform(ref, qry)
        else
            cost_transform(ref, qry; ω=ω, ν=ν)
        end

    return (
        invert     = (β) -> sinkhorn(exp.(-(1 .+ β*Σ))),
        pointcloud = pointcloud,
        database   = (
            data=ref.data,
            gene=ref.gene,
        ),
        match      = match,
        index      = ϕ,
        cost       = Σ,
    )
end

# basic statistics
mean(x)  = sum(x) / length(x)
cov(x,y) = mean(x.*y) .- mean(x)*mean(y)
var(x)   = cov(x,x)
std(x)   = sqrt(abs(var(x)))
cor(x,y) = cov(x,y) / (std(x) * std(y))

"""
    make_objective(ref, qry)

Create an objective function for optimizing the hyperparameters of the projection matrix.
"""
function make_objective(ref, qry)
    function objective(Θ)
        β = 200 # Choice here is arbitary

        N_genes = length(ref.gene)
        ν, ω = Θ[1:N_genes], Θ[N_genes+1:end]
        Σ, ϕ = cost_transform(ref, qry; ω=ω, ν=ν)

        ψ  = sinkhorn(exp.(-(1 .+ β*Σ)))
        ψ *= minimum(size(ψ))

        ι   = findall(.!isnothing.(ϕ))
        db  = ref.real[ι,:]
        est = qry.data[ϕ[ι],:]*ψ'

        return 1-mean(cor(db[i,:], est[i,:]) for i in 1:size(db,1))
    end

    return objective
end

using BlackBoxOptim

function scan_params(qry)
    db = cohortdatabase(6)
    f  = make_objective(db.expression,qry)

    return bboptimize(f, 
                  SearchRange=[(0.1, 10.0) for _ ∈ 1:79],
                  MaxFuncEvals=5000,
                  Method=:generating_set_search,
                  TraceMode=:compact
    )
end

function scan_params(count, genes)
    qry = (
        data = count',
        gene = columns(genes),
    )
    ref, _ = virtualembryo()

    f = make_objective(ref,qry)

    return bboptimize(f,
                  SearchRange=[[(0.01, 10.0) for _ ∈ 1:84]; [(0.1, 2.0) for _ ∈ 1:84]],
                  # SearchRange=[(0.01, 10.0) for _ ∈ 1:(2*84)],
                  MaxFuncEvals=5000,
                  # Method=:adaptive_de_rand_1_bin_radiuslimited,
                  Method=:generating_set_search,
                  TraceMode=:compact
    ) #, Method=:dxnes, NThreads=Threads.nthreads(), )
end

"""
    find_params(ref, qry)

Find the optimal parameters for the cost matrix between `ref` and `qry`.
The parameters are optimized by minimizing the correlation between the estimated and reference distributions.
"""
function find_params(ref, qry)
    f = make_objective(ref,qry)

    return bboptimize(f,
                  SearchRange=[[(0.01, 10.0) for _ ∈ 1:83]; [(0.1, 2.0) for _ ∈ 1:83]],
                  # SearchRange=[(0.01, 10.0) for _ ∈ 1:(2*84)],
                  MaxFuncEvals=5000,
                  # Method=:adaptive_de_rand_1_bin_radiuslimited,
                  Method=:generating_set_search,
                  TraceMode=:compact
    ) #, Method=:dxnes, NThreads=Threads.nthreads(), )
end

end
