module scRNA

using GSL
using Statistics, StatsBase, Distributions, NMF
using SpecialFunctions

import Base:
    size,
    IndexStyle, getindex, setindex!,
    +, -, *, /, %, ^,
    ∪

import LinearAlgebra:
    svd, diag, Diagonal, AbstractMatrix

include("io.jl")
include("mle.jl")
include("util.jl")


using .DataIO: read_mtx, read_barcodes, read_features, read_dge

export barcodes, genes, Count

# ------------------------------------------------------------------------
# globals

BarcodeFile = "barcodes.tsv"
FeatureFile = "features.tsv"
CountMatrix = "matrix.mtx"

const ∞ = Inf
const FitType = NamedTuple{
        (
            :parameters,
            :uncertainty,
            :likelihood,
            :trend,
            :cdf,
            :residuals
        ),
        Tuple{
            Array{Float64,1},
            Array{Float64,1},
            Float64,
            Array{Float64,1},
            Array{Float64,1},
            Array{Float64,1}
        }
}

# ------------------------------------------------------------------------
# utility functions

barcode(name) = occursin("/",name) ? join(split(name,"/")[2:end], "/") : name

function trendline(x, y, n)
    l,r = log(minimum(x)), log(maximum(x))
    bp  = range(l, r, length=n+1)
    x₀  = Array{eltype(x),1}(undef, n)
    μ   = Array{eltype(y),1}(undef, n)
    for i ∈ 1:n
        xₗ, xᵣ = exp(bp[i]), exp(bp[i+1])
        pts = y[xₗ .≤ x .≤ xᵣ]
        if length(pts) > 0
            μ[i] = exp.(mean(log.(pts)))
        else
            μ[i] = μ[i-1]
        end
        x₀[i] = 0.5*(xₗ + xᵣ)
    end

    x₀ = [exp(l); x₀; exp(r)]
    μ  = [y[argmin(x)]; μ; y[argmax(x)]]
    return extrapolate(
        interpolate((x₀,), μ, Gridded(Linear())),
        Line()
    )
end

function clamp!(x, lo, hi)
    x[x .< lo] .= lo
    x[x .> hi] .= hi
    x
end

betainc(a,b,x) = GSL.sf_beta_inc(a,b,x)
gammainc(a,x)  = GSL.sf_gamma_inc_P(a,x)

# ------------------------------------------------------------------------
# type w/ (de)serialization

"""
    struct Count{T <: Real} <: AbstractArray{T,2}
        data :: Array{T,2}
        gene :: Array{AbstractString,1}
        cell :: Array{AbstractString,1}
    end

Data structure used to represent count data obtained during a scRNAseq sequencing experiment.
Individual cells are stored as column vectors while expression of singular genes are obtained as row vectors.
`data` contains the raw/normalized count matrix.
`gene` and `cell` contain the row/column labels respectively.
Genes and cells can be indexed _either_ by integers or names, i.e. strings.
"""
struct Count{T <: Real} <: AbstractArray{T,2}
    data :: Array{T,2}
    gene :: Array{AbstractString,1}
    cell :: Array{AbstractString,1}

    Count(data::Array{T,2}, gene, cell) where T <: Real= begin
        if length(gene) != size(data,1)
            error("number of genes $(length(gene)) not equal to number of rows $(size(data,1))")
        end

        if length(cell) != size(data,2)
            error("number of cells $(length(cell)) not equal to number of columns $(size(data,2))")
        end

        new{T}(data,gene,cell)
    end
end

Count(data::Array{T,1}, gene::S, cell) where {T <: Real, S <: AbstractString} = Count(reshape(data,1,length(data)), [gene], cell)

# ------------------------------------------------------------------------
# simple operators

# data access
genes(seq::Count) = seq.gene
cells(seq::Count) = seq.cell

ngenes(seq::Count) = size(seq,1)
ncells(seq::Count) = size(seq,2)

# arithmetic
+(seq::Count, x::Union{<:Number,<:AbstractMatrix}) = Count(seq.data+x, seq.gene, seq.cell)
-(seq::Count, x::Union{<:Number,<:AbstractMatrix}) = Count(seq.data-x, seq.gene, seq.cell)

*(seq::Count, x::Union{<:Number,<:AbstractMatrix}) = Count(seq.data*x, seq.gene, seq.cell)
/(seq::Count, x::Number) = Count(seq.data/x, seq.gene, seq.cell)
%(seq::Count, x::Number) = Count(seq.data%x, seq.gene, seq.cell)
^(seq::Count, x::Number) = Count(seq.data^x, seq.gene, seq.cell)

# ------------------------------------------------------------------------
# matrix interface

size(seq::Count) = size(seq.data)

# -- indexing

IndexStyle(seq::Count) = IndexCartesian()

# integer indexes
getindex(seq::Count, I::Vararg{Int,2}) = getindex(seq.data, I...)
getindex(seq::Count, I::AbstractArray{<:Integer}, J) = Count(getindex(seq.data,I,J),seq.gene[I],seq.cell[J])
getindex(seq::Count, I, J::AbstractArray{<:Integer}) = Count(getindex(seq.data,I,J),seq.gene[I],seq.cell[J])
getindex(seq::Count, I::AbstractArray{<:Integer}, J::AbstractArray{<:Integer}) = Count(getindex(seq.data,I,J),seq.gene[I],seq.cell[J])

setindex!(seq::Count, v, I::Vararg{Int,2}) = setindex!(seq.data, v, I...)

# string indexes
function getindex(seq::Count, I::Vararg{String,2})
    gene, batch = I
    row = findfirst(seq.gene .== gene)
    col = inbatch(seq, batch)
    return getindex(seq, row, col)
end

ArrayOrOne{T} = Union{<:T, AbstractArray{<:T,1}}

findrows(seq,G) = G isa AbstractArray ? [findfirst(seq.gene .== g) for g in G] : findfirst(seq.gene .== G)
findcols(seq,B) = B isa AbstractArray ? reduce((x,y) -> x .| y, inbatch(seq, b) for b ∈ B) : inbatch(seq, B)

function getindex(seq::Count, genes::ArrayOrOne{AbstractString}, batches::ArrayOrOne{AbstractString})
    row = findrows(seq,genes)
    col = findcols(seq,batches)

    return getindex(seq, row, col)
end

function getindex(seq::Count, genes::ArrayOrOne{<:AbstractString}, batches::Function)
    row = findrows(seq,genes)
    return getindex(seq, row, batches)
end

function getindex(seq::Count, genes::Function, batches::ArrayOrOne{<:AbstractString})
    col = findcols(seq,batches)
    return getindex(seq, genes, col)
end

# mixed indexes

function getindex(seq::Count, row::ArrayOrOne{Integer}, batches::ArrayOrOne{AbstractString})
    col = findcols(seq,batches)
    return getindex(seq, row, col)
end

function getindex(seq::Count, genes::ArrayOrOne{AbstractString}, col::ArrayOrOne{Integer})
    row = findrows(seq,genes)
    return getindex(seq, row, col)
end

# -- matrix operations

svd(seq::Count) = svd(seq.data)

# -- batch subselection

function batches(seq::Count)
    all = map(cells(seq)) do id
        s = split(id, "/")
        return length(s) > 1 ? s[1] : nothing
    end
    all = filter(!isnothing, all)

    unique!(all)
    return all
end

inbatch(seq::Count, batch::AbstractString) = startswith.(cells(seq), batch*"/")

# -- gene subselection

locus(seq::Count, genes::AbstractString...)    = filter((i)->!isnothing(i), [findfirst(seq.gene .== g) for g in genes])
searchloci(seq::Count, prefix::AbstractString) = occursin.(prefix, seq.gene)

# ------------------------------------------------------------------------
# creation operators

"""
    load(dir::AbstractString; batch=missing)

Read in scRNAseq experimental data from directory `dir`.
The directory is expected to contain the following files:
  1. barcodes.tsv : one cell name per line
  2. features.tsv : one gene name per line
  3. matrix.mtx : count matrix in mtx format
If `batch` is not missing, then it will be appended to each cell label.
"""
function load(dir::AbstractString; batch=missing)

    if occursin(".txt", dir) #XXX Added this to support reading from a single DGE file... maybe better as an overload
        open(dir) do io
            counts, genes, cells = read_dge(io; named_rows=true, named_cols=true)
            return Count(counts, genes, cells)
        end

    else
        !isdir(dir) && error("directory '$dir' not found")

        files = readdir(dir)

        BarcodeFile ∉ files && error("'$BarcodeFile' not found in directory '$dir'")
        FeatureFile ∉ files && error("'$FeatureFile' not found in directory '$dir'")
        CountMatrix ∉ files && error("'$CountMatrix' not found in directory '$dir'")

        data     = open(read_mtx,      "$dir/$CountMatrix")
        barcodes = open(read_barcodes, "$dir/$BarcodeFile")
        features = open(read_features, "$dir/$FeatureFile")

        length(barcodes) ≠ size(data,2) && error("number of barcodes $(length(barcodes)) ≠ number of columns $(size(counts,2)). check data")
        length(features) ≠ size(data,1) && error("number of features $(length(features)) ≠ number of rows $(size(counts,1)). check data")

        # TODO: accept sparse inputs?
        prepend = ismissing(batch) ? (bc) -> bc : (bc) -> batch * "/" * bc
        return Count(Matrix(data),features,map(prepend,barcodes))
    end
end


matchperm(a, b) = findfirst.(isequal.(a), (b,))

"""
    ∪(seq₁::Count{T}, seq₂::Count{S}) where {T <: Real, S <: Real}

Collate count matrix `seq₁` and `seq₂` by taking the union across genes.
Reorders rows of `seq₂` to match gene names of `seq₁`.
Additional genes in `seq₂` not contained in `seq₁` are added as augmented rows.
"""
function ∪(seq₁::Count{T}, seq₂::Count{S}) where {T <: Real, S <: Real}
    T₀ = T ≠ S ? promote_rule(T,S) : T
    matches = matchperm(genes(seq₂), genes(seq₁)) # match 2 -> 1

    newgenes = [gene for (gene,match) ∈ zip(genes(seq₂),matches) if isnothing(match) ]
    features = [genes(seq₁) ; newgenes]
    barcodes = [cells(seq₁) ; cells(seq₂)]

    n = 1
    δ = ngenes(seq₁)
    for (i,m) ∈ enumerate(matches)
        if isnothing(m)
            matches[i] = n + δ
            n += 1
        end
    end

    data = [hcat(seq₁.data, zeros(T₀,ngenes(seq₁),ncells(seq₂)));
            zeros(T₀,length(newgenes),ncells(seq₁)+ncells(seq₂))]

    δ = ncells(seq₁)
    for i ∈ 1:ncells(seq₂)
        data[matches,i+δ] = seq₂[:,i]
    end

    return Count(data,features,barcodes)
end

# NOTE: different merging strategy
#       only common genes (intersection) are kept
"""
    ∩(seq₁::Count{T}, seq₂::Count{S}) where {T <: Real, S <: Real}

Collate count matrix `seq₁` and `seq₂` by taking the union across genes.
Reorders rows of `seq₂` to match gene names of `seq₁`.
Only keeps genes present in _both_ `seq₁` and `seq₂`.
"""
function ∩(seq₁::Count{T}, seq₂::Count{S}) where {T <: Real, S <: Real}
    error("need to implement")
end

# ------------------------------------------------------------------------
# filtering

"""
    filtergene(f, seq::Count)

Filters genes of count matrix `seq` based upon row function `f`.
"""
function filtergene(f, seq::Count)
    ι = [f(seq[i,:], seq.gene[i]) for i in 1:ngenes(seq)]
    return seq[ι,:]
end

"""
    filtercells(f, seq::Count)

Filters cells of count matrix `seq` based upon column function `f`.
"""
function filtercell(f, seq::Count)
    ι = [f(seq[:,i], seq.cell[i]) for i in 1:ncells(seq)]
    return seq[:,ι]
end

function filterbarcodes(seq::Count)
    barcodes = open("$ROOT/barcodes") do io
        Set([line for line ∈ eachline(io)])
    end

    seq = filtercell(seq) do _, name
        barcode(name) ∈ barcodes
    end

    return seq
end

"""
    counts(dir::String, subdir::String; chatty = true)

Load and filter scRNAseq data from `dir` and `subdir`.
"""
function counts(dir::String, subdir::String; chatty = true)
    # helper to wrap any filter + logging
    function filter_and_log(f, count, dim::Int, desc::AbstractString)
        if chatty
            n_before = size(count, dim)
        end
        count = f(count)
        if chatty
            n_after = size(count, dim)
            kind = dim == 2 ? "cells" : "genes"
            println("Removed $(n_before - n_after) $desc out of $(n_before) total $kind")
        end
        return count
    end

    # 1) load & union
    count = reduce(∪,
        scRNA.load("$dir/$d") for d ∈ readdir(dir) if occursin(subdir, d)
    )

    # 2) remove non-mel cells
    markers = (
        dvir = scRNA.searchloci(count, "Dvir_"),
    )
    count = filter_and_log(count, 2, "cells with high Dvir expression") do cnt
        scRNA.filtercell(cnt) do cell, _
            sum(cell[markers.dvir]) < .10*sum(cell)
        end
    end

    # 3) remove low-count / non-mel genes
    count = filter_and_log(count, 1, "Dvir genes") do cnt
        scRNA.filtergene(cnt) do row, gene
            !occursin("Dvir_", gene)
        end
    end

    # 4) remove yolk & pole cells
    markers = (
        yolk = scRNA.locus(count, "sisA", "CG8129", "Corp", "CG8195", "CNT1", "ZnT77C"),
        pole = scRNA.locus(count, "pgc"),
    )
    count = filter_and_log(count, 2, "cells with yolk & pole expression") do cnt
        scRNA.filtercell(cnt) do cell, _
            (sum(cell[markers.yolk]) < 10 && sum(cell[markers.pole]) < 3)
        end
    end

    # 5) remove low-count cells
    count = filter_and_log(count, 2, "cells low counts + high number of non-zero counts") do cnt
        scRNA.filtercell(cnt) do cell, _
            (sum(cell) > 1e4 && sum(cell .> 1) > 20)
        end
    end

    # 6) remove lowly expressed genes
    count = filter_and_log(count, 1, "low expression genes") do cnt
        scRNA.filtergene(cnt) do row, _
            sum(row .> 1) > 20
        end
    end

    return count.data, count.gene, count.cell
end


# ------------------------------------------------------------------------
# synthetic data generation

# TODO: gamma distribution?
#       correlated genes?
"""
    generate(ngene, ncell; ρ=(α=Gamma(0.25,2), β=Normal(1,.01), γ=Gamma(3,3)))

Generate scRNAseq data assuming a monoclonal population of cells sampled against a heteroskedastic negative binomial model.
"""
function generate(ngene, ncell; ρ=(α=Gamma(0.25,2), β=Normal(1,.01), γ=Gamma(3,3)))
    seq = zeros(Int, ngene, ncell)
    cdf = zeros(Float64, ngene, ncell)

    z = log.(rand(Gamma(5,1), ncell))

    α = log.(rand(ρ.α, ngene))
    β = rand(ρ.β, ngene)
    γ = rand(ρ.γ, ngene)

    for g ∈ 1:ngene
        μ⃗ = exp.(α[g] .+ β[g].*z)
        for (c, μ) ∈ enumerate(μ⃗)
            λ = rand(Gamma(γ[g], μ/γ[g]),1)[1]

            seq[g,c] = rand(Poisson(λ),1)[1]
        end
        p⃗ = μ⃗./(μ⃗.+γ[g])
        p⃗[p⃗ .< 0] .= 0
        p⃗[p⃗ .> 1] .= 1
        cdf[g,:] = 1 .- betainc.(vec(seq[g,:]).+1.0, γ[g], p⃗)
    end

    # filter out any rows that don't express in at least one cell
    ι = vec(sum(seq,dims=2)) .> 0

    seq = seq[ι, :]
    cdf = cdf[ι, :]

    α = α[ι]
    β = β[ι]
    γ = γ[ι]

    return (
        data = Count(seq, [string(g) for g in 1:size(seq,1)], [string(c) for c in 1:size(seq,2)]),
        cdf  = cdf,
        α    = α,
        β    = β,
        γ    = γ,
        ḡ    = exp.(z),
   )
end

function TestReviseSCRNA()
    return println("TestReviseSCRNA 5:42")
end

end
