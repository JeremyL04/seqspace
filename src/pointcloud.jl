module PointCloud

using LinearAlgebra
# using PyCall

import Base:
    eltype, length, minimum, take!

import ChainRulesCore: 
    rrule, NoTangent

include("queue.jl")
using .PriorityQueue

include("distance.jl")
using .Distances

export embed, upper_tri
export neighborhood, geodesics, mds, isomap, scaling, local_connectors

# ------------------------------------------------------------------------
# globals

const ∞ = Inf

# ------------------------------------------------------------------------
# utility functions

"""
    upper_tri(x)

Returns the upper triangular portion of matrix `x`.
"""
upper_tri(x) = [ x[i.I[1], i.I[2]] for i ∈ CartesianIndices(x) if i.I[1] < i.I[2] ]
function rrule(::typeof(upper_tri), m)
	x = upper_tri(m)
	return x, (∇) -> begin
		∇m = zeros(size(m))
		n = 1
		for i ∈ CartesianIndices(∇m)
			if i.I[1] >= i.I[2]
				continue
			end
			
			∇m[i.I[1],i.I[2]] = ∇[n]
			∇m[i.I[2],i.I[1]] = ∇[n]

			n += 1
		end
		
        (NoTangent(), ∇m)
    end
end

function embed(x, dₒ; σ=0.00)
	y = if dₒ > size(x,1)
			vcat(x, zeros(dₒ-size(x,1), size(x,2)))
		elseif dₒ == size(x,1)
		    x	
		else
			error("cannot embed into smaller dimension")
		end
	
	return y .+ σ.*randn(size(y)...)
end

# ------------------------------------------------------------------------
# types for neighborhood graph

"""
    struct Vertex{T <: Real}
        position :: Array{T}
    end

Represents a single cell within a larger point cloud.
The embedding space is normalized gene expression.
"""
struct Vertex{T <: Real}
    position :: Array{T}
end
Vertex(x) = Vertex{eltype(x)}(x)

eltype(v::Vertex{T}) where T <: Real = T

"""
    struct Edge{T <: Real}
        verts    :: Tuple{Int, Int}
        distance :: T
    end

Connects two neighboring `verts` by an edge of length `distance`.
"""
struct Edge{T <: Real}
    verts    :: Tuple{Int, Int}
    distance :: T
end
Edge(verts, distance) = Edge{typeof(distance)}(verts, distance)

eltype(e::Edge{T}) where T <: Real = T

"""
    struct Graph{T <: Real}
        verts :: Array{Vertex{T},1}
        edges :: Array{Edge{T},1}
    end

A generic graph data structure containing vertices (points in space) stored within `verts` connected by `edges`.
"""
struct Graph{T <: Real}
    verts :: Array{Vertex{T}, 1}
    edges :: Array{Edge{T}, 1}
end
Graph(verts :: Array{Vertex{T},1}) where T <: Real = Graph{T}(verts, [])

length(G :: Graph) = length(G.verts)

# ------------------------------------------------------------------------
# operations

"""
    neighborhood(x, k :: T; D=missing, accept=(d)->true) where T <: Vector{Int}

Constructs a neighborhood graph of the `k` nearest neighbor for each point of cloud `x`.
If `D` is given, it is assumed to be a dense matrix of pairwise distances.
"""
function neighborhood(x, k :: T; D=missing, accept=(d)->true) where T <: Vector{Int64}
    D = ismissing(D) ? euclidean(x) : D
    G = Graph([Vertex(x[:,i]) for i ∈ 1:size(x,2)])
    for i ∈ 1:size(D,1)
        neighbor = sortperm(D[i,:])[2:end]
        append!(G.edges, [Edge((i,j), D[i,j]) for j ∈ neighbor[1:k[i]] if accept(D[i,j])])
    end

    return G
end

"""
    neighborhood(x, k :: T; D=missing, accept=(d)->true) where T <: Integer

Constructs a neighborhood graph of the `k` nearest neighbor for each point of cloud `x`.
If `D` is given, it is assumed to be a dense matrix of pairwise distances.
"""
function neighborhood(x, k :: T; D=missing, accept=(d)->true) where T <: Integer
    D = ismissing(D) ? euclidean(x) : D
    G = Graph([Vertex(x[:,i]) for i ∈ 1:size(x,2)])
    for i ∈ 1:size(D,1)
        neighbor = sortperm(D[i,:])[2:end]
        append!(G.edges, [Edge((i,j), D[i,j]) for j ∈ neighbor[1:k] if accept(D[i,j])])
    end

    return G
end

"""
    neighborhood(x, k :: T; D=missing, accept=(d)->true) where T <: AbstractFloat

Constructs a neighborhood graph of all neighbors within euclidean distance `k` for each point of cloud `x`.
If `D` is given, it is assumed to be a dense matrix of pairwise distances.
"""
function neighborhood(x, k :: T; D=missing, accept=(d)->true) where T <: AbstractFloat
    D = ismissing(D) ? euclidean(x) : D
    G = Graph([Vertex(x[:,i]) for i ∈ 1:size(x,2)])
    for i ∈ 1:size(D,1)
        neighbor = first.(findall(0 .< D[:,i] .< k))
        append!(G.edges, [Edge((i,j), D[i,j]) for j ∈ neighbor if accept(D[i,j])])
    end

    return G
end

"""
    avg_nbhd_distance(x, k :: T) where T <: Integer

Computes the average neighborhood distance for each point in the point cloud `x` using the `k` nearest neighbors.
"""
function avg_nbhd_distance(x, k :: T) where T <: Integer
    N = size(x,2)
    edges = neighborhood(x, k).edges
    verts = [edge.verts[1] for edge in edges]
    distances = [edge.distance for edge in edges]
    NBHD_sizes = Array{Float64}(undef, N)
    for i in eachindex(x[2, :])
        indices = findall(verts .== i)
        NBHD_sizes[i] = sum(distances[indices])
    end
    return NBHD_sizes/maximum(NBHD_sizes)
end

"""
WIP: 
    local_connectors(NBHD_sizes :: Vector{T}, kₒ) where T <: AbstractFloat

Returns the number of connectors for each point in the point cloud `x` based on the average neighborhood distance.
"""


function local_connectors(Ξ, kₒ)
    # This is messy, but it works for now
    if typeof(Ξ) <: Vector
        NBHD_sizes = Ξ
    elseif typeof(Ξ) <: Matrix
        NBHD_sizes = avg_nbhd_distance(Ξ, kₒ)
    else
        error("Ξ must be a vector or matrix")
    end


    κ = Array{Int}(undef, length(NBHD_sizes))
    μ = mean(NBHD_sizes)
    σ = std(NBHD_sizes)
    limits = [μ - 2σ, μ - σ, μ, μ + σ, μ + 2σ]
    for i in eachindex(NBHD_sizes)
        if NBHD_sizes[i] < limits[1]
            κ[i] = kₒ - 2
        elseif NBHD_sizes[i] < limits[2]
            κ[i] = kₒ - 1
        elseif NBHD_sizes[i] < limits[3]
            κ[i] = kₒ
        elseif NBHD_sizes[i] < limits[4]
            κ[i] = kₒ 
        elseif NBHD_sizes[i] < limits[5]
            κ[i] = kₒ
        else
            κ[i] = kₒ
        end
    end
    return κ
end


"""
    adjacency_list(G :: Graph)

Return the flattened adjacency list for graph `G`.
"""
function adjacency_list(G :: Graph)
    adj = [ Tuple{Int, Float64}[] for v ∈ 1:length(G.verts) ]
    for e ∈ G.edges
        v₁, v₂ = e.verts
        if (v₂,e.distance) ∉ adj[v₁]
            push!(adj[v₁], (v₂, e.distance))
        end
        if (v₁,e.distance) ∉ adj[v₂]
            push!(adj[v₂], (v₁, e.distance))
        end
    end

    return adj
end

"""
    dijkstra!(dist, adj, src)

Compute the shortest path from `src` to all other points given adjacency list `adj` and distances `dist` using Dijkstra's algorithm.
"""
function dijkstra!(dist, adj, src)
    dist      .= ∞
    dist[src]  = 0

    Q = RankedQueue((src, 0.0))
    sizehint!(Q, length(dist))

    while length(Q) > 0
        u, d₀ = take!(Q)
        for (v, d₂) ∈ adj[u]
            d₀₂ = d₀ + d₂
            if d₀₂ < dist[v]
                dist[v] = d₀₂
                if v ∈ Q
                    update!(Q, v, d₀₂)
                else
                    insert!(Q, v, d₀₂)
                end
            end
        end
    end
end
"""
    dijkstra_paths(adj, src, target)

Compute the shortest path from `src` to `target` given adjacency list `adj` using Dijkstra's algorithm. Returns array of indices indicating the path.
"""

function dijkstra_paths(adj, src, target)
    dist = fill(∞, length(adj), length(adj))
    dist[src] = 0

    prev = fill(-1, length(dist))  # Array to track the predecessor of each node

    Q = RankedQueue((src, 0.0))
    sizehint!(Q, length(dist))

    while length(Q) > 0
        u, d₀ = take!(Q)
        for (v, d₂) ∈ adj[u]
            d₀₂ = d₀ + d₂
            if d₀₂ < dist[v]
                dist[v] = d₀₂
                prev[v] = u  # Set the predecessor of v to be u
                if v ∈ Q
                    update!(Q, v, d₀₂)
                else
                    insert!(Q, v, d₀₂)
                end
            end
        end
    end

    path = Int[]
    while target != -1
        push!(path, target)
        target = prev[target]
    end

    return reverse!(path) 
end



"""
    floyd_warshall(G :: Graph)

Compute the shortest path from all vertices to all other vertices within graph `G`.
"""
function floyd_warshall(G :: Graph)
    V = length(G.verts)
    D = fill(∞, (V,V))
    # remove diagonal
    for ij ∈ CartesianIndices(D)
        if ij.I[1] == ij.I[2]
            D[ij] = 0
        end
    end

    # all length 1 paths
    for e ∈ G.edges
        D[e.verts[1],e.verts[2]] = e.distance
        D[e.verts[2],e.verts[1]] = e.distance
    end

    # naive V³ paths
    for k ∈ 1:V
        for i ∈ 1:V
            for j ∈ 1:V
                if D[i,j] > D[i,k] + D[k,j]
                    D[i,j] = D[i,k] + D[k,j]
                    D[j,i] = D[i,j]
                end
            end
        end
    end

    return D
end

"""
    geodesics(G :: Graph; sparse=true)

Compute the matrix of pairwise distances, given a neighborhood graph `G`, weighted by local Euclidean distance.
If sparse is true, it will utilize Dijkstra's algorithm, individually for each point.
If sparse is false, it will utilize the Floyd Warshall algorithm.
"""
function geodesics(G :: Graph; sparse=true)
    if sparse
        adj  = adjacency_list(G)
        dist = zeros(length(G), length(G))

        Threads.@threads for v ∈ 1:length(G)
            dijkstra!(view(dist,:,v), adj, v)
        end

        return dist
    else
        return floyd_warshall(G)
    end
end

"""
    geodesics(x, k; D=missing, accept=(d)->true, sparse=true) 

Compute the matrix of pairwise distances, given a pointcloud `x` and neighborhood cutoff `k`, from the resultant neighborhood graph.
If sparse is true, it will utilize Dijkstra's algorithm, individually for each point.
If sparse is false, it will utilize the Floyd Warshall algorithm.
"""
geodesics(x, k; D=missing, accept=(d)->true, sparse=true) = geodesics(neighborhood(x, k; D=D, accept=accept); sparse=sparse)

"""
    geodesic_paths(G :: Graph; sparse=true)

Compute the matrix of pairwise geodesic paths, given a neighborhood graph `G`. Method not implemented with Floyd Warshall.
"""
function geodesic_paths(G :: Graph; sparse=true)
    if sparse
        adj  = adjacency_list(G)
        paths = Array{Array{Int,1},2}(undef, length(G), length(G))

        for v ∈ 1:length(G)
            paths[v,v] = [v]
            for u ∈ v:length(G)
                paths[v,u] = dijkstra_paths(adj, v, u)
                paths[u,v] = reverse(paths[v,u]) # Geodesic paths are symmetric (...hopefully)
            end
        end

    else
        error("Floyd Warshall not implemented for paths")
    end
end

# ------------------------------------------------------------------------
# non ml dimensional reduction

"""
    mds(D², dₒ)

Computes the lowest `dₒ` components from a _Multidimensional Scaling_ analysis given pairwise squared distances `D²`.
"""
function mds(D², dₒ)
    N = size(D²,1)
    C = I - fill(1/N, (N,N))
    B = -1/2 * C*D²*C

    eig = eigen(B)
    ι   = sortperm(real.(eig.values); rev=true)

    λ = eig.values[ι]
    ν = eig.vectors[:,ι]

    return ν[:,1:dₒ] * Diagonal(sqrt.(λ[1:dₒ]))
end

"""
    isomap(x, dₒ; k=12, sparse=true)

Compute the isometric embedding of point cloud `x` into `dₒ` dimensions.
Geodesic distances between all points are estimated by utilizing the shortest path defined by the neighborhood graph.
The embedding is computed from a multidimensional scaling analysis on the resultant geodesics.
"""
function isomap(x, dₒ; k=12, sparse=true)
    G = neighborhood(x, k)
    D = geodesics(G; sparse=sparse)
    return mds(D.^2, dₒ)
end

# ------------------------------------------------------------------------
# dimension estimation

"""
    scaling(D, N)

Estimate the Hausdorff dimension by computing how the number of points contained within balls scales with varying radius.
"""
function scaling(D, N)
	Rₘᵢₙ = minimum(D[D .> 0])
	Rₘₐₓ = maximum(D)
	Rs   = range(Rₘᵢₙ,Rₘₐₓ,length=N)
	
	ϕ = zeros(size(D,1), N)
	for (i,R) ∈ enumerate(Rs)
		ϕ[:,i] = sum(D .<= R, dims=1)
	end
	
	return ϕ, Rs
end

# ------------------------------------------------------------------------
# tests

using Statistics

function test()
    r, ξ = sphere(2000)
    D = spherical_distance(r)

    r = embed(r, size(r,2); σ=0.1)
    ks = collect(4:2:50) 
    ρ  = zeros(length(ks))

    for (i,k) ∈ enumerate(ks)
        D̂ = geodesics(r, k)
        ρ[i] = cor(upper_tri(D̂), upper_tri(D))
    end

    return ks, ρ
end

function dijkstra_paths(adj, src, target)
    dist = fill(∞, length(adj), length(adj))
    dist[src] = 0

    prev = fill(-1, length(dist))  # Array to track the predecessor of each node

    Q = RankedQueue((src, 0.0))
    sizehint!(Q, length(dist))

    while length(Q) > 0
        u, d₀ = take!(Q)
        for (v, d₂) ∈ adj[u]
            d₀₂ = d₀ + d₂
            if d₀₂ < dist[v]
                dist[v] = d₀₂
                prev[v] = u  # Set the predecessor of v to be u
                if v ∈ Q
                    update!(Q, v, d₀₂)
                else
                    insert!(Q, v, d₀₂)
                end
            end
        end
    end

    path = Int[]
    while target != -1
        push!(path, target)
        target = prev[target]
    end

    return reverse!(path) 
end

function geodesic_paths(G :: Graph; sparse=true)
    if sparse
        adj  = adjacency_list(G)
        paths = Array{Array{Int,1},2}(undef, length(G), length(G))

        for v ∈ 1:length(G)
            paths[v,v] = [v]
            for u ∈ v:length(G)
                paths[v,u] = dijkstra_paths(adj, v, u)
                paths[u,v] = reverse(paths[v,u]) # Geodesic paths are symmetric (...hopefully)
            end
        end

    else
        error("Floyd Warshall not implemented for paths")
    end
end



end
