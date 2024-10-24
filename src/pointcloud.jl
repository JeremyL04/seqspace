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
export neighborhood, geodesics, mds, isomap, scaling, geodesic_paths, avg_path_curvature

# ------------------------------------------------------------------------
# globals

const ∞ = Inf

∧(u, v) = u[1] * v[2] - u[2] * v[1]

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
			# ∇m[i.I[2],i.I[1]] = ∇[n] # ChainRulesTestUtils.test_rrule fails with this line

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

Adj = Vector{Vector{Tuple{Int64, Float64}}}

# ------------------------------------------------------------------------
# operations


"""
    neighborhood(x, k :: T; D=missing, accept=(d)->true) where T <: Integer

Constructs a neighborhood graph of the `k` nearest neighbor for each point of cloud `x`.
If `D` is given, it is assumed to be a dense matrix of pairwise distances.
"""
function neighborhood(x, k :: T; D=missing, accept=(d)->true) where T <: Integer
    D = ismissing(D) ? euclidean(x) : D
    G = Graph([Vertex(x[:,i]) for i ∈ axes(x,2)])
    for i ∈ axes(D,1)
        neighbor = sortperm(D[i,:])[2:end]
        append!(G.edges, [Edge((i,j), D[i,j]) for j ∈ neighbor[1:k] if accept(D[i,j])])
    end

    return G
end

"""
    neighborhood(x, r :: T; D=missing, accept=(d)->true) where T <: AbstractFloat

Constructs a neighborhood graph of all neighbors within euclidean distance `r` for each point of cloud `x`.
If `D` is given, it is assumed to be a dense matrix of pairwise distances.
"""
function neighborhood(x, r :: T; D=missing, accept=(d)->true) where T <: AbstractFloat
    D = ismissing(D) ? euclidean(x) : D
    G = Graph([Vertex(x[:,i]) for i ∈ axes(x,2)])
    for i ∈ axes(D,1)
        neighbor = first.(findall(0 .< D[:,i] .< r))
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

function geodesics(adj::Adj; sparse = true)
    if sparse
        adj  = adj
        dist = zeros(length(adj), length(adj))

        Threads.@threads for v ∈ 1:length(adj)
            dijkstra!(view(dist,:,v), adj, v)
        end

        return dist
    else
        return error("Floyd Warshall not implemented for this")
    end
end


# ------------------------------------------------------------------------ Geodesic Paths


"""
    dijkstra_paths(adj, src, target)

Compute the shortest path from `src` to `target` given adjacency list `adj` using Dijkstra's algorithm. Returns array of indices indicating the path.
"""

function dijkstra_paths(adj, src, target = nothing)
    N = length(adj)
    dist = fill(∞, N)  # 1D array for distances from src
    dist[src] = 0
    
    prev = fill(-1, N)  # Array to store previous nodes
    
    Q = RankedQueue((src, 0.0))  # Priority queue with the source node
    sizehint!(Q, N)  # Reduce size hint to N (number of nodes)
    
    while length(Q) > 0
        u, d₀ = take!(Q)
        for (v, d₂) ∈ adj[u]
            d₀₂ = d₀ + d₂
            if d₀₂ < dist[v]
                dist[v] = d₀₂
                prev[v] = u 
                if v ∈ Q
                    update!(Q, v, d₀₂)
                else
                    insert!(Q, v, d₀₂)
                end
            end
        end
    end

    paths = [Vector{Int}() for _ in 1:N]  # Pre-allocate the list of paths

    for target in 1:N
        t = target
        while t != -1
            push!(paths[target], t)  # Push nodes into the path directly
            t = prev[t]
        end
        paths[target] = reverse(paths[target])  # Reverse to get the correct order
    end
    
    if target !== nothing
        return paths[target]
    else
        return paths
    end
    
end

"""
    geodesic_paths(G :: Graph; sparse=true)

Compute the matrix of pairwise geodesic paths, given a neighborhood graph `G`. Method not implemented with Floyd Warshall.
"""
function geodesic_paths(G :: Graph; sparse=true)
    if sparse
        adj = adjacency_list(G)
        return hcat([dijkstra_paths(adj,i) for i ∈ 1:length(G)]...)
    else
        error("Floyd Warshall not implemented for paths")
    end
end

"""
    geodesic_paths(x, k; D=missing, accept=(d)->true, sparse=true)

Compute the matrix of pairwise geodesic paths, given a pointcloud `x` and neighborhood cutoff `k`, from the resultant neighborhood graph.
"""
geodesic_paths(x, k; D=missing, accept=(d)->true, sparse=true) = geodesic_paths(neighborhood(x, k; D=D, accept=accept); sparse=sparse)

"""
    geodesic_paths(adj::Adj; sparse=true)

Compute the matrix of pairwise geodesic paths, given an adjacency list `adj`. Method not implemented with Floyd Warshall.
"""
function geodesic_paths(adj::Adj; sparse = true)
    if sparse
        return hcat([dijkstra_paths(adj,i) for i ∈ 1:length(adj)]...)
    else
        error("Floyd Warshall not implemented for paths")
    end
end

"""
    collect_top_paths(G, N)
    
Collect the top `γ` percent of geodesic paths from the neighborhood graph `G` chosen by path length.
"""
function collect_top_paths(C ::Union{Graph,Adj}; γ = 0.05)
    paths = upper_tri(geodesic_paths(C))
    N = Int(round(γ * length(paths)))
    top_paths = sort(paths, by=length, rev=true)[1:N]
    return top_paths
end

collect_top_paths(data, k; γ = 0.05) = collect_top_paths(neighborhood(data, k), γ = γ)

# ------------------------------------------------------------------------ Path Angles

"""
    calculate_signed_angle(v1, v2)

Compute the angle between two vectors `v1` and `v2`.
"""
calculate_signed_angle(v1, v2) = sign(∧(v1, v2)) * acos((v1 ⋅ v2) / (norm(v1) * norm(v2)))

"""
    angles(path,data)

Compute the angles between the vectors connecting the points in the path.
"""
function angles(path,data)
    n_path = length(path)
    if n_path >= 3
        coords = eachcol(data[:, path]) |> collect
        d = [coords[k] - coords[k+1] for k ∈ 1:n_path - 1]
        angles = [calculate_signed_angle(d[k], d[k+1]) for k ∈ 1:n_path - 2]
        return angles
    else 
        println("Path too short")
    end 
end

function avg_path_curvature(high_data, k, low_data; γ = 0.05)
    G = neighborhood(high_data, k)
    paths = collect_top_paths(G;γ = γ)
    [mean(angles(path,low_data)) for path ∈ paths]
end

function avg_path_curvature(adj::Adj, low_data; γ = 0.05)
    paths = collect_top_paths(adj;γ = γ)
    [mean(angles(path,low_data)) for path ∈ paths]
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

end
