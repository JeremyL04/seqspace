module Radon

using LinearAlgebra
using Statistics
using Roots: find_zero, rrule

#=
TODO Lots of optimization to be done here. Add more arguments to the functions and have them return 
functions to be called in the loss function (i.e. only generate Hamiltonian once).
=#

function Square_CDFRadon(r,θ)
    α = mod(θ + π/4, π/2) + π/4
    C₁ = cot(α)
    C₂ = csc(α)
    A₁ = (-C₂*r+C₁+1)
    A₂ = (C₂*r+C₁+1)
    A₃ = (-C₂*r-C₁+1)
    A₄ = (C₂*r-C₁+1)
    D = 2*(abs(cos(α)-sin(α))+abs(cos(α)+sin(α))) # Missing a factor of 2 here
    return (-abs(A₁)*A₁+abs(A₂)*A₂+abs(A₃)*A₃-abs(A₄)*A₄)/(4*D*C₁*C₂) + 0.5
end

function Square_InvCDFRadon(y,θ)
    g(x) = Square_CDFRadon(x,θ) - y
    return find_zero(g,0)
end

function ϵ_Square(Z)
    N = size(Z,2)
    Nₛ = 6 # Number of slices
    Θ = (0+0.001:π/(Nₛ):π-0.01)
    proj = [[dot( point , [cos(θ),sin(θ)] ) for point ∈ eachcol(Z)] for θ ∈ Θ]
    Y = collect(0:1/(N-1):1)
    InvCDFs = [[Square_InvCDFRadon(y,θ) for y ∈ Y] for θ ∈ Θ]
    return mean([mean( (InvCDFs[i] .- sort(proj[i])).^2 ) for i ∈ 1:length(Θ)].^(1/2)) # Notice this is W₂
end

function generate_boundary_points(N)
    N = Int(N)
    b = zeros(Float32, 2, N)
    for i in 0:N-1
        position = i * 8 / N
        b[:, i + 1] = if position < 2
            [-1 + position, 1.0]
        elseif position < 4
            [1.0, 3 - position]
        elseif position < 6
            [5 - position, -1.0]
        else
            [-1.0, position - 7]
        end
    end
    return b
end

function boundary_loss(z)
    N = 2
    A = collect(hcat(ones(N), 2(rand(N) .- 0.5))')
    B = collect(hcat(-ones(N), 2(rand(N) .- 0.5))')
    C = collect(hcat(2(rand(N) .- 0.5),ones(N))')
    D = collect(hcat(2(rand(N) .- 0.5),-ones(N))')
    E = hcat(A,B,C,D)
    return mean([minimum(norm.(eachcol(abs.(E[:,i] .- z)))) for i ∈ axes(E,2)])
end

function generate_grid(N)
    L = 1
    grid_size = ceil(Int, sqrt(N))
    x = range(-L, stop=L, length=grid_size)
    y = range(-L, stop=L, length=grid_size)
    X = repeat(x', length(y), 1)
    Y = repeat(y, 1, length(x))
    points = hcat(vec(X), vec(Y))'
    selected_points = points[:, 1:N]
    return selected_points
end

grid = generate_grid(15^2)

function rand_gridpoint_loss(z)
    rₘᵢₙ = 1.0
    reduced_grid = grid[:,norm.(eachcol(grid)) .> rₘᵢₙ]
    target_points = hcat(rand(eachcol(reduced_grid),10)...)
    mean([minimum(norm.(eachcol(abs.(target_points[:,i] .- z)))) for i ∈ axes(target_points,2)])    
end

function cell_centers(L, m)
    step = 2L / m
    centers = [-L + ((i - 0.5) * step) for i in 1:m]
    return hcat([[t,q] for t ∈ centers for q ∈ centers]...)
end

centers = cell_centers(1, 6)

function mod²(q)
    return q[1]^2 + q[2]^2
end

function MF_loss(z)
    N = ceil(Int,size(z,2)/(Int(sqrt(size(centers,2)))))
    N == 1 ? N = 2 : nothing
    target_points = centers
    D_full = ([eachcol((target_points[:,i] .- z)) for i ∈ axes(target_points,2)])
    D = [hcat(D_full[i][sortperm(norm.(D_full[i]))][1:N]...) for i ∈ axes(D_full,1)]
    return mean([mean(D[i][1,:])^2 + mean(D[i][2,:])^2 for i ∈ axes(D,1)])
end

        ϵᵤ = let
            #Θ = [0, π/4, π/2, 3π/4] .- 0.001
            Θ = collect(0:π/(8-1):π) .- 0.001
            Nₛ = length(Θ)
            N = size(z,2)
            L = [abs(cos(Θ[i])) + abs(sin(Θ[i])) for i ∈ 1:Nₛ]
            w = [collect(-L[i]:2L[i]/(N-1):L[i]) for i in 1:Nₛ]
            ϵ = 0.1
            W = [10*(σ.(-(w[i].+L[i])/ϵ) + σ.((w[i].-L[i])/ϵ)) .+ 1 for i ∈ 1:Nₛ]
            proj = [[dot( point , [cos(θ),sin(θ)] ) for point ∈ eachcol(z)] for θ ∈ Θ]
            Y = collect(0:1/(N-1):1)
            InvCDFs = [[Radon.Square_InvCDFRadon(y,θ) for y ∈ Y] for θ ∈ Θ]
            mean(mean([ (InvCDFs[i] - sort(proj[i])).^2 for i ∈ 1:length(Θ)]))
        end
        
        # ϵᵤ = let
        #     centered_data = z .- sum(z)/length(z)
        #     covariance_matrix = centered_data * centered_data' / (size(centered_data, 2) - 1)
        #     eigenvalues = eigvals(covariance_matrix)
        #     total_variance = sum(eigenvalues)
        #     relative_importance = eigenvalues / total_variance
        #     (relative_importance[1]/relative_importance[2] - 1)^2
        # end

end