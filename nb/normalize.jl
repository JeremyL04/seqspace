### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 2466c2a4-90c7-11eb-3f9c-5b87a7a35bb6
begin
	using LinearAlgebra, SpecialFunctions
	using Distributions, Statistics, StatsBase, Random
	using Optim, NLSolversBase
	using Clustering, Interpolations
	using Plots, ColorSchemes
	using NMF, Match, GZip, ProgressMeter, ForwardDiff

	import GSL
	
	default(fmt = :png)
end

# ╔═╡ 7cf6be2e-9315-11eb-1cb1-396f2131908b
begin
	using JSServe
	import WGLMakie
end

# ╔═╡ b992e41a-9334-11eb-1919-87967d572a21
using JLD2, FileIO

# ╔═╡ fc2b03f0-924b-11eb-0f20-45edefca4b76
md"""
# Normalization

This notebook serves as a collection of thoughts on how to preprocess scRNAseq data
"""

# ╔═╡ 969b3f50-90bb-11eb-2b67-c784d20c0eb2
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ be981c3a-90bb-11eb-3216-6bed955446f5
scRNA = ingredients("/Users/jeremy/.julia/dev/SeqSpace/src/scrna.jl").scRNA

# ╔═╡ e84220a8-90bb-11eb-2fb8-adb6c87c2faa
const ROOT = "/home/nolln/root/data/seqspace/raw"

# ╔═╡ ce78d71a-917a-11eb-3cdd-b15aad75d147
const SAMPLE = 4

# ╔═╡ 0bb97860-917f-11eb-3dd7-cfd0c7d890cd
begin
	cdfplot(x; kwargs...) = plot(sort(x), range(0,1,length=length(x)); kwargs...)
	cdfplot!(x; kwargs...) = plot!(sort(x), range(0,1,length=length(x)); kwargs...)
end

# ╔═╡ f972b674-9264-11eb-0a1c-0774ce2527e7
const ∞ = Inf

# ╔═╡ 9594926e-91d0-11eb-22de-bdfe3290b19b
function resample(data)
	new = zeros(eltype(data), size(data))
	for g ∈ 1:size(data,1)
		new[g,:] = sample(vec(data[g,:]), size(new,2))
	end
	return new
end

# ╔═╡ d7e28c3e-9246-11eb-0fd3-af6f94ea8562
function generate(ngene, ncell; ρ=(α=Gamma(0.25,2), β=Normal(1,.01)))
    N = zeros(Int, ngene, ncell)
	
    z = log.(rand(Gamma(5,1), ncell))
    α = log.(rand(ρ.α, ngene))
    β = rand(ρ.β, ngene)
	
    for g ∈ 1:ngene
        μ⃗ = exp.(α[g] .+ β[g].*z)
        for (c, μ) ∈ enumerate(μ⃗)
            N[g,c] = rand(Poisson(μ),1)[1]
        end
	end

    ι = vec(sum(N,dims=2)) .> 0

    N = N[ι, :]
    α = α[ι]
    β = β[ι]
		
	return (
		data  = N,
		param = (
			α = α,
			β = β,
			z = z,
		)
	)
end

# ╔═╡ ca2dba38-9184-11eb-1793-f588473daad1
null, params = generate(5000,2000);

# ╔═╡ bf8b0edc-9247-11eb-0ed5-7b1d16e00fc4
md"""
## How to find rank of count matrix?
### Prototype double stochastic
Assume we have a count matrix $X_{iα}$, where $i$ indexes genes and $\alpha$ indexes cells, that can be expressed as a matrix $Z_{i\alpha}$ of low rank $r$ plus a noise matrix of full rank $\mathcal{E}_{i\alpha}$.

``
X_{i\alpha} = Z_{i\alpha} + \mathcal{E}_{i\alpha}
``

By definition, the noise is expected to have zero mean

``
\langle \mathcal{E}_{i\alpha} \rangle = 0 \implies \langle X_{i\alpha} \rangle = Z_{i\alpha}
``

However, in general, we expect the noise matrix to be heteroskedastic, i.e. the variance of a given element of $\mathcal{E}$ will strongly depend upon its position. Thus the usual assumptions of a Marchenko-Pastur law are violated. 

For example, assume genes are Poisson distributed, but with different means depending on the exact gene. The variances of rows will then depend on this hidden variable. The same argument applies for considering columns. Thus, we look for a rescaling $\tilde{\mathcal{E}}$ that satisfies

``
G = \sum\limits_{i=1}^G \langle\tilde{\mathcal{E}}_{i\alpha}^2 \rangle \  \forall \alpha \qquad \text{and} \qquad N = \sum\limits_{\alpha=1}^N \langle\tilde{\mathcal{E}}_{i\alpha}^2 \rangle  \  \forall i
``

We define the rescaling by scaling rows by $\vec{u}$ and columns by $\vec{v}$, i.e.

``
\tilde{X}_{i\alpha} \equiv u_i X_{i\alpha} v_\alpha \implies \tilde{Z}_{i\alpha} \equiv u_i Z_{i\alpha} v_\alpha \quad \text{and} \quad \tilde{\mathcal{E}}_{i\alpha} \equiv u_i \mathcal{E}_{i\alpha} v_\alpha
``

#### Poisson example
Utilizing the Poisson example, we know the variance is equal to the mean and thus 

``
\langle\mathcal{E}_{i\alpha}^2 \rangle = X_{i\alpha} \implies \langle\tilde{\mathcal{E}}_{i\alpha}^2 \rangle = u^2_{i} X_{i\alpha} v^2_{\alpha}
``

Thus we can write down

``
G = \sum\limits_{i=1}^G  u^2_{i} X_{i\alpha} v^2_{\alpha} \  \forall \alpha \qquad \text{and} \qquad N = \sum\limits_{\alpha=1}^N u^2_{i} X_{i\alpha} v^2_{\alpha}  \  \forall i
``

We can solve for $\vec{u}$ and $\vec{v}$ by employing the Sinkhorn Knopp algorithm. We define the rescaled noise matrix by

``
\tilde{\Sigma}_{\alpha\beta} \equiv G^{-1} \sum\limits_{i} \tilde{\mathcal{E}}_{i\alpha}\tilde{\mathcal{E}}_{i\beta}
``

The distribution of eigenvalues of $\Sigma$ converge to Marchenko-Pastur with parameter of $\gamma = N/G$ and variance $1$, see *Girko, V. Theory of stochastic canonical equations 2001*. The largest eigenvalue is expected to be 

``\lambda_{max} \approx (1+\sqrt{\gamma})^2``

Thus the rank is given by the number of eigenvectors of $\Sigma$ greater than $\lambda_{max}$.
"""

# ╔═╡ f387130c-924f-11eb-2ada-794dfbf4d30a
function sinkhorn(A; r=[], c=[], maxᵢₜ=1000, δ=1e-4, verbose=false)
	if length(r) == 0
		r = size(A,2)
	end
	
	if length(c) == 0
		c = size(A,1)
	end
	
	x = ones(size(A,1))
	y = ones(size(A,2))
	for i ∈ 1:maxᵢₜ
		δr = maximum(abs.(x.*(A*y)  .- r))
		δc = maximum(abs.(y.*(A'*x) .- c))
		
		if verbose
			@show minimum(A'*x), minimum(A*y)
			@show r, c
			@show i, δr, δc
		end
		
		(isnan(δr) || isnan(δc)) && return x, y, false
		(δr < δ && δc < δ) 		 && return x, y, true
		
		y = c ./ (A'*x)
		x = r ./ (A*y)
		if verbose
			@show mean(x), mean(y)
		end
	end
	
	return x, y, false
end

# ╔═╡ eed136bc-924d-11eb-3e3a-374d21772e4b
u², v² = sinkhorn(null; verbose=true)

# ╔═╡ 7f38ceba-9253-11eb-000f-25045179e841
X̃ = (Diagonal(.√u²) * null * Diagonal(.√v²)); Σ̃ = X̃'*X̃ / size(X̃,1);

# ╔═╡ b14e4f0e-9253-11eb-171a-053dcc942240
let
	λ = eigvals(Σ̃)
	p = cdfplot(λ, xscale=:log10, linewidth=2, label="Empirical distribution")
	
	vline!([(1+sqrt(size(X̃,2)/size(X̃,1)))^2], linewidth=2, linestyle=:dashdot, label="MP maximum eigenvalue")
	title!("poisson noise")
	xaxis!("eigenvalue")
	yaxis!("CDF")
	p
end

# ╔═╡ cbcbb038-9255-11eb-0fc3-3ba4d95cee62
function generate_mean(null, rank)
	A = exp.(rand(Uniform(-1,2), (size(null,1),rank)))
	B = rand(Uniform(0,1), (rank,size(null,2)))

	return first.(rand.(Poisson.(A*B),1))
end

# ╔═╡ b8bb97aa-9256-11eb-253f-a16885888c5f
let
	X = generate_mean(null, 100) + null
	u², v² = sinkhorn(X; verbose=true)
	X̃ = (Diagonal(.√u²) * X * Diagonal(.√v²))
	Σ̃ = X̃'*X̃ / size(X̃,1);
	
	λ = eigvals(Σ̃)
	p = cdfplot(λ, xscale=:log10, linewidth=2, label="Empirical distribution")
	
	vline!([(1+sqrt(size(X̃,2)/size(X̃,1)))^2], linewidth=2, linestyle=:dashdot, label="MP maximum eigenvalue")
	title!("rank 100")
	xaxis!("eigenvalue")
	yaxis!("CDF")
	
	p
end

# ╔═╡ 5b25177c-925d-11eb-1aec-f52c8c52ec93
md"""
#### Beyond Poisson
At first glance this looks great. Unfortunately, we don't expect our count data to be Poisson distributed - scRNAseq data is generically overdispersed. Thus the relation that allowed us to equate the variance to the observed count matrix doesn't hold. 

For example, consider a negative binomial distribution (NB1) where 
``
\sigma^2 = \mu(1+\gamma)
``
In our previous notation,
``
\langle \mathcal{E}_{i\alpha}^2 \rangle = (1+\gamma_i) X_{i\alpha}
``
, where we have allowed for a gene-dependent overdispersion parameter. Working through the rescaling
``
\langle \tilde{\mathcal{E}}_{i\alpha}^2 \rangle = (1+\gamma_i) u^2_i X_{i\alpha} v^2_\alpha 
``
such that

``
G = \sum\limits_{i=1}^G  u^2_{i} (1+\gamma_i) X_{i\alpha} v^2_{\alpha} \  \forall \alpha \qquad \text{and} \qquad N = \sum\limits_{\alpha=1}^N u^2_{i} (1+\gamma_i) X_{i\alpha} v^2_{\alpha}  \  \forall i
``

This would require us to estimate the gene-specific overdispersion before performing further analyses. Similarly, for the NB2 parameterization, where
``
\sigma^2 = \mu(1+\gamma^{-1}\mu)
``, 
such that
``
\langle \mathcal{E}_{i\alpha}^2 \rangle = (1+\gamma_i^{-1}\langle X_{i\alpha} \rangle) \langle X_{i\alpha} \rangle
``
. Unbiased estimates of the variance give us

``
\langle \mathcal{E}_{i\alpha}^2 \rangle = \frac{X_{i\alpha}(X_{i\alpha} + \gamma_i)}{1+\gamma_i} \implies \langle \tilde{\mathcal{E}}_{i\alpha}^2 \rangle = u^2_{i} \frac{X_{i\alpha}(X_{i\alpha} + \gamma_i)}{1+\gamma_i} v^2_{\alpha}
``

In general, given an unbiased estimation for the variance we simply apply Sinkhorn-Knopp to obtain

``
G = \sum\limits_{i=1}^G  u^2_{i} \,\text{Var}[X]_{i\alpha} \, v^2_{\alpha} \  \forall \alpha \qquad \text{and} \qquad N = \sum\limits_{\alpha=1}^N u^2_{i} \, \text{Var}[X]_{i\alpha} \, v^2_{\alpha}  \  \forall i
``

A priori it's unclear if we could reliably detect $\gamma_i$ at the low counts we observe in-vivo

#### GLM formulation
scRNAseq data will have systematic variations in cell-depth simply as a technological consequence. Thus we must fit our distributions sensitive to this fact. The simple solution would be to divide all cells by their total sequencing depth. We opt for a similar, yet more flexible approach, namely we allow for the mean of the negative binomial to be dependent on the sequencing depth for each cell, $z_\alpha$. Specifically, we take the mean of the negative binomial distribution for gene $i$ and cell $\alpha$ to be

``
\text{log}(\mu_{i\alpha}) = \alpha_i + \beta_i * \text{log}(z_\alpha)
``

``\alpha_i`` and ``\beta_i``, along with the overdispersion parameter $\gamma_i$ define 3 parameters per gene we fit from raw count data using a Maximum Likelihood formalism
"""

# ╔═╡ c65e8a86-9259-11eb-29bb-3bf5c089746f
function generate_nb2(ngene, ncell; ρ=(α=Gamma(0.25,2), β=Normal(1,.01), γ=Gamma(3,3)))
    N = zeros(Int, ngene, ncell)
	
    z = log.(rand(Gamma(5,1), ncell))
    α = log.(rand(ρ.α, ngene))
    β = rand(ρ.β, ngene)
	γ = rand(ρ.γ, ngene)
	
    for g ∈ 1:ngene
        μ⃗ = exp.(α[g] .+ β[g].*z)
        for (c, μ) ∈ enumerate(μ⃗)
			λ = rand(Gamma(γ[g], μ/γ[g]),1)[1]
            N[g,c] = rand(Poisson(λ),1)[1]
        end
	end

	@show maximum(N, dims=2)
    ι = (vec(sum(N,dims=2)) .> 0) .& (vec(maximum(N,dims=2)) .> 1)

    N = N[ι, :]
    α = α[ι]
    β = β[ι]
	γ = γ[ι]
	
	return (
		data  = N,
		param = (
			α = α,
			β = β,
			γ = γ,
			z = z,
		),
	)
end

# ╔═╡ 5ebfd944-9262-11eb-3740-37ba8930e1c6
begin

function loss_nb1(x⃗, z⃗, β̄, δβ¯²)
	function f(Θ)
		α, β, γ = Θ
		
		Mu = (exp(+α + β*z) for z ∈ z⃗)
        S  = (loggamma(x+γ*μ) 
			- loggamma(x+1) 
			- loggamma(γ*μ) 
			+ x*log(γ) 
			- (x+γ*μ)*log(1+γ) for (x,μ) ∈ zip(x⃗,Mu))

        return -mean(S) + 0.5*δβ¯²*(β-β̄)^2
	end
	
	return f
end
	
function loss_nb2(x⃗, z⃗, β̄, δβ¯²)
	function f(Θ)
		α, β, γ = Θ
		
        S  = (loggamma(x+γ) 
			- loggamma(x+1) 
			- loggamma(γ) 
			+ x*(α+β*z)
			+ γ*log(γ)
			- (x+γ)*log(exp(α+β*z)+γ) for (x,z) ∈ zip(x⃗,z⃗))

        return -sum(S) + 0.5*δβ¯²*(β-β̄)^2
	end
	
	return f
end

function fit1(x, z, β̄, δβ¯²)
	μ  = mean(x)
	Θ₀ = [
		log(μ),
		β̄,
		var(x)/μ - 1,
	]
	
    if Θ₀[end] < 0 || isinf(Θ₀[end])
        Θ₀[end] = 1
	end
	
	loss = 	TwiceDifferentiable(
        loss_nb2(x, z, β̄, δβ¯²),
		Θ₀;
		autodiff=:forward
	)
	
	constraint = TwiceDifferentiableConstraints(
		[-∞, -∞, 0],
		[+∞, +∞, +∞],
	)

	soln = optimize(loss, constraint, Θ₀, IPNewton())
	
	Θ̂  = Optim.minimizer(soln)
	Ê  = Optim.minimum(soln)
	δΘ̂ = diag(inv(hessian!(loss, Θ̂)))
	
	# pearson residuals
	α̂, β̂, γ̂ = Θ̂
	μ̂ = exp.(α̂ .+ β̂*z)
	σ̂ = .√(μ̂ .+ μ̂.^2 ./ γ̂)
    ρ = (x .- μ̂) ./ σ̂

	return (
		parameters=Θ̂, 
		uncertainty=δΘ̂, 
		likelihood=Ê,
		trend=μ̂,
		residuals=ρ,
	)
end

function fit(data; β̄=1, δβ¯²=1e-2)
	z = log.(vec(mean(data, dims=1)))
	χ = log.(vec(mean(data, dims=2)))

    fits = [begin 
				@show i 
				fit1(vec(data[i,:]), z, β̄, δβ¯²) 
			end for i ∈ 1:size(data,1)]
	
	return vcat((fit.residuals' for fit ∈ fits)...),
        (
            likelihood  = map((f)->f.likelihood,  fits),

            α  = map((f)->f.parameters[1],  fits),
            β  = map((f)->f.parameters[2],  fits),
            γ  = map((f)->f.parameters[3],  fits),

            δα = map((f)->f.uncertainty[1], fits),
            δβ = map((f)->f.uncertainty[2], fits),
            δγ = map((f)->f.uncertainty[3], fits),

            μ̂ = map((f)->f.trend,  fits),
            χ = vec(mean(data, dims=2)),
            M = vec(maximum(data, dims=2))
        )
end
	
end

# ╔═╡ b563c86c-9264-11eb-01c2-bb42d74c3e69
E, p = generate_nb2(2000,1000);

# ╔═╡ 4f65301e-9186-11eb-1faa-71977e8fb097
let p
	p = cdfplot(
		vec(mean(null,dims=2)),
		xscale=:log10,
		legend=false,
		linewidth=3
	)
	xaxis!("mean expression/cell")
	yaxis!("CDF")
	p
end

# ╔═╡ c423eca6-9264-11eb-375f-953e02fc7ec4
Z, p̂ = fit(E; δβ¯²=10);

# ╔═╡ fb056230-9265-11eb-1a98-33c234a0f959
let
	p = scatter(p.α, p̂.α, alpha=0.1, marker_z=log10.(p̂.χ), label=false)
	xaxis!("true α")
	yaxis!("estimated α")
	p
end

# ╔═╡ c8c4e6b4-9266-11eb-05e6-917b82a580ab
let
	p = scatter(p.β, p̂.β, alpha=0.1, marker_z=log10.(p̂.χ), label=false)
	xaxis!("true β")
	yaxis!("estimated β")
	p
end

# ╔═╡ 622247ba-9268-11eb-3c0b-07f9cd3c6236
let
	p = scatter(p.γ, p̂.γ, xscale=:log10, yscale=:log10, marker_z=log10.(p̂.M), alpha=0.1, label=false)
	xaxis!("true γ")
	yaxis!("estimated γ")
	p
end

# ╔═╡ d9880e94-92f6-11eb-3f1e-cf4462c3b89a
let
	p = scatter(p̂.χ, p̂.γ, xscale=:log10, yscale=:log10, marker_z=log10.(p̂.M), alpha=0.1, label=false)
	xaxis!("expression per cell")
	yaxis!("estimated γ")
	vline!([1], label="cutoff", linestyle=:dashdot, linewidth=3)

	p
end

# ╔═╡ 17395b62-9272-11eb-0237-430f2e6499d6
let
	V = E.*(E.+p̂.γ) ./ (1 .+ p̂.γ)
	u², v² = sinkhorn(V; verbose=true)
	X̃ = (Diagonal(.√u²) * E * Diagonal(.√v²))
	Σ̃ = X̃'*X̃ / size(X̃,1);
	
	λ = eigvals(Σ̃)
	p = cdfplot(λ, xscale=:log10, linewidth=2, label="Empirical distribution")
	
	vline!([(1+sqrt(size(X̃,2)/size(X̃,1)))^2], linewidth=2, linestyle=:dashdot, label="MP maximum eigenvalue")
	title!("rank 0 (overdispersed)")
	xaxis!("eigenvalue")
	yaxis!("CDF")
	
	p
end

# ╔═╡ 3886d34c-9279-11eb-31e6-0fd49a2694aa
let
	E, p = generate_nb2(2000,1000);
	X 	 = generate_mean(E, 100) + E
	
	Z, p̂ = fit(X; δβ¯²=10);
	V 	 = X.*(X.+p̂.γ) ./ (1 .+ p̂.γ)

	u², v² = sinkhorn(V; verbose=true)

	X̃ = (Diagonal(.√u²) * X * Diagonal(.√v²))
	Ṽ = (Diagonal(u²) * V * Diagonal(v²))
	#Σ̃ = V'*V / size(X̃,1);
	
	λ = svdvals(X̃) #eigvals(Σ̃)
	p = cdfplot(λ, xscale=:log10, linewidth=2, label="Empirical distribution")
	k = sum(λ .>= (sqrt(size(X̃,1))+sqrt(size(X̃,2))-12))
	
	vline!([(sqrt(size(X̃,1))+sqrt(size(X̃,2))-12)], linewidth=2, linestyle=:dashdot, label="MP maximum eigenvalue")
	title!("rank ≈ $k (overdispersed)")
	xaxis!("singular value")
	yaxis!("CDF")
	
	p
end

# ╔═╡ 3d3a37d6-927d-11eb-16b3-e74d10851013
X = generate_mean(E, 100) + E;

# ╔═╡ 8472a6a6-927d-11eb-0992-897698a13544
Z₂, p̂₂ = fit(X; δβ¯²=10);

# ╔═╡ a82fe0da-926a-11eb-063b-f70bd587a789
R = kmeans(log.(hcat(p̂.χ, p̂.γ)'), 2);

# ╔═╡ a79aaeb0-926b-11eb-32ea-2bc1ead29909
function trendline(x, y; n = 10)
	l,r = log(minimum(x)), log(maximum(x))
    bp  = range(l, r, length=n+1)
	
    x₀ = Array{eltype(x),1}(undef, n)
    μ  = Array{eltype(y),1}(undef, n)
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

# ╔═╡ f53061cc-92f4-11eb-229b-b9b501c6cab8
md"""
## Real data

Let's try this procedure on a single run of scRNAseq on Drosophila. We first run our negative binomial fitting with no priors put on parameter estimates.
"""

# ╔═╡ f5ef8128-90bb-11eb-1f4b-053ed41f5038
begin 
	seq = scRNA.process(scRNA.load("$ROOT/rep$SAMPLE"));
	seq = scRNA.filtergene(seq) do gene, _
		sum(gene) >= 1e-2*length(gene) && length(unique(gene)) > 2
	end
end;

# ╔═╡ f6f99ff4-92f5-11eb-2577-e51ab1cacfa6
S, p̃ = fit(seq; δβ¯²=100);

# ╔═╡ 65224940-92f6-11eb-3045-4fc4b7b29a6c
let
	p = scatter(p̃.χ, p̃.α, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(p̃.M), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated α")
	p
end

# ╔═╡ 992dd34e-92f6-11eb-2642-bb9862148733
let
	p = scatter(p̃.χ, p̃.β, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(p̃.M), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated β")
	p
end

# ╔═╡ a453ad84-92f6-11eb-0350-d1fbfbbbfda0
let
	p = scatter(p̃.χ, p̃.γ, 
		alpha=0.1, 
		xscale=:log10,
		yscale=:log10,
		marker_z=log10.(p̃.M), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated γ")
	vline!([1], label="cutoff", linestyle=:dashdot, linewidth=3)
	p
end

# ╔═╡ f9d549ac-92f6-11eb-306a-77bf90c8eb33
md"""
We observe a pattern in fitting negative binomial distributions in both the synthetic data (where we know γ does not depend upon χ) and in the real data. Specifically, there exists a knee at around average expression of $1$ where γ becomes independent of expression. Below, there is a roughly linear decrease with a large spread.

Thus we make the ansatz that γ does not actually depend upon α (they are independent variables) but rather this is an artifact of sparse sampling. As such we find the distribution of γ from highly expressed genes and subsequently use our estimate as a prior in the inference of the distribution for the remaining genes to help constrain our inference.
"""

# ╔═╡ 94b6d52a-92f8-11eb-226f-27e28f4a4d4b
function fit_gamma(data)
	L = (Θ) -> let
		k, θ = Θ
		return -sum((k-1)*log.(data) .- (data./θ) .- loggamma(k) .- k*log(θ))
	end
	
	μ  = mean(data)
	σ² = var(data)
	Θ₀ = [
		μ^2/σ²,
		σ²/μ,
	]
	
	loss = TwiceDifferentiable(
        L,
		Θ₀;
		autodiff=:forward
	)
	
	constraint = TwiceDifferentiableConstraints(
		[0,  0],
		[+∞, +∞],
	)

	soln = optimize(loss, constraint, Θ₀, IPNewton())

	return Optim.minimizer(soln)
end

# ╔═╡ 4f96560a-92fd-11eb-19f7-7b11f0ee46bf
function fit_lognormal(data)
	L = (Θ) -> let
		μ, σ = Θ
		return sum(log(σ) .+ (log.(data) .- μ).^2 ./ (2*σ^2))
	end
	
	μ = mean(log.(data))
	σ = std(log.(data))
	Θ₀ = [
		μ,
		σ
	]
	
	loss = TwiceDifferentiable(
        L,
		Θ₀;
		autodiff=:forward
	)
	
	constraint = TwiceDifferentiableConstraints(
		[0,  0],
		[+∞, +∞],
	)

	soln = optimize(loss, constraint, Θ₀, IPNewton())

	return Optim.minimizer(soln)
end

# ╔═╡ 1f2bdf42-92ff-11eb-0e1f-e7c4e1e78058
function fit_gennormal(data)
	L = (Θ) -> let
		μ, σ, β = Θ
		return sum(loggamma(1/β) + log(σ) .+ (abs.(data .- μ)./ σ).^β .- log(β))
	end
	
	μ = mean(data)
	σ = std(data)
	Θ₀ = [
		μ,
		σ,
		2
	]
	
	loss = TwiceDifferentiable(
        L,
		Θ₀;
		autodiff=:forward
	)
	
	constraint = TwiceDifferentiableConstraints(
		[0,  0,  0],
		[+∞, +∞, +∞],
	)

	soln = optimize(loss, constraint, Θ₀, IPNewton())

	return Optim.minimizer(soln)
end

# ╔═╡ bc1990b6-92f7-11eb-0015-310aefc5041e
let
	rank(x) = invperm(sortperm(x))
	
 	Y = p̃.γ[p̃.χ .> 1.05]
	
	k̂, θ̂ = fit_gamma(Y)
	@show k̂, θ̂
	
	μ̂, σ̂ = fit_lognormal(Y)
	@show μ̂, σ̂
	
	ŷ, α̂, β̂ = fit_gennormal(log.(Y))
	@show ŷ, α̂, β̂

	GG(x)  = first(gamma_inc(k̂, (x/θ̂), 0))
	LN(x) = .5*(1+erf((log(x) .- μ̂) ./ (sqrt(2)*σ̂)))
	GN(x) = .5*(1+sign(log(x)-ŷ)*first(gamma_inc((1/β̂), abs(log(x)-ŷ)/(α̂^β̂), 0)))

	x = range(minimum(Y),maximum(Y),length=100)
	
	scatter(rank(Y)/length(Y),  GG.(Y), alpha=0.5, label="gamma distribution")
	scatter!(rank(Y)/length(Y), LN.(Y), alpha=0.5, label="log-normal distribution")
	scatter!(rank(Y)/length(Y), GN.(Y), alpha=0.5, label="log-generalized-normal distribution")

	plot!(0:1, 0:1, linewidth=2, linecolor=:black, linestyle=:dashdot, label="ideal", legend=:bottomright)
	
	xaxis!("Empirical CDF")
	yaxis!("Fit CDF")
end

# ╔═╡ 9f56e6fc-9303-11eb-2bdd-5da078b4d9c3
md"""
We try out three different distributions to see which best estimates the empirical distribution.
  * Gamma distribution appears too heavy right tailed.
  * Log-normal is a better approximation, however we see it is still too heavy tailed (lack of values in the bulk of the distribution implies we are chasing the tails)
  * Generalized log-normal (which tunes the contribution of the tails) appears adequate.
"""

# ╔═╡ 37afad28-9304-11eb-2f4d-8397fca7bb99
begin

function loss_nb2_constrained(x⃗, z⃗, β̄, δβ¯², σ̂, ν̂, μ̂)
	function f(Θ)
		α, β, γ = Θ
		
        S  = (loggamma(x+γ) 
			- loggamma(x+1) 
			- loggamma(γ) 
			+ x*(α+β*z)
			+ γ*log(γ)
			- (x+γ)*log(exp(α+β*z)+γ) for (x,z) ∈ zip(x⃗,z⃗))

        return -sum(S) + 0.5*δβ¯²*(β-β̄)^2 + (abs.(log(γ)-μ̂)/σ̂)^ν̂
	end
	
	return f
end

function fit1_constrained(x, z, β̄, δβ¯², σ̂, ν̂, μ̂)
	μ  = mean(x)
	Θ₀ = [
		log(μ),
		β̄,
		var(x)/μ - 1,
	]
	
    if Θ₀[end] < 0 || isinf(Θ₀[end])
        Θ₀[end] = 1
	end
	
	loss = 	TwiceDifferentiable(
        loss_nb2_constrained(x, z, β̄, δβ¯², σ̂, ν̂, μ̂),
		Θ₀;
		autodiff=:forward
	)
	
	constraint = TwiceDifferentiableConstraints(
		[-∞, -∞, 0],
		[+∞, +∞, +∞],
	)

	soln = optimize(loss, constraint, Θ₀, IPNewton())
	
	Θ̂  = Optim.minimizer(soln)
	Ê  = Optim.minimum(soln)
	δΘ̂ = diag(inv(hessian!(loss, Θ̂)))
	
	# pearson residuals
	α̂, β̂, γ̂ = Θ̂
	μ̂ = exp.(α̂ .+ β̂*z)
	σ̂ = .√(μ̂ .+ μ̂.^2 ./ γ̂)
    ρ = (x .- μ̂) ./ σ̂

	return (
		parameters=Θ̂, 
		uncertainty=δΘ̂, 
		likelihood=Ê,
		trend=μ̂,
		residuals=ρ,
	)
end

function fit_constrained(data; β̄=1, δβ¯²=1e-2, σ̂, ν̂, μ̂)
	z = log.(vec(mean(data, dims=1)))
	χ = log.(vec(mean(data, dims=2)))

    fits = [begin 
				@show i 
				fit1_constrained(vec(data[i,:]), z, β̄, δβ¯², σ̂, ν̂, μ̂) 
			end for i ∈ 1:size(data,1)]
	
	return vcat((fit.residuals' for fit ∈ fits)...),
        (
            likelihood  = map((f)->f.likelihood,  fits),

            α  = map((f)->f.parameters[1],  fits),
            β  = map((f)->f.parameters[2],  fits),
            γ  = map((f)->f.parameters[3],  fits),

            δα = map((f)->f.uncertainty[1], fits),
            δβ = map((f)->f.uncertainty[2], fits),
            δγ = map((f)->f.uncertainty[3], fits),

            μ̂ = map((f)->f.trend,  fits),
            χ = vec(mean(data, dims=2)),
            M = vec(maximum(data, dims=2))
        )
end
	
end

# ╔═╡ d2e5b674-9305-11eb-1254-5b9fa030e8ee
Sᵪ, p̃ᵪ = let
	Y = p̃.γ[p̃.χ .> 1.05]
	μ, σ, ν = fit_gennormal(log.(Y))
	fit_constrained(seq; δβ¯²=100, μ̂=μ, σ̂=σ, ν̂=ν);
end;

# ╔═╡ 70fc4538-9306-11eb-1509-3dfa1034d8aa
let
	p = scatter(p̃ᵪ.χ, p̃ᵪ.α, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(p̃ᵪ.δα), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated α")
	p
end

# ╔═╡ 861ca2b6-9306-11eb-311c-69c6984faa26
let
	p = scatter(p̃ᵪ.χ, p̃ᵪ.β, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(p̃ᵪ.δβ./p̃ᵪ.β), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated β")
	p
end

# ╔═╡ 97ded136-9306-11eb-019b-636b739a61d6
let
	p = scatter(p̃ᵪ.χ, p̃ᵪ.γ, 
		alpha=0.1, 
		xscale=:log10,
		yscale=:log10,
		marker_z=log10.(p̃ᵪ.δγ./p̃ᵪ.γ), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated γ")
	vline!([1], label="cutoff", linestyle=:dashdot, linewidth=3)
	p
end

# ╔═╡ e05d2a98-9306-11eb-252e-5baf6fc59f8f
let 
	V = seq.*(seq.+p̃ᵪ.γ) ./ (1 .+ p̃ᵪ.γ)
	u², v² = sinkhorn(V; verbose=true)

	X̃ = (Diagonal(.√u²) * seq * Diagonal(.√v²))
	Ṽ = (Diagonal(u²) * V * Diagonal(v²))
	
	λ = svdvals(X̃)
	p = cdfplot(λ, xscale=:log10, linewidth=2, label="Empirical distribution")
	
	ϵ = 0
	k = sum(λ .> (sqrt(size(X̃,1))+sqrt(size(X̃,2))) - ϵ)
	
	vline!([(sqrt(size(X̃,1))+sqrt(size(X̃,2))-ϵ)], linewidth=2, linestyle=:dashdot, label="MP maximum eigenvalue")
	title!("scRNAseq data (k=$k)")
	xaxis!("singular value")
	yaxis!("CDF")
end

# ╔═╡ bdffde54-9307-11eb-2ccb-ed48777f28f8
md"""
Remarkably this works beautifully! We recover similar (but smaller as expected) estimates for the true rank of the matrix. Furthermore, we have estimated a scaling of our matrix such that every row and column sum have unit variance. We treat this as our processed matrix going forward.

##### Idea:
One could imagine futher iterating on the estimate for the γ prior as we do see some expression dependence in the inferred values even though the prior does not. This would be tantamount to repeating the same procedure in bins and then interpolating the fit values.

But for now we consider this a success. We should throw out genes with excessive uncertainty in $γ$.
"""

# ╔═╡ 91742bde-9309-11eb-2acc-836cc1ab1aee
S̃, Ṽ = let 
	V = seq.*(seq.+p̃ᵪ.γ) ./ (1 .+ p̃ᵪ.γ)
	u², v² = sinkhorn(V; verbose=true)

	(Diagonal(.√u²) * seq * Diagonal(.√v²)), (Diagonal(u²) * V * Diagonal(v²))
end

# ╔═╡ df6e73de-9309-11eb-3bd3-3f9f511744cf
md"""
## Data imputation
Can we utilize the large number of genes to help "impute" the dropout? We tread lightly here, any averaging over cells will introduce non-trivial correlations in the data.

The basic idea is to compute a distance matrix between cells $D_{\alpha\beta}$. You can then use this to define a Gaussian kernel $K_{\alpha\beta} \sim e^{-D^2_{\alpha\beta}}$ where $K$ is assumed to be suitably normalized. Data is then "imputed" by averaging, i.e. a given count matrix $X_{i\alpha}$ is averaged $\tilde{X}_{i\alpha} = X_{i\alpha} K^t_{\alpha\beta}$ where $t$ is the diffusion time.

This assumes there is a low-dimensional manifold on which our data lives. The above considerations do imply this. Thus the thought experiment is as follows: if we smooth our data and then fit our distributions, can we improve our fits of the underlying distributions, i.e. deal with our sparse sampling problem?

Let's take our scaled matrix (of unit variance) and see what we can find. In all distance measures tested, we only utilize them within a small neighborhood - geodesic distances are used outside.
"""

# ╔═╡ 071f2c26-930c-11eb-2745-93cb2001e76b
PointCloud = ingredients("../src/geo.jl").PointCloud

# ╔═╡ c8227e0e-9326-11eb-2724-cb588170c7c2
Inference = ingredients("../src/infer.jl").Inference

# ╔═╡ 8ffaa54e-930b-11eb-2f1f-9907008b76d2
md"""
#### Euclidean
"""

# ╔═╡ dd9986f4-930f-11eb-33b8-67dbc4d1d087
S̃ₐ = let d = 35
	F = svd(S̃);
	F.U[:,1:d]*Diagonal(F.S[1:d])*F.Vt[1:d,:]
end;

# ╔═╡ 3c979aa0-930c-11eb-3e6e-9bdf7f3029b5
Gₑ = PointCloud.geodesics(S̃ₐ, 6); size(Gₑ)

# ╔═╡ ed6333c6-930c-11eb-25c0-c551592746e0
let
	ρ, Rs = PointCloud.scaling(Gₑ, 1000)

	p = plot(Rs, ρ', alpha=0.03, color=:cyan3, label="")
	plot!(Rs, mean(ρ, dims=1)', linewidth=5, alpha=1, color=:cyan3, label="mean")

	xaxis!("radius", :log10)
	yaxis!("number of points", :log10)

	plot!(Rs, 5e-4*Rs.^3, linestyle=:dashdot, linewidth=2, alpha=1, color=:coral3, label="αR³", legend=:bottomright)

	xaxis!("radius", :log10, (20, 400))
	yaxis!("number of points", :log10, (6, 1200))

	p
end

# ╔═╡ 76b9c454-9316-11eb-1548-bdef70d532d6
md"""
#### Aside:
Embedding estimate for points
"""

# ╔═╡ 585bed74-9319-11eb-2854-9d74e1d592c2
GENE = findfirst(seq.gene .== "sna")

# ╔═╡ 91147112-9315-11eb-0a71-8ddfdb2497a6
Page()

# ╔═╡ e96bc054-9310-11eb-25ca-576caeb336b8
Kₑ = let
	σ = .15*mean(Gₑ[:])
	K = exp.(-(Gₑ./σ).^2)
	u, v = sinkhorn(K; r=1, c=1)
	Diagonal(u) * K * Diagonal(v)
end;

# ╔═╡ f7d6acc2-9314-11eb-03f3-f9f221ded27c
let
	ξ = PointCloud.mds(Gₑ.^2, 3)
	WGLMakie.scatter(ξ[:,1], ξ[:,2], ξ[:,3], color=Kₑ*Sᵪ[GENE,:], markersize=4000)
end

# ╔═╡ 7a3b0202-9311-11eb-101e-99fbc40c6633
let
	S = seq.data ./ sum(seq.data,dims=1)
	Ñ = (S*Kₑ) .* sum(seq.data,dims=1)
	c = [ cor(vec(Ñ[i,:]), vec(seq[i,:])) for i in 1:size(Ñ,1) ]

	cdfplot(c, linewidth=2, label="")
	
	xaxis!("correlation before/after smoothing")
	yaxis!("CDF")
end

# ╔═╡ c9496d2a-931b-11eb-0228-5f08b4e1ff9f
md"""
##### Normalizing imputed values
Interestingly, when we smooth, we lose our "discrete" count values thus a negative binomial is no longer applicable. Let's try to fit Gamma distributions to our gene distributions and see how we do
"""

# ╔═╡ e75442f4-931b-11eb-0a5f-91e8c58ab47e
begin
	function clamp(value, lo, hi)
		if value < lo
			return lo
		elseif value > hi
			return hi
		else
			return value
		end
	end
	
	function gamma_loss(x⃗, z⃗, β̄, δβ¯²)
		function f(Θ)
			α, β, γ = Θ

			M = (exp(α+β*z) for z ∈ z⃗)
			Z = (loggamma(μ/γ)+(μ/γ)*log(γ) for μ ∈ M)  

			return -sum(-z + (μ/γ-1)*log(x)-x/γ for (z,μ,x) ∈ zip(Z,M,x⃗)) + 0.5*δβ¯²*(β-β̄)^2
		end
		
		return f
	end
	
	function fit_continuous1(x, z, β̄, δβ¯²)
		μ  = mean(x)
		Θ₀ = [
			log(μ),
			β̄,
			μ^2 / (var(x)-μ),
		]

		if Θ₀[end] < 0 || isinf(Θ₀[end])
			Θ₀[end] = 1
		end

		loss = 	TwiceDifferentiable(
			gamma_loss(x, z, β̄, δβ¯²),
			Θ₀;
			autodiff=:forward
		)

		constraint = TwiceDifferentiableConstraints(
			[-∞, -∞, 0],
			[+∞, +∞, +∞],
		)

		soln = optimize(loss, constraint, Θ₀, IPNewton())

		Θ̂  = Optim.minimizer(soln)
		Ê  = Optim.minimum(soln)
		δΘ̂ = diag(inv(hessian!(loss, Θ̂)))

		# pearson residuals
		α̂, β̂, γ̂ = Θ̂
		μ̂ = exp.(α̂ .+ β̂*z)
		σ̂ = .√(μ̂ .+ μ̂.^2 ./ γ̂)

		# cdf
		k   = μ̂ ./ γ̂
		cdf = GSL.sf_gamma_inc_P.(k, x./γ̂)

		# gaussian residuals
		ρ = erfinv.(clamp.(2 .*cdf .- 1,-1,1))
		ρ[isinf.(ρ)] = 10*sign.(ρ[isinf.(ρ)])

		return (
			parameters=Θ̂, 
			uncertainty=δΘ̂, 
			likelihood=Ê,
			trend=μ̂,
			cdf=cdf,
			residuals=ρ,
		)
	end
	
	function fit_continuous(data; β̄=1e0, δβ¯²=1e1)
		z = log.(vec(mean(data, dims=1)))
		χ = log.(vec(mean(data, dims=2)))

		fits = [begin 
					@show i 
					fit_continuous1(vec(data[i,:]), z, β̄, δβ¯²) 
				end for i ∈ 1:size(data,1)]

		return vcat((fit.residuals' for fit ∈ fits)...),
			(
				likelihood  = map((f)->f.likelihood,  fits),

				α  = map((f)->f.parameters[1],  fits),
				β  = map((f)->f.parameters[2],  fits),
				γ  = map((f)->f.parameters[3],  fits),

				δα = map((f)->f.uncertainty[1], fits),
				δβ = map((f)->f.uncertainty[2], fits),
				δγ = map((f)->f.uncertainty[3], fits),

				μ̂   = map((f)->f.trend,  fits),
				cdf = map((f)->f.cdf,  fits),
			
				χ = vec(mean(data, dims=2)),
				M = vec(maximum(data, dims=2))
			)
		end
end

# ╔═╡ 94fa4c96-931c-11eb-1a1e-556bb10223f5
Sᵧ, pᵧ = let
	S = seq.data ./ sum(seq.data,dims=1)
	Ñ = (S*Kₑ) .* sum(seq.data,dims=1)
	fit_continuous(Ñ)
end

# ╔═╡ 4ff1b8b4-9321-11eb-249b-1f35bc1facce
let
	p = scatter(pᵧ.χ, pᵧ.α, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(pᵧ.δα), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated α")
	p
end

# ╔═╡ 5a9982a6-9321-11eb-37a5-0be0a5b05d42
let
	p = scatter(pᵧ.χ, pᵧ.β, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(pᵧ.δβ), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated β")
	p
end

# ╔═╡ 7220bdb0-9321-11eb-2a89-79bcaa28726a
let
	p = scatter(pᵧ.χ, pᵧ.γ, 
		alpha=0.1, 
		xscale=:log10,
		yscale=:log10, 
		marker_z=log10.(pᵧ.δγ), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated γ")
	p
end

# ╔═╡ 87d39e64-9321-11eb-0170-834e21b45cc4
let
	cm = ColorSchemes.inferno
	χ  = log10.(pᵧ.χ)
	χ  = (χ .- minimum(χ)) ./ (maximum(χ) - minimum(χ))
	
	p = cdfplot(pᵧ.cdf[1], color=get(cm,χ[1]), alpha=0.01, label="")
	for i ∈ 2:5:length(pᵧ.cdf)
		cdfplot!(pᵧ.cdf[i], color=get(cm,χ[i]), alpha=0.01, label="")
	end
	
	plot!(0:1, 0:1, linestyle=:dashdot, color=:coral2, label="ideal", legend=:bottomright, linewidth=2)
	p
end

# ╔═╡ bb4aac90-9323-11eb-3562-afdd70610e24
let
	F = svd(Sᵧ)
	cdfplot(F.S, linewidth=2, xscale=:log10, label="empirical")
	vline!([sqrt(size(Sᵧ,1)) + sqrt(size(Sᵧ,2))], linestyle=:dashdot, linewidth=2, label="MP maximum")
	
	k = sum(F.S .> (sqrt(size(Sᵧ,1)) + sqrt(size(Sᵧ,2))))

	xaxis!("singular values")
	yaxis!("CDF")
	title!("rank ≈ $k")
end

# ╔═╡ 45ada9de-9325-11eb-1450-793727639203
S̃ᵧ = let d = 40
	F = svd(Sᵧ);
	F.U[:,1:d]*Diagonal(F.S[1:d])*F.Vt[1:d,:]
end;

# ╔═╡ 5c553b5e-9325-11eb-3f4b-e12c3c1c743b
Gᵧ = PointCloud.geodesics(S̃ᵧ, 12); size(Gᵧ)

# ╔═╡ 14462c08-933c-11eb-3369-3d081358bb35
import PyPlot

# ╔═╡ 17e1e152-933c-11eb-335b-6f2b072168ce
PyPlot.clf(); PyPlot.matshow(Gᵧ); PyPlot.gcf()

# ╔═╡ 58dd7382-9326-11eb-09d6-15251ddbb0bd
let
	ρ, Rs = PointCloud.scaling(Gᵧ, 1000)

	p = plot(Rs, ρ', alpha=0.03, color=:cyan3, label="")
	plot!(Rs, mean(ρ, dims=1)', linewidth=5, alpha=1, color=:cyan3, label="mean")

	xaxis!("radius", :log10)
	yaxis!("number of points", :log10)

	plot!(Rs, 4e-3*Rs.^3, linestyle=:dashdot, linewidth=2, alpha=1, color=:coral3, label="αR³", legend=:bottomright)
	plot!(Rs, 4e-2*Rs.^2, linestyle=:dashdot, linewidth=2, alpha=1, color=:coral3, label="αR²", legend=:bottomright)

	p
end

# ╔═╡ c31f47ee-9334-11eb-0fa1-1be19bf991ce
ν, ω = load("params.jld2", "ν", "ω")

# ╔═╡ d7e8e7a6-9326-11eb-0a3d-e918e72dac08
ψ, embryo, db, Φ = Inference.inversion(Sᵧ, seq.gene; ν=ν, ω=ω);

# ╔═╡ b2104258-9327-11eb-2151-69959f436b69
Ψᵣ = ψ(0.1)*size(Sᵧ,2);

# ╔═╡ 5b7847d6-9329-11eb-0269-dbbcf2d1e563
md"""
##### AP Position
"""

# ╔═╡ 18b52476-9326-11eb-26ac-8ff0d7afe21e
let
	ξ = PointCloud.mds(Gᵧ.^2, 3)
	AP = embryo[:,1]
	WGLMakie.scatter(Ψᵣ*ξ[:,1], Ψᵣ*ξ[:,2], Ψᵣ*ξ[:,3], color=AP, markersize=100)
end

# ╔═╡ 66700b9c-9329-11eb-1cde-87750e3652af
md"""
##### DV Position
"""

# ╔═╡ 1fcae9d8-9328-11eb-13fb-6935aaa65434
let
	ξ  = PointCloud.mds(Gᵧ.^2, 3)
	DV = embryo[:,2]#atan.(embryo[:,2],embryo[:,3])

	WGLMakie.scatter(Ψᵣ*ξ[:,1], Ψᵣ*ξ[:,2], Ψᵣ*ξ[:,3], color=DV, markersize=100)
end

# ╔═╡ f97f0ac6-9335-11eb-3051-336f0f48781f
ξ = PointCloud.mds(Gᵧ.^2, 3);

# ╔═╡ be47ae4c-9335-11eb-1deb-1732ce57ed4b
seq.gene[sortperm([cor(vec(Sᵧ[i,:]), ξ[:,1]) for i in 1:size(seq,1)])][1:50]

# ╔═╡ 0838eadc-9336-11eb-0079-23c7230a7bbc
seq.gene[sortperm([cor(vec(Sᵧ[i,:]), ξ[:,1]) for i in 1:size(seq,1)])][end-50:end]

# ╔═╡ 1a3fbf94-9336-11eb-2e48-35f09a2f14d5
seq.gene[sortperm([cor(vec(Sᵧ[i,:]), ξ[:,2]) for i in 1:size(seq,1)])][1:50]

# ╔═╡ 14aa52a6-9336-11eb-1a3d-a97d2d2dbe07
seq.gene[sortperm([cor(vec(Sᵧ[i,:]), ξ[:,2]) for i in 1:size(seq,1)])][end-50:end]

# ╔═╡ 55715968-9336-11eb-1845-a583df1166f6
seq.gene[sortperm([cor(vec(Sᵧ[i,:]), ξ[:,3]) for i in 1:size(seq,1)])][1:50]

# ╔═╡ 58255396-9336-11eb-23e5-67d8002b5595
seq.gene[sortperm([cor(vec(Sᵧ[i,:]), ξ[:,3]) for i in 1:size(seq,1)])][end-50:end]

# ╔═╡ a958c94a-932d-11eb-247c-1b44b35cf02f
GENE2 = findfirst(seq.gene .== "eve")

# ╔═╡ 497aee7c-932d-11eb-07fe-3b41d04c90f8
let
	WGLMakie.scatter(-embryo[:,1], embryo[:,2], embryo[:,3], color=Ψᵣ*Sᵧ[GENE2,:], markersize=2000)
end

# ╔═╡ 4fcb204a-933b-11eb-3797-571078415e34
let
	WGLMakie.scatter(-embryo[:,1], embryo[:,2], embryo[:,3], color=vec(Ψᵣ[:,1100]), markersize=2000)
end

# ╔═╡ d2339cb4-932a-11eb-293f-d35764e7b8f7
md"""
#### Conclusion
I think this works great. The pipeline in broad strokes is:
  1. Fit highly expressed genes to negative binomial.
  2. Use estimates to obtain prior for γ.
  3. Fit all genes to negative binomial with prior from before. 
  4. Use overdispersion factor γ in the variance rescaling equation to set all row and column variances to 1. 
  5. Find the gap in the spectrum, i.e. the rank of the original count matrix.
  6. Use the obtained low-dimensional manifold to estimate intercellular distances.
  7. Use distances in Gaussian kernel to lightly smooth the original raw data. 
  8. Fit smoothed data to Γ distribution.
  9. Transform into standard normal variables.
"""

# ╔═╡ 8e009626-9316-11eb-039e-fb1183e60421
md"""
#### Correlation
"""

# ╔═╡ 93f9188a-9316-11eb-1c35-2d93a1936277
Dₚ = let S = S̃ₐ
	D = zeros(size(S,2), size(S,2))
	for c₁ ∈ 1:size(S,2)
		for c₂ ∈ (c₁+1):size(S,2)
			D[c₁,c₂] = D[c₂, c₁] = 1 - cor(S[:,c₁], S[:,c₂])
		end
	end
	D
end;

# ╔═╡ 59d58e3c-9317-11eb-18c2-a54f3bfcb976
Gₚ = PointCloud.geodesics(S̃ₐ, 6; D=Dₚ); size(Gₚ)

# ╔═╡ f05f1616-9317-11eb-1a84-1b60e7dae579
let
	ρ, Rs = PointCloud.scaling(Gₚ, 1000)

	p = plot(Rs, ρ', alpha=0.03, color=:cyan3, label="")
	plot!(Rs, mean(ρ, dims=1)', linewidth=5, alpha=1, color=:cyan3, label="mean")

	xaxis!("radius", :log10)
	yaxis!("number of points", :log10)

	plot!(Rs, 1e4*Rs.^3, linestyle=:dashdot, linewidth=2, alpha=1, color=:coral3, label="αR³", legend=:bottomright)

	p
end

# ╔═╡ 5271d60c-931a-11eb-1bbb-c1bddc042136
Kₚ = let
	σ = .05*mean(Gₚ[:])
	K = exp.(-(Gₚ./σ).^2)
	u, v = sinkhorn(K; r=1, c=1)
	Diagonal(u) * K * Diagonal(v)
end;

# ╔═╡ 4fd9ff7c-9318-11eb-197b-9f916f33f983
let
	ξ = PointCloud.mds(Gₚ.^2, 3)
	WGLMakie.scatter(ξ[:,1], ξ[:,2], ξ[:,3], color=Kₚ*Sᵪ[GENE,:], markersize=10)
end

# ╔═╡ 83fe3a24-931a-11eb-0d09-e3eed2b09138
let
	Ñ = seq.data*Kₚ
	c = [ cor(vec(Ñ[i,:]), vec(seq[i,:])) for i in 1:size(Ñ,1) ]
	
	cdfplot(c, linewidth=2, label="")

	xaxis!("correlation before/after smoothing")
	yaxis!("CDF")
end

# ╔═╡ b77cf672-92f4-11eb-0301-8d060297d6f3
md"""
#### Wasserstein (prototype)
"""

# ╔═╡ f79db462-9283-11eb-1d3b-c19f6d9dee03
function cost_matrix(data)
	C = zeros(size(data,1), size(data,1))
	cos(x,y) = x⋅y / (norm(x)*norm(y))
	for i ∈ 1:size(data,1)
		@show i
		for j ∈ (i+1):size(data,1)
			C[i,j] = C[j,i] = 1 - cos(vec(data[i,:]),vec(data[j,:]))
		end
	end
	
	return C
end

# ╔═╡ 2d03d7b4-927f-11eb-0e2c-3b8ea5c52055
function pairwise_dists(data, cost; β=.1)
	D = zeros(size(data,2), size(data,2))
	
	# initialize cost matrix	
	H = exp.(β*cost .- 1)
	
	# initialize compute kernel
	kernel = (a, b) -> begin
		u, v = sinkhorn(H; r=a, c=b)
		p = Diagonal(u)*H*Diagonal(v)
		
		return sum(cost.*p) - sum(p.*log(p))/β
	end
	
	# compute
	Threads.@threads for i ∈ 1:size(data,2)
		@show i
		for j ∈ i:size(data,2)
			@show j
			D[i,j] = D[j,i] = kernel(data[:,i], data[:,j])
		end
	end
	
	D = D - .5*(diagonal(D) .+ diagonal(D)')
	
	return D
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
GSL = "92c85e6c-cbff-5e0c-80f7-495c94daaecd"
GZip = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
JSServe = "824d6782-a2ef-11e9-3a09-e5662e0c26f9"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Match = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
NLSolversBase = "d41bc354-129a-5804-8e4c-c37616107c6c"
NMF = "6ef6ca0d-6ad7-5ff6-b225-e928bfa0a386"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
ProgressMeter = "92933f4c-e287-5a05-a399-4b506db050ca"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"

[compat]
Clustering = "~0.15.8"
ColorSchemes = "~3.29.0"
Distributions = "~0.25.119"
FileIO = "~1.17.0"
ForwardDiff = "~1.0.1"
GSL = "~1.0.1"
GZip = "~0.6.2"
Interpolations = "~0.15.1"
JLD2 = "~0.5.13"
JSServe = "~2.3.1"
Match = "~2.1.0"
NLSolversBase = "~7.9.1"
NMF = "~1.0.3"
Optim = "~1.12.0"
Plots = "~1.40.13"
ProgressMeter = "~1.10.4"
PyPlot = "~2.11.6"
SpecialFunctions = "~2.5.1"
StatsBase = "~0.34.4"
WGLMakie = "~0.11.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "5ca83d4e2c5ef14dd5421e28a6727779cd8502f0"

[[deps.ADTypes]]
git-tree-sha1 = "e2478490447631aedba0823d4d7a80b2cc8cdb32"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.14.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bonito]]
deps = ["Base64", "CodecZlib", "Colors", "Dates", "Deno_jll", "HTTP", "Hyperscript", "LinearAlgebra", "Markdown", "MsgPack", "Observables", "RelocatableFolders", "SHA", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "e48e53213512466cebc99c267e275238aaabad6a"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "4.0.3"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2ac646d71d0d24b44f3f8c84da8c9f4d70fb67df"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.4+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "e771a63cc8b539eca78c85b0cabd9233d6c8f06f"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "b19db3927f0db4151cb86d073689f2428e524576"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "5620ff4ee0084a6ab7097a27ba0c19290200b037"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.4"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Deno_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd6756e833c377e0ce9cd63fb97689a255f12323"
uuid = "04572ae6-984a-583e-9378-9577a1c2574d"
version = "1.33.4+0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "aa87a743e3778d35a950b76fbd2ae64f810a2bb3"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.6.52"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "6d8b535fd38293bc54b88455465a1386f8ac1c3c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.119"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "e7b7e6f178525d17c720ab9c081e4ef04429f860"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.4"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.Extents]]
git-tree-sha1 = "063512a13dbe9c40d999c439268539aa552d1ae6"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.5"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "7de7c78d681078f027389e067864a8d53bd7c3c9"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "f089ab1f834470c525562030c8cfde4025d5e915"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.27.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "910febccb28d493032495b7009dce7d7f7aee554"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.0.1"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "d52e255138ac21be31fa633200b65e4e71d26802"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.6"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "7ffa4049937aeba2e5e1242274dc052b0362157a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.14"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "98fc192b4e4b938775ecd276ce88f539bcec358e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.14+0"

[[deps.GSL]]
deps = ["GSL_jll", "Libdl", "Markdown"]
git-tree-sha1 = "3ebd07d519f5ec318d5bc1b4971e2472e14bd1f0"
uuid = "92c85e6c-cbff-5e0c-80f7-495c94daaecd"
version = "1.0.1"

[[deps.GSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "43608dae16c5c9a77c93fcf6f1d4ea7657300e96"
uuid = "1b77fbbe-d8ee-58f0-85f9-836ddc23a7a4"
version = "2.8.0+0"

[[deps.GZip]]
deps = ["Libdl", "Zlib_jll"]
git-tree-sha1 = "0085ccd5ec327c077ec5b91a5f937b759810ba62"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.6.2"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "8e233d5167e63d708d41f87597433f59a0f213fe"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.4"

[[deps.GeoInterface]]
deps = ["DataAPI", "Extents", "GeoFormatTypes"]
git-tree-sha1 = "294e99f19869d0b0cb71aef92f19d03649d028d5"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.4.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "65e3f5c519c3ec6a4c59f4c3ba21b6ff3add95b0"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.7"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "dc6bed05c15523624909b3953686c5f5ffa10adc"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "f93655dc73d7a0b4a368e3c0bce296ae035ad76e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.16"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "LinearAlgebra", "MacroTools", "OpenBLASConsistentFPCSR_jll", "RoundingEmulator"]
git-tree-sha1 = "2c337f943879911c74bb62c927b65b9546552316"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.29"
weakdeps = ["DiffRules", "ForwardDiff", "IntervalSets", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "8e071648610caa2d3a5351aba03a936a0c37ec61"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.13"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSServe]]
deps = ["Base64", "CodecZlib", "Colors", "Dates", "Deno_jll", "HTTP", "Hyperscript", "LinearAlgebra", "Markdown", "MsgPack", "Observables", "RelocatableFolders", "SHA", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "4bcf2a78f7c80c6f3d594267bb4e7ec03ac9c172"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f9"
version = "2.3.1"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd10d2cc78d34c0e2a3a36420ab607b611debfbb"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.7"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "0318d174aa9ec593ddf6dc340b434657a8f1e068"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.22.4"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "903ef1d9d326ebc4a9e6cf24f22194d8da022b50"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.9.2"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Match]]
deps = ["MacroTools", "OrderedCollections"]
git-tree-sha1 = "5ac5e5267e17ccbd717bc7caaa57c5a20b80261d"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "2.1.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "f5a6805fb46c0285991009b526ec6fae43c6dec2"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.3"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "b14c7be6046e7d48e9063a0053f95ee0fc954176"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.9.1"

[[deps.NMF]]
deps = ["LinearAlgebra", "NonNegLeastSquares", "PrecompileTools", "Printf", "Random", "RandomizedLinAlg", "Statistics", "StatsBase"]
git-tree-sha1 = "d098a41cbc60447b430c32e1ac271193652205ee"
uuid = "6ef6ca0d-6ad7-5ff6-b225-e928bfa0a386"
version = "1.0.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "8a3271d8309285f4db73b4f662b1b290c715e85e"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.21"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.NonNegLeastSquares]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "cdc11138e74a0dd0b82e7d64eb1350fdf049d3b1"
uuid = "b7351bd1-99d9-5c5d-8786-f205a815c4d7"
version = "0.4.1"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "567515ca155d0020a45b05175449b499c63e7015"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.29+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9216a80ff3682833ac4b733caa8c00390620ba5d"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.0+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "EnumX", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "31b3b1b8e83ef9f1d50d74f1dd5f19a37a304a1f"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.12.0"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "0e1340b5d98971513bddaa6bbed470670cebbbfe"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.34"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "cf181f0b1e6a18dfeb0ee8acc4a9d1672499626c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.4"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "44f6c1f38f77cafef9450ff93946c53bd9ca16ff"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "809ba625a00c605f8d00cd2a9ae19ce34fc24d68"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.13"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "13c5103482a8ed1536a54c08d0e742ae3dca2d42"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.4"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "d2c2b8627bbada1ba00af2951946fb8ce6012c05"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.11.6"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RandomizedLinAlg]]
deps = ["LinearAlgebra", "Random", "Test"]
git-tree-sha1 = "de93780c85c207586369522af7525aa3011b09c3"
uuid = "0448d7d9-159c-5637-8537-fd72090fea46"
version = "0.1.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "fea870727142270bdf7624ad675901a1ee3b4c87"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.1"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "35b09e80be285516e52c9054792c884b9216ae3c"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.4.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "8ad2e38cbb812e29348719cc63580ec1dfeb9de4"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.1"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "f21231b166166bebc73b99cea236071eb047525b"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.3"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "cbbebadbcc76c5ca1cc4b4f3b0614b3e603b5000"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.WGLMakie]]
deps = ["Bonito", "Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "LinearAlgebra", "Makie", "Observables", "PNGFiles", "PrecompileTools", "RelocatableFolders", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "01c9f3f96844cd3bd0e6b81921b2d36d34855dc5"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.11.4"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "068dfe202b0a05b8332f1e8e6b4080684b9c7700"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.47+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "d2408cac540942921e7bd77272c32e58c33d8a77"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.5.0+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╠═2466c2a4-90c7-11eb-3f9c-5b87a7a35bb6
# ╟─fc2b03f0-924b-11eb-0f20-45edefca4b76
# ╠═969b3f50-90bb-11eb-2b67-c784d20c0eb2
# ╠═be981c3a-90bb-11eb-3216-6bed955446f5
# ╠═e84220a8-90bb-11eb-2fb8-adb6c87c2faa
# ╠═ce78d71a-917a-11eb-3cdd-b15aad75d147
# ╟─0bb97860-917f-11eb-3dd7-cfd0c7d890cd
# ╟─f972b674-9264-11eb-0a1c-0774ce2527e7
# ╟─9594926e-91d0-11eb-22de-bdfe3290b19b
# ╟─d7e28c3e-9246-11eb-0fd3-af6f94ea8562
# ╠═ca2dba38-9184-11eb-1793-f588473daad1
# ╠═4f65301e-9186-11eb-1faa-71977e8fb097
# ╟─bf8b0edc-9247-11eb-0ed5-7b1d16e00fc4
# ╟─f387130c-924f-11eb-2ada-794dfbf4d30a
# ╠═eed136bc-924d-11eb-3e3a-374d21772e4b
# ╠═7f38ceba-9253-11eb-000f-25045179e841
# ╟─b14e4f0e-9253-11eb-171a-053dcc942240
# ╠═cbcbb038-9255-11eb-0fc3-3ba4d95cee62
# ╟─b8bb97aa-9256-11eb-253f-a16885888c5f
# ╟─5b25177c-925d-11eb-1aec-f52c8c52ec93
# ╟─c65e8a86-9259-11eb-29bb-3bf5c089746f
# ╟─5ebfd944-9262-11eb-3740-37ba8930e1c6
# ╠═b563c86c-9264-11eb-01c2-bb42d74c3e69
# ╠═c423eca6-9264-11eb-375f-953e02fc7ec4
# ╟─fb056230-9265-11eb-1a98-33c234a0f959
# ╟─c8c4e6b4-9266-11eb-05e6-917b82a580ab
# ╟─622247ba-9268-11eb-3c0b-07f9cd3c6236
# ╟─d9880e94-92f6-11eb-3f1e-cf4462c3b89a
# ╟─17395b62-9272-11eb-0237-430f2e6499d6
# ╟─3886d34c-9279-11eb-31e6-0fd49a2694aa
# ╠═3d3a37d6-927d-11eb-16b3-e74d10851013
# ╠═8472a6a6-927d-11eb-0992-897698a13544
# ╠═a82fe0da-926a-11eb-063b-f70bd587a789
# ╟─a79aaeb0-926b-11eb-32ea-2bc1ead29909
# ╟─f53061cc-92f4-11eb-229b-b9b501c6cab8
# ╟─f5ef8128-90bb-11eb-1f4b-053ed41f5038
# ╠═f6f99ff4-92f5-11eb-2577-e51ab1cacfa6
# ╟─65224940-92f6-11eb-3045-4fc4b7b29a6c
# ╟─992dd34e-92f6-11eb-2642-bb9862148733
# ╟─a453ad84-92f6-11eb-0350-d1fbfbbbfda0
# ╟─f9d549ac-92f6-11eb-306a-77bf90c8eb33
# ╟─94b6d52a-92f8-11eb-226f-27e28f4a4d4b
# ╟─4f96560a-92fd-11eb-19f7-7b11f0ee46bf
# ╟─1f2bdf42-92ff-11eb-0e1f-e7c4e1e78058
# ╠═bc1990b6-92f7-11eb-0015-310aefc5041e
# ╟─9f56e6fc-9303-11eb-2bdd-5da078b4d9c3
# ╟─37afad28-9304-11eb-2f4d-8397fca7bb99
# ╠═d2e5b674-9305-11eb-1254-5b9fa030e8ee
# ╠═70fc4538-9306-11eb-1509-3dfa1034d8aa
# ╟─861ca2b6-9306-11eb-311c-69c6984faa26
# ╟─97ded136-9306-11eb-019b-636b739a61d6
# ╟─e05d2a98-9306-11eb-252e-5baf6fc59f8f
# ╟─bdffde54-9307-11eb-2ccb-ed48777f28f8
# ╟─91742bde-9309-11eb-2acc-836cc1ab1aee
# ╟─df6e73de-9309-11eb-3bd3-3f9f511744cf
# ╟─071f2c26-930c-11eb-2745-93cb2001e76b
# ╟─c8227e0e-9326-11eb-2724-cb588170c7c2
# ╟─8ffaa54e-930b-11eb-2f1f-9907008b76d2
# ╠═dd9986f4-930f-11eb-33b8-67dbc4d1d087
# ╠═3c979aa0-930c-11eb-3e6e-9bdf7f3029b5
# ╟─ed6333c6-930c-11eb-25c0-c551592746e0
# ╟─76b9c454-9316-11eb-1548-bdef70d532d6
# ╟─585bed74-9319-11eb-2854-9d74e1d592c2
# ╟─7cf6be2e-9315-11eb-1cb1-396f2131908b
# ╟─91147112-9315-11eb-0a71-8ddfdb2497a6
# ╟─f7d6acc2-9314-11eb-03f3-f9f221ded27c
# ╠═e96bc054-9310-11eb-25ca-576caeb336b8
# ╠═7a3b0202-9311-11eb-101e-99fbc40c6633
# ╟─c9496d2a-931b-11eb-0228-5f08b4e1ff9f
# ╟─e75442f4-931b-11eb-0a5f-91e8c58ab47e
# ╠═94fa4c96-931c-11eb-1a1e-556bb10223f5
# ╠═4ff1b8b4-9321-11eb-249b-1f35bc1facce
# ╟─5a9982a6-9321-11eb-37a5-0be0a5b05d42
# ╟─7220bdb0-9321-11eb-2a89-79bcaa28726a
# ╟─87d39e64-9321-11eb-0170-834e21b45cc4
# ╠═bb4aac90-9323-11eb-3562-afdd70610e24
# ╠═45ada9de-9325-11eb-1450-793727639203
# ╠═5c553b5e-9325-11eb-3f4b-e12c3c1c743b
# ╠═14462c08-933c-11eb-3369-3d081358bb35
# ╠═17e1e152-933c-11eb-335b-6f2b072168ce
# ╟─58dd7382-9326-11eb-09d6-15251ddbb0bd
# ╠═b992e41a-9334-11eb-1919-87967d572a21
# ╠═c31f47ee-9334-11eb-0fa1-1be19bf991ce
# ╠═d7e8e7a6-9326-11eb-0a3d-e918e72dac08
# ╠═b2104258-9327-11eb-2151-69959f436b69
# ╟─5b7847d6-9329-11eb-0269-dbbcf2d1e563
# ╠═18b52476-9326-11eb-26ac-8ff0d7afe21e
# ╟─66700b9c-9329-11eb-1cde-87750e3652af
# ╠═1fcae9d8-9328-11eb-13fb-6935aaa65434
# ╠═f97f0ac6-9335-11eb-3051-336f0f48781f
# ╟─be47ae4c-9335-11eb-1deb-1732ce57ed4b
# ╟─0838eadc-9336-11eb-0079-23c7230a7bbc
# ╟─1a3fbf94-9336-11eb-2e48-35f09a2f14d5
# ╟─14aa52a6-9336-11eb-1a3d-a97d2d2dbe07
# ╟─55715968-9336-11eb-1845-a583df1166f6
# ╟─58255396-9336-11eb-23e5-67d8002b5595
# ╠═a958c94a-932d-11eb-247c-1b44b35cf02f
# ╟─497aee7c-932d-11eb-07fe-3b41d04c90f8
# ╠═4fcb204a-933b-11eb-3797-571078415e34
# ╠═d2339cb4-932a-11eb-293f-d35764e7b8f7
# ╟─8e009626-9316-11eb-039e-fb1183e60421
# ╟─93f9188a-9316-11eb-1c35-2d93a1936277
# ╟─59d58e3c-9317-11eb-18c2-a54f3bfcb976
# ╟─f05f1616-9317-11eb-1a84-1b60e7dae579
# ╟─4fd9ff7c-9318-11eb-197b-9f916f33f983
# ╠═5271d60c-931a-11eb-1bbb-c1bddc042136
# ╟─83fe3a24-931a-11eb-0d09-e3eed2b09138
# ╟─b77cf672-92f4-11eb-0301-8d060297d6f3
# ╟─f79db462-9283-11eb-1d3b-c19f6d9dee03
# ╟─2d03d7b4-927f-11eb-0e2c-3b8ea5c52055
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
