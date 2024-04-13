import Base.~

abstract type AbstractGaussianProcess end
abstract type ExactGaussianProcess <: AbstractGaussianProcess end


struct GaussianProcess{M <: AbstractMatrix, V <: AbstractVector, F <: AbstractFloat} <: ExactGaussianProcess
    k::Union{Kernel, <:Node}
    X::M
    K::M
    L::M
    y::V
    c::V
    σn2::F
end


function GP(k::Union{Kernel, <:Node}, X::AbstractMatrix, y::AbstractVector; noise = 0.)
    if isa(k, Node) k = inorder_traversal(k) end
    K = gram_matrix(k, X, noise=noise)
    L = Matrix(cholesky(Symmetric(K)).L)
    c = L' \ (L \ y)

    return GaussianProcess(k, X, K, L, y, c, noise)
end


function naive_update(gp::GaussianProcess, x::AbstractVector, y::AbstractFloat)
    X = hcat(gp.X, x)
    y = vcat(gp.y, y)
    K = gram_matrix(gp.k, X, noise=gp.σn2)  
    L = Matrix(cholesky(K).L)
    c = L' \ (L \ y)

    return GaussianProcess(gp.k, X, K, L, y, c, gp.σn2)
end


function naive_update(gp::GaussianProcess, X::AbstractMatrix, y::AbstractVector)
    X = hcat(gp.X, X)
    y = vcat(gp.y, y)
    K = gram_matrix(gp.k, X, noise=gp.σn2)
    L = Matrix(cholesky(K).L)
    c = L' \ (L \ y)

    return GaussianProcess(gp.k, X, K, L, y, c, gp.σn2)
end


function schur_update(gp::GaussianProcess, x::AbstractVector, y::AbstractFloat)
    X = hcat(gp.X, x)
    y = vcat(gp.y, y)
    kxX = gp.k(x, gp.X)
    kxx = gp.k(x, x) + WhiteNoise(gp.σn2)(x, x)

    K = [gp.K kxX
         kxX'  kxx]
    L21 = kxX' / gp.L'
    L22 = first(cholesky(kxx - L21 * L21').L) # This schur complement is a constant
    zs = zeros(size(gp.L, 1))
    L = [gp.L  zs
         L21   L22]
    c = L' \ (L \ y)

    return GaussianProcess(gp.k, X, K, L, y, c, gp.σn2)
end


function schur_update(gp::GaussianProcess, Xn::AbstractMatrix, yn::AbstractVector)
    _, N = size(Xn)
    _, M = size(gp.X)
    X = hcat(gp.X, Xn)
    y = vcat(gp.y, yn)
    KxX = gp.k(Xn, gp.X)
    Kxx = gp.k(Xn, noise=gp.σn2)
    K = [gp.K KxX'
         KxX  Kxx]
    L21 = KxX / gp.L'
    L22 = Matrix(cholesky(Kxx - L21 * L21').L)
    zs = zeros(M, N)
    L = [gp.L zs
         L21  L22]
    c = L' \ (L \ y)

    return GaussianProcess(gp.k, X, K, L, y, c, gp.σn2)
end


function log_likelihood(gp::GaussianProcess)
    N = length(gp.y)

    data_fit = -.5(dot(gp.y, gp.c))
    complexity_penalty = -sum(log.(diag(gp.L)))
    normalization_constant = -N / 2 * log(2π)

    return data_fit + complexity_penalty + normalization_constant
end


function δlog_likelihood(gp::GaussianProcess, δθ::AbstractVector)
    d, N = size(gp.X)
    δK = gram_matrix_dθ(gp.k, gp.X, δθ)

    return .5dot(gp.c, δK, gp.c) - .5sum(diag(gp.K \ δK))
end


function ∇log_likelihood(gp::GaussianProcess)
    nθ = length(gp.k.θ)
    δθ = zeros(nθ)
    ∇L = zeros(nθ)

    for j in 1:nθ
        δθ[:] .= 0.
        δθ[j] = 1.
        ∇L[j] = δlog_likelihood(gp, δθ)
    end

    return ∇L
end


function plot1d(gp::GaussianProcess; interval::AbstractRange)
    fx = zeros(length(interval))
    stdx = zeros(length(interval))
    normal_sample = randn()

    for (i, x) in enumerate(interval)
        μx, σx = gp([x])
        fx[i] = μx
        stdx[i] = sqrt(σx)
    end

    p = plot(interval, fx, ribbons=2stdx, label="μ ± 2σ")
    scatter!(gp.X', get_observations(gp), label="Observations")
    return p
end


mutable struct BoundedCapacityGaussianProcess{M <: AbstractMatrix, V <: AbstractVector, F <: AbstractFloat} <: ExactGaussianProcess
    k::Union{Kernel, <:Node}
    X::M
    K::M
    L::M
    y::V
    c::V
    σn2::F
    capacity::Int
    observed::Int
end


function BoundedGP(k::Union{Kernel, <:Node}, X::AbstractMatrix, y::AbstractVector; noise = 0., capacity = 100)
    @assert capacity > 0 "Capacity must be greater than 0."
    @assert size(X, 2) <= capacity "Number of observations must be less than or equal to capacity."
    if isa(k, Node) k = inorder_traversal(k) end
    # Preallocate space for covariance and cholesky matrices
    K = zeros(capacity, capacity)
    L = zeros(capacity, capacity)
    
    d, N = size(X)
    K[1:N, 1:N] = gram_matrix(k, X, noise=noise)
    L[1:N, 1:N] = Matrix(cholesky(K[1:N, 1:N]).L)

    c = L[1:N, 1:N]' \ (L[1:N, 1:N] \ y)

    return BoundedCapacityGaussianProcess(k, X, K, L, y, c, noise, capacity, N)
end


function update!(gp::BoundedCapacityGaussianProcess, x::AbstractVector, y::AbstractFloat)
    if gp.observed == gp.capacity
        gp.X = hcat(gp.X[:, 2:end], x)
        gp.y = vcat(gp.y[2:end], y)
    else
        gp.X = hcat(gp.X, x)
        gp.y = vcat(gp.y, y)
        gp.observed += 1
    end

    # Naive update of covariance and cholesky matrices
    gp.K[1:gp.observed, 1:gp.observed] = gram_matrix(gp.k, gp.X[:, 1:gp.observed], noise=gp.σn2)
    gp.L[1:gp.observed, 1:gp.observed] = cholesky(gp.K[1:gp.observed, 1:gp.observed]).L
    gp.c = gp.L[1:gp.observed, 1:gp.observed]' \ (gp.L[1:gp.observed, 1:gp.observed] \ gp.y)

    return gp
end


function predictive_mean(gp::ExactGaussianProcess, x::AbstractVector)
    KxX = gp.k(x, gp.X)
    return dot(KxX, gp.c)
end

function predictive_mean(gp::BoundedCapacityGaussianProcess, x::AbstractVector)
    KxX = gp.k(x, gp.X[:, 1:gp.observed])
    return dot(KxX, gp.c[1:gp.observed])
end

function predictive_variance(gp::ExactGaussianProcess, x::AbstractVector)
    kxx = gp.k(x, x)
    KxX = gp.k(x, gp.X)
    w = gp.L' \ (gp.L \ KxX)

    return kxx - dot(KxX', w)
end

function predictive_variance(gp::BoundedCapacityGaussianProcess, x::AbstractVector)
    kxx = gp.k(x, x)
    KxX = gp.k(x, gp.X[:, 1:gp.observed])
    w = gp.L[1:gp.observed, 1:gp.observed]' \ (gp.L[1:gp.observed, 1:gp.observed] \ KxX)

    return kxx - dot(KxX', w)
end

function predict(gp::ExactGaussianProcess, x::AbstractVector)
    return predictive_mean(gp, x), predictive_variance(gp, x)
end

function (gp::ExactGaussianProcess)(x::AbstractVector)
    return predict(gp, x)
end


function sample(gp::ExactGaussianProcess, x::AbstractVector; gaussian=nothing)
    μ, σ = predict(gp, x)
    u = isnothing(gaussian) ? randn() : gaussian
    
    return μ + sqrt(σ) * u
end

function ~(payload, gp::ExactGaussianProcess)
    x, gaussian = typeof(payload) <: Tuple ? payload : (payload, nothing)

    return sample(gp, x, gaussian=gaussian)
end

function get_observations(gp::ExactGaussianProcess)
    return gp.y
end

function plot1d(gp::ExactGaussianProcess; interval::AbstractRange)
    fx = zeros(length(interval))
    stdx = zeros(length(interval))
    normal_sample = randn()

    for (i, x) in enumerate(interval)
        μx, σx = gp([x])
        fx[i] = μx
        stdx[i] = sqrt(σx)
    end

    p = plot(interval, fx, ribbons=2stdx, label="μ ± 2σ")
    scatter!(gp.X', get_observations(gp), label="Observations")
    return p
end