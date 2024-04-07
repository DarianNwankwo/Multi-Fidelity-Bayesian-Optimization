import Base.~

abstract type AbstractGaussianProcess end
abstract type ExactGaussianProcess <: AbstractGaussianProcess end


struct GaussianProcess <: ExactGaussianProcess
    k::Union{Kernel, <:Node}
    X::AbstractMatrix
    K::AbstractMatrix
    L::AbstractMatrix
    y::AbstractVector
    c::AbstractVector
    σn2::AbstractFloat
end

function predictive_mean(gp::GaussianProcess, x::AbstractVector)
    KxX = gp.k(x, gp.X)
    return dot(KxX, gp.c)
end

function predictive_variance(gp::GaussianProcess, x::AbstractVector)
    kxx = gp.k(x, x)
    KxX = gp.k(x, gp.X)
    w = gp.L' \ (gp.L \ KxX)

    return kxx - dot(KxX', w)
end

function predict(gp::GaussianProcess, x::AbstractVector)
    return predictive_mean(gp, x), predictive_variance(gp, x)
end

function (gp::GaussianProcess)(x::AbstractVector)
    return predict(gp, x)
end

function sample(gp::GaussianProcess, x::AbstractVector; gaussian=nothing)
    μ, σ = predict(gp, x)
    u = isnothing(gaussian) ? randn() : gaussian
    
    return μ + sqrt(σ) * u
end

function ~(payload, gp::GaussianProcess)
    x, gaussian = typeof(payload) <: Tuple ? payload : (payload, nothing)

    return sample(gp, x, gaussian=gaussian)
end

function get_observations(gp::GaussianProcess)
    return gp.y
end


function GP(k::Union{Kernel, <:Node}, X::AbstractMatrix, y::AbstractVector; noise = 0.)
    if isa(k, Node) k = inorder_traversal(k) end
    K = gram_matrix(k, X, noise=noise)
    L = cholesky(Symmetric(K)).L
    c = L' \ (L \ y)

    return GaussianProcess(k, X, K, L, y, c, noise)
end


function update(gp::GaussianProcess, x::AbstractVector, y::AbstractFloat)
    X = hcat(gp.X, x)
    y = vcat(gp.y, y)
    K = gram_matrix(gp.k, X, noise=gp.σn2)  
    L = cholesky(K).L
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


struct BoundedCapacityGaussianProcess <: ExactGaussianProcess
    k::Union{Kernel, <:Node}
    X::AbstractMatrix
    K::AbstractMatrix
    L::AbstractMatrix
    y::AbstractVector
    c::AbstractVector
    σn2::AbstractFloat
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
    L[1:N, 1:N] = cholesky(K[1:N, 1:N]).L

    c = L[1:N, 1:N]' \ (L[1:N, 1:N] \ y)

    return BoundedCapacityGaussianProcess(k, X, K, L, y, c, noise, capacity, observed)
end