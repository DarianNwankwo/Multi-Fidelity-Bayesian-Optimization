import Base.~

include("./kernels.jl")

abstract type AbstractGaussianProcess end
abstract type ExactGaussianProcess <: AbstractGaussianProcess end


struct GaussianProcess <: ExactGaussianProcess
    k::Kernel
    X::AbstractMatrix
    K::AbstractMatrix
    L::AbstractMatrix
    y::AbstractVector
    c::AbstractVector
end

function predictive_mean(gp::GaussianProcess, x::AbstractVector)
    KxX = gp.k(x, gp.X)
    return dot(KxX, gp.c) #+ gp.ymean
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


function GP(k::Kernel, X::AbstractMatrix, y::AbstractVector; noise = 0.)
    K = gram_matrix(k, X, noise=noise)
    L = cholesky(K).L
    ymean = mean(y)
    # y = y .- ymean
    c = L' \ (L \ y)

    return GaussianProcess(k, X, K, L, y, c)#, ymean)
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