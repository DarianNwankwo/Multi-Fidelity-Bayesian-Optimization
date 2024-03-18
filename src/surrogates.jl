include("./kernels.jl")


struct ZeroMeanGaussianProcess
    k::Kernel
    X::AbstractMatrix
    K::AbstractMatrix
    L::AbstractMatrix
    y::AbstractVector
    c::AbstractVector
    ymean::AbstractFloat
end

function predictive_mean(zmgp::ZeroMeanGaussianProcess, x::AbstractVector)
    KxX = kernel_vector(zmgp.k, x, zmgp.X)
    return dot(KxX, zmgp.c) + zmgp.ymean
end

function predictive_variance(zmgp::ZeroMeanGaussianProcess, x::AbstractVector)
    kxx = zmgp.k(x, x)
    KxX = kernel_vector(zmgp.k, x, zmgp.X)
    w = zmgp.L' \ (zmgp.L \ KxX)

    return kxx - dot(KxX', w)
end

function predict(zmgp::ZeroMeanGaussianProcess, x::AbstractVector)
    return predictive_mean(zmgp, x), predictive_variance(zmgp, x)
end

function (zmgp::ZeroMeanGaussianProcess)(x::AbstractVector)
    return predict(zmgp, x)
end
# (TODO) Add syntatic sugar for sampling from the Gaussian process
function sample(gp::ZeroMeanGaussianProcess, x::AbstractVector)
end


function get_observations(zmgp::ZeroMeanGaussianProcess)
    return zmgp.y .+ zmgp.ymean
end


function ZeroMeanGP(k::Kernel, X::AbstractMatrix, y::AbstractVector; noise = 0.)
    K = gram_matrix(k, X, noise=noise)
    L = cholesky(K).L
    ymean = mean(y)
    y = y .- ymean
    c = L' \ (L \ y)

    return ZeroMeanGaussianProcess(k, X, K, L, y, c, ymean)
end

