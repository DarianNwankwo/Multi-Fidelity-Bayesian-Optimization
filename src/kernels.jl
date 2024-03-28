import Base:+,*

using LinearAlgebra

struct Kernel
    θ::AbstractVector
    ψ::Function
    dψdθ::Function
end

function Kernel(θ::AbstractVector, ψ::Function)
    return Kernel(θ, ψ, (x, y) -> nothing)
end

(k::Kernel)(x::AbstractVector, y::AbstractVector) = k.ψ(x, y)
+(k1::Kernel, k2::Kernel) = Kernel([k1.θ; k2.θ], (x, y) -> k1(x, y) + k2(x, y))
*(k1::Kernel, k2::Kernel) = Kernel([k1.θ; k2.θ], (x, y) -> k1(x, y) * k2(x, y))

function *(α::Real, k::Kernel)
    @assert α >= 0 "α must be non-negative"
    return Kernel(k.θ, (x, y) -> α * k(x, y))
end


function SquaredExponential(lengthscales::AbstractVector)
    function squared_exponential(x, y)
        M = Diagonal(lengthscales .^ -2)
        r = x - y
        d = dot(r', M, r)
        return exp(-.5d)
    end

    return Kernel(lengthscales, squared_exponential)
end


function SquaredExponential(lengthscale::AbstractFloat = 1.)
    function squared_exponential(x, y)
        M = Diagonal((lengthscale ^ -2.) * ones(length(x)))
        r = x - y
        d = dot(r', M, r)
        return exp(-.5d)
    end

    return Kernel([lengthscale], squared_exponential)
end

function LinearKernel(y_var::Float64, var::Float64, c::AbstractVector)
        function linear_kernel(x::AbstractVector, y::AbstractVector)
            return y_var + var * dot(x - c, y - c)
    end
    return Kernel([y_var, var, c], linear_kernel)
end


function Periodic(lengthscale::AbstractFloat = 1., period::AbstractFloat = 3.14)
    function periodic(x, y)
        return exp(-2 * sin(pi * norm(x - y) / period) ^ 2 / lengthscale ^ 2)
    end

    return Kernel([lengthscale, period], periodic)
end


function Exponential(lengthscale::AbstractFloat = 1.)
    function exponential(x, y)
        return exp(-norm(x - y) / lengthscale)
    end

    return Kernel([lengthscale], exponential)
end


function GammaExponential(lengthscale::AbstractFloat = 1., γ::AbstractFloat = 1.)
    function gamma_exponential(x, y)
        return exp((-norm(x - y) / lengthscale) ^ γ)
    end

    return Kernel([lengthscale, γ], gamma_exponential)
end


function WhiteNoise(σ::AbstractFloat = 1e-6)
    function white_noise(x, y)
        return x == y ? σ : 0.
    end

    return Kernel([σ], white_noise)
end


function gram_matrix(k::Kernel, X::AbstractMatrix; noise = 0.)
    d, N = size(X)
    G = zeros(N, N)
    k0 = k(zeros(d), zeros(d))
    noise = WhiteNoise(noise)

    for j = 1:N
        G[j, j] = k0 + noise(X[:, j], X[:, j])
        for i = j+1:N
            G[i, j] = k(X[:, i], X[:, j])
            G[j, i] = G[i, j]
        end
    end

    return G
end


function kernel_vector(k::Kernel, x::AbstractVector, X::AbstractMatrix)
    d, N = size(X)
    KxX = zeros(N)

    for j = 1:N
        KxX[j] = k(x, X[:, j])
    end

    return KxX
end

(k::Kernel)(x::AbstractVector, X::AbstractMatrix) = kernel_vector(k, x, X)