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


function gram_matrix(k::Kernel, X::AbstractMatrix; noise = 0.)
    d, N = size(X)
    G = zeros(N, N)
    k0 = k(zeros(d), zeros(d))

    for j = 1:N
        G[j, j] = k0
        for i = j+1:N
            G[i, j] = k(X[:, i], X[:, j])
            G[j, i] = G[i, j]
        end
    end

    return G + noise * I
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

#kernels from https://www.cs.toronto.edu/~duvenaud/cookbook/

# Matern 5/2 kernel function
function matern_52_kernel(x1::Vector{T}, x2::Vector{T}, l::T, σ::T) where T<:Real
    r = norm(x1 .- x2) / l
    sqrt(5) * r * (1 + 5/3*r^2) * exp(-sqrt(5) * r)
end

# Squared exponential (RBF) kernel function
function squared_exponential_kernel(x1::Vector{T}, x2::Vector{T}, l::T, σ::T) where T<:Real
    exp(-0.5 * norm(x1 .- x2)^2 / l^2)
end

# Quadratic kernel function
function quadratic_kernel(x::Float64, y::Float64, a::Float64, b::Float64, c::Float64)
    (x * y + a)^2 + b * x * y + c
end

# Linear kernel function
function linear_kernel(x::Float64, y::Float64, a::Float64, b::Float64)
    x * y + a * x + b * y
end

# Periodic kernel function
function periodic_kernel(x::Float64, y::Float64, l::Float64, p::Float64)
    exp(-2*sin(pi*abs(x-y)/p)^2/l^2)
end

function get_random_kernel()
    testkernels = [
        (matern_52_kernel, "Matern 52"),
        (squared_exponential_kernel, "Squared Exponential"),
        (quadratic_kernel, "Quadratic"),
        (linear_kernel, "Linear"),
        (periodic_kernel, "Periodic"),
    ]
    return testkernels[rand(1:length(testkernels))]
end