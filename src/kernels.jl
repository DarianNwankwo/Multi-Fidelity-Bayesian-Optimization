import Base:+,*

using LinearAlgebra
using ForwardDiff

struct Kernel
    θ::AbstractVector
    ψ::Function
    ψh::Function
    dψdx::Function
    dψdy::Function
    dψdθ::Function
    ψconstructor::Function

    function Kernel(θ, ψ, ψh, dψdx, dψdy, dψdθ, ψconstructor)
        new(θ, ψ, ψh, dψdx, dψdy, dψdθ, ψconstructor)
    end
end


function KernelGeneric(kernel_constructor, θ::AbstractVector)
    if typeof(θ[1]) <: AbstractVector
        # Corresponds to dealing with summation and product of kernels
        # We need to remember which hyperparameters correspond to which kernel
        kernel_function, kernel_function_hypers = kernel_constructor(θ)
    else
        kernel_function, kernel_function_hypers = kernel_constructor(θ...)
    end

    return Kernel(
        θ,
        kernel_function,
        kernel_function_hypers,
        (x, y) -> ForwardDiff.gradient(x -> kernel_function(x, y), x),
        (x, y) -> ForwardDiff.gradient(y -> kernel_function(x, y), y),
        (x, y) -> ForwardDiff.gradient(θ -> kernel_function_hypers(x, y, θ), θ),
        kernel_constructor
    ) 
end


(k::Kernel)(x::AbstractVector, y::AbstractVector) = k.ψ(x, y)
function +(k1::Kernel, k2::Kernel)
    # function sum_kernel_constructor(θ)
    #     function sum_kernel(x, y)
    #         return k1(x, y) + k2(x, y)
    #     end

    #     function sum_kernel_hypers(x, y, θ)
    #         return k1.ψh(x, y, θ[1]...) + k2.ψh(x, y, θ[2]...)
    #     end
        
    #     return (sum_kernel, sum_kernel_hypers)
    # end

    # return KernelGeneric(sum_kernel_constructor, [k1.θ, k2.θ])
    return Kernel(
        [k1.θ; k2.θ],
        (x, y) -> k1(x, y) + k2(x, y),
        (x, y, θ) -> k1.ψh(x, y, θ[1:length(k1.θ)]) + k2.ψh(x, y, θ[length(k1.θ)+1:end]),
        (x, y) -> ForwardDiff.gradient(x -> k1(x, y) + k2(x, y), x),
        (x, y) -> ForwardDiff.gradient(y -> k1(x, y) + k2(x, y), y),
        (x, y) -> ForwardDiff.gradient(θ -> k1.ψh(x, y, θ[1:length(k1.θ)]) + k2.ψh(x, y, θ[length(k1.θ)+1:end]), [k1.θ; k2.θ]),
        k1.ψconstructor
    )
end
# *(k1::Kernel, k2::Kernel) = Kernel([k1.θ; k2.θ], (x, y) -> k1(x, y) * k2(x, y))

function *(α::Real, k::Kernel)
    @assert α >= 0 "α must be non-negative"
    # return Kernel(k.θ, (x, y) -> α * k(x, y))
    kernel_function, kernel_function_hypers = k.ψconstructor(k.θ...)

    return Kernel(
        k.θ,
        (x, y) -> α * k(x, y),
        (x, y, θ) -> α * k.ψh(x, y, θ),
        (x, y) -> ForwardDiff.gradient(x -> α * kernel_function(x, y), x),
        (x, y) -> ForwardDiff.gradient(y -> α * kernel_function(x, y), y),
        (x, y) -> ForwardDiff.gradient(θ -> α * kernel_function_hypers(x, y, θ), k.θ),
        k.ψconstructor
    )
end


function SquaredExponentialConstructor(lengthscales...)
    ls = [lengthscales...]
    function squared_exponential(x, y)
        M = Diagonal(ls .^ -2)
        r = x - y
        d = dot(r', M, r)
        return exp(-.5d)
    end

    function squared_exponential_hypers(x, y, θ::AbstractVector)
        M = Diagonal(θ .^ -2)
        r = x - y
        d = dot(r', M, r)
        return exp(-.5d)
    end

    return (squared_exponential, squared_exponential_hypers)
end
SquaredExponential(lengthscales::AbstractVector) = KernelGeneric(SquaredExponentialConstructor, lengthscales)


function SquaredExponentialConstructor(lengthscale::AbstractFloat = 1.)
    function squared_exponential(x, y)
        M = Diagonal((lengthscale ^ -2.) * ones(length(x)))
        r = x - y
        d = dot(r', M, r)
        return exp(-.5d)
    end

    function squared_exponential_hypers(x, y, θ::AbstractVector)
        M = Diagonal((θ[1] ^ -2.) * ones(length(x)))
        r = x - y
        d = dot(r', M, r)
        return exp(-.5d)
    end

    return (squared_exponential, squared_exponential_hypers)
end
SquaredExponential(lengthscale::AbstractFloat = 1.) = KernelGeneric(SquaredExponentialConstructor, [lengthscale])


function PeriodicConstructor(lengthscale::AbstractFloat = 1., period::AbstractFloat = 1.)
    function periodic(x, y)
        return exp(-2 * sin(pi * norm(x - y) / period) ^ 2 / lengthscale ^ 2)
    end

    function periodic_hypers(x, y, θ::AbstractVector)
        return exp(-2 * sin(pi * norm(x - y) / θ[2]) ^ 2 / θ[1] ^ 2)
    end

    return (periodic, periodic_hypers)
end
Periodic(lengthscale, period) = KernelGeneric(PeriodicConstructor, [lengthscale, period])


function ExponentialConstructor(lengthscale::AbstractFloat = 1.)
    function exponential(x, y)
        return exp(-norm(x - y) / lengthscale)
    end

    function exponential_hypers(x, y, θ::AbstractVector)
        return exp(-norm(x - y) / θ[1])
    end

    return (exponential, exponential_hypers)
end
Exponential(lengthscale) = KernelGeneric(ExponentialConstructor, [lengthscale])


function GammaExponentialConstructor(lengthscale::AbstractFloat = 1., γ::AbstractFloat = 1.)
    function gamma_exponential(x, y)
        return exp((-norm(x - y) / lengthscale) ^ γ)
    end

    function gamma_exponential_hypers(x, y, θ::AbstractVector)
        return exp((-norm(x - y) / θ[1]) ^ θ[2])
    end

    return (gamma_exponential, gamma_exponential_hypers)
end
GammaExponential(lengthscale, γ) = KernelGeneric(GammaExponentialConstructor, [lengthscale, γ])


function RationalQuadraticConstructor(lengthscale::AbstractFloat = 1., α::AbstractFloat = 1.)
    function rational_quadratic(x, y)
        return (1 + norm(x - y) ^ 2 / (2 * α * lengthscale ^ 2)) ^ -α
    end

    function rational_quadradic_hypers(x, y, θ::AbstractVector)
        return (1 + norm(x - y) ^ 2 / (2 * θ[2] * θ[1] ^ 2)) ^ -θ[2]
    end

    return (rational_quadratic, rational_quadradic_hypers)
end
RationalQuadratic(lengthscale, α) = KernelGeneric(RationalQuadraticConstructor, [lengthscale, α])


function Matern12Constructor(lengthscale::AbstractFloat = 1.)
    function matern12(x, y)
        return exp(-norm(x - y) / lengthscale)
    end

    function matern12_hypers(x, y, θ::AbstractVector)
        return exp(-norm(x - y) / θ[1])
    end

    return (matern12, matern12_hypers)
end
Matern12(lengthscale = 1.) = KernelGeneric(Matern12Constructor, [lengthscale])


function Matern32Constructor(lengthscale::AbstractFloat = 1.)
    function matern32(x, y)
        r = norm(x - y) / lengthscale
        return (1 + sqrt(3) * r) * exp(-sqrt(3) * r)
    end

    function matern32_hypers(x, y, θ::AbstractVector)
        r = norm(x - y) / θ[1]
        return (1 + sqrt(3) * r) * exp(-sqrt(3) * r)
    end

    return (matern32, matern32_hypers)
end
Matern32(lengthscale = 1.) = KernelGeneric(Matern32Constructor, [lengthscale])


function Matern52Constructor(lengthscale::AbstractFloat = 1.)
    function matern52(x, y)
        r = norm(x - y) / lengthscale
        return (1 + sqrt(5) * r + 5 * r ^ 2 / 3) * exp(-sqrt(5) * r)
    end

    function matern52_hypers(x, y, θ::AbstractVector)
        r = norm(x - y) / θ[1]
        return (1 + sqrt(5) * r + 5 * r ^ 2 / 3) * exp(-sqrt(5) * r)
    end

    return (matern52, matern52_hypers)
end
Matern52(lengthscale = 1.) = KernelGeneric(Matern52Constructor, [lengthscale])


function WhiteNoiseConstructor(σ::AbstractFloat = 1e-6)
    function white_noise(x, y)
        return x == y ? σ : 0.
    end

    function white_noise_hypers(x, y, θ::AbstractVector)
        return x == y ? θ[1] : 0.
    end

    return (white_noise, white_noise_hypers)
end
WhiteNoise(σ = 1e-6) = KernelGeneric(WhiteNoiseConstructor, [σ])


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


function gram_matrix_dθ(k::Kernel, X::AbstractMatrix, δθ::AbstractVector)
    d, N = size(X)
    δG = zeros(N, N)
    δk0 = dot(k.dψdθ(zeros(d), zeros(d)), δθ)

    for j = 1:N
        δG[j, j] = δk0
        for i = j+1:N
            δGij = dot(k.dψdθ(X[:, i], X[:, j]), δθ)
            δG[i, j] = δGij
            δG[j, i] = δGij
        end
    end

    return δG
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