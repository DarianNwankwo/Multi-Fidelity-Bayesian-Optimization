abstract type AbstractKernel end

# Add support for maintaining kernel bounds
struct Kernel
    θ
    ψ
    ψh
    dψdx
    dψdy
    dψdθ
    """The function that performs the kernel construction."""
    ψconstructor
    """The function that constructs the kernel object."""
    constructor
    """Maintain the length of each kernels hyperparameters."""
    lengths

    function Kernel(θ, ψ, ψh, dψdx, dψdy, dψdθ, ψconstructor, constructor, lengths)
        return new(θ, ψ, ψh, dψdx, dψdy, dψdθ, ψconstructor, constructor, lengths)
    end
end


length(k::Kernel) = length(k.θ)
hyperparameters(k::Kernel) = copy(k.θ)


function KernelGeneric(kernel_constructor, θ::AbstractVector, constructor, lengths)
    kernel_function, kernel_function_hypers = kernel_constructor(θ...)

    return Kernel(
        θ,
        kernel_function,
        kernel_function_hypers,
        (x, y) -> ForwardDiff.gradient(x -> kernel_function(x, y), x),
        (x, y) -> ForwardDiff.gradient(y -> kernel_function(x, y), y),
        (x, y) -> ForwardDiff.gradient(θ -> kernel_function_hypers(x, y, θ), θ),
        kernel_constructor,
        constructor,
        lengths,
    )
end


(k::Kernel)(x::AbstractVector, y::AbstractVector) = k.ψ(x, y)
function +(k1::Kernel, k2::Kernel)
    return Kernel(
        [k1.θ; k2.θ],
        (x, y) -> k1(x, y) + k2(x, y),
        (x, y, θ) -> k1.ψh(x, y, θ[1:length(k1.θ)]) + k2.ψh(x, y, θ[length(k1.θ)+1:end]),
        (x, y) -> ForwardDiff.gradient(x -> k1(x, y) + k2(x, y), x),
        (x, y) -> ForwardDiff.gradient(y -> k1(x, y) + k2(x, y), y),
        (x, y) -> ForwardDiff.gradient(θ -> k1.ψh(x, y, θ[1:length(k1.θ)]) + k2.ψh(x, y, θ[length(k1.θ)+1:end]), [k1.θ; k2.θ]),
        (θ) -> k1.ψconstructor(θ[1:length(k1.θ)]) + k2.ψconstructor(θ[length(k1.θ)+1:end]),
        nothing,
        vcat(k1.lengths, k2.lengths),
    )
end

function *(k1::Kernel, k2::Kernel)
    return Kernel(
        [k1.θ; k2.θ],
        (x, y) -> k1(x, y) * k2(x, y),
        (x, y, θ) -> k1.ψh(x, y, θ[1:length(k1.θ)]) * k2.ψh(x, y, θ[length(k1.θ)+1:end]),
        (x, y) -> ForwardDiff.gradient(x -> k1(x, y) * k2(x, y), x),
        (x, y) -> ForwardDiff.gradient(y -> k1(x, y) * k2(x, y), y),
        (x, y) -> ForwardDiff.gradient(θ -> k1.ψh(x, y, θ[1:length(k1.θ)]) * k2.ψh(x, y, θ[length(k1.θ)+1:end]), [k1.θ; k2.θ]),
        (θ) -> k1.ψconstructor(θ[1:length(k1.θ)]) * k2.ψconstructor(θ[length(k1.θ)+1:end]),
        nothing,
        vcat(k1.lengths, k2.lengths),
    )
end


function *(α::Real, k::Kernel)
    @assert α >= 0 "α must be non-negative"
    kernel_function, kernel_function_hypers = k.ψconstructor(k.θ...)

    return Kernel(
        k.θ,
        (x, y) -> α * k(x, y),
        (x, y, θ) -> α * k.ψh(x, y, θ),
        (x, y) -> ForwardDiff.gradient(x -> α * kernel_function(x, y), x),
        (x, y) -> ForwardDiff.gradient(y -> α * kernel_function(x, y), y),
        (x, y) -> ForwardDiff.gradient(θ -> α * kernel_function_hypers(x, y, θ), k.θ),
        (θ) -> α * k.ψconstructor(θ), # Should be a function that returns the kernel constructor scaled by α
        (θ) -> α * k.constructor(θ),
        k.lengths,
    )
end


################################################################################
# Logic for reconstructing sum and product kernels using inorder traversal of 
# the expression tree.
################################################################################
KERNEL_ADD = +
KERNEL_MULTIPLY = *

mutable struct Node{T <: Union{Function, Kernel}}
    payload::T
    left::Union{<:Node, Nothing}
    right::Union{<:Node, Nothing}
end
Node(payload::T, left=nothing, right=nothing) where T <: Union{Function, Kernel} = Node{T}(payload, left, right)
Node(payload::T, left, right) where T <: Union{Function, Kernel} = Node{T}(payload, left, right)
is_leaf(node::Node) = isnothing(node.left) && isnothing(node.right)
is_internal(node::Node) = !is_leaf(node)

"""
We use inorder_traversal to reconstruct a kernel object from an expression tree with user provided hyperparameters.
We also need a mechanism for constructing the kernel object
"""
function build_kernel(node::Node, θ::AbstractVector, previous_observed::Int = 0)
    if is_leaf(node)
        slice = previous_observed+1:previous_observed+length(node.payload.θ)
        return node.payload.constructor(θ[slice], nodify=false)
    else
        left = build_kernel(node.left, θ, previous_observed)
        right = build_kernel(node.right, θ, previous_observed + length(left.θ))
        return node.payload(left, right)
    end
end

function build_kernel(node::Node)
    if is_leaf(node)
        return node.payload
    else
        left = build_kernel(node.left)
        right = build_kernel(node.right)
        return node.payload(left, right)
    end
end

function show_expression(node::Node, current_index = [1])
    if is_leaf(node)
        current_index[1] += 1
        return "k$(current_index[1] - 1)"
    else
        left = show_expression(node.left, current_index)
        right = show_expression(node.right, current_index)
        return "($left $(string(node.payload)) $right)"
    end
end


+(left::Node, right::Node) = Node(KERNEL_ADD, left, right)
+(left::Node, right::Kernel) = Node(KERNEL_ADD, left, Node(right))
+(left::Kernel, right::Node) = Node(KERNEL_ADD, Node(left), right)
*(left::Node, right::Node) = Node(KERNEL_MULTIPLY, left, right)
*(left::Node, right::Kernel) = Node(KERNEL_MULTIPLY, left, Node(right))
*(left::Kernel, right::Node) = Node(KERNEL_MULTIPLY, Node(left), right)
(n::Node)(x::AbstractVector, y::AbstractVector) = build_kernel(n)(x, y)
length(node::Node) = length(build_kernel(node).θ)


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


function SquaredExponential(lengthscales::AbstractVector; nodify=true)
    if nodify
        return Node(
            KernelGeneric(
                SquaredExponentialConstructor, lengthscales, SquaredExponential, [length(lengthscales)]
            )
        )
    end
    
    return KernelGeneric(
        SquaredExponentialConstructor, lengthscales, SquaredExponential, [length(lengthscales)]
    )
end


function SquaredExponential(lengthscales...; nodify=true)
    if nodify
        return Node(
            KernelGeneric(
                SquaredExponentialConstructor, [lengthscales...], SquaredExponential, [length(lengthscales)]
            )
        )
    end

    return KernelGeneric(
        SquaredExponentialConstructor, [lengthscales...], SquaredExponential, [length(lengthscales)]
    )
end

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

function SquaredExponential(lengthscale::AbstractFloat = 1.; nodify=true)
    if nodify
        return Node(
            KernelGeneric(
                SquaredExponentialConstructor, [lengthscale], SquaredExponential, [1]
            )
        )
    end

    return KernelGeneric(
        SquaredExponentialConstructor, [lengthscale], SquaredExponential, [1]
    )
end


function PeriodicConstructor(lengthscale::AbstractFloat = 1., period::AbstractFloat = 1.)
    function periodic(x, y)
        return exp(-2 * sin(pi * norm(x - y) / period) ^ 2 / lengthscale ^ 2)
    end

    function periodic_hypers(x, y, θ::AbstractVector)
        return exp(-2 * sin(pi * norm(x - y) / θ[2]) ^ 2 / θ[1] ^ 2)
    end

    return (periodic, periodic_hypers)
end

function Periodic(lengthscale::AbstractFloat = 1., period::AbstractFloat = 1.; nodify=true)
    if nodify
        return Node(
            KernelGeneric(
                PeriodicConstructor, [lengthscale, period], Periodic, [2]
            )
        )
    end

    return KernelGeneric(
        PeriodicConstructor, [lengthscale, period], Periodic, [2]
    )
end

function Periodic(hyperparameters::AbstractVector; nodify=true)
    if nodify
        return Node(
            KernelGeneric(
                PeriodicConstructor, hyperparameters, Periodic, [length(hyperparameters)]
            )
        )
    end

    return KernelGeneric(
        PeriodicConstructor, hyperparameters, Periodic, [length(hyperparameters)]
    )
end


function ExponentialConstructor(lengthscale::AbstractFloat = 1.)
    function exponential(x, y)
        return exp(-norm(x - y) / lengthscale)
    end

    function exponential_hypers(x, y, θ::AbstractVector)
        return exp(-norm(x - y) / θ[1])
    end

    return (exponential, exponential_hypers)
end


function Exponential(lengthscale::AbstractFloat = 1.; nodify=true)
    if nodify
        return Node(
            KernelGeneric(
                ExponentialConstructor, [lengthscale], Exponential, [1]
            )
        )
    end

    return KernelGeneric(
        ExponentialConstructor, [lengthscale], Exponential, [1]
    )
end


function GammaExponentialConstructor(lengthscale::AbstractFloat = 1., γ::AbstractFloat = 1.)
    function gamma_exponential(x, y)
        return exp((-norm(x - y) / lengthscale) ^ γ)
    end

    function gamma_exponential_hypers(x, y, θ::AbstractVector)
        return exp((-norm(x - y) / θ[1]) ^ θ[2])
    end

    return (gamma_exponential, gamma_exponential_hypers)
end


function GammaExponential(lengthscale::AbstractFloat = 1., γ::AbstractFloat = 1.; nodify=true)
    if nodify
        return Node(
            KernelGeneric(
                GammaExponentialConstructor, [lengthscale, γ], GammaExponential, [2]
            )
        )
    end

    return KernelGeneric(
        GammaExponentialConstructor, [lengthscale, γ], GammaExponential, [2]
    )
end


function RationalQuadraticConstructor(lengthscale::AbstractFloat = 1., α::AbstractFloat = 1.)
    function rational_quadratic(x, y)
        return (1 + norm(x - y) ^ 2 / (2 * α * lengthscale ^ 2)) ^ -α
    end

    function rational_quadradic_hypers(x, y, θ::AbstractVector)
        return (1 + norm(x - y) ^ 2 / (2 * θ[2] * θ[1] ^ 2)) ^ -θ[2]
    end

    return (rational_quadratic, rational_quadradic_hypers)
end


function RationalQuadratic(lengthscale::AbstractFloat = 1., α::AbstractFloat = 1.; nodify=true)
    if nodify
        return Node(
            KernelGeneric(
                RationalQuadraticConstructor, [lengthscale, α], RationalQuadratic, [2]
            )
        )
    end

    return KernelGeneric(
        RationalQuadraticConstructor, [lengthscale, α], RationalQuadratic, [2]
    )
end


function Matern12Constructor(lengthscale::AbstractFloat = 1.)
    function matern12(x, y)
        return exp(-norm(x - y) / lengthscale)
    end

    function matern12_hypers(x, y, θ::AbstractVector)
        return exp(-norm(x - y) / θ[1])
    end

    return (matern12, matern12_hypers)
end


function Matern12(lengthscale::AbstractFloat = 1.; nodify=true)
    if nodify
        return Node(
            KernelGeneric(
                Matern12Constructor, [lengthscale], Matern12, [1]
            )
        )
    end

    return KernelGeneric(
        Matern12Constructor, [lengthscale], Matern12, [1]
    )
end


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


function Matern32(lengthscale::AbstractFloat = 1.; nodify=true)
    if nodify
        return Node(
            KernelGeneric(
                Matern32Constructor, [lengthscale], Matern32, [1]
            )
        )
    end

    return KernelGeneric(
        Matern32Constructor, [lengthscale], Matern32, [1]
    )
end


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


function Matern52(lengthscale::AbstractFloat = 1.; nodify=true)
    if nodify
        return Node(
            KernelGeneric(
                Matern52Constructor, [lengthscale], Matern52, [1]
            )
        )
    end

    return KernelGeneric(
        Matern52Constructor, [lengthscale], Matern52, [1]
    )
end


function WhiteNoiseConstructor(σ::AbstractFloat = 1e-6)
    function white_noise(x, y)
        return x == y ? σ : 0.
    end

    function white_noise_hypers(x, y, θ::AbstractVector)
        return x == y ? θ[1] : 0.
    end

    return (white_noise, white_noise_hypers)
end
WhiteNoise(σ = 1e-6) = KernelGeneric(
    WhiteNoiseConstructor, [σ], WhiteNoise, [1]
)


function gram_matrix(k::Union{Kernel, <:Node}, X::AbstractMatrix; noise = 0.)
    if isa(k, Node) k = build_kernel(k) end
    d, N = size(X)
    G = zeros(N, N)
    k0 = k(zeros(d), zeros(d))
    noise = WhiteNoise(noise)

    for j = 1:N
        G[j, j] = k0 + noise((@view X[:, j]), (@view X[:, j]))
        for i = j+1:N
            G[i, j] = k((@view X[:, i]), (@view X[:, j]))
            G[j, i] = G[i, j]
        end
    end

    return G
end
(k::Union{Kernel, <:Node})(X::AbstractMatrix; noise = 0.) = gram_matrix(k, X, noise=noise)

function gram_matrix_dθ(k::Union{Kernel, <:Node}, X::AbstractMatrix, δθ::AbstractVector)
    if isa(k, Node) k = build_kernel(k) end
    d, N = size(X)
    δG = zeros(N, N)
    δk0 = dot(k.dψdθ(zeros(d), zeros(d)), δθ)

    for j = 1:N
        δG[j, j] = δk0
        for i = j+1:N
            δGij = dot(
                k.dψdθ((@view X[:, i]), (@view X[:, j])),
                δθ
            )
            δG[i, j] = δGij
            δG[j, i] = δGij
        end
    end

    return δG
end


function kernel_vector(k::Union{Kernel, <:Node}, x::AbstractVector, X::AbstractMatrix)
    if isa(k, Node) k = build_kernel(k) end
    d, N = size(X)
    KxX = zeros(N)

    for j = 1:N
        KxX[j] = k(x, (@view X[:, j]))
    end

    return KxX
end
(k::Union{Kernel, <:Node})(x::AbstractVector, X::AbstractMatrix) = kernel_vector(k, x, X)


function covariance_matrix(k::Union{Kernel, <:Node}, X::AbstractMatrix, Y::AbstractMatrix)
    if isa(k, Node) k = build_kernel(k) end
    d, N = size(X)
    _, M = size(Y)
    K = zeros(N, M)

    for i = 1:N
        K[i, :] = k((@view X[:, i]), Y)
    end

    return K
end
(k::Union{Kernel, <:Node})(X::AbstractMatrix, Y::AbstractMatrix) = covariance_matrix(k, X, Y)