struct Kernel
    θ::AbstractVector
    ψ::Function
end
(k::Kernel)(x, y) = k.ψ(x, y)


function SquaredExponential(lengthscales::AbstractVector)
    function squared_exponential(x, y)
        M = Diagonal(lengthscales .^ -2)
        r = x - y
        d = dot(r', M, r)
        return exp(-.5d)
    end

    return Kernel(lengthscales, squared_exponential)
end


function SquaredExponential(lengthscale::AbstractFloat)
    function squared_exponential(x, y)
        M = Diagonal((lengthscale ^ -2.) * ones(length(x)))
        r = x - y
        d = dot(r', M, r)
        return exp(-.5d)
    end

    return Kernel([lengthscale], squared_exponential)
end


function gram_matrix(k::Kernel, X::AbstractMatrix; noise = 0.)
    d, N = size(X)
    G = zeros(N, N)

    for j = 1:N
        for i = j:N
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