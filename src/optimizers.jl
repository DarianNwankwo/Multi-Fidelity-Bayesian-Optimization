include("kernels.jl")
include("surrogates.jl")


function UCB_constructor(s::ZeroMeanGaussianProcess)
    function UCB(β::AbstractFloat)
        function UCB(x::AbstractVector)
            μ, σ = predict(s, x)
            return μ + β * sqrt(σ)
        end
    end

    return UCB
end


function log_likelihood_constructor(kernel_constructor, X, y; noise=0.)
    function _log_likelihood(θ::AbstractVector)
        sur = GP(kernel_constructor(θ), X, y, noise=noise)

        return -log_likelihood(sur)
    end
    
    function _∇log_likelihood!(g, θ)
        sur = GP(kernel_constructor(θ), X, y, noise=noise)
        
        g[:] = -∇log_likelihood(sur)
    end

    return (_log_likelihood, _∇log_likelihood!)
end

function optimize_surrogate(;
    gp::ZeroMeanGaussianProcess,
    kernel_constructor::Function,
    lbs::AbstractVector,
    ubs::AbstractVector,
    noise::AbstractFloat = 0.,
    random_restarts = 6,
    max_iterations::Int = 100)
    f, g! = log_likelihood_constructor(kernel_constructor, gp.X, get_observations(gp), noise=noise)
    results = []

    for _ in 1:random_restarts
        θ0 = rand(length(lbs)) .* (ubs .- lbs) .+ lbs
        result = optimize(
            f, g!, lbs, ubs, θ0, Fminbox(LBFGS()), Optim.Options(iterations = max_iterations)
        )
        push!(results, result)
    end

    minimizers = Optim.minimizer.(results)
    minimums = Optim.minimum.(results)
    global_minimizer = minimizers[argmin(minimums)]

    return GP(kernel_constructor(global_minimizer), gp.X, get_observations(gp), noise=noise)
end