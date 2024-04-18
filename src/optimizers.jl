"""
We need the kernel object so we can reconstruct it with different paramters during optimization.
The kernel object, if composed of sum or products, maintains all of the necessary information
to reconstruct a new kernel object with different hyperparameters.
"""
function log_likelihood_constructor(kernel_expression_tree::Node, X::AbstractMatrix, y::AbstractVector; noise=0.)
    function _log_likelihood(θ::AbstractVector)
        kernel = build_kernel(kernel_expression_tree, θ, 0)
        sur = GP(kernel, X, y, noise=noise)

        return -log_likelihood(sur)
    end
    
    function _∇log_likelihood!(g, θ)
        kernel = build_kernel(kernel_expression_tree, θ, 0)
        sur = GP(kernel, X, y, noise=noise)
        
        g[:] = -∇log_likelihood(sur)
    end

    return (_log_likelihood, _∇log_likelihood!)
end

function optimize_surrogate(;
    gp::GaussianProcess,
    kernel_expression_tree::Node,
    lbs::AbstractVector,
    ubs::AbstractVector,
    noise::AbstractFloat = 0.,
    random_restarts = 6,
    optim_options = Optim.Options())
    f, g! = log_likelihood_constructor(kernel_expression_tree, gp.X, get_observations(gp), noise=noise)
    results = []
    θ0s = randsample(random_restarts, length(lbs), lbs, ubs)

    for i in 1:random_restarts
        result = optimize(
            f, g!, lbs, ubs, θ0s[:, i], Fminbox(LBFGS()), optim_options
        )
        push!(results, result)
    end

    minimizers = Optim.minimizer.(results)
    minimums = Optim.minimum.(results)
    global_minimizer = minimizers[argmin(minimums)]

    kernel = build_kernel(kernel_expression_tree, global_minimizer, 0)
    return GP(kernel, gp.X, get_observations(gp), noise=noise)
end