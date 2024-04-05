include("kernels.jl")
include("surrogates.jl")


function UCB_constructor(s::ZeroMeanGaussianProcess)
    function UCB_param_constructor(β::AbstractFloat)
        function UCB(x::AbstractVector)
            μ, σ = predict(s, x)
            return μ + β * sqrt(σ)
        end
    end

    return UCB_param_constructor
end


function LCB_constructor(s::ZeroMeanGaussianProcess)
    function LCB_params(β::AbstractFloat)
        function LCB(x::AbstractVector)
            μ, σ = predict(s, x)
            return μ - β * sqrt(σ)
        end
    end

    return LCB_params
end

