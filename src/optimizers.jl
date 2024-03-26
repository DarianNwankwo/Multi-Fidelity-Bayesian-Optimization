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