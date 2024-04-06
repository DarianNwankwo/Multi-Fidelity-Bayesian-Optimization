import Base:+, *, length


using LinearAlgebra
using ForwardDiff
using Distributions
using Optim
using Plots


include("testfns.jl")
include("constants.jl")
include("utils.jl")
include("kernels.jl")
include("surrogates.jl")
include("acquisitions.jl")
include("optimizers.jl")