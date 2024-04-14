abstract type AbstractAcquisitionFunction end
abstract type DifferentiableAcquisitionFunction <: AbstractAcquisitionFunction end
abstract type AcquisitionFunctionOnly <: AbstractAcquisitionFunction end

struct AcquisitionFunction{T <: Function, V <: AbstractVector} <: AcquisitionFunctionOnly
    f::T
    lowerbounds::V
    upperbounds::V
end

z(μ, σ, f⁺; β) = (f⁺-μ-β)/σ

function UCB(s::GaussianProcess; lbs, ubs, β=1., err=1e-6)
    UCBx(x::AbstractVector) = begin
        μ, σ = predict(s, x)
        if σ <= err
            return NaN # Return NaN when standard deviation is too low
        end
        return μ + β * σ
    end
    return AcquisitionFunction(UCBx, lbs, ubs)
end

function LCB(s::GaussianProcess; lbs, ubs, β=1., err=1e-6)
    LCBx(x::AbstractVector) = begin
        μ, σ = predict(s, x)
        if σ <= err
            return NaN # Return NaN when standard deviation is too low
        end
        return μ - β * σ
    end
    return AcquisitionFunction(LCBx, lbs, ubs)
end

function POI(s::GaussianProcess; lbs, ubs, β=1., err=1e-6)
    f⁺ = minimum(s.y)
    POIx(x::AbstractVector) = begin
        μ, σ = predict(s, x)
        if σ <= err
            return 0 # Return 0 when standard deviation is too low
        end
        Φx = cdf(Normal(), z(μ, σ, f⁺; β=β))
        return Φx
    end
    return AcquisitionFunction(POIx, lbs, ubs)
end

function EI(s::GaussianProcess; lbs, ubs, β=1., err=1e-6)
    f⁺ = minimum(s.y)
    EIx(x::AbstractVector) = begin
        μ, σ = predict(s, x)
        # if σ <= err
        #     return 0 # Return 0 when standard deviation is too low
        # end
        zx = z(μ, σ, f⁺; β=β)
        Φx = cdf(Normal(), zx)
        ϕx = pdf(Normal(), zx)
        return -(zx * σ) * Φx + σ * ϕx
    end
    return AcquisitionFunction(EIx, lbs, ubs)
end


function optimize_acquisition(a::AcquisitionFunctionOnly; random_restarts=16, optim_options=Optim.Options())
    ALL_STARTS = randsample(random_restarts, length(a.lowerbounds), a.lowerbounds, a.upperbounds)
    results = []

    for i in 1:random_restarts
        initial_start = ALL_STARTS[:, i]
        push!(
            results,
            optimize(
                a.f, a.lowerbounds, a.upperbounds, initial_start, Fminbox(LBFGS()), optim_options
            )
        )
    end

    minimizer_locations = Optim.minimizer.(results)
    minimizers = Optim.minimum.(results)
    trash, minimizer_index = findmin(minimizers)
    return minimizer_locations[minimizer_index]
end

function get_acquisition_functions(sur::GaussianProcess; β=1., err=1e-6)
    return Dict(
        "Expected Improvement" => EI(sur; β=β, err=err),
        "Probability of Improvement" => POI(sur; β=β, err=err),
        "Upper Confidence Bound" => UCB(sur; β=β, err=err),
        "Lower Confidence Bound" => LCB(sur; β=β, err=err)
    )
end

function get_random_acquisitionfn(sur::GaussianProcess; β=1., err=1e-6)
    testfns = get_acquisition_functions(sur; β=β, err=err)
    names = collect(keys(testfns))
    rand_name = names[rand(1:length(names))]
    return testfns[rand_name], rand_name
end

function plotaf1d(af::AcquisitionFunction; interval::AbstractRange)
    fx = zeros(length(interval))
    for i in 1:length(interval)
        xi = [interval[i]]
        fx[i] = af.f(xi)
    end
    # p = plot(interval, fx, label="${af.name}")
    p = plot(interval, fx)
    # scatter!(af.X', get_observations(gp), label="Observations")
    return p
end
