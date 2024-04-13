struct AcquisitionFunction{T}
    f::T
end

function UCB(s::GaussianProcess; β=1., err=1e-6)
    UCBx(x::AbstractVector) = begin
        μ, σ = predict(s, x)
        if σ <= err
            return NaN # Return NaN when standard deviation is too low
        end
        return μ + β * σ
    end
    return AcquisitionFunction(UCBx)
end

function LCB(s::GaussianProcess; β=1., err=1e-6)
    LCBx(x::AbstractVector) = begin
        μ, σ = predict(s, x)
        if σ <= err
            return NaN # Return NaN when standard deviation is too low
        end
        return μ - β * σ
    end
    return AcquisitionFunction(LCBx)
end

function POI(s::GaussianProcess; β=1., err=1e-6)
    f⁺ = minimum(s.y)
    POIx(x::AbstractVector) = begin
        μ, σ = predict(s, x)
        if σ <= err
            return 0 # Return 0 when standard deviation is too low
        end
        Φx = cdf(Normal(), z(μ, σ, f⁺; β=β))
        return Φx
    end
    return AcquisitionFunction(POIx)
end

function EI(s::GaussianProcess; β=1., err=1e-6)
    f⁺ = minimum(s.y)
    EIx(x::AbstractVector) = begin
        μ, σ = predict(s, x)
        if σ <= err
            return 0 # Return 0 when standard deviation is too low
        end
        zx = z(μ, σ, f⁺; β=β)
        Φx = cdf(Normal(), zx)
        ϕx = pdf(Normal(), zx)
        return (zx * σ) * Φx + σ * ϕx
    end
    return AcquisitionFunction(EIx)
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
    stdx = zeros(length(interval))
    normal_sample = randn()
    p = plot(interval, fx, ribbons=2stdx, label="μ ± 2σ")
    # scatter!(af.X', get_observations(gp), label="Observations")
    return p
end
