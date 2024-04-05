struct AcquisitionFunction{T}
    f::T
end

function UCB(s::ZeroMeanGaussianProcess; β=1., err=1e-6)
    UCBx(x::Float64) = begin
        sx = s
        if sx.σ <= err
            return NaN # Return NaN when standard deviation is too low
        end
        return sx.μ + β * sx.σ
    end
    return AcquisitionFunction(UCBx)
end

function LCB(s::ZeroMeanGaussianProcess; β=1., err=1e-6)
    LCBx(x::Float64) = begin
        sx = s
        if sx.σ <= err
            return NaN # Return NaN when standard deviation is too low
        end
        return sx.μ - β * sx.σ
    end
    return AcquisitionFunction(LCBx)
end

function POI(s::ZeroMeanGaussianProcess; β=1., err=1e-6)
    f⁺ = minimum(s.y)
    POIx(x::Float64) = begin
        sx = s
        if sx.σ <= err
            return 0 # Return 0 when standard deviation is too low
        end
        Φx = cdf(Normal(), z(sx.μ, sx.σ, f⁺; β=β))
        return Φx
    end
    return AcquisitionFunction(POIx)
end

function EI(s::ZeroMeanGaussianProcess; β=1., err=1e-6)
    f⁺ = minimum(s.y)
    EIx(x::Float64) = begin
        sx = s
        if sx.σ <= err
            return 0 # Return 0 when standard deviation is too low
        end
        zx = z(sx.μ, sx.σ, f⁺; β=β)
        Φx = cdf(Normal(), zx)
        ϕx = pdf(Normal(), zx)
        return (zx * sx.σ) * Φx + sx.σ * ϕx
    end
    return AcquisitionFunction(EIx)
end

function get_acquisition_functions(sur::ZeroMeanGaussianProcess; β=1., err=1e-6)
    return Dict(
        "Expected Improvement" => EI(sur; β=β, err=err),
        "Probability of Improvement" => POI(sur; β=β, err=err),
        "Upper Confidence Bound" => UCB(sur; β=β, err=err),
        "Lower Confidence Bound" => LCB(sur; β=β, err=err)
    )
end

function get_random_acquisitionfn(sur::ZeroMeanGaussianProcess; β=1., err=1e-6)
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
