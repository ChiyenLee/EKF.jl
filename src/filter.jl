struct ErrorStateFilter{S<:State, ES<:ErrorState, IN<:Input,
                        Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T}
    est_state::MVector{Nₛ, T}
    est_cov::MMatrix{Nₑₛ, Nₑₛ, T,  Lₑₛ}
    process_cov::MMatrix{Nₑₛ, Nₑₛ, T,  Lₑₛ}  # Dynamics Noise Covariance

    function ErrorStateFilter{S, ES, IN}(est_state::AbstractVector{T},
                                         est_cov::AbstractMatrix{T},
                                         process_cov::AbstractMatrix{T}) where {S, ES, IN, T}

        Nₛ, Nₑₛ, Nᵢₙ = length.([S, ES, IN])
        Lₑₛ = Nₑₛ * Nₑₛ

        est_state = MVector(est_state)
        est_cov = MMatrix{Nₑₛ, Nₑₛ, T, Lₑₛ}(est_cov)
        process_cov = MMatrix{Nₑₛ, Nₑₛ, T, Lₑₛ}(process_cov)

        return new{S, ES, IN, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T}(est_state, est_cov, process_cov)
    end
end

function prediction!(ekf::ErrorStateFilter{S, ES, IN},
                     uₖ::IN,
                     dt::Float64,
                     )::Nothing where {S<:State, ES<:ErrorState, IN<:Input}
    W = SMatrix(ekf.process_cov)
    Pₖₗₖ = SMatrix(ekf.est_cov)
    xₖₗₖ = S(ekf.est_state)

    xₖ₊₁ₗₖ = process(xₖₗₖ, uₖ, dt)
    Aₖ = error_process_jacobian(xₖₗₖ, uₖ, dt)
    Pₖ₊₁ₗₖ = Aₖ * Pₖₗₖ * (Aₖ)' + W

    ekf.est_state .= xₖ₊₁ₗₖ
    ekf.est_cov .= Pₖ₊₁ₗₖ

    return nothing
end

function innovation(ekf::ErrorStateFilter{S, ES, IN, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T},
                    xₖ₊₁ₗₖ::S,
                    Pₖ₊₁ₗₖ::SMatrix{Nₑₛ, Nₑₛ, T, Lₑₛ},
                    oₖ::Observation{M},
                    ) where {S<:State, ES<:ErrorState, IN<:Input, M<:Measurement, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T}
    # Relabeling
    yₖ = getMeasurement(oₖ)
    V = getCovariance(oₖ)

    # # Innovation
    zₖ₊₁ = measurement_error(yₖ, measure(M, xₖ₊₁ₗₖ))
    Cₖ₊₁ = error_measure_jacobian(M, xₖ₊₁ₗₖ)

    Sₖ₊₁ = Cₖ₊₁ * Pₖ₊₁ₗₖ * (Cₖ₊₁)' + V

    # Kalman Gain
    Lₖ₊₁ = Pₖ₊₁ₗₖ * (Cₖ₊₁)' / (Sₖ₊₁)

    return zₖ₊₁, Cₖ₊₁, Lₖ₊₁
end

function update!(ekf::ErrorStateFilter{S, ES, IN},
                 oₖ::Observation,
                 )::Nothing where {S<:State, ES<:ErrorState, IN<:Input}
    Pₖ₊₁ₗₖ = SMatrix(ekf.est_cov)
    xₖ₊₁ₗₖ = S(ekf.est_state)

    zₖ₊₁, Cₖ₊₁, Lₖ₊₁ = innovation(ekf, xₖ₊₁ₗₖ, Pₖ₊₁ₗₖ, oₖ)

    # Update
    xₖ₊₁ₗₖ₊₁ = state_composition(xₖ₊₁ₗₖ, ES(Lₖ₊₁ * zₖ₊₁))
    Pₖ₊₁ₗₖ₊₁ = Pₖ₊₁ₗₖ - Lₖ₊₁ * Cₖ₊₁ * Pₖ₊₁ₗₖ

    ekf.est_state .= xₖ₊₁ₗₖ₊₁
    ekf.est_cov .= Pₖ₊₁ₗₖ₊₁

    return nothing
end


function estimateState!(ekf::ErrorStateFilter{S, ES, IN},
                        input::IN,
                        measurement::M,
                        dt::Float64
                        )::Nothing where {S<:State, ES<:ErrorState, IN<:Input, M<:Measurement}
    # Relabeling
    uₖ = input
    yₖ = measurement

    # Predict
    prediction!(ekf, uₖ, dt)

    # Update
    update!(ekf, yₖ)

    return nothing
end
