mutable struct ErrorStateFilter{S<:State, ES<:ErrorState, IN<:Input,
                        Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T}
    est_state::SVector{Nₛ, T}
    est_cov::SMatrix{Nₑₛ, Nₑₛ, T,  Lₑₛ}
    process_cov::SMatrix{Nₑₛ, Nₑₛ, T,  Lₑₛ}  # Dynamics Noise Covariance

    function ErrorStateFilter{S, ES, IN}(est_state::AbstractVector{T},
                                         est_cov::AbstractMatrix{T},
                                         process_cov::AbstractMatrix{T}) where {S, ES, IN, T}

        Nₛ, Nₑₛ, Nᵢₙ = length.([S, ES, IN])
        Lₑₛ = Nₑₛ * Nₑₛ

        est_state = SVector(est_state)
        est_cov = SMatrix{Nₑₛ, Nₑₛ, T, Lₑₛ}(est_cov)
        process_cov = SMatrix{Nₑₛ, Nₑₛ, T, Lₑₛ}(process_cov)

        return new{S, ES, IN, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T}(est_state, est_cov, process_cov)
    end
end

function updateProcessCov!(ekf::ErrorStateFilter{S, ES, IN, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T},
                           process_cov::SMatrix{Nₑₛ, Nₑₛ, T,  Lₑₛ},
                           ) where {S<:State, ES<:ErrorState, IN<:Input, Nₛ, Nₑₛ, Nᵢₙ, T,  Lₑₛ}
    ekf.process_cov = process_cov
end

function prediction!(ekf::ErrorStateFilter{S, ES, IN, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T},
                     uₖ::IN,
                     dt::T,
                     )::Nothing where {S<:State, ES<:ErrorState, IN<:Input, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T}
    xₖₗₖ = S(SVector{Nₛ,T}(ekf.est_state))
    Pₖₗₖ = ekf.est_cov
    W = ekf.process_cov

    xₖ₊₁ₗₖ = process(xₖₗₖ, uₖ, dt)
    Aₖ = error_process_jacobian(xₖₗₖ, uₖ, dt)
    Pₖ₊₁ₗₖ = Aₖ * Pₖₗₖ * (Aₖ)' + W

    ekf.est_state = xₖ₊₁ₗₖ
    ekf.est_cov = Pₖ₊₁ₗₖ

    return nothing
end

function innovation(ekf::ErrorStateFilter{S, ES, IN, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T},
                    xₖ₊₁ₗₖ::S,
                    Pₖ₊₁ₗₖ::SMatrix{Nₑₛ, Nₑₛ, T, Lₑₛ},
                    oₖ::Observation{M, Nₘ, Nₑₘ, T},
                    ) where {S<:State, ES<:ErrorState, IN<:Input, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, M<:Measurement, Nₘ, Nₑₘ, T,}
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

function update!(ekf::ErrorStateFilter{S, ES, IN, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T},
                 oₖ::Observation{M, Nₘ, Nₑₘ, T},
                 )::Nothing where {S<:State, ES<:ErrorState, IN<:Input, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, M<:Measurement, Nₘ, Nₑₘ, T,}
    xₖ₊₁ₗₖ = S(SVector{Nₛ,T}(ekf.est_state))
    Pₖ₊₁ₗₖ = ekf.est_cov
    R = getCovariance(oₖ)

    zₖ₊₁, Cₖ₊₁, Lₖ₊₁ = innovation(ekf, xₖ₊₁ₗₖ, Pₖ₊₁ₗₖ, oₖ)

    # Update
    xₖ₊₁ₗₖ₊₁ = state_composition(xₖ₊₁ₗₖ, ES(SVector{Nₑₛ,T}(Lₖ₊₁ * zₖ₊₁)))

    # Joseph form covariance update
    A = (I - Lₖ₊₁ * Cₖ₊₁)
    Pₖ₊₁ₗₖ₊₁ = A * Pₖ₊₁ₗₖ * A' + Lₖ₊₁ * R * Lₖ₊₁'

    ekf.est_state = xₖ₊₁ₗₖ₊₁
    ekf.est_cov = Pₖ₊₁ₗₖ₊₁

    return nothing
end


function estimateState!(ekf::ErrorStateFilter{S, ES, IN, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T},
                        input::IN,
                        measurement::M,
                        dt::T
                        )::Nothing where {S<:State, ES<:ErrorState, IN<:Input, Nₛ, Nₑₛ, Nᵢₙ, Lₑₛ, T, M<:Measurement}
    # Relabeling
    uₖ = input
    yₖ = measurement

    # Predict
    prediction!(ekf, uₖ, dt)

    # Update
    update!(ekf, yₖ)

    return nothing
end
