struct ErrorStateFilter{S<:State, ES<:ErrorState, IN<:Input, 
                        Nₛ, Nₑₛ, Nᵢ, Lₑₛ, T}
    est_state::MVector{Nₛ, T}
    est_cov::MMatrix{Nₑₛ, Nₑₛ, T,  Lₑₛ}
    process_cov::MMatrix{Nₑₛ, Nₑₛ, T,  Lₑₛ}  # Dynamics Noise Covariance

    function ErrorStateFilter{S, ES, IN}(est_state::AbstractVector, est_cov::Matrix,
                                                process_cov::Matrix) where {S, ES, IN}

        Nₛ, Nₑₛ, Nᵢ = length.([S, ES, IN])
        Lₑₛ, T = Nₑₛ * Nₑₛ, Float64

        est_state = MVector(est_state)
        est_cov = MMatrix{Nₑₛ, Nₑₛ, T, Lₑₛ}(est_cov)
        process_cov = MMatrix{Nₑₛ, Nₑₛ, T, Lₑₛ}(process_cov)

        return new{S, ES, IN, Nₛ, Nₑₛ, Nᵢ, Lₑₛ, T}(est_state, est_cov, process_cov)
    end
end

function prediction!(ekf::ErrorStateFilter{S, ES, IN}, uₖ::IN, dt::Float64
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

function innovation(ekf::ErrorStateFilter{S, ES, IN},
                    xₖ₊₁ₗₖ::S, Pₖ₊₁ₗₖ::SMatrix, Oₖ::O
                    ) where {S<:State, ES<:ErrorState, IN<:Input, O <: Observation}
    # Relabeling
    V = SMatrix(Oₖ.measure_cov)
    yₖ = Oₖ.measurement

    # Innovation
    zₖ₊₁ = measurement_error(yₖ, measure(O, xₖ₊₁ₗₖ))
    Cₖ₊₁ = error_measure_jacobian(O, xₖ₊₁ₗₖ)
    Sₖ₊₁ = Cₖ₊₁ * Pₖ₊₁ₗₖ * (Cₖ₊₁)' + V

    # Kalman Gain
    Lₖ₊₁ = Pₖ₊₁ₗₖ * (Cₖ₊₁)' / (Sₖ₊₁)

    return zₖ₊₁, Cₖ₊₁, Lₖ₊₁
end

function update!(ekf::ErrorStateFilter{S, ES, IN}, Oₖ::Observation
                    )::Nothing where {S<:State, ES<:ErrorState, IN<:Input,
                                      M<:Measurement}
    Pₖ₊₁ₗₖ = SMatrix(ekf.est_cov)
    xₖ₊₁ₗₖ = S(ekf.est_state)

    zₖ₊₁, Cₖ₊₁, Lₖ₊₁= innovation(ekf, xₖ₊₁ₗₖ, Pₖ₊₁ₗₖ, Oₖ)

    # Update
    xₖ₊₁ₗₖ₊₁ = state_composition(xₖ₊₁ₗₖ, ES(Lₖ₊₁ * zₖ₊₁))
    Pₖ₊₁ₗₖ₊₁ = Pₖ₊₁ₗₖ - Lₖ₊₁ * Cₖ₊₁ * Pₖ₊₁ₗₖ

    ekf.est_state .= xₖ₊₁ₗₖ₊₁
    ekf.est_cov .= Pₖ₊₁ₗₖ₊₁

    return nothing
end


function estimateState!(ekf::ErrorStateFilter{S, ES, IN},
                        input::IN, measurement::M, dt::Float64
                        )::Nothing where {S<:State, ES<:ErrorState, IN<:Input,
                                          M<:Measurement}
    # Relabeling
    uₖ = input
    yₖ = measurement

    # Predict
    prediction!(ekf, uₖ, dt)

    # Update
    update!(ekf, yₖ)

    return nothing
end
