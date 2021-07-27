module EKF
    using StaticArrays: length
using Base: Float64
export State, ErrorState, Input, Measurement, ErrorMeasurement
    export ErrorStateFilter, estimateState!, prediction, innovation, update!
    export measure, process, error_measure_jacobian, error_process_jacobian, ⊖ₘ, ⊕ₛ
    export prediction!, update!

    using StaticArrays
    using LinearAlgebra: inv, I, issymmetric, isposdef, length
    using LinearAlgebra

    include("abstract_states.jl")

    const state_size = 0::Int64
    const error_state_size = 0::Int64
    const error_measure_size = 0::Int64

    struct ErrorStateFilter{S{T}<:State, ES{T}<:ErrorState, IN{T}<:Input,
                            M{T}<:Measurement, EM{T}<:ErrorMeasurement}
        est_state::MVector
        est_cov::MMatrix

        process_cov::MMatrix  # Dynamics Noise Covariance
        measure_cov::MMatrix  # Measurement Noise Covariance

        function ErrorStateFilter{S, ES, IN, M, EM}(est_state::S, est_cov::Matrix,
                                                    process_cov::Matrix, measure_cov::Matrix
                                                    ) where {S, ES, IN, M, EM}
            try
                process(est_state, rand(IN), rand())
            catch MethodError
                error("User must define the `process` function: `process(state::S, input::IN, dt::Float64)`")
            end
            try
                measure(est_state)
            catch MethodError
                error("User must define the `measure` function: `measure(state::S)`")
            end
            try
                error_process_jacobian(est_state, rand(IN), rand())
            catch MethodError
                error("User must define the `error_process_jacobian` function: `error_process_jacobian(state::S, input::IN, dt::Float64)`")
            end
            try
                error_measure_jacobian(est_state)
            catch MethodError
                error("User must define the `error_measure_jacobian` function: `error_measure_jacobian(state::S)`")
            end

            est_state = MVector{length(est_state)}(est_state)
            est_cov = MMatrix{size(est_cov)...}(est_cov)
            process_cov = MMatrix{size(process_cov)...}(process_cov)
            measure_cov = MMatrix{size(measure_cov)...}(measure_cov)

            return new{S, ES, IN, M, EM}(est_state, est_cov, process_cov, measure_cov)
        end
    end

    function prediction!(ekf::ErrorStateFilter{S, ES, IN, M, EM}, uₖ::IN, dt::Float64
                        )::Nothing where {S<:State, ES<:ErrorState, IN<:Input,
                                          M<:Measurement, EM<:ErrorMeasurement}
        # return nothing
        xₖₗₖ = S(ekf.est_state)
        Pₖₗₖ = SMatrix(ekf.est_cov)
        W = SMatrix(ekf.process_cov)

        xₖ₊₁ₗₖ = process(xₖₗₖ, uₖ, dt)
        Aₖ = error_process_jacobian(xₖₗₖ, uₖ, dt)
        Pₖ₊₁ₗₖ = Aₖ * Pₖₗₖ * (Aₖ)' + W

        ekf.est_state .= xₖ₊₁ₗₖ
        ekf.est_cov .= Pₖ₊₁ₗₖ

        return nothing
    end

    function innovation(ekf::ErrorStateFilter{S, ES, IN, M, EM},
                        xₖ₊₁ₗₖ::S, Pₖ₊₁ₗₖ::SMatrix, yₖ::M
                        ) where {S<:State, ES<:ErrorState, IN<:Input,
                                 M<:Measurement, EM<:ErrorMeasurement}
        # Relabeling
        V = ekf.measure_cov

        # Innovation
        zₖ₊₁ = measurement_error(yₖ, measure(xₖ₊₁ₗₖ))
        Cₖ₊₁ = error_measure_jacobian(xₖ₊₁ₗₖ)
        Sₖ₊₁ = Cₖ₊₁ * Pₖ₊₁ₗₖ * (Cₖ₊₁)' + V

        # Kalman Gain
        Lₖ₊₁ = Pₖ₊₁ₗₖ * (Cₖ₊₁)' / (Sₖ₊₁)

        return zₖ₊₁, Cₖ₊₁, Lₖ₊₁
    end

    function update!(ekf::ErrorStateFilter{S, ES, IN, M, EM},
                     yₖ::M) where {S<:State, ES<:ErrorState, IN<:Input,
                              M<:Measurement, EM<:ErrorMeasurement}
        Pₖ₊₁ₗₖ = ekf.est_cov
        xₖ₊₁ₗₖ = ekf.est_state

        zₖ₊₁, Cₖ₊₁, Lₖ₊₁= innovation(ekf, xₖ₊₁ₗₖ, Pₖ₊₁ₗₖ, yₖ)

        # Update
        xₖ₊₁ₗₖ₊₁ = state_composition(xₖ₊₁ₗₖ, ES(Lₖ₊₁ * zₖ₊₁))
        Pₖ₊₁ₗₖ₊₁ = Pₖ₊₁ₗₖ - Lₖ₊₁ * Cₖ₊₁ * Pₖ₊₁ₗₖ

        ekf.est_state = xₖ₊₁ₗₖ₊₁
        ekf.est_cov = Pₖ₊₁ₗₖ₊₁
    end


    function estimateState!(ekf::ErrorStateFilter{S, ES, IN, M, EM},
                            input::IN, measurement::M, dt::Float64
                            ) where {S<:State, ES<:ErrorState, IN<:Input,
                                     M<:Measurement, EM<:ErrorMeasurement}
        # Relabeling
        xₖₗₖ = ekf.est_state
        Pₖₗₖ = ekf.est_cov
        uₖ = input
        yₖ = measurement

        # Predict
        prediction!(ekf, uₖ, dt=dt)

        # Update
        xₖ₊₁ₗₖ₊₁, Pₖ₊₁ₗₖ₊₁ = update!(ekf, yₖ)

        return xₖ₊₁ₗₖ₊₁, Pₖ₊₁ₗₖ₊₁
    end
end # module ErrorStateEKF
