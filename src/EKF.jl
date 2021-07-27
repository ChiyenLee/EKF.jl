module EKF
    export State, ErrorState, Input, Measurement, ErrorMeasurement
    export ErrorStateFilter, estimateState!, prediction, innovation, update!
    export measure, process, error_measure_jacobian, error_process_jacobian, ⊖ₘ, ⊕ₛ
    export prediction!, update!

    using StaticArrays
    using LinearAlgebra: inv, I, issymmetric, isposdef
    using LinearAlgebra

    include("abstract_states.jl")

    struct ErrorStateFilter{S<:State, ES<:ErrorState, IN<:Input,
                            M<:Union{Measurement, Vector{Measurement}}, EM<:ErrorMeasurement}
        est_state::S
        est_cov::Matrix

        process_cov::Matrix  # Dynamics Noise Covariance
        measure_cov::Matrix  # Measurement Noise Covariance

        function ErrorStateFilter{S, ES, IN, M, EM}(est_state::S, est_cov::SMatrix,
                                                    process_cov::SMatrix, measure_cov::SMatrix
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
                error_process_jacobian(est_state, rand(IN), rand())
                error("User must define the `error_process_jacobian` function: `error_process_jacobian(state::S, input::IN, dt::Float64)`")
            end
            try
                error_measure_jacobian(est_state)
            catch MethodError
                error_measure_jacobian(est_state)
                error("User must define the `error_measure_jacobian` function: `error_measure_jacobian(state::S)`")
            end


            return new{S, ES, IN, M, EM}(est_state, est_cov, process_cov, measure_cov)
        end
    end

    function get_attitude_jacobian(ekf)
        return att_jacobian
    end 

    function prediction!(ekf::ErrorStateFilter{S, ES, IN, M, EM}, uₖ::IN; dt=0.1
                        ) where {S<:State, ES<:ErrorState, IN<:Input,
                                 M<:Measurement, EM<:ErrorMeasurement}
        W = ekf.process_cov
        Pₖₗₖ = ekf.est_cov
        xₖₗₖ = ekf.est_state

        xₖ₊₁ₗₖ = process(xₖₗₖ, uₖ, dt)
        Aₖ = error_process_jacobian(xₖₗₖ, uₖ, dt)
        Pₖ₊₁ₗₖ = Aₖ * Pₖₗₖ * (Aₖ)' + W

        ekf.est_state .= xₖ₊₁ₗₖ
        ekf.est_cov .= Pₖ₊₁ₗₖ
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

        ekf.est_state .= xₖ₊₁ₗₖ₊₁
        ekf.est_cov .= Pₖ₊₁ₗₖ₊₁
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
        prediction!(ekf, uₖ; dt=dt)

        # Update
        xₖ₊₁ₗₖ₊₁, Pₖ₊₁ₗₖ₊₁ = update!(ekf, yₖ)

        return xₖ₊₁ₗₖ₊₁, Pₖ₊₁ₗₖ₊₁
    end
end # module ErrorStateEKF
