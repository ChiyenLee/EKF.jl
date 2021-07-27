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
                            M<:Measurement, EM<:ErrorMeasurement, Nₛ, Nₑₛ, Nᵢ, Nₘ, Nₑₘ, Lₑₛ,Lₑₘ, T}  
        est_state::MVector{Nₛ, T}
        est_cov::MMatrix{Nₑₛ, Nₑₛ, T,  Lₑₛ}

        process_cov::MMatrix{Nₑₛ, Nₑₛ, T,  Lₑₛ}  # Dynamics Noise Covariance
        measure_cov::MMatrix{Nₑₘ, Nₑₘ, T,  Lₑₘ}  # Measurement Noise Covariance

        function ErrorStateFilter{S, ES, IN, M, EM}(est_state::AbstractVector, est_cov::Matrix,
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
                error_process_jacobian(est_state, rand(IN), rand())
                error("User must define the `error_process_jacobian` function: `error_process_jacobian(state::S, input::IN, dt::Float64)`")
            end
            try
                error_measure_jacobian(est_state)
            catch MethodError
                error_measure_jacobian(est_state)
                error("User must define the `error_measure_jacobian` function: `error_measure_jacobian(state::S)`")
            end

            est_state = MVector(est_state)
            est_cov = MMatrix{length(ES), length(ES), Float64, length(ES)*length(ES)}(est_cov)
            process_cov = MMatrix{length(ES), length(ES), Float64, length(ES)*length(ES)}(process_cov)
            measure_cov = MMatrix{length(EM), length(EM), Float64, length(EM)*length(EM)}(measure_cov)

            return new{S, ES, IN, M, EM, length(S), length(ES),length(IN), length(M), length(EM), length(ES)*length(ES), length(EM)*length(EM), Float64}(est_state, est_cov, process_cov, measure_cov)
        end
    end

    function get_attitude_jacobian(ekf)
        return att_jacobian
    end 

    function prediction!(ekf::ErrorStateFilter{S, ES, IN, M, EM}, uₖ::IN; dt=0.1
                        ) where {S<:State, ES<:ErrorState, IN<:Input,
                                 M<:Measurement, EM<:ErrorMeasurement}
        W = SMatrix(ekf.process_cov)
        Pₖₗₖ = SMatrix(ekf.est_cov)
        xₖₗₖ = S(ekf.est_state)

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
        V = SMatrix(ekf.measure_cov)

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
        Pₖ₊₁ₗₖ = SMatrix(ekf.est_cov)
        xₖ₊₁ₗₖ = S(ekf.est_state)

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
