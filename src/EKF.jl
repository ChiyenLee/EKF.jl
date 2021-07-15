module EKF
    export State, ErrorState, Input, Measurement, ErrorMeasurement
    export TrunkState, TrunkError, ImuInput, Vicon, ViconError
    export ErrorStateFilter, estimateState!, prediction, innovation, update!
    export measure, process, error_measure_jacobian, error_process_jacobian, ⊖ₘ, ⊕ₛ

    using StaticArrays
    using LinearAlgebra: inv, I, issymmetric, isposdef
    using LinearAlgebra

    include("abstract_states.jl") 

    struct ErrorStateFilter{S<:State, ES<:ErrorState, IN<:Input, 
                            M<:Measurement, EM<:ErrorMeasurement} 
        est_state::S
        est_cov::Matrix

        process_cov::Matrix  # Dynamics Noise Covariance
        measure_cov::Matrix  # Measurement Noise Covariance

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
                error_measure_jacobian(est_state, rand(M))
            catch MethodError
                error("User must define the `error_measure_jacobian` function: `error_measure_jacobian(state::S)`")
            end
            # try 
            #     temp = ⊕ₛ(rand(S), rand(ES))
            #     @assert temp isa S
            # catch MethodError
            #     error("User must define the `⊕ₛ` function: `⊕ₛ(state::S, err_state::ES)::S`")
            # end
            # try 
            #     temp = ⊖ₘ(rand(M), rand(M))
            #     @assert temp isa EM
            # catch MethodError
            #     error("User must define the `⊖ₘ` function: `⊖ₘ(measurement1::M, measurement2::M)::EM`")
            # end
            
            return new{S, ES, IN, M, EM}(est_state, est_cov, process_cov, measure_cov)
        end
    end


    function prediction(ekf::ErrorStateFilter{S, ES, IN, M, EM}, 
                        xₖₗₖ::S, Pₖₗₖ::Matrix, uₖ::IN; dt=0.1
                        ) where {S<:State, ES<:ErrorState, IN<:Input, 
                                 M<:Measurement, EM<:ErrorMeasurement}
        # Relabeling
        W = ekf.process_cov

        xₖ₊₁ₗₖ = process(xₖₗₖ, uₖ, dt)
        Aₖ = error_process_jacobian(xₖₗₖ, uₖ, dt)

        Pₖ₊₁ₗₖ = Aₖ * Pₖₗₖ * (Aₖ)' + W

        return xₖ₊₁ₗₖ, Pₖ₊₁ₗₖ
    end


    function innovation(ekf::ErrorStateFilter{S, ES, IN, M, EM}, 
                        xₖ₊₁ₗₖ::S, Pₖ₊₁ₗₖ::Matrix, yₖ::M
                        ) where {S<:State, ES<:ErrorState, IN<:Input, 
                                 M<:Measurement, EM<:ErrorMeasurement}
        # Relabeling
        V = ekf.measure_cov

        # Innovation
        zₖ₊₁ = measurement_error(measure(xₖ₊₁ₗₖ), yₖ)
        Cₖ₊₁ = error_measure_jacobian(xₖ₊₁ₗₖ, yₖ)
        Sₖ₊₁ = Cₖ₊₁ * Pₖ₊₁ₗₖ * (Cₖ₊₁)' + V

        # Kalman Gain
        Lₖ₊₁ = Pₖ₊₁ₗₖ * (Cₖ₊₁)' / (Sₖ₊₁)

        return zₖ₊₁, Cₖ₊₁, Lₖ₊₁
    end


    function update!(ekf::ErrorStateFilter{S, ES, IN, M, EM}, 
                     xₖ₊₁ₗₖ::S, Pₖ₊₁ₗₖ::Matrix, zₖ₊₁::EM, 
                     Cₖ₊₁::Matrix, Lₖ₊₁::Matrix
                     ) where {S<:State, ES<:ErrorState, IN<:Input, 
                              M<:Measurement, EM<:ErrorMeasurement}
        # Update
        xₖ₊₁ₗₖ₊₁ = state_composition(xₖ₊₁ₗₖ, ES(Lₖ₊₁ * zₖ₊₁))
        Pₖ₊₁ₗₖ₊₁ = Pₖ₊₁ₗₖ - Lₖ₊₁ * Cₖ₊₁ * Pₖ₊₁ₗₖ

        ekf.est_state .= xₖ₊₁ₗₖ₊₁
        ekf.est_cov .= Pₖ₊₁ₗₖ₊₁

        return xₖ₊₁ₗₖ₊₁, Pₖ₊₁ₗₖ₊₁
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
        xₖ₊₁ₗₖ, Pₖ₊₁ₗₖ = prediction(ekf, xₖₗₖ, Pₖₗₖ, uₖ, dt=dt)

        # Innovation
        zₖ₊₁, Cₖ₊₁, Lₖ₊₁ = innovation(ekf, xₖ₊₁ₗₖ, Pₖ₊₁ₗₖ, yₖ)

        # Update
        xₖ₊₁ₗₖ₊₁, Pₖ₊₁ₗₖ₊₁ = update!(ekf, xₖ₊₁ₗₖ, Pₖ₊₁ₗₖ, zₖ₊₁, Cₖ₊₁, Lₖ₊₁)

        ekf.est_state .= xₖ₊₁ₗₖ₊₁
        ekf.est_cov .= Pₖ₊₁ₗₖ₊₁

        return xₖ₊₁ₗₖ₊₁, Pₖ₊₁ₗₖ₊₁
    end
end # module ErrorStateEKF
