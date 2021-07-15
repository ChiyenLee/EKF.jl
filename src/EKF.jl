module EKF

    using StaticArrays
    using LinearAlgebra: inv, I, issymmetric, isposdef
    using ForwardDiff: jacobian
    using Rotations 
    using Rotations: rotation_error, CayleyMap, RotationError, add_error

    export ErrorStateFilter, State, ErrorState, Measurement, ErrorMeasurement
    export Input, estimateState!
    export error_state_jacobian, error_measurement_jacobian, getComponents

    include("abstract_states.jl") 
    # include("states/trunkstate.jl")   


    # struct ErrorStateFilter{S <: State{a} where a, ES <: ErrorState{c} where {c}, In <: Input, 
                            # M <: Measurement, EM <: ErrorMeasurement}
    struct ErrorStateFilter{S<:State, ES<:ErrorState, 
                            IN<:Input, 
                            M<:Measurement, EM<:ErrorMeasurement} 
        est_state::S
        est_cov::MMatrix

        process_cov::MMatrix  # Dynamics Noise Covariance
        measure_cov::MMatrix  # Measurement Noise Covariance

        function ErrorStateFilter{S, ES, IN, M, EM}(est_state::S, est_cov::MMatrix, 
                                                    process_cov::MMatrix, measure_cov::MMatrix
                                                    ) where {S, ES, IN, M, EM}
            try 
                process(est_state, rand(IN), rand())
            catch MethodError
                println("User must define the `process` function: \n`process(state::S, input::IN, dt::Float64)`")
            end
            try 
                measure(est_state)
            catch MethodError
                println("User must define the `measure` function: \n`measure(state::S)`")
            end
            try 
                error_process_jacobian(est_state, rand(IN), rand())
            catch MethodError
                println("User must define the `error_process_jacobian` function: \n`error_process_jacobian(state::S, input::IN, dt::Float64)`")
            end
            try 
                error_measure_jacobian(est_state)
            catch MethodError
                println("User must define the `process` function: \n`error_measure_jacobian(state::S)`")
            end

            return new{S, ES, In, M, EM}(est_state, est_cov, process_cov, measure_cov)
        end
    end


    function prediction(ekf::ErrorStateFilter{S, ES, In, M, EM}, 
                        xₖₗₖ::S, Pₖₗₖ::MMatrix, uₖ::In; 
                        dt=0.1) where {S<:State, ES<:ErrorState, In<:Input, 
                                       M<:Measurement, EM<:ErrorMeasurement}
        # Relabeling
        # xₖₗₖ = ekf.est_state
        # Pₖₗₖ = ekf.est_cov
        # uₖ = input
        W = ekf.process_cov

        xₖ₊₁ₗₖ = ekf.process(xₖₗₖ, uₖ, dt)
        Jₖ = ekf.error_state_jacobian(xₖₗₖ)         # ∂(xₖ₋₁)/∂(dxₖ₋₁)
        Jₖ₊₁ = ekf.error_state_jacobian(xₖ₊₁ₗₖ)     # ∂(dxₖ)/∂xₖ
        # Aₖ = ∂(dxₖ)/∂xₖ * ∂f(xₖ,uₖ)/∂(xₖ₋₁) * ∂(xₖ₋₁)/∂(dxₖ₋₁)
        Aₖ = (Jₖ₊₁)' * ekf.process_jacobian(xₖₗₖ, uₖ, dt) * Jₖ
        Pₖ₊₁ₗₖ = Aₖ * Pₖₗₖ * (Aₖ)' + W

        return xₖ₊₁ₗₖ, Pₖ₊₁ₗₖ
    end


    function innovation(ekf::ErrorStateFilter{S, ES, In, M, EM}, 
                        xₖ₊₁ₗₖ::S, Pₖ₊₁ₗₖ::MMatrix, 
                        yₖ::M) where {S<:State, ES<:ErrorState, In<:Input, 
                                     M<:Measurement, EM<:ErrorMeasurement}
        # Relabeling
        # xₖ₊₁ₗₖ = state
        # Pₖ₊₁ₗₖ = errorCov
        # yₖ = measurement
        V = ekf.measure_cov

        # Innovation
        zₖ₊₁ = ekf.measure(xₖ₊₁ₗₖ) ⊖ₘ yₖ
        Jₖ₊₁ = ekf.error_state_jacobian(xₖ₊₁ₗₖ)          # ∂(xₖₗₖ₋₁)/∂(dxₖₗₖ₋₁)
        Gₖ₊₁ = ekf.error_measurement_jacobian(yₖ)       # ∂(dyₖ)/∂yₖ
        # Cₖ₊₁ = ∂(dz)/∂z * ∂z/∂x * ∂x/∂(dx) |_{xₖₗₖ₋₁, zₖ}
        Cₖ₊₁ = (Gₖ₊₁)' * ekf.measure_jacobian(xₖ₊₁ₗₖ) * Jₖ₊₁
        Sₖ₊₁ = Cₖ₊₁ * Pₖ₊₁ₗₖ * (Cₖ₊₁)' + V

        # Kalman Gain
        Lₖ₊₁ = Pₖ₊₁ₗₖ * (Cₖ₊₁)' / (Sₖ₊₁)

        return zₖ₊₁, Cₖ₊₁, Lₖ₊₁
    end


    function update!(ekf::ErrorStateFilter{S, ES, In, M, EM}, 
                     xₖ₊₁ₗₖ::S, Pₖ₊₁ₗₖ::MMatrix, zₖ₊₁::EM, 
                     Cₖ₊₁::MMatrix, Lₖ₊₁::MMatrix) where {S<:State, ES<:ErrorState, 
                                                         In<:Input, 
                                                         M<:Measurement, EM<:ErrorMeasurement}
        # Update
        xₖ₊₁ₗₖ₊₁ = xₖ₊₁ₗₖ ⊕ₛ ES(Lₖ₊₁ * zₖ₊₁)
        Pₖ₊₁ₗₖ₊₁ = Pₖ₊₁ₗₖ - Lₖ₊₁ * Cₖ₊₁ * Pₖ₊₁ₗₖ

        ekf.est_state .= xₖ₊₁ₗₖ₊₁
        ekf.est_cov .= Pₖ₊₁ₗₖ₊₁

        return xₖ₊₁ₗₖ₊₁, Pₖ₊₁ₗₖ₊₁
    end


    function estimateState!(ekf::ErrorStateFilter{S, ES, In, M, EM}, 
                            input::In, measurement::M, 
                            dt::Float64) where {S<:State, ES<:ErrorState, 
                                                In<:Input, 
                                                M<:Measurement, EM<:ErrorMeasurement}
        # Relabeling
        xₖₗₖ = ekf.est_state
        Pₖₗₖ = ekf.est_cov
        uₖ = input
        yₖ = measurement
        W = ekf.process_cov
        V = ekf.measure_cov

        # Predict
        xₖ₊₁ₗₖ = ekf.process(xₖₗₖ, uₖ, dt)
        Jₖ = ekf.error_state_jacobian(xₖₗₖ)         # ∂(xₖ₋₁)/∂(dxₖ₋₁)
        Jₖ₊₁ = ekf.error_state_jacobian(xₖ₊₁ₗₖ)     # ∂(dxₖ)/∂xₖ
        # Aₖ = ∂(dxₖ)/∂xₖ * ∂f(xₖ₋₁,uₖ₋₁)/∂(xₖ₋₁) * ∂(xₖ₋₁)/∂(dxₖ₋₁)
        # Aₖ = ∂(dxₖ)/∂xₖ * ∂(xₖ)/∂(xₖ₋₁) * ∂(xₖ₋₁)/∂(dxₖ₋₁)
        Aₖ = (Jₖ₊₁)' * ekf.process_jacobian(xₖₗₖ, uₖ, dt) * Jₖ
        Pₖ₊₁ₗₖ = Aₖ * Pₖₗₖ * (Aₖ)' + W

        # Innovation
        zₖ₊₁ = ekf.measure(xₖ₊₁ₗₖ) ⊖ₘ yₖ
        Jₖ₊₁ = ekf.error_state_jacobian(xₖ₊₁ₗₖ)          # ∂(xₖₗₖ₋₁)/∂(dxₖₗₖ₋₁)
        Gₖ₊₁ = ekf.error_measurement_jacobian(yₖ)       # ∂(dyₖ)/∂(yₖ)
        # Cₖ₊₁ = ∂(dyₖ)/∂(yₖ) * ∂h(yₖ,uₖ)/∂(yₖ) * ∂(yₖ)/∂(dyₖ)
        # Cₖ₊₁ = ∂(dyₖ)/∂(yₖ) * ∂(yₖ)/∂(yₖ) * ∂(yₖ)/∂(dyₖ)
        Cₖ₊₁ = (Gₖ₊₁)' * ekf.measure_jacobian(xₖ₊₁ₗₖ) * Jₖ₊₁
        Sₖ₊₁ = Cₖ₊₁ * Pₖ₊₁ₗₖ * (Cₖ₊₁)' + V

        # Kalman Gain
        Lₖ₊₁ = Pₖ₊₁ₗₖ * (Cₖ₊₁)' / (Sₖ₊₁)

        # Update
        println(zₖ₊₁)
        println(Lₖ₊₁ * zₖ₊₁)
        xₖ₊₁ₗₖ₊₁ = xₖ₊₁ₗₖ ⊕ₛ ES(Lₖ₊₁ * zₖ₊₁)
        Pₖ₊₁ₗₖ₊₁ = Pₖ₊₁ₗₖ - Lₖ₊₁ * Cₖ₊₁ * Pₖ₊₁ₗₖ

        ekf.est_state .= xₖ₊₁ₗₖ₊₁
        ekf.est_cov .= Pₖ₊₁ₗₖ₊₁

        return xₖ₊₁ₗₖ₊₁, Pₖ₊₁ₗₖ₊₁
    end
end # module ErrorStateEKF
