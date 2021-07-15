using EKF
using StaticArrays: length
using LinearAlgebra: I 

include("imu_states.jl")
include("imu_dynamics.jl")


est_state = rand(ImuState)
est_cov = MMatrix{length(ImuErrorState), length(ImuErrorState)}(.3 * I(length(ImuErrorState)))

process_cov = MMatrix{length(ImuErrorState), length(ImuErrorState)}(.3 * I(length(ImuErrorState)))
measure_cov = MMatrix{length(ViconErrorMeasurement), length(ViconErrorMeasurement)}(.3 * I(length(ViconErrorMeasurement)))

ekf = ErrorStateFilter{ImuState, ImuErrorState, ImuInput, ViconMeasurement, 
                       ViconErrorMeasurement}(est_state, est_cov, process_cov, measure_cov)

input = rand(ImuInput)
measurement = rand(ViconMeasurement)

println(EKF.measure(ekf.est_state) ⊖ₘ measurement)
println(prediction(ekf, est_state, est_cov, input; dt=0.1))