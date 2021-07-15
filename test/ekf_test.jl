using Pkg 
using StaticArrays
using Revise
using EKF
using LinearAlgebra

include("imu_states.jl")
include("imu_dynamics.jl")

state = ImuState{Float64}(zeros(16)); state.q𝑤 = 1
state_err = ImuErrorState{Float64}(zeros(15)); 
input = ImuInput{Float64}(zeros(6))
vicon = ViconMeasurement{Float64}(zeros(7)); vicon.q𝑤 = 1
vicon_err = ViconErrorMeasurement{Float64}(zeros(6))

P = Matrix{Float64}(1.0I(length(state_err))) * 1e-2
W = Matrix{Float64}(1.0I(length(state_err))) * 1e-2
R = Matrix{Float64}(1.0I(length(vicon_err))) * 1e-2

ekf = ErrorStateFilter{ImuState, ErrorState, ImuInput,
          ViconMeasurement, ViconErrorMeasurement}(state, P, W, R)

input.v̇𝑥 = 10

# x_next, P_next = prediction(ekf, state, P, input, dt=0.01)
# innovation(ekf, x_next, P, vicon)
estimateState!(ekf, input, vicon, 0.01)
