using Pkg 
Pkg.activate(".")
using StaticArrays
using Revise
using EKF

state = TrunkState{Float64}(zeros(16)); state.qw = 1.0
state_error = TrunkError{Float64}(zeros(15))

imuIn = ImuInput{Float64}(zeros(6))
vicon = Vicon{Float64}(zeros(7)); vicon.qw = 1.0

# h = 0.01
# state_next = EKF.process(state, imuIn, 0.01)
# F = EKF.error_process_jacobian(state, imuIn, h)
# vicon_meas = EKF.measure(state_next)
# H = EKF.error_measurement_jacobian(vicon_meas, state_next)

# EKF.⊖ₘ(vicon, vicon_meas)

