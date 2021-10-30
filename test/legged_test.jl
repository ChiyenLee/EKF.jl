using Revise 
using EKF
using LinearAlgebra
using ForwardDiff
using StaticArrays
using Rotations 
# using BenchmarkTools
# include("$(@__DIR__)/legged_states.jl")


# %%
## Init state and input 
s_init = zeros(length(LeggedState)); s_init[4] = 1.0
state = LeggedState(s_init);
input_init = zeros(6); input_init[3] = 9.81; #input_init[1] = 0.2
input = ImuInput(input_init)

s_error = LeggedError(zero(LeggedError))
R1 = @SMatrix [1e-2 0.0 0.0;	
			   0.0 1e-2 0.0;
			   0.0 0.0 1e-2]
## Init Observation 
contact1 = ContactObservation1(ContactMeasure([-1.0, 0.5, 0.2]), 
							   ErrorContactMeasure([0.0, 0.0, 0.0]),
							   R1)

contact2 = ContactObservation2(ContactMeasure([-0.5, -0.5, 0.1]), 
							   ErrorContactMeasure([0.0, 0.0, 0.0]),
							   R1)

contact3 = ContactObservation3(ContactMeasure([0.5, 0.5, 0.3]), 
							   ErrorContactMeasure([0.0, 0.0, 0.0]),
							   R1)

P = Matrix(1.0I(length(LeggedError))) * 1e2; 
P[1:6, 1:6] .= I(6) * 0.01
W = Matrix(1.0I(length(LeggedError))) * 1e-1;
# R = Matrix(1.0I(length(ViconError))) * 1e-3;

# q_rand = normalize(randn(4))
# vicon = Vicon([0. 0. 0. q_rand...])

ekf = EKF.ErrorStateFilter{LeggedState, LeggedError, ImuInput}(state, P, W)
h = 0.005 


EKF.process(state, input, h)
EKF.measure(ContactObservation1, state)
EKF.error_measure_jacobian(ContactObservation1, state)
EKF.error_process_jacobian(state, input, h)
EKF.prediction!(ekf, input, h)
N = 5000

p1s = zeros(N, 3)
p2s = zeros(N,3)
p3s = zeros(N, 3)
qs = zeros(N,4)
rs = zeros(N, 3)
for i in 1:N
	input = ImuInput(randn(6) + [0; 0; 9.81; 0; 0;0 ])
	EKF.prediction!(ekf, input, h)

	p1s[i, :] = ekf.est_state[11:13]
	p2s[i, :] = ekf.est_state[14:16]
	p3s[i, :] = ekf.est_state[17:19]
	qs[i,:] = ekf.est_state[4:7]
	rs[i,:] = ekf.est_state[1:3]
	EKF.update!(ekf, contact1)
	EKF.update!(ekf, contact2)
	EKF.update!(ekf, contact3)

end 
# @benchmark process($state, $input, $h)
# @benchmark error_process_jacobian($state, $input, $h)
# @benchmark prediction!($ekf, $input, dt=$h)
# @benchmark measure($state)
# @benchmark error_measure_jacobian($state)
# @benchmark update!($ekf, $vicon)


# for i in 1:1000
	# @time SVector(1,2,3)
	# @time SVector{2000,Float64}(randn(2000))
	# @time @SVector randn(2000)
	# prediction!(ekf, input, dt=h)
	# t = @time update!(ekf, vicon)
	# if t > 0.0001
	# 	println(i)
	# end
# end 

