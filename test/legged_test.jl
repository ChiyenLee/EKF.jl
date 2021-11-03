using Revise 
using EKF
using LinearAlgebra
using ForwardDiff
using StaticArrays
using Rotations 
# using BenchmarkTools
# include("$(@__DIR__)/legged_states.jl")
include("$(@__DIR__)/legged_utility.jl")


# %%
## Init state and input 
x_stand = [0.9999982624338724, -0.00023974074669699303, 0.0013050801550827176, -0.0013093583927702452, 0.0059343864706899304, -0.0008723071131919254, 0.33093171820437417, 0.004529120639336368, 0.004529120639336368, 0.004529120639336368, 0.0016202404770875548, 0.614106250308167, 0.614106250308167, 0.614106250308167, 0.6100752478388638, -1.1995702781987116, -1.1995702781987116, -1.1995702781987116, -1.1874673846879344]
s_init = zeros(length(LeggedState)); s_init[4] = 1.0
s_init[1:3] = x_stand[5:7]
state = LeggedState(s_init);
input_init = zeros(6); input_init[3] = 9.81; #input_init[1] = 0.2
input = ImuInput(input_init)

s_error = LeggedError(zero(LeggedError))
R = Diagonal(ones(12)) * 1e-3
R1 = @SMatrix [1e-2 0.0 0.0;	
			   0.0 1e-2 0.0;
			   0.0 0.0 1e-2]
## Init Observation 
p = fk(x_stand[8:19])

contact1 = ContactObservation1(ContactMeasure(p[1:3]), 
							   R1)

contact2 = ContactObservation2(ContactMeasure(p[4:6]), 
							   R1)

contact3 = ContactObservation3(ContactMeasure(p[7:9]), 
							   R1)

contact4 = ContactObservation4(ContactMeasure(p[10:12]), 
							   R1)

P = Matrix(1.0I(length(LeggedError))) * 1e-1; 
P[10:21, 10:21] .= I(12) * 1e2
W = Matrix(1.0I(length(LeggedError))) * 1e-1;

ekf = EKF.ErrorStateFilter{LeggedState, LeggedError, ImuInput}(state, P, W)
h = 0.005 


# EKF.process(state, input, h)
# EKF.measure(ContactObservation1{Float64}, state)
# EKF.error_measure_jacobian(ContactObservation1{Float64}, state)
# EKF.error_process_jacobian(state, input, h)
# EKF.prediction!(ekf, input, h)
# EKF.update!(ekf, contact2)
r, q, v, p1, p2, p3 ,p4, α, β = getComponents(LeggedState(ekf.est_state))
N = 5000

p1s = zeros(N, 3)
p2s = zeros(N,3)
p3s = zeros(N, 3)
vs = zeros(N,3)
qs = zeros(N,4)
rs = zeros(N, 3)
for i in 1:N
	input = ImuInput(randn(6)*0.01 + [0; 0; 9.81; 0; 0;0 ])
	EKF.prediction!(ekf, input, h)

	p1s[i, :] = ekf.est_state[11:13]
	p2s[i, :] = ekf.est_state[14:16]
	p3s[i, :] = ekf.est_state[17:19]
	qs[i,:] = ekf.est_state[4:7]
	rs[i,:] = ekf.est_state[1:3]
	vs[i,:] = ekf.est_state[8:10]

	J = dfk(x_stand[8:19])
	J1 = J[1:3,:]
	contact1.measure_cov = SMatrix{3,3,Float64}(J1 * R * J1') 
	J2 = J[4:6,:]
	contact2.measure_cov = SMatrix{3,3,Float64}(J2 * R * J2') 
	J3 = J[7:9,:]
	contact3.measure_cov = SMatrix{3,3,Float64}(J3 * R * J3') 
	J4 = J[10:12,:]
	contact4.measure_cov = SMatrix{3,3,Float64}(J4 * R * J4') 

	# EKF.update!(ekf, contact1)
	EKF.update!(ekf, contact2)
	# EKF.update!(ekf, contact3)
	EKF.update!(ekf, contact4)

end 

