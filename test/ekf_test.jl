using Revise 
using EKF
using LinearAlgebra
using ForwardDiff
using BenchmarkTools
include("$(@__DIR__)/imu_grav_comp/imu_dynamics_discrete.jl")

# %%
s_init = zeros(16); s_init[7] = 1.0
state = TrunkState(s_init);
v_init = zeros(7); v_init[4] = 1.0
vicon = Vicon(v_init); 
input_init = zeros(6); input_init[1] = 10
input = ImuInput(input_init)

P = Matrix(1.0I(length(TrunkError))) * 1e-4; 
W = Matrix(1.0I(length(TrunkError))) * 1e4;
R = Matrix(1.0I(length(ViconError))) * 1e-3;

q_rand = normalize(randn(4))
vicon = Vicon([0. 0. 0. q_rand...])

ekf = ErrorStateFilter{TrunkState, TrunkError, ImuInput, Vicon, ViconError}(state, P, W, R)
h = 0.005


# process(state, input, h)
# @benchmark process($state, $input, $h)
# @benchmark error_process_jacobian($state, $input, $h)
@benchmark prediction!($ekf, $input, dt=$h)
# prediction!(ekf, input, dt=h)

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

