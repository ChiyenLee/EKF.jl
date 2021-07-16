using EKF
using LinearAlgebra

include("$(@__DIR__)/imu_grav_comp/imu_dynamics_discrete.jl")

# %%
state = zeros(TrunkState); state.qw = 1.0
vicon = zeros(Vicon); vicon.qw = 1.0
input = zeros(ImuInput); input.fx = 10

P = Matrix(1.0I(length(TrunkError))) * 1e-4; 
W = Matrix(1.0I(length(TrunkError))) * 1e4;
R = Matrix(1.0I(length(ViconError))) * 1e-3;

q_rand = normalize(randn(4))
vicon = Vicon([0. 0. 0. q_rand...])

ekf = ErrorStateFilter{TrunkState, TrunkError, ImuInput, Vicon, ViconError}(state, P, W, R)

# %%

state_prior, P_prior = prediction(ekf, state, P, input, dt=0.01)
vicon_error, H, L = innovation(ekf, state_prior, P_prior, vicon) # H::measure jacobian, L::Kalman Gain
state_post, P_post = update!(ekf, state_prior, P, vicon_error, H, L)

r, v, q, α, β = getComponents(state_post)
println("r: ", r)
println("q: ", q)
