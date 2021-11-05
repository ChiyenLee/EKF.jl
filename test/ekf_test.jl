import EKF
using StaticArrays
using Rotations
using Test
using ForwardDiff
using LinearAlgebra


@testset "Mulitplicative Extended Kalman Filter" begin
	struct OriVel{T} <: EKF.State{7, T}
		q𝑤::T; q𝑥::T; q𝑦::T; q𝑧::T
		ω𝑥::T; ω𝑦::T; ω𝑧::T
	end
	struct OriVelErr{T} <: EKF.ErrorState{6, T}
		𝕕q𝑥::T; 𝕕q𝑦::T; 𝕕q𝑧::T
		𝕕ω𝑥::T; 𝕕ω𝑦::T; 𝕕ω𝑧::T
	end
	struct Tor{T} <: EKF.Input{3, T}
		τ𝑥::T; τ𝑦::T; τ𝑧::T
	end
	struct Ori{T} <: EKF.Measurement{4, T}
		q𝑤::T; q𝑥::T; q𝑦::T; q𝑧::T
	end
	struct OriErr{T} <: EKF.ErrorMeasurement{3, T}
		𝕕q𝑥::T; 𝕕q𝑦::T; 𝕕q𝑧::T
	end


	dt = .1
	state = OriVel(1., 0., 0., 0., .1, .1, .2)
	state_err = OriVelErr(.1, 0., 0., .1, .1, .2)
	input = Tor(0.5, 0., 0.)
	meas = Ori(1., 0., 0., 0.)
	meas_cov = @SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:3]
	obs = EKF.Observation(meas, meas_cov)


	@testset "Observations" begin
		meas2 = Ori(sqrt(.5), 0., sqrt(.5), 0.)
		meas_cov2 = @SMatrix [i==j ? .5 : 0. for i = 1:7, j = 1:7]
		obs2 = EKF.Observation(meas, meas_cov)

		@test EKF.getMeasurement(obs2) == meas
		@test EKF.getCovariance(obs2) == meas_cov

		EKF.setMeasurement(obs2, meas2)
		EKF.setCovariance(obs2, meas_cov2)

		@test EKF.getMeasurement(obs2) == meas2
		@test EKF.getCovariance(obs2) == meas_cov2
	end


	@testset "Default Errors" begin
		@test_throws EKF.EkfError EKF.process(state, input, dt)
		@test_throws EKF.EkfError EKF.error_process_jacobian(state, input, dt)
		@test_throws EKF.EkfError EKF.measure(Ori, state)
		@test_throws EKF.EkfError EKF.error_measure_jacobian(Ori, state)
		@test_throws EKF.EkfError EKF.state_composition(state, state_err)
		@test_throws EKF.EkfError EKF.measurement_error(meas, meas)
	end


	function EKF.state_composition(x::OriVel, dx::OriVelErr)
	    q = Rotations.UnitQuaternion(x.q𝑤, x.q𝑥, x.q𝑦, x.q𝑧)
		ω = SA[x.ω𝑥, x.ω𝑦, x.ω𝑧]
		tmp = SA[dx.𝕕q𝑥, dx.𝕕q𝑦, dx.𝕕q𝑧]
		𝕕q = Rotations.RotationError(tmp, Rotations.CayleyMap())
		𝕕ω = SA[dx.𝕕ω𝑥, dx.𝕕ω𝑦, dx.𝕕ω𝑧]
		ori = Rotations.add_error(q, 𝕕q)
		vel = ω + 𝕕ω
		return OriVel(Rotations.params(ori)..., vel...)
	end

	# # Compute the error state between two states
	function EKF.measurement_error(m2::Ori, m1::Ori)
		q₁ = Rotations.UnitQuaternion(m1.q𝑤, m1.q𝑥, m1.q𝑦, m1.q𝑧)
		q₂ = Rotations.UnitQuaternion(m2.q𝑤, m2.q𝑥, m2.q𝑦, m2.q𝑧)
		ori_er = Rotations.rotation_error(q₂, q₁, Rotations.CayleyMap())

		return OriErr(ori_er...)
	end


	@testset "State Composition and Measurement Error" begin
		state1 = OriVel(1., 0., 0., 0., .1, .1, .2)
		state_err1 = OriVelErr(0., 0., 0., 0., 0., 0.)

		@test EKF.state_composition(state1, state_err1) == state1

		meas1 = Ori(1., 0., 0., 0.)
		meas_err1 = OriErr(0., 0., 0.)
		@test EKF.measurement_error(meas1, meas1) == meas_err1
	end


	function EKF.process(state::OriVel, input::Tor, dt::Float64)
		# No torque applied just kinematics
		q′ = Rotations.kinematics(UnitQuaternion(state[1:4]), dt*state[5:7])
		ω = SA[state[5], state[6], state[7]]
		return OriVel([q′;  ω])
	end
	function EKF.error_process_jacobian(xₖ::OriVel, uₖ::Tor, dt::Float64)
		A = ForwardDiff.jacobian(st->EKF.process(OriVel(st), uₖ, dt), SVector(xₖ))
		# Get various compoents
		xₖ₊₁ = EKF.process(xₖ, uₖ, dt)
		qₖ = Rotations.UnitQuaternion(xₖ.q𝑤, xₖ.q𝑥, xₖ.q𝑦, xₖ.q𝑧)
		qₖ₊₁ = Rotations.UnitQuaternion(xₖ₊₁.q𝑤, xₖ₊₁.q𝑥, xₖ₊₁.q𝑦, xₖ₊₁.q𝑧)

		J(q) = [[Rotations.∇differential(q)  @SMatrix zeros(4, 3)];
				@SMatrix [i+3==j ? 1. : 0. for i = 1:3, j = 1:6]];
		Jₖ = J(qₖ); Jₖ₊₁ = J(qₖ₊₁);

		return Jₖ₊₁' * A * Jₖ
	end
	function EKF.measure(::Type{<:Ori}, x::OriVel)
		return Ori(x.q𝑤, x.q𝑥, x.q𝑦, x.q𝑧)
	end
	function EKF.error_measure_jacobian(::Type{<:Ori}, xₖ::OriVel)
		A = ForwardDiff.jacobian(st->EKF.measure(Ori, OriVel(st)), SVector(xₖ))

		qₖ = Rotations.UnitQuaternion(xₖ.q𝑤, xₖ.q𝑥, xₖ.q𝑦, xₖ.q𝑧)
		Jₖ = [[Rotations.∇differential(qₖ)  @SMatrix zeros(4, 3)];
			  @SMatrix [i+3==j ? 1. : 0. for i = 1:3, j = 1:6]];
		ŷ = EKF.measure(Ori, xₖ)
		q̂ = Rotations.UnitQuaternion(ŷ.q𝑤, ŷ.q𝑥, ŷ.q𝑦, ŷ.q𝑧)
		Gₖ = Rotations.∇differential(q̂)

		return Gₖ' * A * Jₖ
	end


	@testset "Filter Functions" begin
		state1 = OriVel(1., 0., 0., 0., .1, .1, .2)
		est_cov = @MMatrix [i==j ? 1.5 : 0. for i = 1:6, j = 1:6]
		process_cov = @MMatrix [i==j ? .3 : 0. for i = 1:6, j = 1:6]

		ekf = EKF.ErrorStateFilter{OriVel, OriVelErr, Tor}(state1, est_cov, process_cov)

		input = Tor(0.5, 0., 0.)
		meas = Ori(1., 0., 0., 0.)
		meas_cov = @SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:3]
		oriObs = EKF.Observation(meas, meas_cov)

		EKF.prediction!(ekf, input, dt)
		EKF.update!(ekf, oriObs)
	end



end

# %%
