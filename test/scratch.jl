using Revise
import EKF
using StaticArrays
using Rotations
using Test
using ForwardDiff
using LinearAlgebra
using BenchmarkTools

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

@btime begin
    $meas = Ori(sqrt(.5), 0., sqrt(.5), 0.)
    $meas_cov = @SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:3]

    EKF.Observation($meas, $meas_cov)
end

# b = @benchmark EKF.getMeasurement($obs)

# b = @benchmark EKF.getMeasurement($obs)
# @test maximum(b.gctimes) == 0  # no garbage collection
# @test b.memory == 0            # no dynamic memory allocations
# b = @benchmark EKF.getCovariance($obs)
# @test maximum(b.gctimes) == 0  # no garbage collection
# @test b.memory == 0            # no dynamic memory allocations


# b = @benchmark EKF.setMeasurement!($obs, $meas)
# @test maximum(b.gctimes) == 0  # no garbage collection
# @test b.memory == 0            # no dynamic memory allocations
# b = @benchmark EKF.setCovariance!($obs, $meas_cov)
# @test maximum(b.gctimes) == 0  # no garbage collection
# @test b.memory == 0            # no dynamic memory allocations

