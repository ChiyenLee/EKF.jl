#
using Pkg
Pkg.activate("$(@__DIR__)/..")

using EKF
using StaticArrays
using BenchmarkTools

include("$(@__DIR__)/imu_states.jl")


state = rand(ImuState)
input = rand(ImuInput)
measurement = rand(ViconMeasurement)
