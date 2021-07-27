#
using Pkg
# if Base.active_project() != "$(dirname(@__DIR__))/Project.toml"
#     Pkg.activate("$(@__DIR__)/..")
# end
Pkg.activate()

# using Revise
# using EKF
using StaticArrays
using BenchmarkTools

# include("$(@__DIR__)/imu_states.jl")

struct MyStruct{S <: SArray{Tuple{S1, S2} where {S1, S2}, T where T, 2, L where L}}
    mat::S
    val::S1

    function MyStruct()

    end
end

# %%

# dt = .1
# state = rand(ImuState)
# stateError = rand(ImuError)
# input = rand(ImuInput)
# measurement = rand(Vicon)

# est_cov = MMatrix{length(ImuError),length(ImuError)}(2.2 * I(length(ImuError)))
# process_cov = MMatrix{length(ImuError),length(ImuError)}(2.2 * I(length(ImuError)))
# measure_cov = MMatrix{length(ViconError),length(ViconError)}(2.2 * I(length(ViconError)))

# est_cov = Matrix(2.2 * I(length(ImuError)))
# process_cov = Matrix(2.2 * I(length(ImuError)))
# measure_cov = Matrix(2.2 * I(length(ViconError)))

# ekf = ErrorStateFilter{ImuState, ImuError, ImuInput, Vicon, ViconError}(state, est_cov, process_cov, measure_cov)

# function main()
#     @btime prediction!($ekf, $input, $dt)
# end

# @btime prediction!($ekf, $input, $dt)
# @code_warntype prediction!(ekf, input, dt)

# var = @btime EKF.state_composition($state, $stateError)
# var = @btime EKF.measurement_error($measurement, $measurement)

# var = @btime EKF.process($state, $input, $dt)
# var = @btime EKF.error_process_jacobian($state, $input, $dt)
# var = @btime EKF.measure($state)
# var = @btime EKF.error_measure_jacobian($state)



