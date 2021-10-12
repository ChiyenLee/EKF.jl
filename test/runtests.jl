using Base: Float64
using EKF
using LinearAlgebra
include("imu/imu_states.jl")

u = rand(6)
x = rand(3)
v = rand(3)
# q = rand(4); q = q./sqrt(sum(q.^2))
q = rand(UnitQuaternion)
α = rand(3)
β = rand(3)


x1 = EKF.process(TrunkState(x..., v..., params(q)..., α..., β...),
                 ImuInput(u...), .1)
x2 = EKF.process(ImuState(x..., params(q)..., (q' * v)..., α..., β...),
                 ImuInput2(u...), .1)

p1, v1, q1, α1, β1 = getComponents(x1)
p2, q2, v2, α2, β2 = getComponents(x2)

println(p1)
println(p2)
println(v1)
println(v2)
println(q1)
println(params(q2))
# println(α1)
# println(α2)
# println(β1)
# println(β2)