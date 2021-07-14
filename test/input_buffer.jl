using Dates



###############################################################################
#
###############################################################################
mutable struct Buffer{T}
    inputs::Vector{T}
    times::Vector{DateTime}
    # times::Vector{Float64}
    len::Int64

    function Buffer{T}(; size=100) where T
        inputs = Vector{T}(undef, size)
        times = Vector{DateTime}(undef, size)
        # times = Vector{Float64}(undef, len)
        len = 1

        return new{T}(inputs, times, len)
    end
end

function push!(buff::Buffer{T}, input::T, time::DateTime=now()) where T
    if buff.len <= length(buff.inputs)
        buff.inputs[buff.len] = input
        buff.times[buff.len] = time
        # buff.times[N] = datetime2unix(now())
    else
        Base.push!(buff.inputs, input)
        Base.push!(buff.times, time)
        # push!(buff.times, datetime2unix(now()))
    end

    buff.len = buff.len + 1
    return buff
end

function pop!(buff::Buffer{T}, time::DateTime) where T
    n_min = argmin(abs.(buff.times[1:buff.len-1] .- time))
    time_min = buff.times[n_min]
    input_min = buff.inputs[n_min]

    buff.len = 1  # Reset counter
    return input_min
end

function pop!(buff::Buffer{T}, time::Float64) where T  # Unix time version
    return pop!(buff, unix2datetime(time))
end
