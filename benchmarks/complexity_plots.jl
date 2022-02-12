using AssemblyCalculus
using Graphs
using Plots

function time_sim(assem_size::Int; mean_degree=100)
    assembly_sizes = [assem_size, assem_size]
    area_sizes = assembly_sizes .^ 2
    plasticities = [0.01, 0.01]
    n = sum(area_sizes)
    p = mean_degree / n
    g = erdos_renyi(n, p, is_directed=true)
    ba = BrainAreas(g, area_sizes, assembly_sizes, plasticities)
    g = nothing
    ion_currents = [random_current(ba[i]) for i in 1:2]
    assems = Assembly{Float64}[]
    timesteps = 50
    td = @timed as, sp = simulate!(ion_currents, assems, timesteps)
    return td.time
end

# Precompile
time_sim(5)

assem_sizes = collect(100:50:350)
times = time_sim.(assem_sizes)
neurons = 2 .* assem_sizes .^ 2
β = [neurons ones(size(neurons))] \ times
m = round(β[1], sigdigits=1)
b = round(β[2], sigdigits=2)

plot(neurons, times, linewidth=3, color=:teal, legend=:bottomright)
plot!(neurons, times, smooth=true, label="Best Fit: $m x + $b")
xticks!(neurons, string.(neurons))
xlabel!("Number of neurons")
ylabel!("Seconds to run 50 iterations")
title!("Time Complexity of AssemblyCalculus.jl")
