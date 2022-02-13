using AssemblyCalculus
using Graphs
using Plots

function time_sim(assem_size::Int; mean_degree=100)
    n = assem_size^2
    p = mean_degree / n
    β = 0.1
    ba = BrainAreas(num_areas=1, n=n, k=assem_size, p=p, β=β)
    ion_currents = [random_current(ba[1])]
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

plot(neurons, times, linewidth=3, color=:teal, legend=:bottomright)
xticks!(neurons, string.(neurons))
xlabel!("Number of neurons")
ylabel!("Seconds to run 50 iterations (PartialNeuralArea)")
title!("Time Complexity of Generative Synapses")
