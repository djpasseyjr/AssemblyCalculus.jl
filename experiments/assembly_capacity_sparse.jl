"""Script to compute the memory capacity of a AssemblyCalculus.BrainArea
when the network has mean degree 2 or less.

Script projects random stimuli into the brain area and compares assembly that
form in the region. This is an attempt to answer the question, how many
suitably unique assemblies can a region generate.
"""

using AssemblyCalculus
using AssemblyCalculus: overlap, forget_parents!
using Graphs
using Plots

NUM_BRAINS = 3
MAX_ASSEMBLIES = 1000
MAX_ITERS = 100
TOL = 3
N = 400
K = 20
EDGE_PROBS = [.5f0/N, 1f0/N, 2f0/N]
β = 0.01f0

assemblies = [Assembly{Float32}[] for i in 1:NUM_BRAINS]
stims = [Stimulus{Float32}[] for i in 1:NUM_BRAINS]
max_distances = [Int[] for i in 1:NUM_BRAINS]
iters_taken = [Int[] for i in 1:NUM_BRAINS]

for brain_num in 1:NUM_BRAINS
    g = erdos_renyi(N, EDGE_PROBS[brain_num], is_directed=true)
    brain = Brain(g, [N], [K], [β])
    for assem_num in 1:MAX_ASSEMBLIES
        stim = rand_stim(brain[1])
        assems, _, dists = simulate!([stim], tol=TOL, max_iters=MAX_ITERS)
        a = assems[1]
        forget_parents!(a)
        iters = size(dists)[2]
        push!(iters_taken[brain_num], iters)
        if assem_num != 1
            if dists[iters] <= TOL
                common_neurons = map(x -> overlap(a, x), assemblies[brain_num])
                push!(max_distances[brain_num], maximum(common_neurons))
            else
                push!(max_distances[brain_num], -1)
            end
        end
        push!(assemblies[brain_num], a)

    end
end

p1 = plot(legend=:bottomright)
colors = [:teal, :pink, :green, :orange]
for (i, ep) in enumerate(EDGE_PROBS)
    max_ds = max_distances[i]
    mean_max_iters = cumsum(max_ds) ./ collect(1:length(max_ds))
    plot!(p1, mean_max_iters, label="p = $ep", linewidth=3, color=colors[i])
end
xlabel!(p1, "Number of Assemblies Created")
ylabel!(p1, "Average max overlap \n with previous assemblies")

p2 = plot(legend=:bottomright)
for (i, ep) in enumerate(EDGE_PROBS)
    plot!(p2, max_distances[i], label="p = $ep", linewidth=3, color=colors[i])
end
xlabel!(p2, "Number of Assemblies Created")
ylabel!(p2, "Max overlap with \n previous assemblies")

l = @layout [a b]
plot(p1, p2, layout=l, size=(1200, 400))
title!("Assembly capacity for different edge probabilities")
