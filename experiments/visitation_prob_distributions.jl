using AssemblyCalculus
using AssemblyCalculus: overlap, freeze_synapses!, synapse_noise!
using Graphs
using Plots
using StatsBase
using LinearAlgebra: norm

K = 20
N = K^2
MEAN_DEGREE = K
P = MEAN_DEGREE / N
BETA = 0.0
TIMESTEPS = 10_000

g = erdos_renyi(N, P, is_directed=true)
brain = Brain(g, [N], [K], [BETA])
synapse_noise!(brain)

zstim = zero_stim(brain[1])

_, sp1 = simulate!([zstim], TIMESTEPS)
_, sp2 = simulate!([zstim], TIMESTEPS)

spikes1 = sp1[1]
spikes2 = sp2[1]

function visit_prob(neurons::Array{Int, N}, max_idx::Int) where N
    neuron_counts = counts(reshape(neurons, :), max_idx)
    neruon_prob = neuron_counts ./ sum(neuron_counts)
    return neruon_prob
end


prob1 = visit_prob(spikes1, N)
prob2 = visit_prob(spikes2, N)
adj = adjacency_matrix(g)
normalized_row_sums = sum(adj', dims=2) ./ sum(adj)

p1 = bar(prob1, alpha=0.5, label="Run 1")
bar!(p1, prob2, alpha=0.5, label="Run 2")
bar!(p1, normalized_row_sums, alpha=0.5, label="Normalized In-Degree")
xlabel!(p1, "Neuron ID")
ylabel!(p1, "Visitation Probability")
title!(p1, "Neuron Visitation Probability\n Same Network, Different Initial Condition, 10,000 iterations.")

p2 = histogram(prob1, alpha=0.5, label="Run 1")
histogram!(p2, prob2, alpha=0.5, label="Run 2")
histogram!(p2, normalized_row_sums, alpha=0.5, label="Normalized In-Degree")
xlabel!(p2, "Visitation Probaility")
ylabel!(p2, "Number of Neurons")



plot(p1, p2; size=(1200, 400), layout=@layout [a; b])
