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
TIMESTEPS = 1000

g = erdos_renyi(N, P, is_directed=true)
brain = Brain(g, [N], [K], [BETA])
synapse_noise!(brain)
freeze_synapses!(brain)

stim = rand_stim(brain[1])
zstim = zero_stim(brain[1])

_, zsp = simulate!([zstim], TIMESTEPS)
_, sp = simulate!([stim], TIMESTEPS)

zspikes = zsp[1]
spikes = sp[1]

function visit_prob(neurons::Array{Int, N}, max_idx::Int) where N
    neuron_counts = counts(reshape(neurons, :), max_idx)
    neruon_prob = neuron_counts ./ sum(neuron_counts)
    return neruon_prob
end

function prob_conv(final_prob::Array{T, 1}, probs::Array{T, 2}) where T
    return reshape(mapslices(x -> maximum(abs.(x - final_prob)), probs, dims=1), :)
end

function ktop_compare(x::Array{T, 1}, y::Array{T, 1}) where T
    xperm = sortperm(x)
    yperm = sortperm(y)
    compare = [length(intersect(xperm[1:i], yperm[1:i])) / i for i in 1:length(x)]
    return compare
end

final_z_prob = visit_prob(zspikes, N)
final_stim_prob = visit_prob(spikes, N)

z_prob = mapreduce(i -> visit_prob(zspikes[:, 1:i], N), hcat, 1:TIMESTEPS)
stim_prob = mapreduce(i -> visit_prob(spikes[:, 1:i], N), hcat, 1:TIMESTEPS)

rand_prob = rand(N)
rand_prob ./= sum(rand_prob)
baseline_conv = prob_conv(rand_prob, z_prob)

z_conv = prob_conv(final_z_prob, z_prob)
stim_conv = prob_conv(final_stim_prob, stim_prob)

adj = convert(Array{Float64, 2}, adjacency_matrix(g))
row_prob = reshape(sum(adj', dims=2) ./ sum(adj), :)

row_current_prob = reshape(sum(adj', dims=2), :) .+ stim.currents
row_current_prob ./= sum(row_current_prob)

row_conv = prob_conv(row_prob, z_prob)
row_current_conv = prob_conv(row_current_prob, stim_prob)
row_stim_conv = prob_conv(row_prob, stim_prob)

eigen_prob = eigenvector_centrality(DiGraph(adj'))
eigen_prob ./= sum(eigen_prob)
eigen_conv = prob_conv(eigen_prob, z_prob)


p1 = plot()
line_attr = Dict(:alpha=>0.7, :linewidth=>3)

plot!(p1, zeros(TIMESTEPS), linestyle=:dash, color=:gray, label=nothing)

plot!(p1, baseline_conv; color=:gray, label="Baseline", line_attr...)

plot!(p1, z_conv; label="Zero stim to true distr", line_attr...)
plot!(p1, row_conv; label="Zero stim to row sums", line_attr...)
plot!(p1, eigen_conv; label="Zero stim to eigen", line_attr...)

plot!(p1, stim_conv; label="Stim to true distr", line_attr...)
plot!(p1, row_current_conv; label="Stim to row sums + current", line_attr...)
plot!(p1, row_stim_conv; label="Stim to row sums", line_attr...)

p2 = bar(final_z_prob, alpha=0.5, label="True")
bar!(p2, eigen_prob, alpha=0.5, label="Row Sum")

p3 = plot(legend=:bottomright)
plot!(p3, ktop_compare(eigen_prob, final_z_prob); label="Eigencentrality", line_attr...)
plot!(p3, ktop_compare(row_prob, final_z_prob); label="In-Degree", line_attr...)
plot!(p3, ktop_compare(rand(N), final_z_prob); label="Random", color=:gray, line_attr...)
ylabel!(p3, "Accuracy")
xlabel!(p3, "K Largest")
title!(p3, "Accuracy of Centrality Methods at Predicting the K Most Visited Neurons")
plot(p2, p1, p3; size=(1200, 400), layout=@layout [a; b c])
