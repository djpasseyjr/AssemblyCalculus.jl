using AssemblyCalculus
using AssemblyCalculus: overlap, freeze_synapses!
using Graphs
using Plots
using StatsBase
using LinearAlgebra: norm

K = 20
N = K^2
MEAN_DEGREE = K
P = MEAN_DEGREE / N
BETA = 0.0
TIMESTEPS = 500

g = erdos_renyi(N, P, is_directed=true)
brain = Brain(g, [N], [K], [BETA])

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


final_z_prob = visit_prob(zspikes, N)
final_stim_prob = visit_prob(spikes, N)

z_prob = mapreduce(i -> visit_prob(zspikes[:, 1:i], N), hcat, 1:TIMESTEPS)
stim_prob = mapreduce(i -> visit_prob(spikes[:, 1:i], N), hcat, 1:TIMESTEPS)

rand_prob = rand(N)
rand_prob ./= sum(rand_prob)
baseline_conv = prob_conv(rand_prob, z_prob)

z_conv = prob_conv(final_z_prob, z_prob)
stim_conv = prob_conv(final_stim_prob, stim_prob)

adj = adjacency_matrix(g)
row_prob = reshape(sum(adj, dims=2) ./ sum(adj), :)

row_current_prob = reshape(sum(adj, dims=2), :) .+ stim.currents
row_current_prob ./= sum(row_current_prob)

row_conv = prob_conv(row_prob, z_prob)
row_current_conv = prob_conv(row_current_prob, stim_prob)
row_stim_conv = prob_conv(row_prob, stim_prob)

eigen_prob = eigenvector_centrality(g)
eigen_prob ./= sum(eigen_prob)
eigen_conv = prob_conv(eigen_prob, z_prob)


p = plot()
line_attr = Dict(:alpha=>0.7, :linewidth=>3)

plot!(p, zeros(TIMESTEPS), linestyle=:dash, color=:gray, label=nothing)

plot!(p, baseline_conv; color=:gray, label="Baseline", line_attr...)

plot!(p, z_conv; label="Zero stim to true distr", line_attr...)
plot!(p, row_conv; label="Zero stim to row sums", line_attr...)
plot!(p, eigen_conv; label="Zero stim to eigen", line_attr...)

plot!(p, stim_conv; label="Stim to true distr", line_attr...)
plot!(p, row_current_conv; label="Stim to row sums + current", line_attr...)
plot!(p, row_stim_conv; label="Stim to row sums", line_attr...)

pb = bar(final_z_prob, alpha=0.5, label="True")
bar!(pb, row_prob, alpha=0.5, label="Row Sum")

plot(p, pb; size=(1200, 400), layout=@layout [a{.25w} b])
