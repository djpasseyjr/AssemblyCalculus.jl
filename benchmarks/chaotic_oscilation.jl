"""
A script checking if the oscilatory assemblies are a chaotic attractor.

We see the exponential increase in error even though the initial condition
is only one neuron different.

We see about the same distribution of visitations to each neuron.

Works without making assemblies in the areas! It is just about the assembly
input creating an attractor.

When the assemblies are projected, the attractor actually oscillated through the
assemblies, but this is probably because some of the neurons incorperated into
the projected assemblies already belonged to the attractor and this is why they
were incorperated into the assemblies in the first place.

We are just watching an attractor move through space.

Without an input stimulus, the orbit appears to wander, but with a stimulus,
this wandering is constrained to visit the same nodes with the same frequencies.

Amazing!

Questions:
1. Can we predict the attractor frequencies via undriven network frequencies and
   the stimulus vector? All the stimulus vector does is increase the probability
   that the given neurons will win.
2. If used in combination with the network frequencies will the stimulus vector
   give us the probabilities that the neuron ends up in the attractor?
"""

using AssemblyCalculus
using AssemblyCalculus: overlap, freeze_synapses!
using Plots
using StatsBase

K = 20
N = K^2
MEAN_DEGREE = K
P = MEAN_DEGREE / N

MAX_ITERS = 1000
TOL = floor(Int, 0.95 * K)

brain = Brain(num_areas=3, n=N, k=K, β=0.001, p=P)
assems, sp, ds = simulate!([rand_stim(brain[3])], max_iters=MAX_ITERS)
a0 = assems[1]


# Project alternating pattern across areas 1 and 2
# assems, sp, ds = simulate!([rand_stim(brain[1])], [a0], tol = TOL, max_iters=MAX_ITERS)
# a1 = assems[1]
# assems, sp, ds = simulate!([rand_stim(brain[2])], [a0], tol = TOL, max_iters=MAX_ITERS)
# a2 = assems[1]
# assems, sp, ds = simulate!([rand_stim(brain[1])], [a0], tol = TOL, max_iters=MAX_ITERS)
# a3 = assems[1]
# assems, sp, ds = simulate!([rand_stim(brain[2])], [a0], tol = TOL, max_iters=MAX_ITERS)
# a4 = assems[1]

println("overlap a1, a3: $(overlap(a1, a3))")
println("overlap a2, a4: $(overlap(a2, a4))")

freeze_synapses!(brain)
map(random_firing!, brain.areas[1:2])
init_fire = deepcopy([brain[1].firing, brain[2].firing])

timesteps = 500

assems, sp, ds = simulate!(
   [zero_stim(brain[1]), zero_stim(brain[2])],
   [a0],
   max_iters=timesteps,
   tol=0,
   random_initial=false,
)

delta_init = init_fire
delta_init[1][1] = rand(1:N)
brain[1].firing = delta_init[1]
brain[2].firing = delta_init[2]

assems, sp_delta, ds_delta = simulate!(
   [zero_stim(brain[1]), zero_stim(brain[2])],
   [a0],
   max_iters=timesteps,
   tol=0,
   random_initial=false,
)

function spike_to_vec(firing::Array{Int, 1}, n::Int)
   x = zeros(n)
   x[firing] .= 1.0
   return x
end

function spike_to_vec(spikes::Array{Array{Int, 2}, N}, n::Int) where N
   mapreduce(A -> mapslices(x -> spike_to_vec(x, n), A, dims=1), vcat, spikes)
end

norm(x) = sum(x[x .> 0])
V = spike_to_vec(sp, N)
Vdelta = spike_to_vec(sp_delta, N)
diff = abs.(V .- Vdelta)
shared = V .* Vdelta

err = reshape(mapslices(norm, V - Vdelta, dims=1), :)

p1 = plot()
plot!(
   p1,
   err,
   linewidth=3,
   color=:purple,
   label="Orbit error",
   legend=:bottomright
)
xlabel!("Timestep")
ylabel!("Difference in Assembly")

p2 = plot()
histogram!(p2, reshape(sp[1], :), alpha=0.5, bins=N)
histogram!(p2, reshape(sp_delta[1], :), alpha=0.5, bins=N)

p3 = plot()
histogram!(p3, reshape(sp[2], :), alpha=0.5, bins=N)
histogram!(p3, reshape(sp_delta[2], :), alpha=0.5, bins=N)

plot(p1, p2, p3, size=(1200, 400), layout=@layout [a{0.25w} [b; c]])
