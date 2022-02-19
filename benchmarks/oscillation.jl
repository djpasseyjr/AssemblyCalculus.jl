"""
A script attempting to make oscilatory assemblies in an AssemblyCalculus.Brain

Notes:

First attempt

1. Create a0 in area 3
2. Use four random inputs and a0 to create assemblies a1, a3 in area 1 and a2, a4
in area 2
3. Reciprocal project w/o a0
      a1 <-> a2
      a3 <-> a2
      a3 <-> a4
      a1 <-> a4

4. Run with zero input to area1 and area2 and a0 for 100 timesteps

In most of the simulations I tried, the final result was not oscilation, but
rather a fixed point that encompassed a number of neurons from both possible
area assemblies, but not all.

When I made β very small, and the edge probability very high, I was able to
create some noisy form of oscialting assemblies, but it was rough and at most
only 25% of the assemblies I created were present at a time.

This behavior occurs if the iterations are not allowed to fully converge or if
they do fully converge. It also occurs without step 3!

Updated notes:

Dropped step 3. The simulation appears to create non-noise oscillation
if β is small enough so that the assemblies don't converge too quickly
and they are stopped early. Hypothesis is that β is a matter of resolution
and what really matters is stopping early in the convergence process.

This could be a chaotic attractor!
"""

using AssemblyCalculus
using AssemblyCalculus: overlap, freeze_synapses!
using Plots
using StatsBase

K = 100
N = K^2
MEAN_DEGREE = K
P = MEAN_DEGREE / N

MAX_ITERS = 1000
TOL = floor(Int, 0.15 * K)

brain = Brain(num_areas=3, n=N, k=K, β=0.001, p=P)
assems, sp, ds = simulate!([rand_stim(brain[3])], max_iters=MAX_ITERS)
a0 = assems[1]


# Project alternating pattern across areas 1 and 2
assems, sp, ds = simulate!([rand_stim(brain[1])], [a0], tol = TOL, max_iters=MAX_ITERS)
a1 = assems[1]
assems, sp, ds = simulate!([rand_stim(brain[2])], [a0], tol = TOL, max_iters=MAX_ITERS)
a2 = assems[1]
assems, sp, ds = simulate!([rand_stim(brain[1])], [a0], tol = TOL, max_iters=MAX_ITERS)
a3 = assems[1]
assems, sp, ds = simulate!([rand_stim(brain[2])], [a0], tol = TOL, max_iters=MAX_ITERS)
a4 = assems[1]

println("overlap a1, a3: $(overlap(a1, a3))")
println("overlap a2, a4: $(overlap(a2, a4))")

freeze_synapses!(brain)
T = 200
assems, sp, ds = simulate!(
   [zero_stim(brain[1]), zero_stim(brain[2])],
   [a0],
   max_iters=T,
   tol=0,
)

assems = [a1, a2, a3, a4]
# Repeat spike to match assemby areas
sp = hcat(sp, sp)
assem_presence = [mapslices(x -> overlap(assems[i].neurons, x), sp[i], dims=1) for i in 1:length(assems)]
assem_presence = map(x -> reshape(x, :), assem_presence)

colors = [:teal, :pink, :purple, :orange]
names = ["a1", "a2", "a3", "a4"]
p1 = plot()
for (i, c, name) in zip(1:4, colors, names)
           plot!(p1, assem_presence[i], color=c, label=name, linewidth=3)
end
ylabel!("Area Overlap With Assembly")
xlabel!("Timestep")

p2 = plot()
plot!(p2, ds[1, :], color=colors[1], linewidth=3, label="Area 1")
plot!(p2, ds[2, :], color=colors[2], linewidth=3, label="Area 2")
xlabel!("Timestep")
ylabel!("Change in Area Neurons")

ap1, ap2, ap3, ap4 = assem_presence

p3 = plot()
plot!(crosscor(ap1, ap2), label="a2 Cross Correlation", linewidth=3)
plot!(crosscor(ap1, ap3), label="a3 Cross Correlation", linewidth=3)
plot!(crosscor(ap1, ap4), label="a4 Cross Correlation", linewidth=3)
ylabel!("Cross correlation with a1")
xlabel!("Lags")

p4 = plot()
plot!(crosscor(ap2, ap1), label="a1 Cross Correlation", linewidth=3)
plot!(crosscor(ap2, ap3), label="a3 Cross Correlation", linewidth=3)
plot!(crosscor(ap2, ap4), label="a4 Cross Correlation", linewidth=3)
ylabel!("Cross correlation with a2")
xlabel!("Lags")

plot(p1, p2, p3, p4, size=(1200, 1000), layout=@layout [a b; c d])


# assems_r = [a1r, a2r, a3r, a4r]
# # Repeat spike to match assemby areas
# sp = hcat(sp, sp)
# assem_presence = [mapslices(x -> overlap(assems_r[i].neurons, x), sp[i], dims=1) for i in 1:length(assems)]
#
# colors = [:teal, :pink, :purple, :orange]
# names = ["a1r", "a2r", "a3r", "a4r"]
# p2 = plot()
# for (i, c, name) in zip(1:4, colors, names)
#            plot!(p2, reshape(assem_presence[i], :), color=c, label=name, linewidth=3)
# end
# ylabel!("Area Overlap With Assembly")
# xlabel!("Timestep")
#
# plot(p1, p2, size=(1200, 400), layout=@layout [a b])
