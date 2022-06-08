# AssemblyCalculus.jl

This is a library for simulating brain computation according to the
assembly calculus model proposed by
[Papadimitriou et. al..](https://www.pnas.org/content/117/25/14464)
Assembly calculus is built on the assumption that an object is represented
in a brain region by a small number of active neurons in that region called an assembly.
This is called the assembly hypothesis and is a form of sparse encoding.
Papadimitriou et. al. proposed simple biologically plausible mechanisms by which
assemblies in different regions can be associated with eachother to create
computational assembly circuits. This "calculus" is Turing complete.

While very intriguing, the limititations of assembly calculus are not well understood.
The [original assembly calculus](https://github.com/dmitropolsky/assemblies)
package was developed in Python and explicit simulations are very slow. This package
was developed for faster and more extensible assembly calculus simulations. This
package also takes steps towards a simplified API for building brain regions and
running simulations.

## Installation Instructions

Until the package is added to the Julia registry, install by starting the Julia
REPL and typing

```
] add https://github.com/djpasseyjr/AssemblyCalculus.jl
```
or
```
using Pkg
Pkg.add("https://github.com/djpasseyjr/AssemblyCalculus.jl")
```

## Quick Start

Once the package is installed, building brain areas and running simulations is simple
```
using AssemblyCalculus

ba = Brain(num_areas=3, n=10_000, k=100, β=.01, p=.01)
```

The code above will create three brain regions, each containing `n = 10_000` neurons with
an assembly size of `k = 100` neurons, a hebbian plasticity parameter of `β=0.01` and
an edge probability of `p=0.01`.

To run a simulation, create input currents to stimulate the brain areas. The following
code creates stimulus current into the second and third brain areas, and runs a simulation
```
stims = [rand_stim(ba[2]), rand_stim(ba[3])]
assemblies, spikes, convergence = simulate!(stims)
```

The `simulate!` function runs until both brain areas converge on an assembly
or for 50 iterations (since this is the default for the `max_iters` keyword argument).
Once the simulation is finished, the `assemblies` variable contains a list of the
active assemblies in each simulated area. The `spikes` variable is a list of arrays that
contain which neurons spiked at each timestep of the simulation. The `convergence`
variable is an array with dimensions (number of active areas) x (simulation timesteps)
and `convergence[i, j]` contains the number of neurons in the `i`th area that were
active at timestep `j` but were not active at timestep `j - 1`. If this number is zero,
then there was no change in the active assembly (and the area has converged).

## Important Details

The simulation API considers all regions inhibited unless they
receive input current. To disinhibit an area without sending current
to the neurons, use `zero_stim(region)`.

To reduce function arguments, many of the types in `AssemblyCalculus.jl`
contain pointers to each other. This can prevent garbage collection when
hidden references to an object are not deleted.

Here are the relationships between types
1. `Neuron` types point to `Neuron` objects they synapse onto
2. `Neuron` types point to `BrainArea` objects that house `Neuron` objects onto which they synapse.
3. `BrainArea` types point to all `Neuron` objects they house
4. `Stimulus` types point to the `BrainArea` objects that they stimulate
5. `Assembly` types point to the `BrainArea` where they reside
6. `Assembly` types point to their parent `Stimulus` and `Assembly` objects
7. `Brain` types point to all `BrainArea` types they house.

In short, be sure to delete **all** objects related to `AssemblyCalculus.jl`
types you wish to discard or they may inadvertently remain in memory.
