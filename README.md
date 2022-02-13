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

ba = BrainAreas(num_areas=3, n=10_000, k=100, β=.01, p=.01)
```

The code above will create two brain regions, each containing `n = 10_000` neurons with 
an assembly size of `k = 100` neurons, a hebbian plasticity parameter of `β=0.01` and
an edge probaility of `p=0.01`.
