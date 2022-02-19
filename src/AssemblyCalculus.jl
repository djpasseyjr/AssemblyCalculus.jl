module AssemblyCalculus

    using Distributions: Binomial, Bernoulli
    using Graphs
    using StatsBase
    using ThreadsX

    include("types.jl")
    include("simulation_functions.jl")
    include("partial_neural_area.jl")

    export Brain, Assembly, Stimulus
    export simulate!, random_firing!, rand_stim, zero_stim

end  # module AssemblyCalculus
