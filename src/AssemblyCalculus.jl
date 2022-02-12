module AssemblyCalculus

    using Distributions: Binomial, Bernoulli
    using Graphs
    using StatsBase
    using ThreadsX

    include("types.jl")
    include("simulation_functions.jl")
    include("partial_neural_area.jl")

    export BrainAreas, Assembly, IonCurrent
    export simulate!, random_firing!, random_current, zero_current

end  # module AssemblyCalculus
