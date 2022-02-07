module AssemblyCalculus

    using Distributions: Binomial
    using Graphs
    using StatsBase
    using ThreadsX

    include("types.jl")
    include("simulation_functions.jl")


    export BrainAreas, Assembly, IonCurrent
    export simulate!

end  # module AssemblyCalculus
