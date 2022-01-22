using AssemblyCalculus
using Test

findparam(n::AssemblyCalculus.Neuron{T}) where T = T

@testset "Neuron" begin
    neuronf64 = AssemblyCalculus.Neuron()
    neuronf32 = AssemblyCalculus.Neuron(Float32)
    @test findparam(neuronf64) == Float64
    @test findparam(neuronf32) == Float32
end
