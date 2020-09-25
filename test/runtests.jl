using MBTR
using Test, LinearAlgebra, BenchmarkTools
using JuLIP, JuLIP.Testing
using MBTR: mbtr

@testset "MBTR.jl" begin
    include("test.jl")
end

